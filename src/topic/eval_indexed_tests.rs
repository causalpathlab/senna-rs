//! The masked eval's dense block read, its concurrency, and its hold-out draw.

//////////////////////////////////////////////////////////////
// The dense block, and how many of them run at once         //
//////////////////////////////////////////////////////////////

/// One block of the window-free read equals the straightforward dense
/// conversion of the same sparse columns.
///
/// Three things a scatter gets wrong quietly, all planted here: an **empty
/// column** (a cell with nothing in it must come out as a zero row, not as the
/// previous cell's row or a shifted one), a column whose **nonzeros are out of
/// row order** (the block is built by scattering, so nothing may depend on the
/// order they arrive in), and a **remap that sends two query genes to one
/// training gene** (the fill has to ADD, or the second silently replaces the
/// first). The reference below is the obvious `[n, D_train]` fill, written out
/// so the comparison is against arithmetic rather than against another copy of
/// the same scatter.
#[test]
fn a_dense_block_equals_the_straightforward_conversion() {
    use crate::candle_core::Device;
    use nalgebra_sparse::CscMatrix;

    const D_QUERY: usize = 6;
    const D_TRAIN: usize = 4;
    const N: usize = 4;

    // Column 1 is empty; column 2's rows are given out of order; column 3
    // repeats nothing but lands on a remapped gene.
    let col_offsets = vec![0, 3, 3, 6, 8];
    let row_indices = vec![
        0, 2, 5, /* col 2, unsorted: */ 4, 1, 0, /* col 3: */ 3, 5,
    ];
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_dn =
        CscMatrix::try_from_unsorted_csc_data(D_QUERY, N, col_offsets, row_indices, values.clone())
            .expect("a valid CSC block");

    // Two query genes (0 and 5) land on training gene 0; query gene 4 is not on
    // the training axis at all and must be dropped, not folded into gene 0.
    let remap: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3), None, Some(0)];

    // `None` means the columns are already on the training axis, so it is read
    // at the block's own width; the remap is what crosses axes.
    for (map, d_out) in [(None, D_QUERY), (Some(remap.as_slice()), D_TRAIN)] {
        // The reference: walk the triplets and add, exactly as the definition
        // reads, with no tensor in sight.
        let mut want = vec![vec![0.0f32; d_out]; N];
        for (r, c, v) in x_dn.triplet_iter() {
            let g = match map {
                Some(m) => m[r],
                None => Some(r),
            };
            if let Some(g) = g {
                want[c][g] += v;
            }
        }

        let got: Vec<Vec<f32>> = super::csc_block_to_dense(&x_dn, d_out, map, &Device::Cpu)
            .expect("the block builds")
            .to_vec2()
            .expect("[n, D_out]");

        assert_eq!(got.len(), N, "one row per cell in the block");
        assert_eq!(
            got,
            want,
            "the block is not the dense conversion (remap: {})",
            map.is_some()
        );
        assert!(
            got[1].iter().all(|&v| v == 0.0),
            "the empty column must come out as a zero row"
        );
    }

    // Reading an off-axis block without a remap is a wiring mistake, not
    // something to index past the end of the buffer for.
    let msg = super::csc_block_to_dense(&x_dn, D_TRAIN, None, &Device::Cpu)
        .expect_err("D_query > D_train with no remap is not a readable block")
        .to_string();
    assert!(
        msg.contains("gene_remap"),
        "the refusal must name what is missing; got: {msg}"
    );

    // ... and the planted cases are actually present, or the test proves nothing.
    let nnz_per_col: Vec<usize> = (0..N).map(|j| x_dn.col(j).nnz()).collect();
    assert!(nnz_per_col.contains(&0), "no empty column was planted");
    let with_remap: Vec<Vec<f32>> =
        super::csc_block_to_dense(&x_dn, D_TRAIN, Some(&remap), &Device::Cpu)
            .unwrap()
            .to_vec2()
            .unwrap();
    assert_eq!(
        with_remap[0][0], 4.0,
        "query genes 0 and 5 of cell 0 must ADD onto training gene 0"
    );
}

/// The dense read holds a handful of blocks at once.
///
/// A dense block's `[n, D]` working set is the encoder chain over it, and its
/// own matmul already asks rayon for every core — so running one per thread
/// costs more than it buys (measured: 395 s of summed block wall wide open
/// against 19 s held at 8).
#[test]
fn the_dense_read_bounds_how_many_blocks_run_at_once() {
    use crate::candle_core::Device;

    let dense = super::masked_block_concurrency(&Device::Cpu, 100, 34008);
    assert!(
        (1..=super::DENSE_BLOCKS_IN_FLIGHT).contains(&dense),
        "a whole-transcriptome dense block must be capped, got {dense}"
    );
}

/////////////////////////////////////////////////
// The query head is no longer wired anywhere  //
/////////////////////////////////////////////////

/// A model trained with a query decoder cannot be scored by this build.
///
/// Its `dec_query.*` tensors ride in the checkpoint, and `VarMap::load` fills
/// only the vars a rebuild has already registered — anything else is skipped in
/// silence. With the head unwired nothing registers them, so scoring such a
/// model would quietly use a different rate from the one it was trained on.
/// Refuse by name instead.
#[test]
fn a_model_with_a_query_head_is_refused_by_name() {
    use crate::topic::model_metadata::ensure_query_head_not_wired;

    assert!(ensure_query_head_not_wired(None).is_ok());
    assert!(
        ensure_query_head_not_wired(Some(0)).is_ok(),
        "rank 0 is no head at all"
    );
    let msg = ensure_query_head_not_wired(Some(32))
        .expect_err("a model whose weights would be silently skipped must be refused")
        .to_string();
    for needle in ["query", "32"] {
        assert!(
            msg.contains(needle),
            "the message must name {needle}; got: {msg}"
        );
    }
}

/// `query_rank` stays deserialisable: an old model records it, this build reads
/// it back unchanged (so it can refuse), and a model written now has none.
#[test]
fn the_query_rank_round_trips_with_and_without_a_value() {
    use crate::topic::model_metadata::TopicModelMetadata;

    let base = TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_MASKED_VAE.into(),
        decoder_types: vec!["nb".into()],
        decoder_weights: vec![1.0],
        n_features_encoder: 6,
        n_features_full: 6,
        n_topics: 2,
        encoder_hidden: vec![8],
        num_levels: 1,
        level_decoder_dims: vec![6],
        adj_method: "residual".into(),
        batch_null_in_training: false,
        has_coarsening: false,
        embedding_dim: Some(4),
        theta_mean: None,
        n_train_cells: None,
        n_gene_modules: None,
        enc_context_size: None,
        query_rank: None,
    };
    let dir = tempfile::tempdir().unwrap();

    let plain = dir.path().join("plain").to_string_lossy().into_owned();
    base.save(&plain).unwrap();
    assert_eq!(
        TopicModelMetadata::load(&plain).unwrap().query_rank,
        None,
        "a model this build writes has no query head"
    );

    let old = dir.path().join("old").to_string_lossy().into_owned();
    let mut m = base;
    m.query_rank = Some(32);
    m.save(&old).unwrap();
    let back = TopicModelMetadata::load(&old).unwrap();
    assert_eq!(
        back.query_rank,
        Some(32),
        "the field has to survive, or the refusal cannot fire"
    );
    assert!(crate::topic::model_metadata::ensure_query_head_not_wired(back.query_rank).is_err());
}

/// The seeded hold-out draw, against the loop it replaced, kept inline as the
/// reference: the helper has to reproduce it exactly, RNG stream included, or
/// a held-out number moves.
#[test]
fn the_holdout_mask_matches_its_reference_draw() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    // A planted buffer with zeros scattered through it: a zero is never held
    // out and must never consume a draw, which is what pins the RNG stream.
    let values: Vec<f32> = (0..37)
        .map(|i| {
            if i % 4 == 0 {
                0.0
            } else {
                (i % 7) as f32 + 0.5
            }
        })
        .collect();
    let seed = 42u64;
    let lb = 512u64;
    let rate = 0.4;

    // The draw as it was written: 1 everywhere, 0 on a held-out slot.
    let dense_ref = {
        let mut rng = StdRng::seed_from_u64(seed ^ lb);
        let mut vis = vec![1f32; values.len()];
        for (slot, &v) in values.iter().enumerate() {
            if v > 0.0 && rng.random::<f64>() < rate {
                vis[slot] = 0.0;
            }
        }
        vis
    };
    let held = super::seeded_holdout_mask(&values, seed ^ lb, rate);
    let vis: Vec<f32> = held.iter().map(|m| 1.0 - m).collect();
    assert_eq!(vis, dense_ref, "the dense arm's visible mask");
    assert!(
        held.iter().any(|&m| m > 0.0) && held.contains(&0.0),
        "a fixture that holds out everything or nothing proves nothing"
    );
}
