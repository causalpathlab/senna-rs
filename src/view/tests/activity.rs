use super::*;

#[test]
fn below_median_baselines_are_dropped_and_others_kept() {
    let mut scores = vec![1.0, 2.0, 3.0, 4.0];
    let b = [0.1, 5.0, 3.0, f32::NAN];
    assert_eq!(median_of(&b), Some(3.0));
    drop_below_median(&mut scores, Some((&b, 3.0)));
    assert_eq!(scores[0], f32::NEG_INFINITY);
    assert_eq!(scores[1], 2.0);
    assert_eq!(scores[2], 3.0);
    assert_eq!(scores[3], f32::NEG_INFINITY);
    let mut untouched = vec![1.0, 2.0];
    drop_below_median(&mut untouched, None);
    assert_eq!(untouched, vec![1.0, 2.0]);
}

/// An activity over a tiny model: five cells, three features.
fn tiny(topic: bool) -> Activity {
    let names = |p: &str, n: usize| (0..n).map(|i| format!("{p}{i}").into()).collect();
    let cells: Vec<Box<str>> = names("c", 5);
    let z = Mat::from_row_slice(5, 2, &[0.1, 1.0, 0.4, 0.2, 0.9, 0.3, 0.2, 0.8, 0.7, 0.5]);
    let rho = Mat::from_row_slice(3, 2, &[1.0, -0.5, 0.3, 2.0, -1.0, 0.4]);
    let model = if topic {
        Model::Topic {
            log_theta: z.map(f32::ln),
            log_beta: rho.map(|v| v.abs().ln()),
        }
    } else {
        Model::Embedding {
            z,
            rho,
            bias: Some(vec![0.5, 0.1, 0.9]),
        }
    };
    let mut loads = RunLoads::none();
    loads.model = Arc::new(ModelTables::ready(Ok(Expected::new(
        Axis::new(names("GENE", 3)),
        cells,
        model,
    ))));
    Activity::new(
        RunManifest::new(senna::run_manifest::RunKind::Topic, "r"),
        PathBuf::new(),
        loads,
        true,
    )
}

#[test]
fn cached_group_contrasts_equal_the_direct_ones() {
    for topic in [false, true] {
        let mut a = tiny(topic);
        // View order differs from the model's; one cell has no group,
        // one is not in the model.
        let view: Vec<Box<str>> = ["c3", "c0", "c4", "c1", "c2", "cx"]
            .map(Into::into)
            .to_vec();
        let groups = [1, 0, NO_ROW, 1, 0, 0];
        let sums = a.group_sums(0, &view, &groups, 3).unwrap();
        let close = |x: &[f32], y: &[f32]| {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a == b || (a - b).abs() < 1e-5)
        };
        for chosen in [vec![0], vec![1], vec![0, 1]] {
            let mask: Vec<bool> = groups.iter().map(|g| chosen.contains(g)).collect();
            let (direct, _) = a.contrast(0, &view, Some(&mask)).unwrap();
            let direct = direct.clone();
            let (cached, _) = a
                .union_contrast(&sums, |g| chosen.contains(&(g as u32)))
                .unwrap();
            assert!(close(&direct, &cached), "{direct:?} vs {cached:?}");
        }
        // An empty group has no contrast, as before.
        assert!(a.union_contrast(&sums, |g| g == 2).is_err());
        // Per cluster, the rest is the other groups only.
        let (per, _) = a.cluster_contrasts(&sums).unwrap();
        let per = per.clone();
        let grouped: Vec<Box<str>> = ["c3", "c0", "c1", "c2"].map(Into::into).to_vec();
        let (direct, _) = a
            .contrast(1, &grouped, Some(&[false, true, false, true]))
            .unwrap();
        assert!(close(&direct, &per[0]));
        assert!(per[2].is_empty());
    }
}

#[test]
fn cells_near_a_feature_rank_by_its_loading_against_the_average_cell() {
    // GENE0 is ρ = (1, -0.5); z · ρ per cell is -0.4, 0.3, 0.75, -0.2,
    // 0.45, and the average cell scores 0.18.
    let near = tiny(false).near_feature("GENE0", 3).unwrap();
    let got: Vec<&str> = near.iter().map(|(n, _)| n.as_ref()).collect();
    assert_eq!(got, ["c2", "c4", "c1"]);
    assert!((near[0].1 - 0.57).abs() < 1e-5);
    assert!(tiny(false).near_feature("NOPE", 3).is_err());
    assert!(tiny(true).near_feature("GENE0", 3).is_err());
}

#[test]
fn cells_near_a_place_rank_by_direction_as_the_map_is_laid_out() {
    // z per cell: c4 (0.7, 0.5) points nearest (1.4, 1.0)'s way, though
    // c2 (0.9, 0.3) sits closer to (0.7, 0.5) by plain distance.
    let mut a = tiny(false);
    let at = Vector::from_vec(vec![1.4, 1.0]);
    let near = a.cells_near(&at, 2).unwrap();
    assert_eq!(&*near[0].0, "c4");
    assert!(near[0].1.abs() < 1e-6, "{near:?}");
    // Where c1 and c3 point together: the mean of their unit vectors.
    let both = a.cells_direction(["c1", "c3", "not-in-model"]).unwrap();
    let (c1, c3) = (
        unit(Vector::from_vec(vec![0.4, 0.2])),
        unit(Vector::from_vec(vec![0.2, 0.8])),
    );
    assert!((&both - unit(c1 + c3)).norm() < 1e-6);
    assert!((both.norm() - 1.0).abs() < 1e-6);
    assert!(a.cells_direction(["cx"]).is_err());
}

#[test]
fn features_near_a_cluster_rank_by_its_mean_against_the_average_cell() {
    // Cells c1, c2 average z = (0.65, 0.25); the average cell is
    // (0.46, 0.56), so the cluster leans (0.19, -0.31). ρ · that is
    // 0.345 for GENE0, -0.563 for GENE1, -0.314 for GENE2; GENE1's
    // baseline (0.1) is below the median (0.5), so it is left out.
    let cluster = ["c1", "c2", "not-in-model"];
    let near = tiny(false).near_cells(cluster, 3).unwrap();
    let got: Vec<&str> = near.iter().map(|(n, _)| n.as_ref()).collect();
    assert_eq!(got, ["GENE0", "GENE2"]);
    assert!((near[0].1 - 0.345).abs() < 1e-5, "{near:?}");
    assert!(tiny(false).near_cells(["cx"], 3).is_err());
    assert!(tiny(true).near_cells(cluster, 3).is_err());
}

#[test]
fn covariance_and_quadratic_forms_match_by_hand() {
    // Two dims; rows 0..3. Only dim 0 varies: values 0, 2, 4.
    let m = Mat::from_row_slice(3, 2, &[0.0, 1.0, 2.0, 1.0, 4.0, 1.0]);
    let all = [0, 1, 2];
    let mean = row_mean(&m, &all);
    assert!((mean[0] - 2.0).abs() < 1e-6 && (mean[1] - 1.0).abs() < 1e-6);
    let c = row_cov(&m, &all);
    assert!((c[(0, 0)] - 4.0).abs() < 1e-5);
    assert!(c[(1, 1)].abs() < 1e-6 && c[(0, 1)].abs() < 1e-6);
    // A feature loading only on dim 0 has variance ρ² · 4; one on dim 1 has none.
    let rho = Mat::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 3.0]);
    let q = quadratic_forms(&rho, &c);
    assert!((q[0] - 1.0).abs() < 1e-5);
    assert!(q[1].abs() < 1e-6);
}

/// Write `rows × k` table `file` into `dir`, rows named `{p}{i}`.
fn table(dir: &Path, file: &str, p: &str, data: &[f32], k: usize) {
    let rows: Vec<Box<str>> = (0..data.len() / k)
        .map(|i| format!("{p}{i}").into())
        .collect();
    let cols: Vec<Box<str>> = (0..k).map(|j| format!("T{j}").into()).collect();
    Mat::from_row_slice(rows.len(), k, data)
        .to_parquet_with_names(
            &dir.join(file).to_string_lossy(),
            (Some(&rows), Some("name")),
            Some(&cols),
        )
        .unwrap();
}

/// A run of `kind` whose `latent` and `dictionary` are the given tables.
fn run_of(
    kind: senna::run_manifest::RunKind,
    dir: &Path,
    latent: &[f32],
    dict: &[f32],
) -> Activity {
    table(dir, "r.latent.parquet", "c", latent, 2);
    table(dir, "r.dictionary.parquet", "GENE", dict, 2);
    let mut m = RunManifest::new(kind, "r");
    m.outputs.latent = Some("r.latent.parquet".into());
    m.outputs.dictionary = Some("r.dictionary.parquet".into());
    let loads = RunLoads::new(&m, dir);
    Activity::new(m, dir.to_path_buf(), loads, true)
}

#[test]
fn a_vae_run_predicts_from_its_latent_and_loadings() {
    // π = softmax_d(z · W + b): the latent and the dictionary are z and W.
    let dir = tempfile::tempdir().unwrap();
    let z = [0.5, -1.0, 2.0, 0.0];
    let w = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut a = run_of(senna::run_manifest::RunKind::Vae, dir.path(), &z, &w);
    let e = a.expected().unwrap();
    let Model::Embedding {
        z: got_z,
        rho,
        bias,
    } = &e.model
    else {
        panic!("a vae run is z · W");
    };
    assert_eq!(got_z.as_slice(), Mat::from_row_slice(2, 2, &z).as_slice());
    assert_eq!(rho.as_slice(), Mat::from_row_slice(3, 2, &w).as_slice());
    assert!(bias.is_none());
}

#[test]
fn a_masked_vae_run_reads_its_latent_through_a_softmax() {
    // Its latent is the raw z; the decoder reads log θ = log_softmax(z)
    // against the log β of its dictionary.
    let dir = tempfile::tempdir().unwrap();
    let z = [0.5, -1.0, 2.0, 0.0];
    let log_beta = [-0.1, -2.0, -3.0, -0.2, -2.5, -2.6];
    let mut a = run_of(
        senna::run_manifest::RunKind::MaskedVae,
        dir.path(),
        &z,
        &log_beta,
    );
    let e = a.expected().unwrap();
    let Model::Topic { log_theta, .. } = &e.model else {
        panic!("a masked-vae run is a topic model of softmax(z)");
    };
    for r in 0..2 {
        let total: f32 = log_theta.row(r).iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "row {r} sums to {total}");
    }
    // The softmax keeps the order within a cell.
    assert!(log_theta[(0, 0)] > log_theta[(0, 1)]);
}

#[test]
fn a_run_with_no_model_of_expression_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = run_of(
        senna::run_manifest::RunKind::Svd,
        dir.path(),
        &[1.0; 4],
        &[1.0; 6],
    );
    let Err(e) = a.expected() else {
        panic!("svd has no model of expression");
    };
    assert!(e.contains("observed"), "{e}");
}

#[test]
fn a_feature_shows_only_where_its_source_has_values_for_the_view() {
    let mut a = tiny(false);
    let listed: Vec<(Box<str>, f32)> = vec![("GENE0".into(), 1.0), ("NOPE".into(), 0.5)];
    let view: Vec<Box<str>> = ["c0", "c3"].map(Into::into).to_vec();
    // Not in the model: nothing to draw.
    assert_eq!(
        a.drawable(&listed, Source::Expected, 0, &view),
        [true, false]
    );
    // None of the view's cells is in the model: missing everywhere.
    let elsewhere: Vec<Box<str>> = ["x0", "x1"].map(Into::into).to_vec();
    assert_eq!(
        a.drawable(&listed, Source::Expected, 1, &elsewhere),
        [false, false]
    );
}

/// Names as boxed strings.
fn boxed(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&n| n.into()).collect()
}

/// A backend at `path`: genes `GENE1..3`, cells `columns`; cell `i`
/// has `i + 1` reads of GENE1 and, on even cells, 3 of GENE3.
fn counts_file(path: &Path, columns: &[&str]) {
    use senna::sparse_output::{write_backend, Entries};
    let mut triplets = Vec::new();
    for i in 0..columns.len() as u64 {
        triplets.push((0, i, (i + 1) as f32));
        if i % 2 == 0 {
            triplets.push((2, i, 3.0));
        }
    }
    write_backend(
        path.to_str().unwrap(),
        Entries::Triplets(triplets),
        &boxed(&["GENE1", "GENE2", "GENE3"]),
        &boxed(columns),
    )
    .unwrap();
}

#[test]
fn an_extra_file_shows_features_and_per_cell_stats_on_the_run_s_cells() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("other.zarr.zip");
    // The file's columns in another order, one run cell missing, one
    // column not the run's.
    counts_file(&path, &["c2", "c0", "zz", "c1"]);
    let run = boxed(&["c0@lib", "c1@lib", "c2@lib", "c3@lib"]);
    let read = read_extra(&[path], &run).unwrap();
    assert_eq!(
        (read.matched(), read.of, &*read.name),
        (3, 4, "other.zarr.zip")
    );

    let mut a = Activity::new(
        RunManifest::new(senna::run_manifest::RunKind::Svd, "r"),
        dir.path().to_path_buf(),
        RunLoads::none(),
        true,
    );
    a.set_extra(Arc::new(ExtraFiles::ready(Ok(read))));
    // Points in the run's own order; the last has no column.
    let (l, spelled) = a.levels("gene1", Source::Extra, 0, &run).unwrap();
    assert_eq!(&*spelled, "GENE1");
    // Columns c2, c0, zz, c1 hold 1, 2, 3, 4 reads of GENE1.
    let want = [2f32, 4.0, 1.0].map(f32::ln_1p);
    assert_eq!(&l.values[..3], &want);
    assert!(l.values[3].is_nan() && l.zero_is_off);

    let total = a
        .stat_levels(&Stat::Total, &[], Source::Extra, 0, &run)
        .unwrap();
    // c0 (column 1) has 2 reads only; c2 (column 0) has 1 + 3.
    assert_eq!(total.values[0], 2f32.ln_1p());
    assert_eq!(total.values[2], 4f32.ln_1p());
    let detected = a
        .stat_levels(&Stat::Detected, &[], Source::Extra, 0, &run)
        .unwrap();
    assert_eq!(&detected.values[..3], &[1.0, 1.0, 2.0]);
    let set = boxed(&["GENE3", "NOT_THERE"]);
    let share = a
        .stat_levels(&Stat::Share("CT1".into()), &set, Source::Extra, 0, &run)
        .unwrap();
    assert_eq!(&share.values[..3], &[0.0, 0.0, 0.75]);
    // The model's expectation has no counts to summarise.
    assert!(a
        .stat_levels(&Stat::Total, &[], Source::Expected, 0, &run)
        .is_err());
}

#[test]
fn several_extra_files_are_read_together() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (dir.path().join("a.zarr.zip"), dir.path().join("b.zarr.zip"));
    counts_file(&a, &["c0", "c1"]);
    counts_file(&b, &["c2", "zz"]);
    let run = boxed(&["c0", "c1", "c2", "c3"]);
    let read = read_extra(&[a, b], &run).unwrap();
    assert_eq!(
        (read.matched(), read.of, &*read.name),
        (3, 4, "a.zarr.zip + 1 more")
    );
}

#[test]
fn a_file_with_none_of_the_run_s_cells_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("other.zarr.zip");
    counts_file(&path, &["q0", "q1"]);
    let Err(e) = read_extra(&[path], &boxed(&["c0", "c1"])) else {
        panic!("a file with none of the run's cells was read");
    };
    assert!(e.to_string().contains("none of the run's 2 cells"), "{e}");
}

#[test]
fn a_scene_opens_an_extra_file_and_steps_through_its_stats() {
    use crate::view::Pick;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("other.zarr.zip");
    counts_file(&path, &["c1", "c2", "c3", "c4"]);
    let mut s = crate::view::tests::scene();
    s.data.run = Some((
        RunManifest::new(senna::run_manifest::RunKind::Svd, "r"),
        dir.path().to_path_buf(),
    ));
    s.open_extra(&[path]);
    // Shown at once, as each cell's total.
    assert_eq!(s.source, Source::Extra);
    assert_eq!(s.pick, Some(Pick::Stat(Stat::Total)));
    let shown = s.shown.as_ref().expect("totals shown");
    assert!(shown.levels.values.iter().all(|v| v.is_finite()));
    s.cycle_stat();
    assert_eq!(s.pick, Some(Pick::Stat(Stat::Detected)));
    // No group in focus: no markers' share, back to group colours.
    s.cycle_stat();
    assert_eq!(s.pick, None);
    // `o` goes through the extra source, then back to the model.
    s.source = Source::Expected;
    s.cycle_source();
    s.cycle_source();
    assert_eq!(s.source, Source::Extra);
    s.cycle_source();
    assert_eq!(s.source, Source::Expected);
}

#[test]
fn the_run_s_own_totals_are_read_on_the_side() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.zarr.zip");
    counts_file(&path, &["c0", "c1"]);
    let cells = boxed(&["c0", "c1"]);
    let observed = read_extra(&[path], &cells).unwrap().observed;
    // Read but not yet summed: `read_extra` sums its own file at once.
    let observed = Observed::new(observed.data, observed.cell_index).unwrap();
    let mut loads = RunLoads::none();
    loads.files = Arc::new(RunFiles::ready(Ok(RunRead {
        observed,
        batches: None,
    })));
    loads.totals = totals_of(&loads.files);
    let mut a = Activity::new(
        RunManifest::new(senna::run_manifest::RunKind::Svd, "r"),
        dir.path().to_path_buf(),
        loads.clone(),
        false,
    );
    // The screen does not wait: asked first, the totals are coming,
    // unless the worker has already summed so small a file.
    let first = a.stat_levels(&Stat::Total, &[], Source::Observed, 0, &cells);
    if let Err(e) = &first {
        assert_eq!(e, legume_tui::work::LOADING);
    }
    assert!(loads.totals.get().is_ok());
    let total = a
        .stat_levels(&Stat::Total, &[], Source::Observed, 0, &cells)
        .unwrap();
    assert_eq!(total.values, [4f32.ln_1p(), 2f32.ln_1p()]);
}
