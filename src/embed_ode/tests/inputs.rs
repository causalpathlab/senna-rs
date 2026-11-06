use crate::embed_ode::inputs::{
    count_rows, load_data, partition_on, pb_sums, read_branch_times, read_delta, read_pb_time,
};
use crate::embed_ode::model::QMass;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use nalgebra::DMatrix;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| s.into()).collect()
}

/// A backend at `dir/{name}.zarr` of `m` (`rows × cells`).
fn backend(dir: &std::path::Path, name: &str, m: &DMatrix<f32>, rows: &[Box<str>]) -> String {
    let path = dir
        .join(format!("{name}.zarr"))
        .to_string_lossy()
        .into_owned();
    let mut b = create_sparse_from_dmatrix(m, Some(&path), Some(&SparseIoBackend::Zarr)).unwrap();
    let cols: Vec<Box<str>> = (0..m.ncols()).map(|c| format!("c{c}").into()).collect();
    b.register_row_names_vec(rows);
    b.register_column_names_vec(&cols);
    path
}

#[test]
fn a_two_track_backend_is_read_on_its_spliced_rows_by_gene() {
    let rows = names(&[
        "GENE0/count/spliced",
        "GENE0/count/unspliced",
        "GENE1/count/unspliced",
        "GENE1/count/spliced",
        "GENE2/count/unspliced",
    ]);
    let c = count_rows(&rows);
    assert_eq!(c.keep, vec![true, false, false, true, false]);
    assert_eq!(c.genes, names(&["GENE0", "GENE1"]));

    // Without tracks every row is a gene as it is.
    let plain = names(&["GENE0", "GENE1", "GENE2"]);
    let c = count_rows(&plain);
    assert_eq!(c.keep, vec![true; 3]);
    assert_eq!(c.genes, plain);
}

#[test]
fn a_partition_is_put_on_the_datas_cells_by_name_and_compacted() {
    let src = names(&["a", "b", "c", "d"]);
    // The finest level's ode ids.
    let finest = vec![7, 3, 7, 5];
    let data = names(&["d", "x", "a", "c"]);
    let p = partition_on(&finest, &src, &data).unwrap();
    // Sorted ode ids, the cells' local pseudobulks; a cell off the
    // partition has none.
    assert_eq!(p.ids, vec![5, 7]);
    assert_eq!(p.of_cell, vec![Some(0), None, Some(1), Some(1)]);

    // None of the data's cells in it: an error.
    assert!(partition_on(&finest, &src, &names(&["x", "y"])).is_err());
}

#[test]
fn pb_time_is_keyed_by_the_ode_ids_and_gives_every_lineages_duration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.pb_time.parquet");
    let path = path.to_string_lossy();
    // Positions 0..4 of 5, phase k/4; lineage 1's duration 2, lineage 2's 3.
    let pb = names(&["5", "5", "7", "9", "7"]);
    let lineage = names(&["L1", "L2", "L2", "L1", "L1"]);
    let position = vec![1i32, 2, 4, 0, 3];
    let phase: Vec<f32> = position.iter().map(|&k| k as f32 / 4.0).collect();
    let time: Vec<f32> = lineage
        .iter()
        .zip(&phase)
        .map(|(l, u)| u * if l.as_ref() == "L1" { 2.0 } else { 3.0 })
        .collect();
    let q = vec![0.3f32, 0.6, 0.5, 1.0, 0.5];
    write_named_table(
        &path,
        "pb",
        &pb,
        &[
            ("lineage".into(), Column::Str(&lineage)),
            ("position".into(), Column::I32(&position)),
            ("phase".into(), Column::F32(&phase)),
            ("time".into(), Column::F32(&time)),
            ("q".into(), Column::F32(&q)),
        ],
    )
    .unwrap();
    // The data's pseudobulks: ode's 5 and 7, not 9.
    let t = read_pb_time(&path, &[5, 7], 2, 5).unwrap();
    let mut got = t.q.clone();
    got.sort_by_key(|m| (m.pb, m.lineage, m.position));
    assert_eq!(
        got,
        vec![
            QMass {
                pb: 0,
                lineage: 0,
                position: 1,
                q: 0.3
            },
            QMass {
                pb: 0,
                lineage: 1,
                position: 2,
                q: 0.6
            },
            QMass {
                pb: 1,
                lineage: 0,
                position: 3,
                q: 0.5
            },
            QMass {
                pb: 1,
                lineage: 1,
                position: 4,
                q: 0.5
            },
        ]
    );
    let d = t.duration;
    assert!((d[0].unwrap() - 2.0).abs() < 1e-5, "{d:?}");
    assert!((d[1].unwrap() - 3.0).abs() < 1e-5, "{d:?}");

    // A lineage or position off the grid is an error.
    assert!(read_pb_time(&path, &[5, 7], 1, 5).is_err());
    assert!(read_pb_time(&path, &[5, 7], 2, 4).is_err());
}

#[test]
fn branch_times_are_read_by_lineage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.branch_time.tsv");
    std::fs::write(
        &path,
        "lineage\tbranch_time\tbranch_time_sd\trhat\tess\tflagged\n\
         L3\t0.5\t0.1\t1.0\t100\t0\n\
         L2\t0.25\t0.1\t1.0\t100\t0\n",
    )
    .unwrap();
    let path = path.to_string_lossy();
    assert_eq!(read_branch_times(&path, 3).unwrap(), vec![0.25, 0.5]);
    // A lineage missing: an error.
    assert!(read_branch_times(&path, 4).is_err());
}

/// Two files, each its own batch: every cell's spliced reads divided by its
/// batch's effect, scaled back to its own reads, then summed by
/// pseudobulk; the unspliced rows never read.
#[test]
fn pseudobulks_sum_their_cells_spliced_reads_over_the_batch_effect() {
    let dir = tempfile::tempdir().unwrap();
    let rows = names(&[
        "GENE0/count/spliced",
        "GENE0/count/unspliced",
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
    ]);
    // Rows × cells.
    let a = DMatrix::from_row_slice(4, 2, &[2.0, 4.0, 9.0, 9.0, 2.0, 0.0, 9.0, 9.0]);
    let b = DMatrix::from_row_slice(4, 2, &[1.0, 3.0, 9.0, 9.0, 3.0, 1.0, 9.0, 9.0]);
    let files: Vec<Box<str>> = vec![
        backend(dir.path(), "a", &a, &rows).into(),
        backend(dir.path(), "b", &b, &rows).into(),
    ];
    let loaded = load_data(&files, None, false).unwrap();
    assert_eq!(loaded.genes, names(&["GENE0", "GENE1"]));
    let cells = loaded.data.column_names().unwrap();
    assert_eq!(cells.len(), 4);
    let batches = loaded.data.batch_names().unwrap();
    assert_eq!(batches.len(), 2);

    // Gene 0 twice as high in batch 0; gene 1 even.
    let delta_path = dir.path().join("run.delta.parquet");
    let delta_path = delta_path.to_string_lossy();
    let (mut gene, mut batch, mut mean) = (Vec::new(), Vec::new(), Vec::new());
    for r in &rows {
        for (i, bn) in batches.iter().enumerate() {
            gene.push(r.clone());
            batch.push(bn.clone());
            mean.push(if r.starts_with("GENE0") && i == 0 {
                2.0f32
            } else {
                1.0
            });
        }
    }
    write_named_table(
        &delta_path,
        "gene",
        &gene,
        &[
            ("batch".into(), Column::Str(&batch)),
            ("mean".into(), Column::F32(&mean)),
        ],
    )
    .unwrap();
    let delta = read_delta(&delta_path, &loaded.data).unwrap();
    assert_eq!(delta.shape(), (2, 2));

    // Cells are file a's two then file b's; batch 0's cells each their own
    // pseudobulk, batch 1's two together.
    let batch_of = loaded.data.get_batch_membership(0..4);
    assert_eq!(batch_of[0], batch_of[1]);
    assert_ne!(batch_of[0], batch_of[2]);
    let (first, second, base) = if batch_of[0] == 0 {
        (&a, &b, 0)
    } else {
        (&b, &a, 2)
    };
    let of_cell: Vec<Option<usize>> = (0..4)
        .map(|c| Some(if (c >= 2) == (base == 2) { c - base } else { 2 }))
        .collect();
    let y = pb_sums(&loaded.data, &of_cell, 3, 1, Some(&delta)).unwrap();
    let adjust = |s0: f32, s1: f32| {
        let (u0, u1) = (s0 / 2.0, s1);
        let k = (s0 + s1) / (u0 + u1);
        [u0 * k, u1 * k]
    };
    let mut want = vec![0f32; 6];
    for c in 0..2 {
        let v = adjust(first[(0, c)], first[(2, c)]);
        want[c * 2] = v[0];
        want[c * 2 + 1] = v[1];
    }
    want[4] = second[(0, 0)] + second[(0, 1)];
    want[5] = second[(2, 0)] + second[(2, 1)];
    for (g, w) in y.iter().zip(&want) {
        assert!((g - w).abs() < 1e-4, "{y:?} vs {want:?}");
    }

    // With no effect, the reads as they are.
    let raw = pb_sums(&loaded.data, &[Some(0); 4], 1, 3, None).unwrap();
    assert_eq!(
        raw,
        vec![a[(0, 0)] + a[(0, 1)] + b[(0, 0)] + b[(0, 1)], 6.0]
    );
}
