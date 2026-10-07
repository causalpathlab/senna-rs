use super::hvg_row_weights;
use crate::tde::tracks::assign_tracks;
use data_beans::alg::hvg::HvgCliArgs;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use graph_embedding_util as ge;
use nalgebra::DMatrix;

fn boxes(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// A single-file synthetic axis: `rows` x `cells`, one row per `values`
/// entry (must match `rows.len()` x `cells.len()`, row-major).
fn fixture(
    dir: &std::path::Path,
    rows: &[&str],
    cells: &[&str],
    values: &[f32],
) -> ge::UnifiedData {
    let path = dir.join("fixture.zarr");
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_row_slice(rows.len(), cells.len(), values);
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create synthetic backend");
    b.register_row_names_vec(&boxes(rows));
    b.register_column_names_vec(&boxes(cells));
    drop(b);

    ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: vec![path],
        feature_kind: Some(ge::FeatureNameKind::Exact),
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::Disjoint,
        ..Default::default()
    })
    .expect("load_unified_data")
}

/// GENE1's spliced row is flat and all its variance sits on its unspliced
/// row; GENE2's spliced row varies a little; GENE3 is flat.
const ROWS: [&str; 4] = [
    "GENE1/count/spliced",
    "GENE1/count/unspliced",
    "GENE2/count/spliced",
    "GENE3/count/spliced",
];
const CELLS: [&str; 6] = ["C1", "C2", "C3", "C4", "C5", "C6"];
#[rustfmt::skip]
const VALUES: [f32; 24] = [
    // GENE1/count/spliced: flat.
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    // GENE1/count/unspliced: strongly alternating.
    1.0, 30.0, 1.0, 30.0, 1.0, 30.0,
    // GENE2/count/spliced: mildly alternating.
    3.0, 7.0, 3.0, 7.0, 3.0, 7.0,
    // GENE3/count/spliced: flat.
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
];

fn axis(dir: &std::path::Path) -> ge::UnifiedData {
    fixture(dir, &ROWS, &CELLS, &VALUES)
}

fn selection(n_hvg: usize) -> HvgCliArgs {
    HvgCliArgs {
        n_hvg,
        feature_list_file: None,
        must_train_features: None,
    }
}

/// Genes rank on their spliced rows only: the unspliced row's variance does
/// not earn its gene a place.
#[test]
fn genes_rank_on_their_spliced_rows_alone() {
    let dir = tempfile::tempdir().expect("tempdir");
    let unified = axis(dir.path());
    let plan = assign_tracks(&unified.feature_names).expect("assign_tracks");

    let w = hvg_row_weights(&unified, &plan, &selection(1), None)
        .expect("hvg_row_weights")
        .expect("selection is on");

    assert_eq!(
        w,
        vec![0.0, 0.0, 1.0, 0.0],
        "GENE2's spliced row, not GENE1"
    );
}

#[test]
fn only_base_rows_carry_weight() {
    let dir = tempfile::tempdir().expect("tempdir");
    let unified = axis(dir.path());
    let plan = assign_tracks(&unified.feature_names).expect("assign_tracks");

    let w = hvg_row_weights(&unified, &plan, &selection(3), None)
        .expect("hvg_row_weights")
        .expect("selection is on");

    assert_eq!(
        w,
        vec![1.0, 0.0, 1.0, 1.0],
        "the unspliced row never weighs"
    );
}

#[test]
fn selection_off_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let unified = axis(dir.path());
    let plan = assign_tracks(&unified.feature_names).expect("assign_tracks");

    let w = hvg_row_weights(&unified, &plan, &selection(0), None).expect("hvg_row_weights");
    assert_eq!(w, None);
}
