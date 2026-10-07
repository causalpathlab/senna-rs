use super::*;
use data_beans::aux::feature_types::write_feature_types;
use graph_embedding_util::{LoraSpec, PresetMode};
use legume_numeric::matrix::traits::IoOps;
use nalgebra::DMatrix;

/// A `senna fne`-shaped run: genes, one term and one word share the embedding
/// table, and `feature_types.parquet` says which is which.
fn write_fne_like(dir: &std::path::Path, with_types: bool) -> String {
    let prefix = dir.join("run").to_string_lossy().into_owned();
    let names: Vec<Box<str>> = ["TP53", "GATA1", "GO:0006915", "apoptosis"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let mut e = DMatrix::<f32>::zeros(4, 3);
    for i in 0..4 {
        for k in 0..3 {
            e[(i, k)] = (i * 3 + k) as f32 * 0.5 - 1.0;
        }
    }
    e.to_parquet_with_names(
        &format!("{prefix}.feature_embedding.parquet"),
        (Some(&names), Some("feature")),
        None,
    )
    .unwrap();
    if with_types {
        let types: Vec<Box<str>> = ["gene", "gene", "term", "word"]
            .iter()
            .map(|s| Box::from(*s))
            .collect();
        write_feature_types(&prefix, &names, &types).unwrap();
    }
    prefix
}

#[test]
fn gene_rows_match_the_axis_by_canonical_name_and_other_types_are_left_out() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    // The word `apoptosis` also names a feature on this axis; only the GENE row
    // of the source may pin anything.
    let axis: Vec<Box<str>> = ["ENSG1_GATA1", "ENSG2_MYC", "apoptosis", "ENSG3_TP53"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let f = load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' },
    )
    .unwrap()
    .0;
    assert_eq!(f.ids, vec![0, 3]);
    // GATA1 is source row 1, TP53 source row 0.
    assert_eq!(f.rows, vec![0.5, 1.0, 1.5, -1.0, -0.5, 0.0]);
}

#[test]
fn without_a_types_table_every_row_is_a_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), false);
    let axis: Vec<Box<str>> = ["apoptosis", "ENSG3_TP53"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let f = load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' },
    )
    .unwrap()
    .0;
    assert_eq!(f.ids, vec![0, 1]);
}

#[test]
fn no_matching_gene_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    let axis: Vec<Box<str>> = vec![Box::from("ENSG2_MYC")];
    assert!(load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' }
    )
    .is_err());
}

#[test]
fn the_mode_is_carried_and_a_rank_the_table_cannot_hold_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    let axis: Vec<Box<str>> = vec![Box::from("ENSG3_TP53")];
    let kind = ge::FeatureNameKind::Gene { delim: '_' };
    for mode in [
        PresetMode::Freeze,
        PresetMode::Init,
        PresetMode::Lora(LoraSpec {
            rank: 2,
            lr_ratio: 16.0,
            ridge: 0.0,
        }),
    ] {
        assert_eq!(
            load_preset_genes(&prefix, mode, &axis, &kind)
                .unwrap()
                .0
                .mode,
            mode
        );
    }
    // The table is H = 3 wide: rank 3 is no residual at that width, but is
    // low-rank once the run trains wider.
    let rank3 = || {
        Some(
            load_preset_genes(
                &prefix,
                PresetMode::Lora(LoraSpec {
                    rank: 3,
                    lr_ratio: 1.0,
                    ridge: 0.0,
                }),
                &axis,
                &kind,
            )
            .unwrap()
            .0,
        )
    };
    assert!(resolve_dim(ge::EmbeddingDim::Auto, &mut rank3(), &mut None).is_err());
    assert_eq!(
        resolve_dim(ge::EmbeddingDim::Fixed(8), &mut rank3(), &mut None).unwrap(),
        8
    );
}

/// Under a pinning mode the source rows that matched nothing come back to be
/// carried through: genes the axis lacks and every non-gene row, each with
/// its type, in source order — except a row whose name the axis already
/// holds. Under init nothing is carried.
#[test]
fn the_unused_rows_are_carried_under_a_pinning_mode_only() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    let kind = ge::FeatureNameKind::Gene { delim: '_' };
    let lora = PresetMode::Lora(LoraSpec {
        rank: 1,
        lr_ratio: 1.0,
        ridge: 0.0,
    });

    let axis: Vec<Box<str>> = vec![Box::from("ENSG3_TP53")];
    for mode in [PresetMode::Freeze, lora] {
        let (_, carried) = load_preset_genes(&prefix, mode, &axis, &kind).unwrap();
        let c = carried.expect("three rows are left over");
        let names: Vec<&str> = c.names.iter().map(AsRef::as_ref).collect();
        assert_eq!(names, ["GATA1", "GO:0006915", "apoptosis"]);
        let types: Vec<&str> = c.types.iter().map(AsRef::as_ref).collect();
        assert_eq!(types, ["gene", "term", "word"]);
        assert_eq!(c.rows.nrows(), 3);
        assert_eq!(
            c.rows.row(0).iter().copied().collect::<Vec<f32>>(),
            vec![0.5, 1.0, 1.5],
            "GATA1 is source row 1"
        );
        assert_eq!(c.source, format!("{prefix}.feature_embedding.parquet"));
    }
    assert!(load_preset_genes(&prefix, PresetMode::Init, &axis, &kind)
        .unwrap()
        .1
        .is_none());

    // The word `apoptosis` also names a feature of this axis: not carried, or
    // the output would hold that name twice.
    let axis: Vec<Box<str>> = ["ENSG1_GATA1", "apoptosis", "ENSG3_TP53"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let (_, carried) = load_preset_genes(&prefix, PresetMode::Freeze, &axis, &kind).unwrap();
    let c = carried.unwrap();
    assert_eq!(c.names, vec![Box::from("GO:0006915")]);
    assert_eq!(c.types, vec![Box::from("term")]);

    // Everything matched or taken: nothing to carry.
    let axis: Vec<Box<str>> = ["GATA1", "apoptosis", "TP53", "GO:0006915"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let (_, carried) = load_preset_genes(&prefix, PresetMode::Freeze, &axis, &kind).unwrap();
    assert!(carried.is_none());
}

/// Without a types table every source row is a gene, so the carried rows are
/// typed `gene`.
#[test]
fn carried_rows_of_an_untyped_source_are_genes() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), false);
    let axis: Vec<Box<str>> = vec![Box::from("ENSG3_TP53")];
    let (_, carried) = load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' },
    )
    .unwrap();
    let c = carried.unwrap();
    assert_eq!(c.names.len(), 3);
    assert!(c.types.iter().all(|t| t.as_ref() == "gene"));
}

/// The flag may name the run's table or its manifest instead of its prefix,
/// as `senna run`'s picker fills it: the types table is still found.
#[test]
fn the_types_are_read_when_the_flag_names_a_table_or_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    let axis: Vec<Box<str>> = vec![Box::from("ENSG3_TP53")];
    let kind = ge::FeatureNameKind::Gene { delim: '_' };
    for given in [
        format!("{prefix}.feature_embedding.parquet"),
        format!("{prefix}.senna.json"),
    ] {
        let (_, carried) = load_preset_genes(&given, PresetMode::Freeze, &axis, &kind).unwrap();
        let c = carried.unwrap();
        let types: Vec<&str> = c.types.iter().map(AsRef::as_ref).collect();
        assert_eq!(types, ["gene", "term", "word"], "{given}");
    }
}

fn write_run_table(prefix: &str, suffix: &str, row_axis: &str, names: &[&str]) -> String {
    let path = format!("{prefix}.{suffix}");
    let names: Vec<Box<str>> = names.iter().map(|s| Box::from(*s)).collect();
    let m = DMatrix::<f32>::from_fn(names.len(), 3, |i, k| (i * 10 + k) as f32);
    let cols: Vec<Box<str>> = ["h0", "h1", "h2"].iter().map(|s| Box::from(*s)).collect();
    m.to_parquet_with_names(&path, (Some(&names), Some(row_axis)), Some(&cols))
        .unwrap();
    path
}

fn carried_fixture() -> CarriedRows {
    CarriedRows {
        names: vec![Box::from("GATA1"), Box::from("GO:0006915")],
        types: vec![Box::from("gene"), Box::from("term")],
        rows: DMatrix::<f32>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
        source: "src.parquet".into(),
    }
}

/// Appending keeps the run's rows first, its row axis and column names, and
/// writes a types table over every row: the run's rows `gene` when it wrote
/// no types of its own.
#[test]
fn appending_writes_the_full_table_and_types_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let path = write_run_table(&out, "feature_embedding.parquet", "gene", &["A", "B"]);
    carried_fixture()
        .append_to(&out, "feature_embedding.parquet")
        .unwrap();

    let t = DMatrix::<f32>::from_parquet(&path).unwrap();
    let names: Vec<&str> = t.rows.iter().map(AsRef::as_ref).collect();
    assert_eq!(names, ["A", "B", "GATA1", "GO:0006915"]);
    assert_eq!(
        t.mat.row(1).iter().copied().collect::<Vec<f32>>(),
        vec![10.0, 11.0, 12.0]
    );
    assert_eq!(
        t.mat.row(3).iter().copied().collect::<Vec<f32>>(),
        vec![-1.0, -2.0, -3.0]
    );
    let fields = legume_numeric::matrix::parquet::peek_parquet_field_names(&path).unwrap();
    assert_eq!(fields[0].as_ref(), "gene", "the row axis is kept");
    assert_eq!(&t.cols[..], &["h0".into(), "h1".into(), "h2".into()]);
    let types = data_beans::aux::feature_types::read_feature_types(&out)
        .unwrap()
        .unwrap();
    let types: Vec<&str> = types.iter().map(|(_, t)| t.as_ref()).collect();
    assert_eq!(types, ["gene", "gene", "gene", "term"]);
}

/// A run that wrote its own types (fne) keeps them for its rows.
#[test]
fn appending_keeps_the_run_s_own_types() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    write_run_table(&out, "feature_embedding.parquet", "feature", &["A", "CT0"]);
    write_feature_types(
        &out,
        &[Box::from("A"), Box::from("CT0")],
        &[Box::from("gene"), Box::from("cell_type")],
    )
    .unwrap();
    carried_fixture()
        .append_to(&out, "feature_embedding.parquet")
        .unwrap();
    let types = data_beans::aux::feature_types::read_feature_types(&out)
        .unwrap()
        .unwrap();
    let types: Vec<&str> = types.iter().map(|(_, t)| t.as_ref()).collect();
    assert_eq!(types, ["gene", "cell_type", "gene", "term"]);
}

/// Set `path`'s modification time `secs` seconds back.
fn age(path: &str, secs: u64) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(secs))
        .unwrap();
}

/// A types table left at the output prefix by an older run, for other rows,
/// is replaced rather than refusing the finished run; one written since the
/// run's table that disagrees with it is refused.
#[test]
fn appending_replaces_a_types_table_another_run_left() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    write_run_table(&out, "feature_embedding.parquet", "gene", &["A", "B"]);
    write_feature_types(&out, &[Box::from("X")], &[Box::from("word")]).unwrap();
    assert!(carried_fixture()
        .append_to(&out, "feature_embedding.parquet")
        .is_err());
    age(
        &data_beans::aux::feature_types::feature_types_path(&out),
        60,
    );
    carried_fixture()
        .append_to(&out, "feature_embedding.parquet")
        .unwrap();
    let types = data_beans::aux::feature_types::read_feature_types(&out)
        .unwrap()
        .unwrap();
    let types: Vec<&str> = types.iter().map(|(_, t)| t.as_ref()).collect();
    assert_eq!(types, ["gene", "gene", "gene", "term"]);
}

/// A cell type named like a gene, earlier in the table, neither pins the
/// gene nor comes through as one.
#[test]
fn a_cell_type_named_like_a_gene_does_not_stand_in_for_it() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let names: Vec<Box<str>> = ["CD4", "CD4"].iter().map(|s| Box::from(*s)).collect();
    DMatrix::<f32>::from_row_slice(2, 2, &[9.0, 9.0, 1.0, -1.0])
        .to_parquet_with_names(
            &format!("{prefix}.feature_embedding.parquet"),
            (Some(&names), Some("feature")),
            None,
        )
        .unwrap();
    write_feature_types(
        &prefix,
        &names,
        &[Box::from("cell_type"), Box::from("gene")],
    )
    .unwrap();
    let axis: Vec<Box<str>> = vec![Box::from("ENSG1_CD4")];
    let (f, carried) = load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' },
    )
    .unwrap();
    assert_eq!(f.ids, vec![0]);
    assert_eq!(f.rows, vec![1.0, -1.0]);
    // The cell type is its own row: carried through, as a cell type.
    let c = carried.unwrap();
    assert_eq!(c.names, vec![Box::from("CD4")]);
    assert_eq!(c.types, vec![Box::from("cell_type")]);
}

/// With nothing carried, a types table an earlier run left at the output
/// prefix, for other rows, is removed; the run's own is kept.
#[test]
fn a_stale_types_table_is_cleared_when_nothing_is_carried() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let rho = write_run_table(&out, "feature_embedding.parquet", "gene", &["A", "B"]);
    let types = data_beans::aux::feature_types::feature_types_path(&out);
    write_feature_types(
        &out,
        &[Box::from("A"), Box::from("B")],
        &[Box::from("gene"), Box::from("word")],
    )
    .unwrap();
    age(&types, 60);
    senna::carried_rows::clear_stale_types(&out, &rho).unwrap();
    assert!(
        std::path::Path::new(&types).is_file(),
        "it lists the rows: kept"
    );
    write_feature_types(&out, &[Box::from("X")], &[Box::from("word")]).unwrap();
    age(&types, 60);
    senna::carried_rows::clear_stale_types(&out, &rho).unwrap();
    assert!(!std::path::Path::new(&types).is_file());
}

/// A carried name the run wrote itself is superseded by the run's row; a
/// width that disagrees is refused and the table is left as written.
#[test]
fn appending_skips_a_superseded_name_and_refuses_a_width_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let path = write_run_table(&out, "feature_embedding.parquet", "gene", &["A", "GATA1"]);
    let mut wide = carried_fixture();
    wide.names = vec![Box::from("X"), Box::from("Y")];
    wide.rows = DMatrix::<f32>::zeros(2, 4);
    assert!(wide.append_to(&out, "feature_embedding.parquet").is_err());
    let t = DMatrix::<f32>::from_parquet(&path).unwrap();
    assert_eq!(t.mat.nrows(), 2, "left as written");

    carried_fixture()
        .append_to(&out, "feature_embedding.parquet")
        .unwrap();
    let t = DMatrix::<f32>::from_parquet(&path).unwrap();
    let names: Vec<&str> = t.rows.iter().map(AsRef::as_ref).collect();
    assert_eq!(names, ["A", "GATA1", "GO:0006915"]);
    assert_eq!(
        t.mat.row(1).iter().copied().collect::<Vec<f32>>(),
        vec![10.0, 11.0, 12.0],
        "the run's own GATA1 row stands"
    );

    // Everything superseded: the table stands and no types table appears.
    let out2 = dir.path().join("run2").to_string_lossy().into_owned();
    let path2 = write_run_table(
        &out2,
        "feature_embedding.parquet",
        "gene",
        &["GATA1", "GO:0006915"],
    );
    carried_fixture()
        .append_to(&out2, "feature_embedding.parquet")
        .unwrap();
    assert_eq!(DMatrix::<f32>::from_parquet(&path2).unwrap().mat.nrows(), 2);
    assert!(data_beans::aux::feature_types::read_feature_types(&out2)
        .unwrap()
        .is_none());
}

/// H is the larger of `--embedding-dim` and the table's width: a narrower
/// table is widened with zero columns, given and carried rows alike, and a
/// smaller H (or `auto`) takes the table's.
#[test]
fn the_width_is_the_larger_of_the_flag_and_the_table() {
    let preset = || {
        Some(ge::PresetRows {
            ids: vec![0, 3],
            rows: vec![0.5, 1.0, 1.5, -1.0, -0.5, 0.0],
            mode: PresetMode::Freeze,
        })
    };
    let mut p = preset();
    let mut c = Some(carried_fixture());
    let h = resolve_dim(ge::EmbeddingDim::Fixed(5), &mut p, &mut c).unwrap();
    assert_eq!(h, 5);
    assert_eq!(
        p.unwrap().rows,
        vec![0.5, 1.0, 1.5, 0.0, 0.0, -1.0, -0.5, 0.0, 0.0, 0.0]
    );
    let c = c.unwrap().rows;
    assert_eq!(c.shape(), (2, 5));
    assert_eq!(
        c.row(1).iter().copied().collect::<Vec<_>>(),
        [-1.0, -2.0, -3.0, 0.0, 0.0]
    );

    for flag in [ge::EmbeddingDim::Fixed(2), ge::EmbeddingDim::Auto] {
        let mut p = preset();
        assert_eq!(resolve_dim(flag, &mut p, &mut None).unwrap(), 3);
        assert_eq!(p.unwrap().rows.len(), 6);
    }
    // Without a table the flag is the width, and `auto` has nothing to take.
    assert_eq!(
        resolve_dim(ge::EmbeddingDim::Fixed(7), &mut None, &mut None).unwrap(),
        7
    );
    assert!(resolve_dim(ge::EmbeddingDim::Auto, &mut None, &mut None).is_err());
}
