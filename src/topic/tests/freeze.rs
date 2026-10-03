use super::*;
use data_beans::aux::feature_types::write_feature_types;

/// An `fne`-like source: a cell type named like a gene, before it, and a
/// word named like a feature of the data. Only gene rows match, in the
/// mask and the host, and the carried rows keep their own types.
#[test]
fn only_the_source_s_gene_rows_match() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let names: Vec<Box<str>> = ["CD4", "TP53", "apoptosis", "CD4"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    DMatrix::<f32>::from_row_slice(4, 2, &[9.0, 9.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
        .to_parquet_with_names(
            &format!("{prefix}.feature_embedding.parquet"),
            (Some(&names), Some("feature")),
            None,
        )
        .unwrap();
    let types: Vec<Box<str>> = ["cell_type", "gene", "word", "gene"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    write_feature_types(&prefix, &names, &types).unwrap();

    let spec = FrozenFeatureSpec::resolve_from_prefix(
        &prefix,
        "--freeze-feature-embedding",
        FeatureNameKind::Gene { delim: '_' },
    )
    .unwrap();
    let axis: Vec<Box<str>> = ["ENSG1_CD4", "apoptosis", "ENSG2_TP53"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    assert_eq!(spec.mask_fn()(&axis).unwrap(), [true, false, true]);

    let kept: Vec<Box<str>> = vec![axis[0].clone(), axis[2].clone()];
    let host = spec.materialize(&kept).unwrap();
    assert_eq!(host.keep_target_indices, [0, 1]);
    assert_eq!(
        host.keep_src_indices,
        [3, 1],
        "CD4 is the gene row, not the cell type"
    );
    assert_eq!(
        host.e_feat.row(0).iter().copied().collect::<Vec<_>>(),
        [3.0, -3.0]
    );
    let carried = spec.carried_types(&host);
    assert_eq!(carried[0].1.as_ref(), "cell_type");
    assert_eq!(carried.len(), 4);
}
