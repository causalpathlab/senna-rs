use super::*;

fn boxed(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

fn citeseq_record() -> RunMultiome {
    RunMultiome {
        modality: strings(&["scRNA", "scADT", "scRNA", "scADT"]),
        group: strings(&["S1", "S1", "S2", "S2"]),
        barcode_tagged: true,
    }
}

fn plain(n: usize) -> ReadSharedRowsArgs {
    ReadSharedRowsArgs {
        data_files: (0..n).map(|i| format!("f{i}.zarr").into()).collect(),
        preload: true,
        ..Default::default()
    }
}

/// A single-modality run records nothing, and the replay is then exactly the
/// loader's own defaults — applying it to a load must change nothing. A
/// record replays its alignment and both suffixes, keeping the caller's own
/// settings; one group needs no barcode tag (replaying one would rename every
/// cell).
#[test]
fn a_recorded_layout_replays_onto_the_load() {
    let out = recorded_layout(None, 4).unwrap().apply(plain(4)).unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Disjoint, "no record");
    assert!(out.feature_kind.is_none(), "no record");
    assert!(out.per_file_feature_suffix.is_none(), "no record");
    assert!(out.per_file_barcode_suffix.is_none(), "no record");

    let l = recorded_layout(Some(&citeseq_record()), 4).unwrap();
    assert!(l.is_multiome());
    let out = l.apply(plain(4)).unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Union);
    assert!(matches!(out.feature_kind, Some(FeatureNameKind::Mixed)));
    assert_eq!(
        out.per_file_feature_suffix.as_deref().unwrap(),
        boxed(&["scRNA", "scADT", "scRNA", "scADT"]).as_slice()
    );
    let bs = out.per_file_barcode_suffix.unwrap();
    assert_eq!(bs[0].as_deref(), Some("S1"));
    assert_eq!(bs[3].as_deref(), Some("S2"));
    assert!(out.preload, "caller's own settings must survive");

    let r = RunMultiome {
        modality: strings(&["rna", "atac"]),
        group: strings(&["g0", "g0"]),
        barcode_tagged: false,
    };
    let out = recorded_layout(Some(&r), 2)
        .unwrap()
        .apply(plain(2))
        .unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Union, "untagged");
    assert!(out.per_file_barcode_suffix.is_none(), "untagged");
}

/// The record is positional. A caller that swapped the file list would
/// silently namespace the wrong rows, so refuse rather than guess.
#[test]
fn record_of_the_wrong_length_is_an_error() {
    let r = citeseq_record();
    let err = recorded_layout(Some(&r), 3).unwrap_err().to_string();
    assert!(err.contains('4') && err.contains('3'), "{err}");
}

/// Query files are a fresh set, so their layout is detected. The detected
/// modality tags come from THEIR filenames and need not match the training
/// run's, so they are renamed by which trained row block their features
/// actually land in.
#[test]
fn query_modalities_are_renamed_to_the_trained_ones() {
    let trained = boxed(&[
        "GENEG/scRNA",
        "GENEQ/scRNA",
        "GENEO/scRNA",
        "ADT-GENEE/scADT",
        "ADT-GENEF/scADT",
    ]);
    let detected = ge::MultiomePlan {
        files: boxed(&["gex_q.zarr", "prot_q.zarr"]),
        modality: boxed(&["gex", "prot"]),
        group: boxed(&["q", "q"]),
        group_sizes: vec![2],
        barcode_tagged: false,
        n_bridge_cells: Some(500),
    };
    let rows = vec![
        boxed(&["GENEG", "GENEQ", "GENEO"]),
        boxed(&["ADT-GENEE", "ADT-GENEF"]),
    ];
    let named = reconcile_modalities(detected, &rows, &trained).unwrap();
    assert_eq!(&*named.modality[0], "scRNA");
    assert_eq!(&*named.modality[1], "scADT");
}

/// A query modality whose features match no trained block keeps its own tag —
/// renaming it would claim an overlap that is not there.
#[test]
fn an_unmatched_query_modality_keeps_its_detected_tag() {
    let trained = boxed(&["GENEG/scRNA", "GENEQ/scRNA"]);
    let detected = ge::MultiomePlan {
        files: boxed(&["gex_q.zarr", "atac_q.zarr"]),
        modality: boxed(&["gex", "atac"]),
        group: boxed(&["q", "q"]),
        group_sizes: vec![2],
        barcode_tagged: false,
        n_bridge_cells: Some(10),
    };
    let rows = vec![
        boxed(&["GENEG", "GENEQ"]),
        boxed(&["chr1:100-200", "chr2:300-400"]),
    ];
    let named = reconcile_modalities(detected, &rows, &trained).unwrap();
    assert_eq!(&*named.modality[0], "scRNA");
    assert_eq!(&*named.modality[1], "atac");
}

/// Two query modalities must not both claim one trained block: that would put
/// two different assays on the same rows.
#[test]
fn two_query_modalities_cannot_claim_one_trained_block() {
    let trained = boxed(&["GENEG/scRNA", "GENEQ/scRNA", "GENEO/scRNA"]);
    let detected = ge::MultiomePlan {
        files: boxed(&["a.zarr", "b.zarr"]),
        modality: boxed(&["a", "b"]),
        group: boxed(&["q", "q"]),
        group_sizes: vec![2],
        barcode_tagged: false,
        n_bridge_cells: Some(10),
    };
    let rows = vec![boxed(&["GENEG", "GENEQ"]), boxed(&["GENEQ", "GENEO"])];
    let err = reconcile_modalities(detected, &rows, &trained)
        .unwrap_err()
        .to_string();
    assert!(err.contains("scRNA"), "{err}");
}

/// The trained modalities are read off the row names. A trained run with no
/// `/modality` rows is single-modality, and so is a partially-tagged one: a
/// query must not be routed down the multiome path on the strength of one
/// stray separator, and is loaded the plain way even if its own files look
/// multi-modal.
#[test]
fn trained_modalities_are_read_off_the_row_names() {
    for (label, rows, want) in [
        ("untagged", &["GENEG", "GENEQ", "GENEO"][..], &[][..]),
        (
            "tagged",
            &["GENEG/scRNA", "ADT-GENEF/scADT", "GENEQ/scRNA"],
            &["scADT", "scRNA"],
        ),
        ("partially tagged", &["A/B", "GENEG", "GENEO"], &[]),
        ("two untagged", &["GENEG", "GENEO"], &[]),
        ("empty", &[], &[]),
        (
            "every row tagged",
            &["GENEG/scRNA", "ADT-GENEF/scADT"],
            &["scADT", "scRNA"],
        ),
    ] {
        let mut m = trained_modalities(&boxed(rows));
        m.sort();
        assert_eq!(m, boxed(want), "{label}");
    }
}

/// The plain load must pass through `apply` untouched — every
/// single-modality call site goes through it and none may change.
#[test]
fn apply_is_a_no_op_for_a_plain_load() {
    let base = ReadSharedRowsArgs {
        data_files: boxed(&["a.zarr", "b.zarr"]),
        preload: true,
        feature_kind: Some(FeatureNameKind::Gene { delim: '_' }),
        ..Default::default()
    };
    let out = ReloadLayout::default().apply(base).unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Disjoint);
    assert!(matches!(
        out.feature_kind,
        Some(FeatureNameKind::Gene { delim: '_' })
    ));
    assert!(out.per_file_feature_suffix.is_none());
    assert!(out.per_file_barcode_suffix.is_none());
    assert!(out.preload);
}

/// faba rows are `{gene}/{modality}/{channel}` — two separators, and the
/// trailing field is a splice channel, not a modality. Reading that as
/// multiome would re-scope such a model's whole feature axis on query.
#[test]
fn a_splice_axis_is_not_a_multiome_model() {
    let tracks = boxed(&["G1/count/spliced", "G1/count/unspliced", "G2/count/spliced"]);
    assert!(trained_modalities(&tracks).is_empty());
    assert!(query_layout(&boxed(&["a.zarr", "b.zarr"]), &tracks)
        .unwrap()
        .is_none());
}

/// A multiome axis carries gene names and chrX:s-e loci together, so only
/// `Mixed` can canonicalize it. An explicit rule there is refused rather than
/// silently overridden — the override would be invisible and would break the
/// axis match against the dictionary. `Mixed` passed explicitly agrees, and a
/// plain load keeps the caller's own rule.
#[test]
fn an_explicit_feature_kind_is_kept_unless_a_multiome_axis_refuses_it() {
    let gene = FeatureNameKind::Gene { delim: '_' };
    for (label, multiome, kind, refused) in [
        (
            "a plain load keeps the caller's rule",
            false,
            gene.clone(),
            false,
        ),
        ("a gene rule on a multiome axis", true, gene, true),
        (
            "mixed on a multiome axis",
            true,
            FeatureNameKind::Mixed,
            false,
        ),
    ] {
        let l = if multiome {
            recorded_layout(Some(&citeseq_record()), 4).unwrap()
        } else {
            ReloadLayout::default()
        };
        let mut base = plain(4);
        base.feature_kind = Some(kind.clone());
        // `ReadSharedRowsArgs` is not Debug, so `unwrap_err` is unavailable.
        match l.apply(base) {
            Ok(out) => {
                assert!(!refused, "{label}: must be refused");
                assert_eq!(out.feature_kind, Some(kind), "{label}");
            }
            Err(e) => {
                let err = e.to_string();
                assert!(refused, "{label}: {err}");
                assert!(err.contains("Gene"), "{label}: {err}");
                assert!(
                    err.contains("scRNA") && err.contains("scADT"),
                    "{label}: {err}"
                );
            }
        }
    }
}

/// Trained rows carry the loader's canonical names, query rows their raw
/// ones: a peak matches its trained block through the shared locus rule.
#[test]
fn query_peaks_match_trained_peaks_by_their_canonical_key() {
    let key = |p: &str| format!("{}/peaks", FeatureNameKind::Mixed.canonicalize(p));
    let trained: Vec<Box<str>> = ["chrX:0-100", "chr1:100-200"]
        .iter()
        .map(|p| key(p).into())
        .chain(["GENEG/scRNA".into()])
        .collect();
    let detected = ge::MultiomePlan {
        files: boxed(&["gex_q.zarr", "atac_q.zarr"]),
        modality: boxed(&["gex", "atac"]),
        group: boxed(&["q", "q"]),
        group_sizes: vec![2],
        barcode_tagged: false,
        n_bridge_cells: Some(10),
    };
    let rows = vec![boxed(&["GENEG"]), boxed(&["chrX:0-100", "chr1:100-200"])];
    let named = reconcile_modalities(detected, &rows, &trained).unwrap();
    assert_eq!(&*named.modality[0], "scRNA");
    assert_eq!(&*named.modality[1], "peaks");
}
