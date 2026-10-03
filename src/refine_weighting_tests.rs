use super::*;
use clap::Parser;
use data_beans::alg::collapse_data::BelowEdge;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    collapse: CollapseArgs,
}

#[test]
fn the_tree_is_the_default_and_marginal_switches_it_off() {
    let cli = Cli::try_parse_from(["senna"]).unwrap();
    let params = cli.collapse.pb_tree_params().expect("refined by default");
    assert!((params.edge_margin - 0.10).abs() < 1e-6);
    assert_eq!(params.below_edge, BelowEdge::Keep);
    assert_eq!(params.max_genes, 2000);
    assert_eq!(params.min_cells_to_split, 8);
    let reassign = params.reassign_cells.expect("cells are reassigned first");
    assert_eq!(reassign.num_gibbs, 3);

    let cli = Cli::try_parse_from(["senna", "--pb-tree", "marginal"]).unwrap();
    assert!(cli.collapse.pb_tree_params().is_none());
}

/// The carrying default must reach a parent recorded before the opt-out
/// existed: the manifest lacks the field, and replaying it must not fall back
/// to the type's zero, which would silently stop the chain from carrying.
#[test]
fn a_fit_recorded_before_the_opt_out_existed_still_carries() {
    let json = r#"{"emit_pb_reference": false}"#;
    let a: CollapseArgs = serde_json::from_str(json).expect("old record replays");
    assert!(
        a.emits_pb_reference(),
        "missing field is the default, which is on"
    );
    let json = r#"{"no_emit_pb_reference": true}"#;
    let a: CollapseArgs = serde_json::from_str(json).expect("opt-out replays");
    assert!(!a.emits_pb_reference());
}

#[test]
fn cnv_clones_flag_parses() {
    let cli = Cli::try_parse_from(["senna", "--cnv-clones", "x.clones.parquet"]).unwrap();
    assert_eq!(cli.collapse.cnv_clones.as_deref(), Some("x.clones.parquet"));
}

/// A recorded fit and one about to run differ in what built the
/// partition, and only there: the sketch width is the larger of
/// `proj_dim` and the latent count, outputs do not count, and a setting
/// the record lacks is passed.
#[test]
fn differences_are_in_what_builds_the_partition() {
    let there = serde_json::json!({
        "collapse": {"proj_dim": 50, "sort_dim": 10, "emit_pb_reference": true},
        "qc": {"min_features": 200},
        "n_latent_topics": 60,
        "epochs": 1000,
    });
    let here = serde_json::json!({
        "collapse": {"proj_dim": 60, "sort_dim": 10, "emit_pb_reference": false},
        "qc": {"min_features": 200, "new_flag": 1},
        "n_latent": 10,
        "epochs": 5,
    });
    assert!(senna::run_manifest::settings_differences(
        &source_settings(&there),
        &source_settings(&here)
    )
    .is_empty());

    let here = serde_json::json!({
        "collapse": {"proj_dim": 50, "sort_dim": 6},
        "qc": {"min_features": 200},
        "n_latent": 10,
    });
    let d = senna::run_manifest::settings_differences(
        &source_settings(&there),
        &source_settings(&here),
    );
    assert_eq!(d.len(), 2, "{d:?}");
    assert!(d
        .iter()
        .any(|x| x.starts_with("collapse.sort_dim: 10 there, 6 here")));
    assert!(d
        .iter()
        .any(|x| x.starts_with("sketch_dim: 60 there, 50 here")));
}

/// A float back from JSON compares equal to the f32 it was, output-only
/// settings are passed over, and a flag that cuts the cells differently
/// (multiome, a feature preset) counts.
#[test]
fn floats_outputs_and_the_feature_axis_in_the_comparison() {
    let x: f32 = 0.012_036_108;
    let there = serde_json::json!({
        "collapse": {"proj_dim": 50, "sort_dim": 10},
        "qc": {"qc_mads": f64::from(x) + 1e-12, "qc_report": "a.tsv"},
        "multiome": false,
    });
    let here = serde_json::json!({
        "collapse": {"proj_dim": 50, "sort_dim": 10},
        "qc": {"qc_mads": x, "qc_report": "b.tsv"},
        "multiome": false,
        "init_feature_embedding": null,
    });
    assert!(senna::run_manifest::settings_differences(
        &source_settings(&there),
        &source_settings(&here)
    )
    .is_empty());

    let here = serde_json::json!({
        "collapse": {"proj_dim": 50, "sort_dim": 10},
        "qc": {"qc_mads": x},
        "multiome": true,
    });
    let d = senna::run_manifest::settings_differences(
        &source_settings(&there),
        &source_settings(&here),
    );
    assert_eq!(d, ["multiome: false there, true here"]);

    // A record with no collapse settings is passed, not refused.
    let bare = serde_json::json!({"epochs": 5});
    assert!(senna::run_manifest::settings_differences(
        &source_settings(&bare),
        &source_settings(&here)
    )
    .is_empty());
}
