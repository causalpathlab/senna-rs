use super::TdeArgs;
use clap::{CommandFactory, Parser};

/// `TdeArgs` is an `Args` group, not a `Parser`; wrap it to parse standalone.
#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: TdeArgs,
}

fn parse(argv: &[&str]) -> TdeArgs {
    let mut full = vec!["senna-tde"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).expect("parse").args
}

fn try_parse(argv: &[&str]) -> Result<TdeArgs, clap::Error> {
    let mut full = vec!["senna-tde"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).map(|c| c.args)
}

#[test]
fn parses_genes() {
    let a = parse(&["a.zarr.zip", "b.zarr.zip", "-o", "out"]);
    assert_eq!(a.genes.len(), 2);
    assert_eq!(&*a.genes[0], "a.zarr.zip");
    assert_eq!(&*a.genes[1], "b.zarr.zip");
}

#[test]
fn defaults() {
    let a = parse(&["a.zarr.zip", "-o", "out"]);
    assert_eq!(a.phase1_cells_per_pb, 16);
    assert_eq!(a.collapse.proj_dim, 50);
    assert_eq!(a.collapse.knn_cells, 10);
    assert_eq!(a.collapse.iter_opt, 30);
    assert_eq!(a.seed, 1);
    assert_eq!(a.hvg.n_hvg, 5000);
    assert_eq!(a.divergence_l2, 30.0);
    assert_eq!(a.divergence_epochs, 50);
}

/// The divergence flags reach the engine's knobs.
#[test]
fn the_divergence_flags_set_the_divergence_config() {
    let a = parse(&[
        "a.zarr.zip",
        "-o",
        "out",
        "--divergence-l2",
        "2",
        "--divergence-epochs",
        "7",
    ]);
    let axis = graph_embedding_util::DivergenceAxis {
        base_backend_row: vec![],
        divergent_backend_row: vec![],
        track_name: "count/unspliced".into(),
    };
    let k = a.divergence(axis);
    assert_eq!(k.l2, 2.0);
    assert_eq!(k.epochs, 7);
}

/// The joint gem's track flags, and the composite engine's before it, fail
/// to parse: tde has one gene axis and a divergent track, no modality tracks
/// and no per-track offsets.
#[test]
fn dropped_flags_fail_to_parse() {
    let dropped = [
        "--modality",
        "--rule-a-prob",
        "--nuisance-rank",
        "--cell-divergence-l2",
        "--divergence-distill-epochs",
        "--divergence-refine-epochs",
        "--offset-l2",
        "--offset-rank",
        "--genes",
        "--delta-l2",
        "--feature-embedding-l2",
        "--nce-objective",
        "--num-opt-iter",
        "--knn-pb",
        "--max-grad-norm",
        "--batches-per-epoch",
        "--no-preload-data",
        "--threads",
    ];
    for flag in dropped {
        let err = try_parse(&["a.zarr.zip", "-o", "out", flag, "x"]);
        assert!(err.is_err(), "{flag} must fail to parse (dropped)");
    }
}

#[test]
fn serde_round_trip_is_stable() {
    let a = parse(&["a.zarr.zip", "-o", "out", "--divergence-l2", "2.5"]);
    let v1 = serde_json::to_value(&a).expect("serialize");
    let a2: TdeArgs = serde_json::from_value(v1.clone()).expect("deserialize");
    let v2 = serde_json::to_value(&a2).expect("serialize again");
    assert_eq!(v1, v2, "a serde round trip must reproduce the same record");
}

/// Every flag `tde/args.rs` declares directly (excludes the flattened shared
/// groups `hvg` / `collapse` / `qc` / `modules`, whose help text belongs to
/// their own source files and is governed there, not here).
const TDE_OWN_ARG_IDS: [&str; 21] = [
    "genes",
    "batch_files",
    "genes_sample_strip",
    "embedding_dim",
    "phase1_cells_per_pb",
    "modules_per_unit",
    "skip_etm",
    "num_topics",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "block_size",
    "preload_data",
    "divergence_l2",
    "divergence_epochs",
    "divergence_learning_rate",
    "seed",
    "device",
    "device_no",
    "out",
];

#[test]
fn tdes_own_flags_have_no_em_dash_and_wrap_under_100_cols() {
    let cmd = Cli::command();
    for id in TDE_OWN_ARG_IDS {
        let arg = cmd
            .get_arguments()
            .find(|a| a.get_id() == id)
            .unwrap_or_else(|| panic!("tde/args.rs must declare --{id}"));
        for text in [arg.get_help(), arg.get_long_help()].into_iter().flatten() {
            let s = text.to_string();
            assert!(
                !s.contains('\u{2014}'),
                "{id}: em dash (U+2014) in help: {s:?}"
            );
            for line in s.lines() {
                assert!(
                    line.chars().count() <= 100,
                    "{id}: help line over 100 columns: {line:?}"
                );
            }
        }
    }
}

/// The `tde` subcommand's own `about`/`long_about` (`main.rs`), which lives
/// outside `TdeArgs` and so isn't covered by the test above.
#[test]
fn tde_subcommand_about_has_no_em_dash_and_wraps_under_100_cols() {
    let cmd = crate::Cli::command();
    let tde = cmd
        .find_subcommand("tde")
        .expect("main.rs must declare the `tde` subcommand");
    for text in [tde.get_about(), tde.get_long_about()]
        .into_iter()
        .flatten()
    {
        let s = text.to_string();
        assert!(
            !s.contains('\u{2014}'),
            "tde subcommand about/long_about: em dash (U+2014): {s:?}"
        );
        for line in s.lines() {
            assert!(
                line.chars().count() <= 100,
                "tde subcommand about/long_about: line over 100 columns: {line:?}"
            );
        }
    }
}

/// `render_long_help()` runs cleanly end to end AND every flattened group
/// reaches the render: one distinctive flag from `TdeArgs` itself plus one
/// from each of `HvgCliArgs`, `refine_weighting::CollapseArgs`, `QcArgs` and
/// `ge::FeatureModuleArgs`.
#[test]
fn full_help_renders() {
    let help = Cli::command().render_long_help().to_string();
    for flag in [
        "--divergence-l2",      // TdeArgs
        "--genes-sample-strip", // TdeArgs
        "--embedding-dim",      // TdeArgs
        "--n-hvg",              // HvgCliArgs
        "--num-levels",         // refine_weighting::CollapseArgs
        "--no-qc",              // QcArgs
        "--feature-modules",    // ge::FeatureModuleArgs
    ] {
        assert!(help.contains(flag), "tde --help is missing {flag}");
    }
}

/// The feature-embedding triple and `--embedding-dim auto` parse on tde
/// exactly as on bge.
#[test]
fn the_feature_embedding_triple_and_the_auto_dim_parse() {
    use graph_embedding_util::{EmbeddingDim, PresetMode};
    let a = parse(&["a.zarr", "-o", "out"]);
    assert_eq!(a.embedding_dim, EmbeddingDim::Fixed(128));
    assert!(a.feature_embedding.resolve().unwrap().is_none());

    let a = parse(&[
        "a.zarr",
        "-o",
        "out",
        "--freeze-feature-embedding",
        "prev",
        "--embedding-dim",
        "auto",
    ]);
    assert_eq!(a.embedding_dim, EmbeddingDim::Auto);
    assert!(matches!(
        a.feature_embedding.resolve().unwrap(),
        Some(("prev", PresetMode::Freeze))
    ));

    let a = parse(&[
        "a.zarr",
        "-o",
        "out",
        "--lora-feature-embedding",
        "prev",
        "--lora-rank",
        "4",
    ]);
    assert!(matches!(
        a.feature_embedding.resolve().unwrap(),
        Some(("prev", PresetMode::Lora(s))) if s.rank == 4
    ));
    assert!(try_parse(&[
        "a.zarr",
        "-o",
        "out",
        "--freeze-feature-embedding",
        "p",
        "--init-feature-embedding",
        "q",
    ])
    .is_err());
}
