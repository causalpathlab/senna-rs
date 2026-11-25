#![allow(
    // `embed_common` is deliberately shaped as a prelude module.
    clippy::wildcard_imports,
    // Counts / dimensions / IDs routinely cross usize↔f32/f64; the
    // values always fit and the casts are intentional.
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // Not every `Result`-returning helper needs a `# Errors` stanza.
    clippy::missing_errors_doc,
    // Training / fit functions are naturally long; splitting them for
    // a line-count lint would fragment logical phases.
    clippy::too_many_lines,
    // CLI struct fields intentionally share a `phate_` prefix so the
    // clap flag names (`--phate-t`, `--phate-mds-iter`) are self-documenting.
    clippy::struct_field_names,
    // Config / args structs are typically built once at the call site
    // and consumed — passing by value is part of the ownership-forward
    // API style used across the crate.
    clippy::needless_pass_by_value,
    // Local `use`/`const`/`enum` items scoped to where they're relevant
    // read more naturally than hoisting them to the top of a function.
    clippy::items_after_statements,
    // Binding-name similarity is noisy for domain-driven names like
    // `dist`/`d`, `stress`/`prev_stress`.
    clippy::similar_names,
    // Math code uses short names (`n`, `i`, `j`, `k`, `d`) where the
    // semantics come from surrounding indices (row/col/dim).
    clippy::many_single_char_names,
)]

// Training on the CPU allocates and frees multi-megabyte buffers every
// step; the system allocator hands them back to the kernel on free and page
// faults them in again on the next step, which cost as much as the arithmetic
// in the elementwise passes. mimalloc keeps large blocks around.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod anchor_common;
mod bge;
mod cluster_bhc;
mod clustering;
mod commands;
mod critique;
mod deconvolve;
mod docs;
mod donor_deviation;
mod empirical_dict;
mod feature_embedding_args;
mod feature_preset;
#[cfg(test)]
#[path = "tests/fit_cli.rs"]
mod fit_cli;
mod fne;
mod hvg;
mod impute;
mod joint_topic;
mod logging;
mod masked_topic;
mod ode_mixture;
mod pbg_train_args;
#[cfg(test)]
#[path = "tests/planted.rs"]
mod planted;
mod postprocess;
mod predict;
#[cfg(test)]
#[path = "tests/predict_null.rs"]
mod predict_null_tests;
mod probe;
mod rbm;
mod rbm_core;
mod refine_weighting;
mod resolve_embedding_space;
#[cfg(feature = "view")]
mod run;
#[cfg(test)]
#[path = "tests/scoring_delta.rs"]
mod scoring_delta_tests;
#[cfg(test)]
#[path = "tests/seed_replay.rs"]
mod seed_replay_tests;
#[cfg(all(test, feature = "view"))]
#[path = "tests/session_cli.rs"]
mod session_cli_tests;
mod simba;
mod svd;
#[cfg(all(test, unix, feature = "view"))]
mod test_program;
mod topic;
mod training_data;
#[cfg(feature = "view")]
mod tui;
mod update;
mod vae;
#[cfg(feature = "view")]
mod view;

use bge::fit_bge;
use clustering::*;
use critique::run_critique;
use docs::run_docs;
use donor_deviation::run_donor_deviation;
use fne::fit_fne;
use impute::impute_model;
use joint_topic::*;
use masked_topic::*;
use postprocess::*;
use predict::predict_model;
use probe::run_probe;
use resolve_embedding_space::resolve_embedding_space;
use senna::embed_common::*;
use senna::multiome_plan;
use simba::fit_simba;
use svd::*;
use topic::cmd::*;
use update::run_update;
use vae::*;

use colored::Colorize;
use commands::{Commands, LayoutCmd};

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('@', &"@".bright_yellow().to_string())
        .replace('◠', &"◠".bright_yellow().to_string())
        .replace('◡', &"◠".bright_yellow().to_string())
        .replace('_', &"_".bright_yellow().to_string())
        .replace('(', &"(".bright_yellow().to_string())
        .replace(')', &")".bright_yellow().to_string())
        .replace('{', &"{".bright_yellow().to_string())
        .replace('}', &"}".bright_yellow().to_string())
        .replace('\\', &"\\".bright_yellow().to_string())
        .replace('/', &"/".bright_yellow().to_string())
        .replace('|', &"|".green().to_string())
        .replace('‖', &"‖".green().to_string())
        .replace('~', &"~".truecolor(101, 67, 33).to_string())
}

fn print_logo() {
    let intro = [
        "",
        "",
        "SENNA",
        "Stochastic data Embedding with",
        "Nearest Neighbourhood Adjustment",
        "",
    ];

    let logo_lines: Vec<_> = LOGO.lines().collect();
    let max_lines = logo_lines.len().max(intro.len());

    for i in 0..max_lines {
        let logo_part = if i < logo_lines.len() {
            colorize_logo_line(logo_lines[i])
        } else {
            " ".repeat(13) // width of logo box
        };

        let text_part = if i < intro.len() { intro[i] } else { "" };

        println!("{logo_part}  {text_part}");
    }
    println!();
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "SENNA — single-cell embedding (SVD / topic models, graph methods,\n\
             clustering, and 2D layout).",
    long_about = "SENNA — Stochastic data Embedding with Nearest Neighbourhood Adjustment.\n\
                  \n\
                  Input: sparse backends in `.zarr` or `.h5`.\n\
                  Convert from Matrix Market with `data-beans from-mtx`.\n\
                  \n\
                  Each step writes its outputs back to the run manifest `{prefix}.senna.json`.\n\
                  Downstream commands read data and batch files from it.\n\
                  Clustering still needs its own --latent / --out.\n\
                  \n  \
                  1. Train embedding   senna topic | masked-topic | svd | bge | simba\n                       \
                  senna joint-topic | joint-svd   (multi-modality)\n                       \
                  senna run   (choose data, methods and flags in the terminal)\n  \
                  2. Held-out inference senna predict            (apply trained model)\n  \
                  3. Cluster cells     senna clustering --from run.senna.json --latent L --out O\n  \
                  4. 2D layout         senna layout {phate|tsne|umap|lle|sugiyama|stress|forceatlas2|diffusion|radial} --from run.senna.json\n\
                  \n\
                  Annotation, trajectory, association, and figures live in `lupin`\n\
                  (`lupin --help`).\n\
                  \n\
                  Bulk deconvolution is a side branch off a `bge` or `simba` run.\n\
                  It needs an annotation (from `lupin annotate`) and bulk counts, plus the single-cell\n\
                  counts the reference profiles are measured from.\n\
                  \n  \
                  senna deconvolve --from bge.senna.json --annotation A --bulk bulk.parquet\n\
                  \n\
                  CNV-aware collapse: `mung clones` writes `{out}.clones.parquet`;\n\
                  pass `--cnv-clones` on topic / masked-* / vae / svd / bge /\n\
                  joint-* so donor-private CN stays out of batch δ.\n\
                  \n\
                  Artifact naming: a slot name fixes the axis, never the numeric scale.\n\
                  `feature_embedding` is the per-gene embedding rho, and it is signed;\n\
                  `feature_coembedding` is rho re-placed onto the cell manifold.\n\
                  `dictionary` is a topic dictionary in LOG space, or SVD signed loadings.\n\
                  Reading one as the other yields NaN, so check `kind` before assuming.\n\
                  See senna/docs/deconvolve.md and the run_manifest module docs."
)]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Verbose logging")]
    verbose: bool,

    #[arg(long, global = true, help = "Worker threads (default: all cores)")]
    threads: Option<usize>,

    #[command(subcommand)]
    commands: Commands,
}

fn main() -> anyhow::Result<()> {
    // A session's host runs this binary again: it serves the session and
    // exits here, before any of senna's own work.
    #[cfg(feature = "view")]
    legume_tui::serve_if_host();

    // Show logo if help is requested
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    logging::init_logger(cli.verbose);

    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()?;
    }

    match &cli.commands {
        Commands::Svd(args) => {
            fit_svd(args)?;
        }
        Commands::Bge(args) => {
            fit_bge(args)?;
        }
        Commands::Simba(args) => {
            fit_simba(args)?;
        }
        Commands::Fne(args) => {
            fit_fne(args)?;
        }
        Commands::Ode(args) => {
            ode_mixture::command::run_ode(args)?;
        }
        Commands::Rbm(args) => {
            rbm::run::fit_rbm(args)?;
        }
        Commands::ResolveEmbeddingSpace(args) => {
            resolve_embedding_space(args)?;
        }
        Commands::Topic(args) => {
            fit_topic_model(args)?;
        }
        Commands::MaskedTopic(args) => {
            fit_masked_topic_model(args)?;
        }
        Commands::MaskedVae(args) => {
            fit_masked_vae_model(args)?;
        }
        Commands::MaskedSbp(args) => {
            fit_masked_sbp_model(args)?;
        }
        Commands::Vae(args) => {
            fit_vae_model(args)?;
        }
        Commands::JointTopic(args) => {
            fit_joint_topic_model(args)?;
        }

        Commands::Deconvolve(args) => {
            deconvolve::run(args)?;
        }
        Commands::DonorDeviation(args) => {
            run_donor_deviation(args)?;
        }
        Commands::Predict(args) => {
            predict_model(args)?;
        }
        Commands::Critique(args) => {
            run_critique(args)?;
        }
        Commands::Probe(args) => {
            run_probe(args)?;
        }
        Commands::Update(args) => {
            run_update(args)?;
        }
        Commands::Impute(args) => {
            impute_model(args)?;
        }
        Commands::JointSvd(args) => {
            fit_joint_svd(args)?;
        }
        Commands::Docs(args) => run_docs(args)?,
        Commands::Layout { cmd } => match cmd {
            LayoutCmd::Tsne(args) => {
                fit_layout_tsne(args)?;
            }
            LayoutCmd::Umap(args) => {
                fit_layout_umap(args)?;
            }
            LayoutCmd::Phate(args) => {
                fit_layout_phate(args)?;
            }
            LayoutCmd::Lle(args) => {
                fit_layout_lle(args)?;
            }
            LayoutCmd::Sugiyama(args) => {
                fit_layout_sugiyama(args)?;
            }
            LayoutCmd::Stress(args) => {
                fit_layout_stress(args)?;
            }
            LayoutCmd::Forceatlas2(args) => {
                fit_layout_forceatlas2(args)?;
            }
            LayoutCmd::Diffusion(args) => {
                fit_layout_diffusion(args)?;
            }
            LayoutCmd::Radial(args) => {
                fit_layout_radial(args)?;
            }
        },
        Commands::Clustering(args) => {
            run_clustering(args)?;
        }
        #[cfg(feature = "view")]
        Commands::View(args) => view::run_view(args)?,
        #[cfg(feature = "view")]
        Commands::Mcp => legume_mcp::serve(
            legume_mcp::ServerInfo {
                name: view::mcp::SERVER.into(),
                version: env!("CARGO_PKG_VERSION").into(),
                instructions: Some(view::mcp::INSTRUCTIONS.into()),
            },
            view::mcp::Tools::default(),
        )?,
        #[cfg(feature = "view")]
        Commands::Run(args) => {
            use clap::CommandFactory;
            let mut cli = Cli::command();
            cli.build();
            run::tui::run(cli, args)?;
        }
        #[cfg(feature = "view")]
        Commands::Attach(args) => legume_tui::cli::attach(&tui::TOOL, args)?,
        #[cfg(feature = "view")]
        Commands::Sessions => legume_tui::cli::sessions(&tui::TOOL)?,
        #[cfg(feature = "view")]
        Commands::Kill(args) => legume_tui::cli::kill(&tui::TOOL, args)?,
        #[cfg(feature = "view")]
        Commands::Rename(args) => legume_tui::cli::rename(&tui::TOOL, args)?,
    }

    info!("Done");
    Ok(())
}
