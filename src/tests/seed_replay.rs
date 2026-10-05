//! A fit replays from its `--seed`: the same seed on the same data writes the
//! same latent, bit for bit.

use crate::planted::planted_zarr;
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;
use senna::embed_common::Mat;

#[derive(Parser)]
struct Cli<A: clap::Args> {
    #[command(flatten)]
    args: A,
}

fn parse<A: clap::Args>(argv: &[&str]) -> A {
    let mut full = vec!["fit"];
    full.extend_from_slice(argv);
    Cli::<A>::try_parse_from(full).expect("valid argv").args
}

fn latent(prefix: &str) -> Mat {
    Mat::from_parquet_with_row_names(&format!("{prefix}.latent.parquet"), Some(0))
        .expect("latent")
        .mat
}

/// Fit `run` into three prefixes, seeds 7, 7 and 8, on the planted data:
/// the first two latents are identical and the third is not.
fn replays<A: clap::Args>(fit: fn(&A) -> anyhow::Result<()>, extra: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let data = planted_zarr(dir.path());
    let fit_seed = |name: &str, seed: &str| -> Mat {
        let out = dir.path().join(name).to_string_lossy().into_owned();
        let mut argv = vec![data.as_str(), "-o", &out, "-i", "3", "--seed", seed];
        argv.extend_from_slice(extra);
        fit(&parse::<A>(&argv)).unwrap();
        latent(&out)
    };
    let (a, b, c) = (fit_seed("a", "7"), fit_seed("b", "7"), fit_seed("c", "8"));
    let gap = |x: &Mat, y: &Mat| (x - y).abs().max();
    assert_eq!(gap(&a, &b), 0.0, "the same seed replays");
    assert!(gap(&a, &c) > 0.0, "another seed draws another fit");
}

#[test]
fn a_masked_topic_fit_replays_from_its_seed() {
    replays(
        crate::masked_topic::fit_masked_topic_model,
        &[
            "-t",
            "3",
            "--minibatch-size",
            "50",
            "--gene-modules",
            "0",
            "--embedding-dim",
            "8",
        ],
    );
}
