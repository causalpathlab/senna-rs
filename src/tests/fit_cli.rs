//! What the end-to-end fit tests share: a command's arguments from an argv,
//! and the latent a run wrote. Kept apart from `planted`, which the
//! integration tests also mount and which must not carry these unused.

use clap::Parser;
use legume_numeric::matrix::traits::IoOps;
use senna::embed_common::Mat;

#[derive(Parser)]
struct Cli<A: clap::Args> {
    #[command(flatten)]
    args: A,
}

/// A command's arguments from `argv` (without the program name).
pub(crate) fn parse<A: clap::Args>(argv: &[&str]) -> A {
    let mut full = vec!["senna"];
    full.extend_from_slice(argv);
    Cli::<A>::try_parse_from(full).expect("valid argv").args
}

/// The latent a run wrote under `prefix`.
pub(crate) fn latent(prefix: &str) -> Mat {
    Mat::from_parquet_with_row_names(&format!("{prefix}.latent.parquet"), Some(0))
        .expect("latent")
        .mat
}
