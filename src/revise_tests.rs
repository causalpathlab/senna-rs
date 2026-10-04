//! `senna revise`'s arguments.

use super::ReviseArgs;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: ReviseArgs,
}

fn parse(extra: &[&str]) -> Result<ReviseArgs, clap::Error> {
    let base = [
        "senna-revise",
        "--model",
        "m",
        "--labels",
        "l.parquet",
        "-o",
        "out",
    ];
    Cli::try_parse_from(base.iter().copied().chain(extra.iter().copied())).map(|c| c.args)
}

/// A revision needs a model, its labels and an output; the rest defaults to a
/// small step that pushes merged pairs to the critique's own "far".
#[test]
fn a_revision_needs_labels_and_defaults_the_rest() {
    let a = parse(&[]).expect("parses");
    assert_eq!(a.far_frac, 0.25);
    assert!(a.pb_from.is_none());
    let r = a.revision().expect("defaults are valid");
    assert_eq!(r.0.labels, "l.parquet");
    assert!(Cli::try_parse_from(["senna-revise", "--model", "m", "-o", "out"]).is_err());
}

#[test]
fn out_of_range_settings_are_refused() {
    for bad in [
        ["--far-frac", "1"],
        ["--far-frac", "0"],
        ["--epochs", "0"],
        ["--pair-batch", "0"],
        ["--learning-rate", "0"],
        ["--max-llik-drop=-0.1", "--epochs=1"],
    ] {
        let a = parse(&bad).expect("parses");
        assert!(a.revision().is_err(), "{} {}", bad[0], bad[1]);
    }
}
