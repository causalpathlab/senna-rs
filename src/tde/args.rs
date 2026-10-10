//! `senna tde`'s command line: every `senna bge` flag, plus where each cell's
//! time comes from.

use crate::bge::BgeArgs;
use clap::Args;

#[derive(Args, Debug)]
pub struct TdeArgs {
    #[command(flatten)]
    pub bge: BgeArgs,

    #[arg(
        long,
        help = "Cells' times: tab-separated, header line, barcodes in the first column",
        long_help = "Each cell's time, as a tab-separated table (optionally gzipped) with a\n\
                     header line and the barcode in the first column. The distinct values\n\
                     of --time-column are the time bins: a known stage, or a binned τ."
    )]
    pub time: Box<str>,

    #[arg(long, help = "The column of --time holding each cell's time bin")]
    pub time_column: Box<str>,

    #[arg(
        long,
        help = "Keep each pseudobulk within one time bin (cells without a time mix freely)",
        long_help = "Pass the time bins to the collapse as strata, as --cnv-clones does for\n\
                     clones: no pseudobulk mixes cells of two bins. Cells without a time are\n\
                     the mixable residual, so a held-out cell's grouping never sees its time."
    )]
    pub collapse_by_time: bool,

    #[arg(
        long,
        help = "Read --time-column as a continuous time (e.g. ode's mean τ), not as bins",
        long_help = "Read --time-column as a number per cell. A unit's time is its cells'\n\
                     mean; time neighbours are weighted by a Gaussian kernel whose width is\n\
                     the pooled spread of τ within the finest pseudobulks (how precisely\n\
                     expression pins time), so nothing is tuned."
    )]
    pub continuous: bool,
}
