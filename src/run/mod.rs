//! `senna run`: set up embedding fits in the terminal and run them.
//!
//! Pick data and batch files, queue one or more methods, change their
//! flags, check the exact commands, and run them in turn. Each command is
//! kept as `{out}.cmd.sh`, a script that runs it again and refuses to run
//! over a result already there.

pub(crate) mod tui;

#[derive(clap::Args, Debug)]
pub struct RunArgs {
    #[arg(default_value = ".", help = "Folder the file browser starts in")]
    pub dir: std::path::PathBuf,
}
