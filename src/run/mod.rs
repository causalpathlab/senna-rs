//! `senna run`: set up embedding fits in the terminal and run them.
//!
//! Pick data and batch files, queue one or more methods, change their
//! flags, check the exact commands, and run them in turn. Each command is
//! kept as `{out}.cmd.sh`, a script that runs it again and refuses to run
//! over a result already there.

pub(crate) mod tui;

#[derive(clap::Args, Debug)]
pub struct RunArgs {
    #[arg(help = "Folder the file browser starts in [default: .]")]
    pub dir: Option<std::path::PathBuf>,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,
}

impl RunArgs {
    /// The session to run in. A folder named starts a new one; started
    /// with none, live sessions are offered first.
    pub fn session(&self) -> legume_tui::Options {
        self.session.options(self.dir.is_some())
    }
}
