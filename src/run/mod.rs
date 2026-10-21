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

    /// A fit given whole (`senna run -- svd data.zarr -o out`): queued at
    /// once on the Run screen, without the form, and senna run ends when
    /// it does. `senna mcp` starts a background fit this way.
    #[arg(last = true, hide = true)]
    pub fit: Vec<String>,
}

impl RunArgs {
    /// The session to run in. A folder or a fit named starts a new one;
    /// started with neither, live sessions are offered first.
    pub fn session(&self) -> legume_tui::Options {
        self.session
            .options(self.dir.is_some() || !self.fit.is_empty())
    }
}
