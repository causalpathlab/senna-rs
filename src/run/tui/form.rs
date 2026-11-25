//! A method's flags as rows of a form, read from its clap definition, and
//! the command line the filled form stands for.
//!
//! Nothing here names a flag of any one method: every row comes from the
//! subcommand's `Arg`s, so a flag added to a method shows up on its own.

use clap::Command;
pub use legume_tui::form::{blamed, Field, Kind};
use legume_tui::form::{emit, many_of, Many};

/// Flags `senna run` fills from its own screens or never passes.
const OWN: &[&str] = &["out", "batch-files", "help", "version", "verbose"];

/// Whether `f` names an earlier run's feature table
/// (`--{freeze,init,lora}-feature-embedding`), so enter browses for one.
#[must_use]
pub fn takes_table(f: &Field) -> bool {
    f.kind == Kind::Text && f.value_name == "PREFIX"
}

/// One embedding method as a form.
#[derive(Clone, Debug)]
pub struct Method {
    pub name: String,
    pub about: String,
    pub fields: Vec<Field>,
    /// How the batch files are passed, when the method takes them.
    batch: Option<Many>,
}

impl Method {
    /// The form for subcommand `name` of `cli`, which must be built, its
    /// words a path to a nested one (`layout tsne`).
    pub fn new(cli: &Command, name: &str) -> anyhow::Result<Self> {
        let path: Vec<&str> = name.split(' ').collect();
        let cmd = crate::tui::subcommand(cli, &path)
            .ok_or_else(|| anyhow::anyhow!("senna has no `{name}` command"))?;
        Ok(Self::new_with(cmd, name, &[]))
    }

    /// The form for `cmd`, the command `name` names, without the flags
    /// `own` names besides [`OWN`]: those the caller fills itself.
    pub fn new_with(cmd: &Command, name: &str, own: &[&str]) -> Self {
        let batch = cmd
            .get_arguments()
            .find(|a| a.get_long() == Some("batch-files"))
            .map(many_of);
        // The method's own flags first, then those of the data steps every
        // method shares, each in clap's order.
        let data = data_flags();
        let (own, shared): (Vec<Field>, Vec<Field>) = cmd
            .get_arguments()
            .filter(|a| {
                a.get_long()
                    .is_some_and(|l| !OWN.contains(&l) && !own.contains(&l))
            })
            .filter_map(Field::from_arg)
            .partition(|f| !data.contains(f.long.as_str()));
        Method {
            name: name.to_string(),
            about: cmd.get_about().map(ToString::to_string).unwrap_or_default(),
            fields: own.into_iter().chain(shared).collect(),
            batch,
        }
    }

    /// The field of flag `--{long}`, when the method has it.
    #[must_use]
    pub fn field(&self, long: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.long == long)
    }

    /// Whether the method reads batch files.
    #[must_use]
    pub fn takes_batches(&self) -> bool {
        self.batch.is_some()
    }

    /// Flags changed from their defaults.
    #[must_use]
    pub fn changed(&self) -> usize {
        self.fields.iter().filter(|f| !f.is_default()).count()
    }

    /// The command line, without the program: method, data files, batch
    /// files, `--out`, then every changed flag.
    #[must_use]
    pub fn argv(&self, data: &[String], batches: &[String], out: &str) -> Vec<String> {
        let mut argv = vec![self.name.clone()];
        argv.extend(data.iter().cloned());
        if let (Some(many), false) = (self.batch, batches.is_empty()) {
            argv.extend(emit("--batch-files", many, batches.to_vec()));
        }
        argv.extend(["--out".to_string(), out.to_string()]);
        for f in &self.fields {
            argv.extend(f.argv());
        }
        argv
    }
}

/// Whether clap takes `argv` (without the program); its complaint if not.
pub fn check(cli: &Command, argv: &[String]) -> Result<(), String> {
    // The method's own command parses its line: no need to copy all of senna.
    // Its leading words name it, a nested one (`layout tsne`) by several.
    let words: Vec<&str> = argv.iter().map(String::as_str).collect();
    let named = (1..=words.len())
        .map_while(|n| crate::tui::subcommand(cli, &words[..n]).map(|m| (n, m)))
        .last();
    let Some((n, method)) = named else {
        return Err(format!(
            "senna has no `{}` command",
            words.first().copied().unwrap_or_default()
        ));
    };
    legume_tui::form::check(method, &argv[n - 1..])
}

/// The long names of the data steps' flags (`--from`, `--multiome`, quality
/// control, highly variable features, the collapse into pseudobulks): the same
/// in every method that loads counts, so a method's page lists them after its
/// own.
fn data_flags() -> &'static std::collections::HashSet<String> {
    static FLAGS: std::sync::OnceLock<std::collections::HashSet<String>> =
        std::sync::OnceLock::new();
    FLAGS.get_or_init(build_data_flags)
}

fn build_data_flags() -> std::collections::HashSet<String> {
    use clap::Args;
    let mut c = Command::new("data");
    c = crate::refine_weighting::CollapseArgs::augment_args(c);
    c = crate::refine_weighting::PbFromArgs::augment_args(c);
    c = data_beans::alg::hvg::HvgCliArgs::augment_args(c);
    c = senna::embed_common::QcArgs::augment_args(c);
    c.get_arguments()
        .filter_map(|a| a.get_long().map(str::to_string))
        .chain(["from", "multiome", "preload-data", "block-size"].map(String::from))
        .collect()
}

#[cfg(test)]
#[path = "tests/form.rs"]
mod tests;
