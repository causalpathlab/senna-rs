//! A method's flags as rows of a form, read from its clap definition, and
//! the command line the filled form stands for.
//!
//! Nothing here names a flag of any one method: every row comes from the
//! subcommand's `Arg`s, so a flag added to a method shows up on its own.

use clap::{Arg, ArgAction, Command};

/// Flags `senna run` fills from its own screens or never passes.
const OWN: &[&str] = &["out", "batch-files", "help", "version", "verbose"];

/// What a row holds and how it is changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A switch, given or not. `on` is the value it sets when given.
    Flag { on: bool },
    /// One of a fixed set of values; `""` stands for unset.
    Choice(Vec<String>),
    /// Free text.
    Text,
}

/// How several values reach the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Many {
    One,
    /// Joined into one argument: `--flag a,b`.
    Joined(char),
    /// Listed after one flag: `--flag a b`.
    Listed,
    /// The flag repeated: `--flag a --flag b`.
    Repeated,
}

/// One flag of a method.
#[derive(Clone, Debug)]
pub struct Field {
    /// The long name, without `--`.
    pub long: String,
    pub help: String,
    pub long_help: String,
    pub kind: Kind,
    /// The value clap would use when the flag is not given; `""` for none.
    pub default: String,
    pub value: String,
    /// Hidden from `--help`: shown only with the advanced rows.
    pub advanced: bool,
    pub required: bool,
    many: Many,
}

impl Field {
    fn from_arg(a: &Arg) -> Option<Self> {
        let long = a.get_long()?.to_string();
        if OWN.contains(&long.as_str()) {
            return None;
        }
        let kind = match a.get_action() {
            ArgAction::SetTrue => Kind::Flag { on: true },
            ArgAction::SetFalse => Kind::Flag { on: false },
            ArgAction::Set | ArgAction::Append => {
                let values: Vec<String> = a
                    .get_possible_values()
                    .iter()
                    .filter(|v| !v.is_hide_set())
                    .map(|v| v.get_name().to_string())
                    .collect();
                if values.is_empty() {
                    Kind::Text
                } else {
                    Kind::Choice(values)
                }
            }
            // Counters, help and version: not something to fill in.
            _ => return None,
        };
        let many = many_of(a);
        let sep = match many {
            Many::Joined(c) => c.to_string(),
            _ => ",".to_string(),
        };
        let default = a
            .get_default_values()
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(&sep);
        let mut kind = kind;
        if let Kind::Choice(values) = &mut kind {
            // A choice with no default can stay unset.
            if default.is_empty() {
                values.insert(0, String::new());
            }
        }
        Some(Field {
            help: a.get_help().map(ToString::to_string).unwrap_or_default(),
            long_help: a
                .get_long_help()
                .or_else(|| a.get_help())
                .map(ToString::to_string)
                .unwrap_or_default(),
            value: default.clone(),
            default,
            advanced: a.is_hide_set(),
            required: a.is_required_set(),
            kind,
            many,
            long,
        })
    }

    /// Whether the value is the one clap would use anyway.
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.value.trim() == self.default
    }

    /// Back to what clap would use.
    pub fn reset(&mut self) {
        self.value.clone_from(&self.default);
    }

    /// Flip a switch.
    pub fn toggle(&mut self) {
        if let Kind::Flag { .. } = self.kind {
            self.value = if self.value == "true" {
                "false"
            } else {
                "true"
            }
            .into();
        }
    }

    /// Step a choice by `d`, wrapping around.
    pub fn cycle(&mut self, d: isize) {
        if let Kind::Choice(values) = &self.kind {
            let n = values.len() as isize;
            let at = values.iter().position(|v| *v == self.value).unwrap_or(0) as isize;
            self.value
                .clone_from(&values[(at + d).rem_euclid(n) as usize]);
        }
    }

    /// The value as shown: `on` / `off` for a switch, `(unset)` for nothing.
    #[must_use]
    pub fn shown(&self) -> String {
        match self.kind {
            Kind::Flag { .. } => if self.value == "true" { "on" } else { "off" }.into(),
            _ if self.value.trim().is_empty() => "(unset)".into(),
            _ => self.value.clone(),
        }
    }

    /// What this row adds to the command line: nothing when it is the
    /// default.
    #[must_use]
    pub fn argv(&self) -> Vec<String> {
        if self.is_default() {
            return Vec::new();
        }
        let flag = format!("--{}", self.long);
        match self.kind {
            Kind::Flag { on } => {
                if (self.value == "true") == on {
                    vec![flag]
                } else {
                    Vec::new()
                }
            }
            Kind::Choice(_) | Kind::Text => emit(&flag, self.many, values(self.many, &self.value)),
        }
    }
}

fn many_of(a: &Arg) -> Many {
    if let Some(c) = a.get_value_delimiter() {
        Many::Joined(c)
    } else if a.get_num_args().is_some_and(|r| r.max_values() > 1) {
        Many::Listed
    } else if matches!(a.get_action(), ArgAction::Append) {
        Many::Repeated
    } else {
        Many::One
    }
}

/// The values typed into a row: split on commas and spaces when the flag
/// takes several, else the whole text.
fn values(many: Many, text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    match many {
        Many::One => vec![text.to_string()],
        _ => text
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .collect(),
    }
}

/// `flag` with values `vs`, laid out the way the flag takes them.
fn emit(flag: &str, many: Many, vs: Vec<String>) -> Vec<String> {
    if vs.is_empty() {
        return Vec::new();
    }
    match many {
        Many::One => vec![flag.to_string(), vs[0].clone()],
        Many::Joined(c) => vec![flag.to_string(), vs.join(&c.to_string())],
        Many::Listed => std::iter::once(flag.to_string()).chain(vs).collect(),
        Many::Repeated => vs.into_iter().flat_map(|v| [flag.to_string(), v]).collect(),
    }
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
    /// The form for subcommand `name` of `cli`, which must be built.
    pub fn new(cli: &Command, name: &str) -> anyhow::Result<Self> {
        let cmd = cli
            .find_subcommand(name)
            .ok_or_else(|| anyhow::anyhow!("senna has no `{name}` command"))?;
        let batch = cmd
            .get_arguments()
            .find(|a| a.get_long() == Some("batch-files"))
            .map(many_of);
        Ok(Method {
            name: name.to_string(),
            about: cmd.get_about().map(ToString::to_string).unwrap_or_default(),
            fields: cmd.get_arguments().filter_map(Field::from_arg).collect(),
            batch,
        })
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
    let method = argv
        .first()
        .and_then(|m| cli.find_subcommand(m))
        .ok_or_else(|| {
            format!(
                "senna has no `{}` command",
                argv.first().map_or("", |m| m.as_str())
            )
        })?;
    match method.clone().try_get_matches_from(argv) {
        Ok(_) => Ok(()),
        Err(e) => Err(complaint(&e.render().to_string())),
    }
}

/// The first line of a clap error, without its `error: ` lead.
fn complaint(rendered: &str) -> String {
    let first = rendered.lines().next().unwrap_or_default().trim();
    first.strip_prefix("error: ").unwrap_or(first).to_string()
}

/// The flag a clap complaint is about, when it names one of `fields`.
#[must_use]
pub fn blamed<'a>(complaint: &str, fields: &'a [Field]) -> Option<&'a str> {
    fields
        .iter()
        .map(|f| f.long.as_str())
        .filter(|l| {
            complaint.contains(&format!("'--{l}")) || complaint.contains(&format!("--{l} "))
        })
        .max_by_key(|l| l.len())
}

#[cfg(test)]
#[path = "tests/form.rs"]
mod tests;
