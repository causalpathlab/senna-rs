//! Another program's subcommand as a clap command, rebuilt from the JSON
//! its `describe` prints (`mung describe clones`). The form and the check
//! then treat it as they treat senna's own methods, while the program
//! itself stays apart: started as a child, never linked.

use clap::builder::{PossibleValuesParser, ValueRange};
use clap::{value_parser, Arg, ArgAction, Command};
use serde::Deserialize;

/// The `describe` format this reads.
const FORMAT: u64 = 1;

#[derive(Debug, Deserialize)]
pub struct Description {
    describe: u64,
    pub program: String,
    pub version: String,
    pub command: String,
    #[serde(default)]
    about: Option<String>,
    args: Vec<Described>,
}

#[derive(Debug, Deserialize)]
struct Described {
    id: String,
    long: Option<String>,
    short: Option<String>,
    #[serde(default)]
    positional: bool,
    help: Option<String>,
    long_help: Option<String>,
    action: String,
    #[serde(default)]
    value_type: Option<String>,
    #[serde(default)]
    values: Vec<String>,
    #[serde(default)]
    default: Vec<String>,
    delimiter: Option<String>,
    num_args_min: Option<usize>,
    num_args_max: Option<usize>,
    #[serde(default)]
    hidden: bool,
    #[serde(default)]
    required: bool,
}

/// Lives as long as the command: clap keeps `'static` names, and a
/// description is read once a run.
fn keep(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

impl Description {
    pub fn parse(json: &str) -> anyhow::Result<Self> {
        let d: Description = serde_json::from_str(json)?;
        anyhow::ensure!(
            d.describe == FORMAT,
            "{} describes its flags in format {}; senna reads {FORMAT}",
            d.program,
            d.describe
        );
        Ok(d)
    }

    /// The program as a built clap command with this one subcommand.
    #[must_use]
    pub fn command(&self) -> Command {
        let mut sub = Command::new(keep(&self.command));
        if let Some(about) = &self.about {
            sub = sub.about(keep(about));
        }
        for a in &self.args {
            if let Some(arg) = a.arg() {
                sub = sub.arg(arg);
            }
        }
        let mut cmd = Command::new(keep(&self.program))
            .version(keep(&self.version))
            .subcommand(sub);
        cmd.build();
        cmd
    }
}

impl Described {
    fn arg(&self) -> Option<Arg> {
        let action = match self.action.as_str() {
            "set_true" => ArgAction::SetTrue,
            "set_false" => ArgAction::SetFalse,
            "set" => ArgAction::Set,
            "append" => ArgAction::Append,
            "count" => ArgAction::Count,
            // Help and version come with every clap command.
            _ => return None,
        };
        let mut a = Arg::new(keep(&self.id)).action(action.clone());
        if !self.positional {
            a = a.long(keep(self.long.as_deref()?));
        }
        if let Some(c) = self.short.as_deref().and_then(|s| s.chars().next()) {
            a = a.short(c);
        }
        if let Some(h) = &self.help {
            a = a.help(keep(h));
        }
        if let Some(h) = &self.long_help {
            a = a.long_help(keep(h));
        }
        if matches!(action, ArgAction::Set | ArgAction::Append) {
            a = if self.values.is_empty() {
                match self.value_type.as_deref() {
                    Some("unsigned") => a.value_parser(value_parser!(u64)),
                    Some("integer") => a.value_parser(value_parser!(i64)),
                    Some("number") => a.value_parser(value_parser!(f64)),
                    _ => a,
                }
            } else {
                let values: Vec<&'static str> = self.values.iter().map(|v| keep(v)).collect();
                a.value_parser(PossibleValuesParser::new(values))
            };
            if let Some(min) = self.num_args_min {
                a = a.num_args(match self.num_args_max {
                    Some(max) => ValueRange::new(min..=max.max(min)),
                    None => ValueRange::new(min..),
                });
            }
            if let Some(c) = self.delimiter.as_deref().and_then(|s| s.chars().next()) {
                a = a.value_delimiter(c);
            }
            if !self.default.is_empty() {
                let d: Vec<&'static str> = self.default.iter().map(|v| keep(v)).collect();
                a = a.default_values(d);
            }
        }
        Some(a.hide(self.hidden).required(self.required))
    }
}

#[cfg(test)]
#[path = "tests/described.rs"]
mod tests;
