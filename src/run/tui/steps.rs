//! Post-fit steps: the senna commands one runs next on a finished fit
//! (its layouts, its clusters, stage 2), each a form read from its clap
//! definition like a fit's, run in the fit's lane once it is done, with
//! `--from` its manifest.

use super::form::Method;
use senna::run_manifest::RunKind;

/// A step `senna run` offers after a fit.
#[derive(Debug)]
pub struct Spec {
    /// The command, its words a path to a nested one.
    pub name: &'static str,
    /// Whether it runs on a fit of this kind.
    offered: fn(RunKind) -> bool,
    /// Whether it reads the fit's batch files (it fails without them).
    pub needs_batches: bool,
}

/// The steps, in the order listed: t-SNE first.
pub const SPECS: [Spec; 5] = [
    Spec {
        name: "layout tsne",
        offered: RunKind::has_cells,
        needs_batches: false,
    },
    Spec {
        name: "layout umap",
        offered: RunKind::has_cells,
        needs_batches: false,
    },
    Spec {
        name: "layout phate",
        offered: RunKind::has_cells,
        needs_batches: false,
    },
    Spec {
        name: "clustering",
        offered: RunKind::has_cells,
        needs_batches: false,
    },
    Spec {
        name: "donor-deviation",
        offered: RunKind::has_stage2_inputs,
        needs_batches: true,
    },
];

/// Flags a step's form leaves out: `senna run` fills them from its fit.
const OWN: &[&str] = &["from", "out"];

/// The kind of run fit `method` (a name of [`super::METHODS`]) makes.
#[must_use]
pub fn kind_of(method: &str) -> Option<RunKind> {
    Some(match method {
        "topic" => RunKind::Topic,
        "masked-topic" | "masked-sbp" => RunKind::Itopic,
        "masked-vae" => RunKind::MaskedVae,
        "vae" => RunKind::Vae,
        "svd" => RunKind::Svd,
        "bge" => RunKind::Bge,
        "simba" => RunKind::Simba,
        "rbm" => RunKind::Rbm,
        "ode" => RunKind::Ode,
        _ => return None,
    })
}

/// One step after a fit: its form, and whether it is queued.
#[derive(Clone, Debug)]
pub struct Step {
    pub spec: &'static Spec,
    pub form: Method,
    pub on: bool,
}

impl Step {
    /// The command line, without the program: the command's words,
    /// `--from` the manifest of the fit writing `fit_out` (the step reads
    /// its cell table there and writes beside it), then every changed flag.
    #[must_use]
    pub fn argv(&self, fit_out: &str) -> Vec<String> {
        let mut argv: Vec<String> = self.spec.name.split(' ').map(String::from).collect();
        argv.extend([
            "--from".to_string(),
            senna::run_manifest::default_path(fit_out),
        ]);
        for f in &self.form.fields {
            argv.extend(f.argv());
        }
        argv
    }
}

/// The steps `cli` has that run on a fit of `method`, in [`SPECS`]' order,
/// none queued.
#[must_use]
pub fn offered(cli: &clap::Command, method: &str) -> Vec<Step> {
    let Some(kind) = kind_of(method) else {
        return Vec::new();
    };
    SPECS
        .iter()
        .filter(|spec| (spec.offered)(kind))
        .filter_map(|spec| {
            let path: Vec<&str> = spec.name.split(' ').collect();
            let cmd = crate::tui::subcommand(cli, &path)?;
            Some(Step {
                spec,
                form: Method::new_with(cmd, spec.name, OWN),
                on: false,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "tests/steps.rs"]
mod tests;
