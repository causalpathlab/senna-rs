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
        offered: takes_donor_deviation,
        needs_batches: true,
    },
];

/// Flags a step's form leaves out: `senna run` fills them from its fit.
const OWN: &[&str] = &["from", "out", "latent"];

/// Whether `senna donor-deviation` runs on a fit of `kind`: one gene axis
/// and a cell → pseudobulk partition recorded. simba, the joint fits, fne
/// and resolve-embedding-space record no partition; ode's counts are two
/// tracks. A full match, so a new kind must answer.
#[must_use]
pub fn takes_donor_deviation(kind: RunKind) -> bool {
    match kind {
        RunKind::Topic
        | RunKind::Itopic
        | RunKind::MaskedVae
        | RunKind::Vae
        | RunKind::Svd
        | RunKind::Bge
        | RunKind::Rbm => true,
        RunKind::JointTopic
        | RunKind::JointSvd
        | RunKind::Fne
        | RunKind::ResolveEmbeddingSpace
        | RunKind::Simba
        | RunKind::Ode => false,
    }
}

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

/// The table a run of `kind` keeps its cells' geometry in, after its
/// prefix: what the manifest's `geometry_latent` names once it is written.
fn cell_table(kind: RunKind) -> &'static str {
    match kind {
        RunKind::Bge | RunKind::Simba | RunKind::Ode | RunKind::ResolveEmbeddingSpace => {
            "cell_embedding.parquet"
        }
        RunKind::Topic
        | RunKind::Itopic
        | RunKind::MaskedVae
        | RunKind::JointTopic
        | RunKind::Vae
        | RunKind::Rbm
        | RunKind::Svd
        | RunKind::JointSvd
        | RunKind::Fne => "latent.parquet",
    }
}

/// One step after a fit: its form, and whether it is queued.
#[derive(Clone, Debug)]
pub struct Step {
    pub spec: &'static Spec,
    pub form: Method,
    pub on: bool,
    kind: RunKind,
    /// Whether it takes `--latent` (the fit's cell table) and needs
    /// `--out`, as its clap definition says.
    latent: bool,
    out: bool,
}

impl Step {
    /// The command line, without the program: the command's words,
    /// `--from` the manifest of the fit writing `fit_out`, its cell table
    /// and `--out` where the command takes them, then every changed flag.
    #[must_use]
    pub fn argv(&self, fit_out: &str) -> Vec<String> {
        let mut argv: Vec<String> = self.spec.name.split(' ').map(String::from).collect();
        argv.extend([
            "--from".to_string(),
            senna::run_manifest::default_path(fit_out),
        ]);
        if self.latent {
            argv.extend([
                "--latent".to_string(),
                format!("{fit_out}.{}", cell_table(self.kind)),
            ]);
        }
        if self.out {
            argv.extend(["--out".to_string(), fit_out.to_string()]);
        }
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
            let arg = |long: &str| cmd.get_arguments().find(|a| a.get_long() == Some(long));
            Some(Step {
                spec,
                form: Method::new_with(cli, spec.name, OWN).ok()?,
                on: false,
                kind,
                latent: arg("latent").is_some(),
                out: arg("out").is_some_and(clap::Arg::is_required_set),
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "tests/steps.rs"]
mod tests;
