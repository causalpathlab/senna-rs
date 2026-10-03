//! Fits that would build the same pseudobulk partition: the first queued
//! builds it, and the rest collapse on it with `--pb-from`, so the cell
//! reassignment, the pb tree and the refinement run once.
//!
//! Which flags shape the partition is read from their clap definitions (the
//! collapse's, the cell QC's and the HVG weighting of the sketch), so a flag
//! added to one of those groups is compared without a change here.

use super::form::Method;
use super::jobs::PB_FROM_FLAG;
use clap::Args;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Flags with which a fit takes its partition from elsewhere, or cuts it
/// by clone: such a fit neither shares nor is shared.
const OWN_PARTITION: &[&str] = &["from", PB_FROM_FLAG, "cnv-clones"];

/// Whether flag `long` changes the partition a fit builds.
fn shapes(long: &str) -> bool {
    static SHAPING: OnceLock<BTreeSet<String>> = OnceLock::new();
    let set = SHAPING.get_or_init(|| {
        let groups = [
            crate::refine_weighting::CollapseArgs::augment_args(clap::Command::new("collapse")),
            data_beans::qc_lib::QcArgs::augment_args(clap::Command::new("qc")),
            crate::hvg::HvgCliArgs::augment_args(clap::Command::new("hvg")),
        ];
        groups
            .iter()
            .flat_map(|c| c.get_arguments())
            .filter_map(|a| a.get_long().map(str::to_string))
            .collect()
    });
    // The masked models' feature-axis restriction (a feature network, or
    // the genes of a given feature table) and multiome load change the
    // cells' sketch too.
    set.contains(long)
        || long == "multiome"
        || long.starts_with("feature-network")
        || long.starts_with("no-feature-network")
        || long.ends_with("-feature-embedding")
}

/// What decides the partition `form` builds: its partition-shaping flags
/// set away from their defaults. `None` when it cannot share one: it takes
/// no `--pb-from` (it builds no partition of this kind), it is given
/// `--cnv-clones` by a queued `mung clones` (`clones`), or it takes its
/// partition from elsewhere.
#[must_use]
pub fn key(form: &Method, clones: bool) -> Option<Vec<(String, String)>> {
    if clones || !form.fields.iter().any(|f| f.long == PB_FROM_FLAG) {
        return None;
    }
    let changed = |f: &&super::form::Field| !f.is_default();
    if form
        .fields
        .iter()
        .filter(changed)
        .any(|f| OWN_PARTITION.contains(&f.long.as_str()))
    {
        return None;
    }
    let mut key: Vec<(String, String)> = form
        .fields
        .iter()
        .filter(changed)
        .filter(|f| shapes(&f.long) && f.long != "proj-dim")
        .map(|f| (f.long.clone(), f.value.trim().to_string()))
        .collect();
    key.push(("proj-dim".into(), sketch_dim(form).to_string()));
    // How row names line up across data files decides the merged feature
    // axis; compared as set, since its default may differ between methods.
    if let Some(f) = form.fields.iter().find(|f| f.long == "feature-name-kind") {
        key.push((f.long.clone(), f.value.trim().to_string()));
    }
    key.sort();
    Some(key)
}

/// The sketch the fit partitions cells on is as wide as `--proj-dim`, or
/// its number of latent topics when that is larger.
fn sketch_dim(form: &Method) -> usize {
    let value = |long: &str| -> Option<usize> {
        form.fields
            .iter()
            .find(|f| f.long == long)?
            .value
            .trim()
            .parse()
            .ok()
    };
    let k = value("n-latent-topics")
        .or_else(|| value("n-latent"))
        .unwrap_or(0);
    value("proj-dim").unwrap_or(0).max(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_collapse_qc_and_hvg_flags_shape_the_partition() {
        for long in [
            "sort-dim",
            "num-levels",
            "pb-tree",
            "ignore-batch",
            "proj-dim",
        ] {
            assert!(shapes(long), "{long}");
        }
        assert!(shapes("feature-network"));
        assert!(shapes("init-feature-embedding"));
        for long in ["epochs", "n-latent-topics", "embedding-dim", "out"] {
            assert!(!shapes(long), "{long}");
        }
    }

    #[test]
    fn more_latent_topics_than_the_projection_widen_the_sketch() {
        let mut cli = clap::Command::new("senna").subcommand(
            clap::Command::new("topic")
                .arg(
                    clap::Arg::new("proj_dim")
                        .long("proj-dim")
                        .default_value("50"),
                )
                .arg(
                    clap::Arg::new("k")
                        .long("n-latent-topics")
                        .default_value("10"),
                )
                .arg(clap::Arg::new("pb_from").long(PB_FROM_FLAG)),
        );
        cli.build();
        let mut form = Method::new(&cli, "topic").unwrap();
        let narrow = key(&form, false).unwrap();
        form.fields[1].value = "30".into();
        assert_eq!(key(&form, false).unwrap(), narrow, "K under proj-dim");
        form.fields[1].value = "100".into();
        assert_ne!(key(&form, false).unwrap(), narrow, "K over proj-dim");
        // Given the clones, or another's partition, it builds its own.
        assert!(key(&form, true).is_none());
        form.fields[2].value = "run".into();
        assert!(key(&form, false).is_none());
    }
}
