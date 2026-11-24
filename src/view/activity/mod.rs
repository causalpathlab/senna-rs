//! One feature's activity per cell, for colouring the cell map by a gene.
//!
//! Two sources:
//!
//! - **expected**: what the model predicts, from tables already on disk and so
//!   instant for any feature. Embedding runs: `ρ_g · z_n`, the feature's
//!   program in each cell without the cell's depth term. Topic runs:
//!   `log Σ_k θ_nk β_gk`. Only the relative level across cells means anything.
//! - **observed**: `ln(1 + count)` from the run's data files, one row read at a
//!   time. The backend is opened on first use.
//! - **extra**: the same from a data file the run was not fitted on, opened in
//!   the view, its cells matched to the run's by name ([`match_cells`]).
//!
//! Values come back aligned to a set of points by cell name, ready to be
//! mapped onto a colour ramp. Which model row each point is, is worked out
//! once per view and cached, not per feature.

use data_beans::utilities::name_matching::GeneIndex;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;
use senna::run_manifest::{self, RunManifest};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod background_loads;
mod embedding_neighbours;
mod group_contrasts;
mod numeric;
mod sources;

pub(crate) use background_loads::*;
pub(super) use numeric::*;
pub(crate) use sources::*;

/// Per group, a score per feature; and the features, in column order.
type GroupContrasts<'a> = (Vec<Vec<f32>>, &'a [Box<str>]);

pub(super) type Vector = nalgebra::DVector<f32>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Source {
    Expected,
    Observed,
    Extra,
}

impl Source {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Source::Expected => "expected",
            Source::Observed => "observed",
            Source::Extra => "extra",
        }
    }

    /// The next source to show, `Extra` only when a file is open for it.
    #[must_use]
    pub fn next(self, extra: bool) -> Self {
        match self {
            Source::Expected => Source::Observed,
            Source::Observed if extra => Source::Extra,
            Source::Observed | Source::Extra => Source::Expected,
        }
    }

    /// Whether the values are counts (`ln(1 + count)`), a zero "not seen".
    fn counts(self) -> bool {
        self != Source::Expected
    }
}

/// A summary of each cell over every feature of a count source, or the
/// share of its counts in a feature set (a group's markers).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stat {
    /// `ln(1 + total count)`.
    Total,
    /// How many features have a count.
    Detected,
    /// The set's share of the cell's counts.
    Share(Box<str>),
}

impl Stat {
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Stat::Total => "total counts".into(),
            Stat::Detected => "features detected".into(),
            Stat::Share(group) => format!("{group} markers' share of counts"),
        }
    }
}

/// A feature axis with the shared name matcher (exact, then symbol, then
/// flexible), so a marker named `GENE1` finds a row named `ENSG…_GENE1`.
pub(super) struct Axis {
    pub(super) names: Vec<Box<str>>,
    pub(super) index: GeneIndex,
}

impl Axis {
    pub(super) fn new(names: Vec<Box<str>>) -> Self {
        let index = GeneIndex::build(&names);
        Self { names, index }
    }
}

pub struct Activity {
    manifest: RunManifest,
    dir: PathBuf,
    /// The run's slow reads, shared with every view of it.
    loads: RunLoads,
    /// Whether to wait for a read not done yet, or answer [`LOADING`]
    /// (the screen does not wait).
    patient: bool,
    /// Row of each point of a view in a source's cell table, by (source,
    /// view key).
    rows: HashMap<(Source, usize), Arc<Vec<u32>>>,
    /// The latent as topic mixtures, read on first use.
    mixture: Option<Result<Mixture, String>>,
}

/// Each cell's topic mixture (θ, rows summing to one) and the row of each
/// cell name.
struct Mixture {
    theta: Mat,
    index: HashMap<Box<str>, u32>,
}

impl std::hash::Hash for Source {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        (*self as u8).hash(h);
    }
}

fn read_table(dir: &Path, rel: &str) -> anyhow::Result<MatWithNames<Mat>> {
    let path = run_manifest::resolve(dir, rel);
    Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))
}

impl Activity {
    #[must_use]
    pub fn new(manifest: RunManifest, dir: PathBuf, loads: RunLoads, patient: bool) -> Self {
        Self {
            manifest,
            dir,
            loads,
            patient,
            rows: HashMap::default(),
            mixture: None,
        }
    }

    /// The run's latent as topic mixtures, when it is one: log θ whose
    /// rows sum to one once exponentiated. Topic runs write it so, and so
    /// does `bge` (its topics resolved from the cell embedding) unless it
    /// was told `--skip-etm`; told by the numbers, not by the run's kind.
    fn mixture(&mut self) -> Result<&Mixture, String> {
        if self.mixture.is_none() {
            self.mixture = Some(self.load_mixture());
        }
        self.mixture
            .as_ref()
            .expect("just set")
            .as_ref()
            .map_err(Clone::clone)
    }

    fn load_mixture(&self) -> Result<Mixture, String> {
        const NONE: &str =
            "a structure plot needs topics: this run's latent is not a topic mixture \
                            (topic runs have one, and bge unless --skip-etm)";
        let rel = self.manifest.outputs.latent.as_deref().ok_or(NONE)?;
        let MatWithNames { rows, mat, .. } =
            read_table(&self.dir, rel).map_err(|e| e.to_string())?;
        // Rows are the simplex here; `detect` classifies columns.
        if run_manifest::ArtifactScale::detect(&mat.transpose())
            != run_manifest::ArtifactScale::LogSimplexColumns
        {
            return Err(NONE.into());
        }
        let theta = mat.map(f32::exp);
        let index = rows
            .into_iter()
            .enumerate()
            .map(|(i, n)| (n, i as u32))
            .collect();
        Ok(Mixture { theta, index })
    }

    /// Whether the run has topic mixtures to draw.
    pub fn has_mixtures(&mut self) -> bool {
        self.mixture().is_ok()
    }

    /// Forget the per-view row maps, when the views themselves changed.
    pub fn forget_views(&mut self) {
        self.rows.clear();
    }

    /// Row of each of `names` in `source`'s cell table (`NO_ROW` if absent),
    /// cached under `key` (the caller's view id).
    fn rows(
        &mut self,
        source: Source,
        key: usize,
        names: &[Box<str>],
    ) -> Result<Arc<Vec<u32>>, String> {
        if let Some(r) = self
            .rows
            .get(&(source, key))
            .filter(|r| r.len() == names.len())
        {
            return Ok(r.clone());
        }
        let lookup = |index: &HashMap<Box<str>, u32>| -> Arc<Vec<u32>> {
            Arc::new(
                names
                    .iter()
                    .map(|n| index.get(n.as_ref()).copied().unwrap_or(NO_ROW))
                    .collect(),
            )
        };
        let rows = match source {
            Source::Expected => lookup(self.expected()?.cell_index()),
            Source::Observed | Source::Extra => lookup(&self.counts(source)?.cell_index),
        };
        self.rows.insert((source, key), rows.clone());
        Ok(rows)
    }

    fn expected(&mut self) -> Result<&Expected, String> {
        self.loads.model.answer(self.patient)
    }

    /// The data files the observed counts need that are not here, even
    /// looked for where a moved run's data would be: each one's place in
    /// `data.input` and the path recorded there.
    #[must_use]
    pub fn missing_inputs(&self) -> Vec<(usize, String)> {
        self.manifest
            .data
            .input
            .iter()
            .enumerate()
            .filter(|(_, rec)| !self.manifest.data_file(&self.dir, rec).exists())
            .map(|(i, rec)| (i, rec.clone()))
            .collect()
    }

    /// How many of `cells` the observed counts have, opening them if need
    /// be: whether a data file is this run's.
    pub fn observed_cells_found(&mut self, cells: &[Box<str>]) -> Result<usize, String> {
        let index = &self.observed()?.cell_index;
        Ok(cells
            .iter()
            .filter(|c| index.contains_key(c.as_ref()))
            .count())
    }

    /// Whether the observed counts could not be opened at all (as opposed
    /// to one feature missing from them).
    pub fn observed_failed(&self) -> bool {
        matches!(self.loads.files.try_get(), Some(Err(_)))
    }

    /// The observed counts, from the run's one read of its data files.
    fn observed(&mut self) -> Result<&Observed, String> {
        self.loads
            .files
            .answer(self.patient)
            .map(|read| &read.observed)
    }

    /// The extra file's counts, once read.
    fn extra(&mut self) -> Result<&ExtraRead, String> {
        self.loads
            .extra
            .as_ref()
            .ok_or("no extra data file is open (ctrl+e opens one)")?
            .answer(self.patient)
    }

    /// The counts behind a count source.
    fn counts(&mut self, source: Source) -> Result<&Observed, String> {
        match source {
            Source::Observed => self.observed(),
            Source::Extra => self.extra().map(|x| &x.observed),
            Source::Expected => Err("the model's expectation is not counts".into()),
        }
    }

    /// Show `extra` as the extra source from now on, in place of any other.
    pub(crate) fn set_extra(&mut self, extra: Arc<ExtraFiles>) {
        self.loads.extra = Some(extra);
        self.rows.retain(|(s, _), _| *s != Source::Extra);
    }

    /// What a count source is called in a message.
    fn where_counts(&self, source: Source) -> String {
        match (source, self.loads.extra.as_ref().and_then(|x| x.try_get())) {
            (Source::Extra, Some(Ok(x))) => x.name.to_string(),
            (Source::Extra, _) => "the extra data file".into(),
            _ => "the data files".into(),
        }
    }

    /// Feature names on `source`'s axis, for search and completion.
    pub fn feature_names(&mut self, source: Source) -> Result<&[Box<str>], String> {
        Ok(match source {
            Source::Expected => &self.expected()?.features.names,
            Source::Observed | Source::Extra => &self.counts(source)?.features.names,
        })
    }

    /// Values of `feature` per source cell, and the name the axis spells the
    /// feature with.
    fn raw(&mut self, feature: &str, source: Source) -> Result<(Vec<f32>, Box<str>), String> {
        match source {
            Source::Expected => {
                let e = self.expected()?;
                let g = e
                    .features
                    .index
                    .match_gene(feature)
                    .ok_or_else(|| format!("no feature {feature} in the model"))?;
                let values = match &e.model {
                    Model::Embedding { z, rho, .. } => {
                        (z * rho.row(g).transpose()).iter().copied().collect()
                    }
                    Model::Topic {
                        log_theta,
                        log_beta,
                    } => {
                        // log Σ_k exp(log θ_nk + log β_gk), two passes, no
                        // per-cell allocation.
                        let b = log_beta.row(g);
                        (0..log_theta.nrows())
                            .into_par_iter()
                            .map(|n| {
                                let t = log_theta.row(n);
                                let terms = || t.iter().zip(b.iter()).map(|(a, b)| a + b);
                                let m = terms().fold(f32::NEG_INFINITY, f32::max);
                                m + terms().map(|v| (v - m).exp()).sum::<f32>().ln()
                            })
                            .collect()
                    }
                };
                Ok((values, e.features.names[g].clone()))
            }
            Source::Observed | Source::Extra => {
                let place = self.where_counts(source);
                let o = self.counts(source)?;
                let g = o
                    .features
                    .index
                    .match_gene(feature)
                    .ok_or_else(|| format!("no feature {feature} in {place}"))?;
                Ok((o.row(g, f32::ln_1p)?, o.features.names[g].clone()))
            }
        }
    }

    /// `values` (per source cell) on the points `rows` maps.
    fn on_points(values: &[f32], rows: &[u32]) -> Vec<f32> {
        rows.iter()
            .map(|&r| {
                if r == NO_ROW {
                    f32::NAN
                } else {
                    values[r as usize]
                }
            })
            .collect()
    }

    /// `feature` on each of the points `names` (the view `key`).
    /// Which of `features` `source` can draw on view `key`: those on its
    /// feature axis, when any of the view's points is among its cells. The
    /// rest would show as missing values everywhere.
    pub fn drawable(
        &mut self,
        features: &[(Box<str>, f32)],
        source: Source,
        key: usize,
        names: &[Box<str>],
    ) -> Vec<bool> {
        let covers = self
            .rows(source, key, names)
            .is_ok_and(|rows| rows.iter().any(|&r| r != NO_ROW));
        let axis = match source {
            Source::Expected => self.expected().map(|e| &e.features),
            Source::Observed | Source::Extra => self.counts(source).map(|o| &o.features),
        };
        match axis {
            Ok(axis) if covers => features
                .iter()
                .map(|(f, _)| axis.index.match_gene(f).is_some())
                .collect(),
            _ => vec![false; features.len()],
        }
    }

    pub fn levels(
        &mut self,
        feature: &str,
        source: Source,
        key: usize,
        names: &[Box<str>],
    ) -> Result<(Levels, Box<str>), String> {
        let rows = self.rows(source, key, names)?;
        let (raw, spelled) = self.raw(feature, source)?;
        let values = Self::on_points(&raw, &rows);
        Ok((Levels::from_values(values, source.counts()), spelled))
    }

    /// `stat` on each of the points `names` (the view `key`), from a count
    /// source; `set` is the feature set a share is of.
    pub fn stat_levels(
        &mut self,
        stat: &Stat,
        set: &[Box<str>],
        source: Source,
        key: usize,
        names: &[Box<str>],
    ) -> Result<Levels, String> {
        if !source.counts() {
            return Err(format!(
                "{} needs counts: o shows the observed (or extra) counts",
                stat.name()
            ));
        }
        let rows = self.rows(source, key, names)?;
        // The run's own totals are a pass over its data: on the side.
        if source == Source::Observed {
            self.loads.totals.answer(self.patient)?;
        }
        let raw = self.counts(source)?.stat(stat, set)?;
        // A share starts at zero; a total or a count of features has no
        // natural origin in view, so its ramp starts at a low quantile.
        let from_zero = matches!(stat, Stat::Share(_));
        Ok(Levels::from_values(Self::on_points(&raw, &rows), from_zero))
    }

    /// Mean ramp position over a feature set: each feature is first placed on
    /// its own ramp, so one highly expressed feature cannot carry the set.
    /// Returns the levels and how many of the features were found.
    pub fn set_levels(
        &mut self,
        features: &[Box<str>],
        source: Source,
        key: usize,
        names: &[Box<str>],
    ) -> Result<(Levels, usize), String> {
        let n_points = names.len();
        let mut sum = vec![0f32; n_points];
        let mut seen = vec![false; n_points];
        let mut used = 0usize;
        for f in features {
            let Ok((levels, _)) = self.levels(f, source, key, names) else {
                continue;
            };
            used += 1;
            for i in 0..n_points {
                if levels.values[i].is_finite() {
                    seen[i] = true;
                    sum[i] += levels.t(i).max(0.0);
                }
            }
        }
        if used == 0 {
            return Err(match source {
                Source::Expected => "none of these features are in the model".into(),
                _ => format!(
                    "none of these features are in {}",
                    self.where_counts(source)
                ),
            });
        }
        let values = sum
            .iter()
            .zip(&seen)
            .map(|(&s, &ok)| if ok { s / used as f32 } else { f32::NAN })
            .collect();
        Ok((Levels::from_values(values, source.counts()), used))
    }
}

#[cfg(test)]
#[path = "../tests/activity.rs"]
mod tests;
