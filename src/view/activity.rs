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
//!
//! Values come back aligned to a set of points by cell name, ready to be
//! mapped onto a colour ramp. Which model row each point is, is worked out
//! once per view and cached, not per feature.

use data_beans::utilities::name_matching::GeneIndex;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;
use senna::run_manifest::{self, RunManifest};
use senna::senna_input::{read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch};
use std::cell::OnceCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Per group, a score per feature; and the features, in column order.
type GroupContrasts<'a> = (Vec<Vec<f32>>, &'a [Box<str>]);

type Vector = nalgebra::DVector<f32>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Source {
    Expected,
    Observed,
}

impl Source {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Source::Expected => "expected",
            Source::Observed => "observed",
        }
    }

    #[must_use]
    pub fn other(self) -> Self {
        match self {
            Source::Expected => Source::Observed,
            Source::Observed => Source::Expected,
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

enum Model {
    /// `z` is N × H, `rho` is D × H.
    /// `bias` is the per-feature baseline log level, when the run has one.
    Embedding {
        z: Mat,
        rho: Mat,
        bias: Option<Vec<f32>>,
    },
    /// `log_theta` is N × K, `log_beta` is D × K.
    Topic { log_theta: Mat, log_beta: Mat },
}

struct Expected {
    features: Axis,
    cells: Vec<Box<str>>,
    model: Model,
    /// Each feature's expected variance over all cells (embedding runs), for
    /// the "varies most here" ranking; computed on first use.
    variance_everywhere: OnceCell<Vec<f32>>,
    /// θ and β out of log space (topic runs); computed on first use.
    topic_linear: OnceCell<(Mat, Mat)>,
    /// Median of the finite baselines (`None`: no usable baseline).
    bias_median: OnceCell<Option<f32>>,
    /// Mean of `z` over all cells (embedding runs).
    z_mean: OnceCell<Vector>,
    /// Row of each cell name.
    cell_index: OnceCell<HashMap<Box<str>, u32>>,
}

impl Expected {
    fn new(features: Axis, cells: Vec<Box<str>>, model: Model) -> Self {
        Self {
            features,
            cells,
            model,
            variance_everywhere: OnceCell::new(),
            topic_linear: OnceCell::new(),
            bias_median: OnceCell::new(),
            z_mean: OnceCell::new(),
            cell_index: OnceCell::new(),
        }
    }

    /// Mean of `z` (this model's cell embedding) over all cells.
    fn z_mean(&self, z: &Mat) -> &Vector {
        self.z_mean
            .get_or_init(|| row_mean(z, &(0..z.nrows()).collect::<Vec<_>>()))
    }

    fn cell_index(&self) -> &HashMap<Box<str>, u32> {
        self.cell_index.get_or_init(|| {
            self.cells
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i as u32))
                .collect()
        })
    }

    /// The per-feature baseline and its median, when the run has one.
    fn baseline(&self) -> Option<(&[f32], f32)> {
        let Model::Embedding { bias: Some(b), .. } = &self.model else {
            return None;
        };
        let median = self.bias_median.get_or_init(|| median_of(b));
        Some((b, (*median)?))
    }

    /// θ and β out of log space (topic runs).
    fn linear(&self) -> Option<&(Mat, Mat)> {
        let Model::Topic {
            log_theta,
            log_beta,
        } = &self.model
        else {
            return None;
        };
        Some(
            self.topic_linear
                .get_or_init(|| (log_theta.map(f32::exp), log_beta.map(f32::exp))),
        )
    }

    /// The per-cell table a group contrast is linear in: `z`, or θ.
    fn table(&self) -> &Mat {
        match (&self.model, self.linear()) {
            (Model::Embedding { z, .. }, _) => z,
            (_, Some((theta, _))) => theta,
            _ => unreachable!("a topic run has linear tables"),
        }
    }

    /// Each feature's score, cells summing to `inside` (`n_in` of them) over
    /// cells summing to `rest`: the expected log fold change.
    fn score(&self, inside: &Vector, n_in: usize, rest: &Vector, n_rest: usize) -> Vec<f32> {
        let a = inside / n_in.max(1) as f32;
        let b = rest / n_rest.max(1) as f32;
        match (&self.model, self.linear()) {
            (Model::Embedding { rho, .. }, _) => {
                let mut v: Vec<f32> = (rho * (a - b)).iter().copied().collect();
                drop_below_median(&mut v, self.baseline());
                v
            }
            (_, Some((_, beta))) => {
                let (x, y) = (beta * a, beta * b);
                x.iter()
                    .zip(y.iter())
                    .map(|(&x, &y)| (x.max(1e-12) / y.max(1e-12)).ln())
                    .collect()
            }
            _ => unreachable!("a topic run has linear tables"),
        }
    }
}

/// Per-group sums of the per-cell table a group contrast is linear in, so
/// any union of groups scores without another pass over the cells.
pub struct GroupSums {
    sums: Vec<Vector>,
    counts: Vec<usize>,
    /// Cells in view with no group; they count as the rest of the view.
    none: Vector,
    none_count: usize,
}

/// A point with no row in a source's cell table.
const NO_ROW: u32 = u32::MAX;

const NO_CONTRAST: &str = "the focused group has no cells, or no other cells are in view";

struct Observed {
    data: SparseIoVec,
    features: Axis,
    cells: Vec<Box<str>>,
}

/// Values of one feature (or feature set) per point, `NaN` where the point
/// has no value, and the range the colour ramp spans.
#[derive(Clone)]
pub struct Levels {
    pub values: Vec<f32>,
    pub lo: f32,
    pub hi: f32,
    /// Observed counts: a zero means "not detected" and is drawn muted rather
    /// than at the bottom of the ramp.
    pub zero_is_off: bool,
}

impl Levels {
    /// Position on the ramp in `[0, 1]`, or a negative number for "off".
    #[inline]
    #[must_use]
    pub fn t(&self, i: usize) -> f32 {
        let v = self.values[i];
        if !v.is_finite() || (self.zero_is_off && v <= 0.0) {
            return -1.0;
        }
        ((v - self.lo) / (self.hi - self.lo).max(1e-12)).clamp(0.0, 1.0)
    }

    fn from_values(values: Vec<f32>, zero_is_off: bool) -> Self {
        let mut sample: Vec<f32> = values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && (!zero_is_off || *v > 0.0))
            .collect();
        let (lo, hi) = if sample.is_empty() {
            (0.0, 1.0)
        } else {
            // Two linear-time selections, not a full sort.
            let mut q = |p: f32| {
                let k = ((sample.len() - 1) as f32 * p) as usize;
                *sample.select_nth_unstable_by(k, f32::total_cmp).1
            };
            let hi = q(0.99);
            // Counts start the ramp at zero; model levels at a low quantile,
            // since their scale has no natural origin.
            let lo = if zero_is_off { 0.0 } else { q(0.02) };
            (lo, hi.max(lo + 1e-6))
        };
        Self {
            values,
            lo,
            hi,
            zero_is_off,
        }
    }
}

pub struct Activity {
    manifest: RunManifest,
    dir: PathBuf,
    expected: Option<Result<Expected, String>>,
    observed: Option<Result<Observed, String>>,
    /// Row of each point of a view in a source's cell table, by (source,
    /// view key).
    rows: HashMap<(Source, usize), Arc<Vec<u32>>>,
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
    pub fn new(manifest: RunManifest, dir: PathBuf) -> Self {
        Self {
            manifest,
            dir,
            expected: None,
            observed: None,
            rows: HashMap::default(),
        }
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
            Source::Observed => {
                let cells = &self.observed()?.cells;
                let index: HashMap<Box<str>, u32> = cells
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), i as u32))
                    .collect();
                lookup(&index)
            }
        };
        self.rows.insert((source, key), rows.clone());
        Ok(rows)
    }

    fn load_expected(&self) -> anyhow::Result<Expected> {
        let m = &self.manifest;
        let o = &m.outputs;
        if let (Some(z_rel), Ok((rho_path, bias_path))) = (
            o.cell_embedding.as_deref(),
            run_manifest::resolve_feature_embedding_for(m, &self.dir),
        ) {
            let z = read_table(&self.dir, z_rel)?;
            let rho = Mat::from_parquet_with_row_names(&rho_path, Some(0))?;
            anyhow::ensure!(
                z.mat.ncols() == rho.mat.ncols(),
                "cell embedding has {} dims, feature embedding {}",
                z.mat.ncols(),
                rho.mat.ncols()
            );
            // Only used when it lines up with the feature table.
            let bias = bias_path
                .and_then(|p| Mat::from_parquet_with_row_names(&p, Some(0)).ok())
                .filter(|b| b.rows == rho.rows && b.mat.ncols() >= 1)
                .map(|b| b.mat.column(0).iter().copied().collect());
            return Ok(Expected::new(
                Axis::new(rho.rows),
                z.rows,
                Model::Embedding {
                    z: z.mat,
                    rho: rho.mat,
                    bias,
                },
            ));
        }
        if let (true, Some(theta_rel), Some(beta_rel)) = (
            m.kind.latent_is_log_simplex(),
            o.latent.as_deref(),
            o.gene_dictionary(),
        ) {
            let theta = read_table(&self.dir, theta_rel)?;
            let beta = read_table(&self.dir, beta_rel)?;
            anyhow::ensure!(
                theta.mat.ncols() == beta.mat.ncols(),
                "latent has {} topics, dictionary {}",
                theta.mat.ncols(),
                beta.mat.ncols()
            );
            return Ok(Expected::new(
                Axis::new(beta.rows),
                theta.rows,
                Model::Topic {
                    log_theta: theta.mat,
                    log_beta: beta.mat,
                },
            ));
        }
        anyhow::bail!("this run has no model tables to predict a feature from")
    }

    fn load_observed(&self) -> anyhow::Result<Observed> {
        let m = &self.manifest;
        anyhow::ensure!(!m.data.input.is_empty(), "the manifest lists no data files");
        let files = m.data_inputs(&self.dir);
        // Observed counts are the one thing here that needs the data.
        if let Some(gone) = files.iter().find(|f| !Path::new(f.as_ref()).exists()) {
            anyhow::bail!("the data is not here ({gone})");
        }
        let reload =
            senna::multiome_layout::recorded_layout(m.data.multiome.as_ref(), files.len())?;
        info!(
            "view: opening {} data file(s) for observed counts",
            files.len()
        );
        let SparseDataWithBatch { data, .. } =
            read_data_on_shared_rows(reload.apply(ReadSharedRowsArgs {
                data_files: files,
                keep_empty_barcodes: true,
                ..Default::default()
            })?)?;
        Ok(Observed {
            features: Axis::new(data.row_names()?),
            cells: data.column_names()?,
            data,
        })
    }

    fn expected(&mut self) -> Result<&Expected, String> {
        if self.expected.is_none() {
            self.expected = Some(self.load_expected().map_err(|e| e.to_string()));
        }
        self.expected
            .as_ref()
            .expect("just set")
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Whether the observed counts could not be opened at all (as opposed
    /// to one feature missing from them).
    pub fn observed_failed(&self) -> bool {
        matches!(self.observed, Some(Err(_)))
    }

    fn observed(&mut self) -> Result<&Observed, String> {
        if self.observed.is_none() {
            self.observed = Some(self.load_observed().map_err(|e| e.to_string()));
        }
        self.observed
            .as_ref()
            .expect("just set")
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Feature names on `source`'s axis, for search and completion.
    pub fn feature_names(&mut self, source: Source) -> Result<&[Box<str>], String> {
        Ok(match source {
            Source::Expected => &self.expected()?.features.names,
            Source::Observed => &self.observed()?.features.names,
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
            Source::Observed => {
                let o = self.observed()?;
                let g = o
                    .features
                    .index
                    .match_gene(feature)
                    .ok_or_else(|| format!("no feature {feature} in the data files"))?;
                let row = o
                    .data
                    .read_rows_csr(std::iter::once(g))
                    .map_err(|e| e.to_string())?;
                let mut values = vec![0f32; o.cells.len()];
                for (&c, &v) in row.col_indices().iter().zip(row.values()) {
                    values[c] = v.ln_1p();
                }
                Ok((values, o.features.names[g].clone()))
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
        Ok((
            Levels::from_values(values, source == Source::Observed),
            spelled,
        ))
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
                Source::Observed => "none of these features are in the data files".into(),
            });
        }
        let values = sum
            .iter()
            .zip(&seen)
            .map(|(&s, &ok)| if ok { s / used as f32 } else { f32::NAN })
            .collect();
        Ok((
            Levels::from_values(values, source == Source::Observed),
            used,
        ))
    }

    /// Every feature's score for a view, in the model's feature order, with
    /// the features' names. `-inf` marks a feature left out.
    ///
    /// `universe` names the cells in view (the view `key`); `group` marks, in
    /// the same order, the focused group's cells. With a group, a feature
    /// scores by its expected log fold change, group over the rest of the
    /// view (for an embedding run, `ρ_g · (mean z_group − mean z_rest)`).
    /// Without one, by how much it varies across the view relative to how
    /// much it varies over all cells, among features at least as variable as
    /// the median, which points at what shapes a zoomed-in layout. Where the
    /// run records a per-feature baseline, features below its median are left
    /// out, so rarely expressed features with noisy loadings do not lead.
    ///
    /// A nudge from the model, not a test: `o` shows the observed counts.
    pub fn contrast(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        group: Option<&[bool]>,
    ) -> Result<(Vec<f32>, &[Box<str>]), String> {
        let rows = self.rows(Source::Expected, key, universe)?;
        let e = self.expected()?;
        let pick = |want: Option<bool>| -> Vec<usize> {
            rows.iter()
                .enumerate()
                .filter(|&(k, &r)| {
                    r != NO_ROW && want.is_none_or(|w| group.is_some_and(|g| g[k] == w))
                })
                .map(|(_, &r)| r as usize)
                .collect()
        };
        let scores: Vec<f32> = match (&e.model, group) {
            (_, Some(_)) => {
                let (inside, rest) = (pick(Some(true)), pick(Some(false)));
                if inside.is_empty() || rest.is_empty() {
                    return Err(NO_CONTRAST.into());
                }
                let table = e.table();
                e.score(
                    &row_sum(table, &inside),
                    inside.len(),
                    &row_sum(table, &rest),
                    rest.len(),
                )
            }
            (Model::Embedding { z, rho, .. }, None) => {
                let cells = pick(None);
                // A view of every cell has nothing to compare against, so
                // its variance ranks alone; a zoomed view ranks by how much
                // more a feature varies there than everywhere.
                let whole = cells.len() == z.nrows();
                let here = quadratic_forms(rho, &row_cov(z, &cells));
                let everywhere = e.variance_everywhere.get_or_init(|| {
                    let all: Vec<usize> = (0..z.nrows()).collect();
                    quadratic_forms(rho, &row_cov(z, &all))
                });
                let mut sorted = everywhere.clone();
                sorted.sort_unstable_by(f32::total_cmp);
                let floor = sorted[sorted.len() / 2];
                let mut ratio: Vec<f32> = here
                    .iter()
                    .zip(everywhere)
                    .map(|(&h, &a)| {
                        if a >= floor && a > 0.0 {
                            if whole {
                                h
                            } else {
                                h / a
                            }
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                drop_below_median(&mut ratio, e.baseline());
                ratio
            }
            (Model::Topic { .. }, None) => {
                return Err("on a topic run, focus a group first to get suggestions".into());
            }
        };
        Ok((scores, &e.features.names))
    }

    /// Per-group sums for `contrast` of groups: `groups` gives each point
    /// of `universe` (the view `key`) its group (`>= n_groups` for none).
    /// The one pass over the cells; the contrasts below are cheap after it.
    pub fn group_sums(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        groups: &[u32],
        n_groups: usize,
    ) -> Result<GroupSums, String> {
        let rows = self.rows(Source::Expected, key, universe)?;
        let table = self.expected()?.table();
        let zero = Vector::zeros(table.ncols());
        let mut out = GroupSums {
            sums: vec![zero.clone(); n_groups],
            counts: vec![0; n_groups],
            none: zero,
            none_count: 0,
        };
        for (&r, &g) in rows.iter().zip(groups) {
            if r == NO_ROW {
                continue;
            }
            let (sum, count) = match out.sums.get_mut(g as usize) {
                Some(s) => (s, &mut out.counts[g as usize]),
                None => (&mut out.none, &mut out.none_count),
            };
            for (a, &b) in sum.iter_mut().zip(table.row(r as usize).iter()) {
                *a += b;
            }
            *count += 1;
        }
        Ok(out)
    }

    /// `contrast` for every group at once, each over the other groups'
    /// cells (cells with no group left out). One score vector per group
    /// (empty for an empty group), and the feature names.
    pub fn cluster_contrasts(&mut self, sums: &GroupSums) -> Result<GroupContrasts<'_>, String> {
        let e = self.expected()?;
        let total = sums
            .sums
            .iter()
            .fold(Vector::zeros(sums.none.len()), |a, b| a + b);
        let all: usize = sums.counts.iter().sum();
        let out = sums
            .sums
            .iter()
            .zip(&sums.counts)
            .map(|(s, &n)| {
                if n == 0 {
                    return Vec::new();
                }
                e.score(s, n, &(&total - s), all - n)
            })
            .collect();
        Ok((out, &e.features.names))
    }

    /// Every feature's expected level in each group's mean cell, from
    /// `sums`: feature-major (`features × groups`), `NaN` for an empty group
    /// and, on an embedding run with a baseline, for features below its
    /// median (rarely expressed, with noisy loadings). Embedding runs:
    /// `ρ_g · z̄ + bias_g`; topic runs: `ln(β_g · θ̄)`. And the features.
    pub fn group_levels(&mut self, sums: &GroupSums) -> Result<(Vec<f32>, &[Box<str>]), String> {
        let e = self.expected()?;
        let n = sums.sums.len();
        let d = e.features.names.len();
        let mut out = vec![f32::NAN; d * n];
        for (g, (s, &count)) in sums.sums.iter().zip(&sums.counts).enumerate() {
            if count == 0 {
                continue;
            }
            let mean = s / count as f32;
            let level: Vec<f32> = match (&e.model, e.linear()) {
                (Model::Embedding { rho, bias, .. }, _) => (rho * &mean)
                    .iter()
                    .enumerate()
                    .map(|(f, &v)| v + bias.as_ref().map_or(0.0, |b| b[f]))
                    .collect(),
                (_, Some((_, beta))) => (beta * &mean).iter().map(|&v| v.max(1e-12).ln()).collect(),
                _ => unreachable!("a topic run has linear tables"),
            };
            for (f, v) in level.into_iter().enumerate() {
                out[f * n + g] = v;
            }
        }
        if let Some((b, median)) = e.baseline().filter(|(b, _)| b.len() == d) {
            for (f, &v) in b.iter().enumerate() {
                if v.is_nan() || v < median {
                    out[f * n..(f + 1) * n].fill(f32::NAN);
                }
            }
        }
        Ok((out, &e.features.names))
    }

    /// Each point's topic mixture (θ, summing to one), `k` values a point,
    /// `NaN` for a point the model has no row for; and `k`. Topic runs only.
    pub fn mixtures(
        &mut self,
        key: usize,
        universe: &[Box<str>],
    ) -> Result<(Vec<f32>, usize), String> {
        let rows = self.rows(Source::Expected, key, universe)?;
        let e = self.expected()?;
        let Some((theta, _)) = e.linear() else {
            return Err("a structure plot needs a topic model (topic, masked-topic, …)".into());
        };
        let k = theta.ncols();
        let mut out = vec![f32::NAN; rows.len() * k];
        for (p, &r) in rows.iter().enumerate() {
            if r == NO_ROW {
                continue;
            }
            let row = theta.row(r as usize);
            let total: f32 = row.iter().sum::<f32>().max(1e-12);
            for (o, &v) in out[p * k..(p + 1) * k].iter_mut().zip(row.iter()) {
                *o = v / total;
            }
        }
        Ok((out, k))
    }

    /// Each feature's mean level in each group of the points in view
    /// (`groups`, `>= n_groups` for none), feature-major (`features.len() ×
    /// n_groups`): the mean `ln(1 + count)` over the group's cells from the
    /// data files, or, when they cannot be read, the model's expected level
    /// of the group's mean cell. Also says which source it used. A feature
    /// neither has is `NaN`.
    pub fn group_means(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        groups: &[u32],
        n_groups: usize,
        features: &[Box<str>],
    ) -> Result<(Vec<f32>, Source), String> {
        if let Ok(v) = self.observed_group_means(key, universe, groups, n_groups, features) {
            return Ok((v, Source::Observed));
        }
        let sums = self.group_sums(key, universe, groups, n_groups)?;
        let (levels, _) = self.group_levels(&sums)?;
        let index = &self.expected()?.features.index;
        let mut out = vec![f32::NAN; features.len() * n_groups];
        for (i, f) in features.iter().enumerate() {
            if let Some(g) = index.match_gene(f) {
                out[i * n_groups..(i + 1) * n_groups]
                    .copy_from_slice(&levels[g * n_groups..(g + 1) * n_groups]);
            }
        }
        Ok((out, Source::Expected))
    }

    fn observed_group_means(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        groups: &[u32],
        n_groups: usize,
        features: &[Box<str>],
    ) -> Result<Vec<f32>, String> {
        let rows = self.rows(Source::Observed, key, universe)?;
        let o = self.observed()?;
        // Each data column's group, and each group's size in columns.
        let mut group_of = vec![u32::MAX; o.cells.len()];
        let mut size = vec![0usize; n_groups];
        for (&r, &g) in rows.iter().zip(groups) {
            if r != NO_ROW && (g as usize) < n_groups {
                group_of[r as usize] = g;
                size[g as usize] += 1;
            }
        }
        let matched: Vec<(usize, usize)> = features
            .iter()
            .enumerate()
            .filter_map(|(i, f)| Some((i, o.features.index.match_gene(f)?)))
            .collect();
        let mut out = vec![f32::NAN; features.len() * n_groups];
        if matched.is_empty() {
            return Ok(out);
        }
        let csr = o
            .data
            .read_rows_csr(matched.iter().map(|&(_, g)| g))
            .map_err(|e| e.to_string())?;
        for (k, &(i, _)) in matched.iter().enumerate() {
            let mut sum = vec![0f32; n_groups];
            let row = csr.row(k);
            for (&c, &v) in row.col_indices().iter().zip(row.values()) {
                if let Some(s) = sum.get_mut(group_of[c] as usize) {
                    *s += v.ln_1p();
                }
            }
            for (j, (s, &n)) in sum.iter().zip(&size).enumerate() {
                if n > 0 {
                    out[i * n_groups + j] = s / n as f32;
                }
            }
        }
        Ok(out)
    }

    /// `contrast` of the union of the groups `chosen` picks, over every
    /// other cell in view (cells with no group included), from `sums`.
    pub fn union_contrast(
        &mut self,
        sums: &GroupSums,
        chosen: impl Fn(usize) -> bool,
    ) -> Result<(Vec<f32>, &[Box<str>]), String> {
        let e = self.expected()?;
        let mut inside = Vector::zeros(sums.none.len());
        let mut rest = sums.none.clone();
        let (mut n_in, mut n_rest) = (0, sums.none_count);
        for (g, (s, &n)) in sums.sums.iter().zip(&sums.counts).enumerate() {
            if chosen(g) {
                inside += s;
                n_in += n;
            } else {
                rest += s;
                n_rest += n;
            }
        }
        if n_in == 0 || n_rest == 0 {
            return Err(NO_CONTRAST.into());
        }
        Ok((e.score(&inside, n_in, &rest, n_rest), &e.features.names))
    }

    /// The `top` best-scoring features of `contrast`, best first.
    pub fn suggest(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        group: Option<&[bool]>,
        top: usize,
    ) -> Result<Vec<(Box<str>, f32)>, String> {
        let (scores, names) = self.contrast(key, universe, group)?;
        Ok(best(&scores, names, top))
    }

    /// Features whose expected level is highest in `cells` (one clicked
    /// cell, or a cluster) relative to the average cell: for an embedding
    /// run, `ρ_g · (mean z of the cells − mean z)`, the genes nearest them in
    /// the embedding. Cells not in the model are skipped; rarely expressed
    /// features are left out as in `contrast`.
    pub fn near_cells<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a str>,
        top: usize,
    ) -> Result<Vec<(Box<str>, f32)>, String> {
        let e = self.expected()?;
        let Model::Embedding { z, rho, .. } = &e.model else {
            return Err("neighbouring features need an embedding run".into());
        };
        let index = e.cell_index();
        let rows: Vec<usize> = cells
            .into_iter()
            .filter_map(|c| index.get(c).map(|&n| n as usize))
            .collect();
        if rows.is_empty() {
            return Err("none of these cells is in the model".into());
        }
        let d = row_mean(z, &rows) - e.z_mean(z);
        let mut scores: Vec<f32> = (rho * d).iter().copied().collect();
        drop_below_median(&mut scores, e.baseline());
        Ok(best(&scores, &e.features.names, top))
    }

    /// Cells whose expected level of `feature` is highest relative to the
    /// average cell (`ρ_g · (z_cell − mean z)`): the cells nearest that
    /// feature in the embedding, the mirror of `near_cell`.
    pub fn near_feature(
        &mut self,
        feature: &str,
        top: usize,
    ) -> Result<Vec<(Box<str>, f32)>, String> {
        let e = self.expected()?;
        let Model::Embedding { z, rho, .. } = &e.model else {
            return Err("neighbouring cells need an embedding run".into());
        };
        let g = e
            .features
            .index
            .match_gene(feature)
            .ok_or_else(|| format!("no feature {feature} in the model"))?;
        let r = rho.row(g).transpose();
        let mean = e.z_mean(z);
        let mut scores = z * &r;
        scores.add_scalar_mut(-mean.dot(&r));
        Ok(best(scores.as_slice(), &e.cells, top))
    }
}

/// The `top` finite scores, best first, with their names.
pub(super) fn best(scores: &[f32], names: &[Box<str>], top: usize) -> Vec<(Box<str>, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .collect();
    // Only the top few are wanted: select them, then sort just those.
    let by_score = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if top < ranked.len() {
        ranked.select_nth_unstable_by(top, by_score);
        ranked.truncate(top);
    }
    ranked.sort_by(by_score);
    ranked
        .into_iter()
        .map(|(g, v)| (names[g].clone(), v))
        .collect()
}

/// The upper median of the finite values.
fn median_of(v: &[f32]) -> Option<f32> {
    let mut finite: Vec<f32> = v.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let k = finite.len() / 2;
    Some(*finite.select_nth_unstable_by(k, f32::total_cmp).1)
}

/// Set scores of features whose baseline is below its median to -inf.
fn drop_below_median(scores: &mut [f32], baseline: Option<(&[f32], f32)>) {
    let Some((b, median)) = baseline.filter(|(b, _)| b.len() == scores.len()) else {
        return;
    };
    for (s, &v) in scores.iter_mut().zip(b) {
        if v.is_nan() || v < median {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Sum of the given rows of `m`, as a column vector.
fn row_sum(m: &Mat, rows: &[usize]) -> Vector {
    let mut acc = Vector::zeros(m.ncols());
    for &i in rows {
        acc += m.row(i).transpose();
    }
    acc
}

/// Mean of the given rows of `m`, as a column vector.
fn row_mean(m: &Mat, rows: &[usize]) -> Vector {
    row_sum(m, rows) / rows.len().max(1) as f32
}

/// Covariance of the given rows of `m` (columns are variables), summed over
/// chunks of rows in parallel.
fn row_cov(m: &Mat, rows: &[usize]) -> Mat {
    let mu = row_mean(m, rows);
    let h = m.ncols();
    let c = rows
        .par_chunks(4096)
        .map(|chunk| {
            let mut c = Mat::zeros(h, h);
            for &i in chunk {
                let d = m.row(i).transpose() - &mu;
                c.ger(1.0, &d, &d, 1.0);
            }
            c
        })
        .reduce(|| Mat::zeros(h, h), |a, b| a + b);
    c / (rows.len().max(2) - 1) as f32
}

/// `x_gᵀ C x_g` for every row `x_g` of `x`.
fn quadratic_forms(x: &Mat, c: &Mat) -> Vec<f32> {
    let xc = x * c;
    xc.row_iter()
        .zip(x.row_iter())
        .map(|(a, b)| a.dot(&b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_median_baselines_are_dropped_and_others_kept() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let b = [0.1, 5.0, 3.0, f32::NAN];
        assert_eq!(median_of(&b), Some(3.0));
        drop_below_median(&mut scores, Some((&b, 3.0)));
        assert_eq!(scores[0], f32::NEG_INFINITY);
        assert_eq!(scores[1], 2.0);
        assert_eq!(scores[2], 3.0);
        assert_eq!(scores[3], f32::NEG_INFINITY);
        let mut untouched = vec![1.0, 2.0];
        drop_below_median(&mut untouched, None);
        assert_eq!(untouched, vec![1.0, 2.0]);
    }

    /// An activity over a tiny model: five cells, three features.
    fn tiny(topic: bool) -> Activity {
        let names = |p: &str, n: usize| (0..n).map(|i| format!("{p}{i}").into()).collect();
        let cells: Vec<Box<str>> = names("c", 5);
        let z = Mat::from_row_slice(5, 2, &[0.1, 1.0, 0.4, 0.2, 0.9, 0.3, 0.2, 0.8, 0.7, 0.5]);
        let rho = Mat::from_row_slice(3, 2, &[1.0, -0.5, 0.3, 2.0, -1.0, 0.4]);
        let model = if topic {
            Model::Topic {
                log_theta: z.map(f32::ln),
                log_beta: rho.map(|v| v.abs().ln()),
            }
        } else {
            Model::Embedding {
                z,
                rho,
                bias: Some(vec![0.5, 0.1, 0.9]),
            }
        };
        let mut a = Activity::new(
            RunManifest::new(senna::run_manifest::RunKind::Topic, "r"),
            PathBuf::new(),
        );
        a.expected = Some(Ok(Expected::new(Axis::new(names("GENE", 3)), cells, model)));
        a
    }

    #[test]
    fn cached_group_contrasts_equal_the_direct_ones() {
        for topic in [false, true] {
            let mut a = tiny(topic);
            // View order differs from the model's; one cell has no group,
            // one is not in the model.
            let view: Vec<Box<str>> = ["c3", "c0", "c4", "c1", "c2", "cx"]
                .map(Into::into)
                .to_vec();
            let groups = [1, 0, NO_ROW, 1, 0, 0];
            let sums = a.group_sums(0, &view, &groups, 3).unwrap();
            let close = |x: &[f32], y: &[f32]| {
                x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a == b || (a - b).abs() < 1e-5)
            };
            for chosen in [vec![0], vec![1], vec![0, 1]] {
                let mask: Vec<bool> = groups.iter().map(|g| chosen.contains(g)).collect();
                let (direct, _) = a.contrast(0, &view, Some(&mask)).unwrap();
                let direct = direct.clone();
                let (cached, _) = a
                    .union_contrast(&sums, |g| chosen.contains(&(g as u32)))
                    .unwrap();
                assert!(close(&direct, &cached), "{direct:?} vs {cached:?}");
            }
            // An empty group has no contrast, as before.
            assert!(a.union_contrast(&sums, |g| g == 2).is_err());
            // Per cluster, the rest is the other groups only.
            let (per, _) = a.cluster_contrasts(&sums).unwrap();
            let per = per.clone();
            let grouped: Vec<Box<str>> = ["c3", "c0", "c1", "c2"].map(Into::into).to_vec();
            let (direct, _) = a
                .contrast(1, &grouped, Some(&[false, true, false, true]))
                .unwrap();
            assert!(close(&direct, &per[0]));
            assert!(per[2].is_empty());
        }
    }

    #[test]
    fn cells_near_a_feature_rank_by_its_loading_against_the_average_cell() {
        // GENE0 is ρ = (1, -0.5); z · ρ per cell is -0.4, 0.3, 0.75, -0.2,
        // 0.45, and the average cell scores 0.18.
        let near = tiny(false).near_feature("GENE0", 3).unwrap();
        let got: Vec<&str> = near.iter().map(|(n, _)| n.as_ref()).collect();
        assert_eq!(got, ["c2", "c4", "c1"]);
        assert!((near[0].1 - 0.57).abs() < 1e-5);
        assert!(tiny(false).near_feature("NOPE", 3).is_err());
        assert!(tiny(true).near_feature("GENE0", 3).is_err());
    }

    #[test]
    fn features_near_a_cluster_rank_by_its_mean_against_the_average_cell() {
        // Cells c1, c2 average z = (0.65, 0.25); the average cell is
        // (0.46, 0.56), so the cluster leans (0.19, -0.31). ρ · that is
        // 0.345 for GENE0, -0.563 for GENE1, -0.314 for GENE2; GENE1's
        // baseline (0.1) is below the median (0.5), so it is left out.
        let cluster = ["c1", "c2", "not-in-model"];
        let near = tiny(false).near_cells(cluster, 3).unwrap();
        let got: Vec<&str> = near.iter().map(|(n, _)| n.as_ref()).collect();
        assert_eq!(got, ["GENE0", "GENE2"]);
        assert!((near[0].1 - 0.345).abs() < 1e-5, "{near:?}");
        assert!(tiny(false).near_cells(["cx"], 3).is_err());
        assert!(tiny(true).near_cells(cluster, 3).is_err());
    }

    #[test]
    fn covariance_and_quadratic_forms_match_by_hand() {
        // Two dims; rows 0..3. Only dim 0 varies: values 0, 2, 4.
        let m = Mat::from_row_slice(3, 2, &[0.0, 1.0, 2.0, 1.0, 4.0, 1.0]);
        let all = [0, 1, 2];
        let mean = row_mean(&m, &all);
        assert!((mean[0] - 2.0).abs() < 1e-6 && (mean[1] - 1.0).abs() < 1e-6);
        let c = row_cov(&m, &all);
        assert!((c[(0, 0)] - 4.0).abs() < 1e-5);
        assert!(c[(1, 1)].abs() < 1e-6 && c[(0, 1)].abs() < 1e-6);
        // A feature loading only on dim 0 has variance ρ² · 4; one on dim 1 has none.
        let rho = Mat::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 3.0]);
        let q = quadratic_forms(&rho, &c);
        assert!((q[0] - 1.0).abs() < 1e-5);
        assert!(q[1].abs() < 1e-6);
    }
}
