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
struct Axis {
    names: Vec<Box<str>>,
    index: GeneIndex,
}

impl Axis {
    fn new(names: Vec<Box<str>>) -> Self {
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
}

/// A point with no row in a source's cell table.
const NO_ROW: u32 = u32::MAX;

struct Observed {
    data: SparseIoVec,
    features: Axis,
    cells: Vec<Box<str>>,
}

/// Values of one feature (or feature set) per point, `NaN` where the point
/// has no value, and the range the colour ramp spans.
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
        let cells: &[Box<str>] = match source {
            Source::Expected => &self.expected()?.cells,
            Source::Observed => &self.observed()?.cells,
        };
        let index: HashMap<&str, u32> = cells
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i as u32))
            .collect();
        let rows: Arc<Vec<u32>> = Arc::new(
            names
                .iter()
                .map(|n| index.get(n.as_ref()).copied().unwrap_or(NO_ROW))
                .collect(),
        );
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
            return Ok(Expected {
                features: Axis::new(rho.rows),
                cells: z.rows,
                model: Model::Embedding {
                    z: z.mat,
                    rho: rho.mat,
                    bias,
                },
                variance_everywhere: OnceCell::new(),
                topic_linear: OnceCell::new(),
            });
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
            return Ok(Expected {
                features: Axis::new(beta.rows),
                cells: theta.rows,
                model: Model::Topic {
                    log_theta: theta.mat,
                    log_beta: beta.mat,
                },
                variance_everywhere: OnceCell::new(),
                topic_linear: OnceCell::new(),
            });
        }
        anyhow::bail!("this run has no model tables to predict a feature from")
    }

    fn load_observed(&self) -> anyhow::Result<Observed> {
        let m = &self.manifest;
        anyhow::ensure!(!m.data.input.is_empty(), "the manifest lists no data files");
        let files: Vec<Box<str>> = m
            .data
            .input
            .iter()
            .map(|p| run_manifest::resolve(&self.dir, p).to_string_lossy().into())
            .collect();
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
            (Model::Embedding { z, rho, bias }, Some(_)) => {
                let (inside, rest) = (pick(Some(true)), pick(Some(false)));
                if inside.is_empty() || rest.is_empty() {
                    return Err(
                        "the focused group has no cells, or no other cells are in view".into(),
                    );
                }
                let delta = row_mean(z, &inside) - row_mean(z, &rest);
                let mut lfc: Vec<f32> = (rho * delta).iter().copied().collect();
                drop_below_median(&mut lfc, bias.as_deref());
                lfc
            }
            (Model::Embedding { z, rho, bias }, None) => {
                let here = quadratic_forms(rho, &row_cov(z, &pick(None)));
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
                            h / a
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                drop_below_median(&mut ratio, bias.as_deref());
                ratio
            }
            (
                Model::Topic {
                    log_theta,
                    log_beta,
                },
                Some(_),
            ) => {
                let (inside, rest) = (pick(Some(true)), pick(Some(false)));
                if inside.is_empty() || rest.is_empty() {
                    return Err(
                        "the focused group has no cells, or no other cells are in view".into(),
                    );
                }
                let (theta, beta) = e
                    .topic_linear
                    .get_or_init(|| (log_theta.map(f32::exp), log_beta.map(f32::exp)));
                let (g, r) = (
                    beta * row_mean(theta, &inside),
                    beta * row_mean(theta, &rest),
                );
                g.iter()
                    .zip(r.iter())
                    .map(|(&a, &b)| (a.max(1e-12) / b.max(1e-12)).ln())
                    .collect()
            }
            (Model::Topic { .. }, None) => {
                return Err("on a topic run, focus a group first to get suggestions".into());
            }
        };
        Ok((scores, &e.features.names))
    }

    /// `contrast` for every group at once: each group over the rest of the
    /// view. `groups` gives each point of `universe` its group (`u32::MAX`
    /// for none). One pass over the cells: each group's rest is the total
    /// minus the group. Returns one score vector per group (empty for an
    /// empty group), in the model's feature order, and the feature names.
    pub fn cluster_contrasts(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        groups: &[u32],
        n_groups: usize,
    ) -> Result<GroupContrasts<'_>, String> {
        let rows = self.rows(Source::Expected, key, universe)?;
        let e = self.expected()?;
        // Per-group sums of the per-cell table the scores are linear in.
        let sums = |m: &Mat| -> (Vec<nalgebra::DVector<f32>>, Vec<usize>) {
            let mut s = vec![nalgebra::DVector::<f32>::zeros(m.ncols()); n_groups];
            let mut n = vec![0usize; n_groups];
            for (k, &r) in rows.iter().enumerate() {
                let g = groups[k];
                if r == NO_ROW || g as usize >= n_groups {
                    continue;
                }
                s[g as usize] += m.row(r as usize).transpose();
                n[g as usize] += 1;
            }
            (s, n)
        };
        let split = |s: &[nalgebra::DVector<f32>], n: &[usize], g: usize| {
            let total: nalgebra::DVector<f32> = s
                .iter()
                .fold(nalgebra::DVector::<f32>::zeros(s[g].len()), |a, b| a + b);
            let all: usize = n.iter().sum();
            let inside = &s[g] / n[g].max(1) as f32;
            let rest = (total - &s[g]) / (all - n[g]).max(1) as f32;
            (inside, rest)
        };
        let out = match &e.model {
            Model::Embedding { z, rho, bias } => {
                let (s, n) = sums(z);
                (0..n_groups)
                    .map(|g| {
                        if n[g] == 0 {
                            return Vec::new();
                        }
                        let (inside, rest) = split(&s, &n, g);
                        let mut v: Vec<f32> = (rho * (inside - rest)).iter().copied().collect();
                        drop_below_median(&mut v, bias.as_deref());
                        v
                    })
                    .collect()
            }
            Model::Topic {
                log_theta,
                log_beta,
            } => {
                let (theta, beta) = e
                    .topic_linear
                    .get_or_init(|| (log_theta.map(f32::exp), log_beta.map(f32::exp)));
                let (s, n) = sums(theta);
                (0..n_groups)
                    .map(|g| {
                        if n[g] == 0 {
                            return Vec::new();
                        }
                        let (inside, rest) = split(&s, &n, g);
                        let (a, b) = (beta * inside, beta * rest);
                        a.iter()
                            .zip(b.iter())
                            .map(|(&x, &y)| (x.max(1e-12) / y.max(1e-12)).ln())
                            .collect()
                    })
                    .collect()
            }
        };
        Ok((out, &e.features.names))
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

    /// Features whose expected level is highest in cell `cell` relative to
    /// the average cell (for an embedding run, `ρ_g · (z_cell − mean z)`):
    /// the genes nearest that cell in the embedding. Rarely expressed
    /// features are left out as in `contrast`.
    pub fn near_cell(&mut self, cell: &str, top: usize) -> Result<Vec<(Box<str>, f32)>, String> {
        let e = self.expected()?;
        let Model::Embedding { z, rho, bias } = &e.model else {
            return Err("neighbouring features need an embedding run".into());
        };
        let n = e
            .cells
            .iter()
            .position(|c| c.as_ref() == cell)
            .ok_or_else(|| format!("{cell} is not in the model"))?;
        let all: Vec<usize> = (0..z.nrows()).collect();
        let d = z.row(n).transpose() - row_mean(z, &all);
        let mut scores: Vec<f32> = (rho * d).iter().copied().collect();
        drop_below_median(&mut scores, bias.as_deref());
        Ok(best(&scores, &e.features.names, top))
    }
}

/// The `top` finite scores, best first, with their names.
fn best(scores: &[f32], names: &[Box<str>], top: usize) -> Vec<(Box<str>, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
        .into_iter()
        .take(top)
        .map(|(g, v)| (names[g].clone(), v))
        .collect()
}

/// Set scores of features whose baseline is below the median to -inf.
fn drop_below_median(scores: &mut [f32], baseline: Option<&[f32]>) {
    let Some(b) = baseline.filter(|b| b.len() == scores.len()) else {
        return;
    };
    let mut sorted: Vec<f32> = b.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return;
    }
    sorted.sort_unstable_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];
    for (s, &v) in scores.iter_mut().zip(b) {
        if v.is_nan() || v < median {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Mean of the given rows of `m`, as a column vector.
fn row_mean(m: &Mat, rows: &[usize]) -> nalgebra::DVector<f32> {
    let mut acc = nalgebra::DVector::<f32>::zeros(m.ncols());
    for &i in rows {
        acc += m.row(i).transpose();
    }
    acc / rows.len().max(1) as f32
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
        drop_below_median(&mut scores, Some(&[0.1, 5.0, 3.0, f32::NAN]));
        assert_eq!(scores[0], f32::NEG_INFINITY);
        assert_eq!(scores[1], 2.0);
        assert_eq!(scores[2], 3.0);
        assert_eq!(scores[3], f32::NEG_INFINITY);
        let mut untouched = vec![1.0, 2.0];
        drop_below_median(&mut untouched, None);
        assert_eq!(untouched, vec![1.0, 2.0]);
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
