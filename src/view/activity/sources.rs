//! The tables a feature's levels are read from: the run's model of
//! expression ([`Expected`]) and the observed counts ([`Observed`]), with the
//! per-group sums a contrast is linear in.

use super::*;
use crate::view::color;
use rustc_hash::FxHashMap as HashMap;
use senna::run_manifest::{self, RunManifest};
use std::borrow::Cow;
use std::path::Path;
use std::sync::OnceLock;

pub(super) enum Model {
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

pub(crate) struct Expected {
    pub(super) features: Axis,
    pub(super) cells: Vec<Box<str>>,
    pub(super) model: Model,
    /// Each feature's expected variance over all cells (embedding runs), for
    /// the "varies most here" ranking; computed on first use.
    pub(super) variance_everywhere: OnceLock<Vec<f32>>,
    /// θ and β out of log space (topic runs); computed on first use.
    pub(super) topic_linear: OnceLock<(Mat, Mat)>,
    /// Median of the finite baselines (`None`: no usable baseline).
    pub(super) bias_median: OnceLock<Option<f32>>,
    /// Mean of `z` over all cells (embedding runs).
    pub(super) z_mean: OnceLock<Vector>,
    /// Row of each cell name.
    pub(super) cell_index: OnceLock<HashMap<Box<str>, u32>>,
}

impl Expected {
    pub(super) fn new(features: Axis, cells: Vec<Box<str>>, model: Model) -> Self {
        Self {
            features,
            cells,
            model,
            variance_everywhere: OnceLock::new(),
            topic_linear: OnceLock::new(),
            bias_median: OnceLock::new(),
            z_mean: OnceLock::new(),
            cell_index: OnceLock::new(),
        }
    }

    /// Mean of `z` (this model's cell embedding) over all cells.
    pub(super) fn z_mean(&self, z: &Mat) -> &Vector {
        self.z_mean
            .get_or_init(|| row_mean(z, &(0..z.nrows()).collect::<Vec<_>>()))
    }

    pub(super) fn cell_index(&self) -> &HashMap<Box<str>, u32> {
        self.cell_index.get_or_init(|| {
            self.cells
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i as u32))
                .collect()
        })
    }

    /// The per-feature baseline and its median, when the run has one.
    pub(super) fn baseline(&self) -> Option<(&[f32], f32)> {
        let Model::Embedding { bias: Some(b), .. } = &self.model else {
            return None;
        };
        let median = self.bias_median.get_or_init(|| median_of(b));
        Some((b, (*median)?))
    }

    /// θ and β out of log space (topic runs).
    pub(super) fn linear(&self) -> Option<&(Mat, Mat)> {
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
    pub(super) fn table(&self) -> &Mat {
        match (&self.model, self.linear()) {
            (Model::Embedding { z, .. }, _) => z,
            (_, Some((theta, _))) => theta,
            _ => unreachable!("a topic run has linear tables"),
        }
    }

    /// Each feature's score, cells summing to `inside` (`n_in` of them) over
    /// cells summing to `rest`: the expected log fold change.
    pub(super) fn score(
        &self,
        inside: &Vector,
        n_in: usize,
        rest: &Vector,
        n_rest: usize,
    ) -> Vec<f32> {
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
    pub(super) sums: Vec<Vector>,
    pub(super) counts: Vec<usize>,
    /// Cells in view with no group; they count as the rest of the view.
    pub(super) none: Vector,
    pub(super) none_count: usize,
}

/// A point with no row in a source's cell table.
pub(super) const NO_ROW: u32 = u32::MAX;

pub(super) const NO_CONTRAST: &str =
    "the focused group has no cells, or no other cells are in view";

pub(crate) struct Observed {
    pub(super) data: SparseIoVec,
    pub(super) features: Axis,
    /// Column of each cell name.
    pub(super) cell_index: HashMap<Box<str>, u32>,
    pub(super) n_cells: usize,
    /// Each column's total count and number of features detected, read on
    /// first use.
    pub(super) totals: OnceLock<(Vec<f32>, Vec<f32>)>,
}

impl Observed {
    pub(super) fn new(
        data: SparseIoVec,
        cell_index: HashMap<Box<str>, u32>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            features: Axis::new(data.row_names()?),
            n_cells: data.num_columns(),
            cell_index,
            data,
            totals: OnceLock::new(),
        })
    }

    /// Row `g`'s counts per column, each through `f`.
    pub(super) fn row(&self, g: usize, f: impl Fn(f32) -> f32) -> Result<Vec<f32>, String> {
        // One row a read: see `observed_group_means`.
        let row = self
            .data
            .read_rows_csr(std::iter::once(g))
            .map_err(|e| e.to_string())?;
        let mut values = vec![0f32; self.n_cells];
        for (&c, &v) in row.col_indices().iter().zip(row.values()) {
            values[c] = f(v);
        }
        Ok(values)
    }

    /// Each column's total count and number of features detected, in one
    /// pass over the columns. No progress bar: the screen is the view's.
    pub(super) fn totals(&self) -> Result<&(Vec<f32>, Vec<f32>), String> {
        if let Some(t) = self.totals.get() {
            return Ok(t);
        }
        let sums = self.column_sums(None)?;
        Ok(self.totals.get_or_init(|| sums))
    }

    /// Each column's total count and number of features detected over the
    /// rows `keep` marks (every row: `None`), in one pass over the columns.
    pub(super) fn column_sums(
        &self,
        keep: Option<&[bool]>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        const BLOCK: usize = 4096;
        let blocks: Vec<usize> = (0..self.n_cells).step_by(BLOCK).collect();
        let kept = |r: &usize| keep.is_none_or(|k| k[*r]);
        let parts = blocks
            .par_iter()
            .map(|&lb| {
                let csc = self
                    .data
                    .read_columns_csc(lb..(lb + BLOCK).min(self.n_cells))?;
                Ok(csc
                    .col_iter()
                    .map(|c| {
                        let (mut total, mut detected) = (0f32, 0f32);
                        for (_, &v) in c
                            .row_indices()
                            .iter()
                            .zip(c.values())
                            .filter(|(r, _)| kept(r))
                        {
                            total += v;
                            detected += f32::from(u8::from(v > 0.0));
                        }
                        (total, detected)
                    })
                    .unzip::<_, _, Vec<f32>, Vec<f32>>())
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| e.to_string())?;
        let (mut total, mut detected) = (Vec::new(), Vec::new());
        for (t, d) in parts {
            total.extend(t);
            detected.extend(d);
        }
        Ok((total, detected))
    }

    /// `stat` per column.
    pub(super) fn stat(&self, stat: &Stat, set: &[Box<str>]) -> Result<Cow<'_, [f32]>, String> {
        let (total, detected) = self.totals()?;
        Ok(match stat {
            Stat::Total => Cow::Owned(total.iter().map(|t| t.ln_1p()).collect()),
            Stat::Detected => Cow::Borrowed(detected),
            Stat::Rows(_) => return Err("a row statistic is read on the side".into()),
            Stat::Share(_) => {
                let rows: std::collections::BTreeSet<usize> = set
                    .iter()
                    .filter_map(|f| self.features.index.match_gene(f))
                    .collect();
                if rows.is_empty() {
                    return Err("none of these features are in the counts".into());
                }
                let mut sum = vec![0f32; self.n_cells];
                for g in rows {
                    // One row a read: see `observed_group_means`.
                    let row = self
                        .data
                        .read_rows_csr(std::iter::once(g))
                        .map_err(|e| e.to_string())?;
                    for (&c, &v) in row.col_indices().iter().zip(row.values()) {
                        sum[c] += v;
                    }
                }
                for (s, &t) in sum.iter_mut().zip(total) {
                    *s = if t > 0.0 { *s / t } else { f32::NAN };
                }
                Cow::Owned(sum)
            }
        })
    }
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
    /// Signed values around 0, drawn on the diverging ramp over `±hi`
    /// (`lo = −hi`) rather than on the sequential one.
    pub diverging: bool,
    /// How much of each point's colour to draw, in `[0, 1]`, the rest muted:
    /// a value drawn faded where it is not well supported. `None`: all of it.
    pub weight: Option<Vec<f32>>,
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

    pub(in crate::view) fn from_values(values: Vec<f32>, zero_is_off: bool) -> Self {
        let mut sample: Vec<f32> = values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && (!zero_is_off || *v > 0.0))
            .collect();
        let (lo, hi) = if sample.is_empty() {
            (0.0, 1.0)
        } else {
            let hi = quantile(&mut sample, 0.99);
            // Counts start the ramp at zero; model levels at a low quantile,
            // since their scale has no natural origin.
            let lo = if zero_is_off {
                0.0
            } else {
                quantile(&mut sample, 0.02)
            };
            (lo, hi.max(lo + 1e-6))
        };
        Self {
            values,
            lo,
            hi,
            zero_is_off,
            diverging: false,
            weight: None,
        }
    }

    /// Signed `values` on the diverging ramp, symmetric about 0 out to their
    /// 99th percentile in size, each drawn with its `weight`.
    pub(in crate::view) fn diverging(values: Vec<f32>, weight: Vec<f32>) -> Self {
        let mut size: Vec<f32> = values
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.abs())
            .collect();
        let hi = if size.is_empty() {
            1.0
        } else {
            quantile(&mut size, 0.99).max(1e-6)
        };
        Self {
            values,
            lo: -hi,
            hi,
            zero_is_off: false,
            diverging: true,
            weight: Some(weight),
        }
    }

    /// The ramp the levels are drawn on: the diverging one for signed
    /// levels, else `sequential`.
    #[must_use]
    pub fn ramp<'a>(&self, sequential: &'a [color::Rgb]) -> &'a [color::Rgb] {
        if self.diverging {
            color::diverging_ramp()
        } else {
            sequential
        }
    }

    /// Point `i`'s place in the draw order, later on top: by level, or for
    /// signed levels by the size of the departure either way times its
    /// weight, so strong departures are not hidden under faded ones.
    #[inline]
    #[must_use]
    pub fn draw_key(&self, i: usize) -> f32 {
        match self.t(i) {
            t if t < 0.0 || !self.diverging => t,
            t => (t - 0.5).abs() * self.weight_of(i),
        }
    }

    /// How much of point `i`'s colour to draw ([`Self::weight`]).
    #[inline]
    #[must_use]
    pub fn weight_of(&self, i: usize) -> f32 {
        self.weight.as_ref().map_or(1.0, |w| w[i])
    }
}

/// The `p`-quantile of `sample` (non-empty), by a linear-time selection
/// that reorders it.
fn quantile(sample: &mut [f32], p: f32) -> f32 {
    let k = ((sample.len() - 1) as f32 * p) as usize;
    *sample.select_nth_unstable_by(k, f32::total_cmp).1
}

/// The run's model of expression, read the way its kind's decoder reads
/// its tables ([`senna::run_manifest::RunKind::expression_model`]).
pub(super) fn load_expected(m: &RunManifest, dir: &Path) -> anyhow::Result<Expected> {
    use senna::run_manifest::ExpressionModel as E;
    let o = &m.outputs;
    let slot = |slot: Option<&str>, what: &str| {
        slot.map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("this {} run records no {what}", m.kind))
    };
    let model = m.kind.expression_model();
    match model {
        E::Embedding => {
            let z_rel = slot(o.cell_embedding.as_deref(), "cell embedding")?;
            let (rho_path, bias_path) =
                run_manifest::resolve_feature_embedding_for(m, dir)?;
            let z = read_table(dir, &z_rel)?;
            let rho = Mat::from_parquet_with_row_names(&rho_path, Some(0))?;
            // Only used when it lines up with the feature table.
            let bias = bias_path
                .and_then(|p| Mat::from_parquet_with_row_names(&p, Some(0)).ok())
                .filter(|b| b.rows == rho.rows && b.mat.ncols() >= 1)
                .map(|b| b.mat.column(0).iter().copied().collect());
            embedding(z, rho, bias)
        }
        E::LatentLoadings => {
            let z = read_table(dir, &slot(o.latent.as_deref(), "latent")?)?;
            let w = read_table(dir, &slot(o.dictionary.as_deref(), "dictionary")?)?;
            embedding(z, w, None)
        }
        E::Topic | E::TopicOfLogits => {
            let mut theta = read_table(dir, &slot(o.latent.as_deref(), "latent")?)?;
            let beta = read_table(dir, &slot(o.gene_dictionary(), "dictionary")?)?;
            anyhow::ensure!(
                theta.mat.ncols() == beta.mat.ncols(),
                "latent has {} topics, dictionary {}",
                theta.mat.ncols(),
                beta.mat.ncols()
            );
            if model == E::TopicOfLogits {
                // The decoder reads the raw latent through a softmax.
                use legume_numeric::candle::vae::masked_topic::LatentHead;
                theta.mat = senna::embed_common::latent_to_theta(&theta.mat, LatentHead::Gaussian)
                    .map(f32::ln);
            }
            Ok(Expected::new(
                Axis::new(beta.rows),
                theta.rows,
                Model::Topic {
                    log_theta: theta.mat,
                    log_beta: beta.mat,
                },
            ))
        }
        E::Unmodelled => anyhow::bail!(
            "a {} run has no model of expression to predict a feature from · o shows the observed counts",
            m.kind
        ),
    }
}

/// An embedding model from cell table `z` and feature table `rho` of one
/// width.
pub(super) fn embedding(
    z: MatWithNames<Mat>,
    rho: MatWithNames<Mat>,
    bias: Option<Vec<f32>>,
) -> anyhow::Result<Expected> {
    anyhow::ensure!(
        z.mat.ncols() == rho.mat.ncols(),
        "cell table has {} dims, feature table {}",
        z.mat.ncols(),
        rho.mat.ncols()
    );
    Ok(Expected::new(
        Axis::new(rho.rows),
        z.rows,
        Model::Embedding {
            z: z.mat,
            rho: rho.mat,
            bias,
        },
    ))
}
