//! The tables a feature's levels are read from: the run's model of
//! expression ([`Expected`]) and the observed counts ([`Observed`]), with the
//! per-group sums a contrast is linear in.

use super::numeric::quantile;
use super::*;
use crate::view::color;
use crate::view::colour_scale::{ColourScale, Missing, Order, Transform};
use rustc_hash::FxHashMap as HashMap;
use senna::run_manifest::{self, RunManifest};
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
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
    /// pass over the columns.
    pub(super) fn totals(&self) -> Result<&(Vec<f32>, Vec<f32>), String> {
        if let Some(t) = self.totals.get() {
            return Ok(t);
        }
        let sums = self.column_stats(None)?;
        Ok(self.totals.get_or_init(|| sums))
    }

    /// Each column's total count and number of features detected over
    /// `rows` (every row: `None`): data-beans' column statistics, as its
    /// QC sums the mito rows.
    pub(super) fn column_stats(
        &self,
        rows: Option<&[usize]>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (detected, total, _, _) =
            data_beans::qc::collect_column_stat_across_vec(&self.data, rows, None)
                .map_err(|e| e.to_string())?
                .to_f32_vecs();
        Ok((total, detected))
    }

    /// `stat` per column.
    pub(super) fn stat(&self, stat: &Stat, set: &[Box<str>]) -> Result<Cow<'_, [f32]>, String> {
        let (total, detected) = self.totals()?;
        Ok(match stat {
            Stat::Total => Cow::Borrowed(total),
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
/// has no value, what they are (counts, a fraction, signed scores), and
/// once [`Levels::fit`] to a [`ColourScale`], how they are coloured: the
/// scale, the range the ramp spans on it, the ramp, each point's place on
/// it, and the key's ticks.
#[derive(Clone)]
pub struct Levels {
    pub values: Vec<f32>,
    /// Observed counts: a zero means "not detected" and is drawn muted rather
    /// than at the bottom of the ramp.
    pub zero_is_off: bool,
    /// The ramp starts at zero (a fraction) unless clipped.
    pub from_zero: bool,
    /// Signed values around 0: drawn on a diverging ramp unless the colour
    /// scale asks for a sequential one.
    pub signed: bool,
    /// The scale the values take when the colour scale names none.
    pub natural: Transform,
    /// How much of each point's colour to draw, in `[0, 1]`, the rest muted:
    /// a value drawn faded where it is not well supported. `None`: all of it.
    pub weight: Option<Vec<f32>>,
    /// Fitted: the ramp's ends, on `transform`'s scale.
    pub lo: f32,
    pub hi: f32,
    pub transform: Transform,
    /// Fitted: drawn on a diverging ramp, its middle at `(lo + hi) / 2`.
    pub diverging: bool,
    /// Fitted: the colour scale's missing points hidden, and draw order.
    pub hide_missing: bool,
    pub order: Order,
    /// Fitted: the ramp, each point's place on it ([`Levels::t`]), whether
    /// any point has no value, and the key's ticks.
    ramp: Arc<[color::Rgb]>,
    pos: Vec<f32>,
    any_missing: bool,
    ticks: Vec<(f32, String)>,
}

/// [`Levels::t`] of a point with no value, and of one that is off (a count
/// of zero, not detected).
pub const MISSING: f32 = -2.0;
pub const OFF: f32 = -1.0;

impl Levels {
    /// Position on the ramp in `[0, 1]`, or [`OFF`] or [`MISSING`], as
    /// fitted.
    #[inline]
    #[must_use]
    pub fn t(&self, i: usize) -> f32 {
        self.pos[i]
    }

    /// Plain values; [`Levels::fit`] colours them.
    pub(in crate::view) fn from_values(values: Vec<f32>, zero_is_off: bool) -> Self {
        Self {
            values,
            zero_is_off,
            from_zero: false,
            signed: false,
            natural: Transform::Linear,
            weight: None,
            lo: 0.0,
            hi: 1.0,
            transform: Transform::Linear,
            diverging: false,
            hide_missing: false,
            order: Order::High,
            ramp: Arc::from(Vec::new()),
            pos: Vec::new(),
            any_missing: false,
            ticks: Vec::new(),
        }
    }

    /// Signed `values` on the diverging ramp, symmetric about 0 out to their
    /// 99th percentile in size, each drawn with its `weight`.
    pub(in crate::view) fn diverging(values: Vec<f32>, weight: Vec<f32>) -> Self {
        Self {
            signed: true,
            weight: Some(weight),
            ..Self::from_values(values, false)
        }
    }

    /// These levels with the ramp from zero, as a fraction's.
    #[must_use]
    pub(in crate::view) fn starting_at_zero(mut self) -> Self {
        self.from_zero = true;
        self
    }

    /// These levels on `natural` unless a colour scale names another.
    #[must_use]
    pub(in crate::view) fn on(mut self, natural: Transform) -> Self {
        self.natural = natural;
        self
    }

    /// These levels fitted to `scale`.
    #[cfg(test)]
    #[must_use]
    pub fn fitted(mut self, scale: &ColourScale) -> Self {
        self.fit(scale);
        self
    }

    /// Colour these levels as `scale` says: its scale, its range (the
    /// percentiles, else 2 to 99, from 0 for counts; a min or max given
    /// overrides its side; a diverging ramp symmetric about its midpoint),
    /// its ramp, order and missing points.
    pub fn fit(&mut self, scale: &ColourScale) {
        self.transform = scale.scale.unwrap_or(self.natural);
        self.diverging = scale.diverging(self.signed);
        self.hide_missing = scale.missing == Missing::Hidden;
        self.order = scale.order;
        self.ramp = Arc::from(scale.ramp(self.diverging));
        let f = |v: f32| self.transform.apply(v);
        let mut sample: Vec<f32> = self
            .values
            .iter()
            .filter(|&&v| !(self.zero_is_off && v <= 0.0))
            .map(|&v| f(v))
            .filter(|v| v.is_finite())
            .collect();
        let [p_lo, p_hi] = scale.clip.map_or(
            [if self.zero_is_off { 0.0 } else { 0.02 }, 0.99],
            |[a, b]| [a / 100.0, b / 100.0],
        );
        let (lo, hi) = if self.diverging {
            let mid = f(scale.midpoint.unwrap_or(0.0));
            let mid = if mid.is_finite() || sample.is_empty() {
                mid
            } else {
                quantile(&mut sample, 0.5)
            };
            let given = [scale.min, scale.max]
                .into_iter()
                .flatten()
                .map(|v| (f(v) - mid).abs())
                .reduce(f32::max);
            // Out to the farther of the clip's two percentiles from the
            // middle, by default the 99th percentile of the departure.
            let half = given.unwrap_or_else(|| match scale.clip {
                _ if sample.is_empty() => 1.0,
                Some(_) => {
                    let (a, b) = (quantile(&mut sample, p_lo), quantile(&mut sample, p_hi));
                    (a - mid).abs().max((b - mid).abs())
                }
                None => {
                    let mut size: Vec<f32> = sample.iter().map(|v| (v - mid).abs()).collect();
                    quantile(&mut size, p_hi)
                }
            });
            let half = half.max(1e-6);
            (mid - half, mid + half)
        } else {
            // Counts and fractions start the ramp at zero unless asked to
            // clip; model levels at a low percentile, their scale having no
            // origin. A min or max given holds even with no value to show.
            let zero = f(0.0);
            let empty = sample.is_empty();
            let lo = match scale.min {
                Some(v) => f(v),
                None if (self.zero_is_off || self.from_zero)
                    && scale.clip.is_none()
                    && zero.is_finite() =>
                {
                    zero
                }
                None if empty => 0.0,
                None => quantile(&mut sample, p_lo),
            };
            let hi = match scale.max {
                Some(v) => f(v),
                None if empty => lo + 1.0,
                None => quantile(&mut sample, p_hi),
            };
            (lo, hi.max(lo + 1e-6))
        };
        (self.lo, self.hi) = (lo, hi);
        let span = (hi - lo).max(1e-12);
        self.pos = self
            .values
            .iter()
            .map(|&v| {
                if self.zero_is_off && v <= 0.0 {
                    return OFF;
                }
                let x = f(v);
                if x.is_finite() {
                    ((x - lo) / span).clamp(0.0, 1.0)
                } else {
                    MISSING
                }
            })
            .collect();
        self.any_missing = self.pos.contains(&MISSING);
        self.ticks = self.key_ticks();
    }

    /// Whether some point has no value (not one that is off), as fitted.
    #[must_use]
    pub fn any_missing(&self) -> bool {
        self.any_missing
    }

    /// The key's ticks, as fitted.
    #[must_use]
    pub fn ticks(&self) -> &[(f32, String)] {
        &self.ticks
    }

    /// The ramp the levels are drawn on.
    #[must_use]
    pub fn ramp(&self) -> &[color::Rgb] {
        &self.ramp
    }

    /// The ramp, shared.
    #[must_use]
    pub fn ramp_arc(&self) -> Arc<[color::Rgb]> {
        self.ramp.clone()
    }

    /// Point `i`'s place in the draw order, later on top, as the order
    /// asks: by level, or for diverging levels by the size of the
    /// departure either way times its weight, so strong departures are not
    /// hidden under faded ones; points with no value or off always under.
    #[inline]
    #[must_use]
    pub fn draw_key(&self, i: usize) -> f32 {
        let t = self.t(i);
        if t < 0.0 {
            return t - 10.0;
        }
        let key = if self.diverging {
            (t - 0.5).abs() * self.weight_of(i)
        } else {
            t
        };
        match self.order {
            Order::High => key,
            Order::Low => -key,
            Order::Random => 0.0,
        }
    }

    /// How much of point `i`'s colour to draw ([`Self::weight`]).
    #[inline]
    #[must_use]
    pub fn weight_of(&self, i: usize) -> f32 {
        self.weight.as_ref().map_or(1.0, |w| w[i])
    }
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
