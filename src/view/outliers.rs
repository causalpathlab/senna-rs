//! Outlying points of a layout, so a few strays do not shrink the map.
//!
//! Two rules, the default by the layout's method:
//!
//! - `isolated` (t-SNE, UMAP, PHATE, force layouts): a point whose
//!   distance to its k-th nearest neighbour in the layout is above
//!   `median + mad × MAD` of that distance over every point, or above its
//!   `quantile`. A tight small cluster is kept wherever it lies; a lone
//!   point between clusters is caught.
//! - `range` (diffusion, LLE and other spectral axes): a point outside the
//!   `[1 − quantile, quantile]` quantiles of x or of y, or with `mad`
//!   outside `median ± mad × MAD` on either axis. A component localised on
//!   a few tight cells, which `isolated` keeps, is caught.
//!
//! `fit` fits the camera to the rest and pins outliers to its edge; `hide`
//! also leaves them out of the drawing.

use super::activity::quantile;
#[cfg(test)]
use super::data::Points;
use legume_tui::work::Loading;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// What is done with outliers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Every point counts.
    #[default]
    Off,
    /// The camera fits the rest; outliers are drawn pinned to its edge.
    Fit,
    /// The camera fits the rest and outliers are not drawn.
    Hide,
}

impl Mode {
    /// The next in `O`'s cycle: off, fit, hide.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Fit,
            Self::Fit => Self::Hide,
            Self::Hide => Self::Off,
        }
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fit => "fit",
            Self::Hide => "hide",
        }
    }
}

/// How outliers are found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Rule {
    /// Far from their own nearest neighbours.
    Isolated,
    /// Outside a robust range of x or y.
    Range,
}

impl Rule {
    /// The rule for a layout `method`: spectral axes by range, the rest by
    /// isolation.
    #[must_use]
    pub fn for_method(method: &str) -> Self {
        let m = method.to_ascii_lowercase();
        if ["diffusion", "lle", "pca", "spectral", "dc"]
            .iter()
            .any(|s| m.contains(s))
        {
            Self::Range
        } else {
            Self::Isolated
        }
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Isolated => "isolated",
            Self::Range => "range",
        }
    }
}

/// Neighbours `isolated` measures to, unless asked.
pub const K: usize = 10;
/// How many MADs past the median is an outlier, unless asked.
pub const MAD: f32 = 5.0;
/// `range`'s quantile unless asked: x and y within 0.5% to 99.5%.
pub const RANGE_QUANTILE: f32 = 0.995;

/// Outliers as asked for: what to do, and how to find them; unset fields
/// are the defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outliers {
    #[serde(default)]
    pub mode: Mode,
    /// [default: by the layout's method]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule: Option<Rule>,
    /// `isolated`'s neighbour [default: 10].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<usize>,
    /// Out past `median + mad × MAD` [default: 5, unless a quantile].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mad: Option<f32>,
    /// Out past this quantile instead [range's default: 0.995].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantile: Option<f32>,
}

impl Outliers {
    /// Refused: a k of none or over 100, a MAD that is not positive, a
    /// quantile not between 0.5 and 1, or both a MAD and a quantile.
    pub fn check(&self) -> Result<(), String> {
        if self.k.is_some_and(|k| !(1..=100).contains(&k)) {
            return Err("k is 1 to 100 neighbours".into());
        }
        if self.mad.is_some_and(|m| !(m.is_finite() && m > 0.0)) {
            return Err("mad is a positive number of MADs".into());
        }
        if self.quantile.is_some_and(|q| !(q > 0.5 && q < 1.0)) {
            return Err("quantile is between 0.5 and 1, as 0.995".into());
        }
        if self.mad.is_some() && self.quantile.is_some() {
            return Err("a mad or a quantile, not both".into());
        }
        Ok(())
    }

    /// The outliers among the points at `xy` (`order`: the finite ones),
    /// a layout by `method`, as this asks; `bounds` is the layout's.
    #[must_use]
    pub fn find(&self, xy: &[[f32; 2]], order: &[u32], bounds: [f32; 4], method: &str) -> Flags {
        let rule = self.rule.unwrap_or_else(|| Rule::for_method(method));
        let out = match rule {
            Rule::Isolated => self.isolated(xy, order),
            Rule::Range => self.range(xy, order),
        };
        let n = order.iter().filter(|&&i| out[i as usize]).count();
        let kept = |i: usize| !out[i] && xy[i].iter().all(|v| v.is_finite());
        Flags {
            bounds: super::panel::kept_bounds(xy, kept, bounds),
            out: out.into(),
            n,
            of: order.len(),
            rule,
            mode: self.mode,
        }
    }

    /// Distance to the k-th neighbour above the cut.
    fn isolated(&self, xy: &[[f32; 2]], finite: &[u32]) -> Vec<bool> {
        let mut out = vec![false; xy.len()];
        let k = self.k.unwrap_or(K);
        if finite.len() <= k {
            return out;
        }
        let rows = nalgebra::DMatrix::from_fn(finite.len(), 2, |r, c| xy[finite[r] as usize][c]);
        let (_, dists) = legume_numeric::matrix::knn::knn_rows(&rows, k);
        let d: Vec<f32> = dists
            .iter()
            .map(|d| d.last().copied().unwrap_or(f32::NAN))
            .collect();
        let cut = match self.quantile {
            Some(q) => {
                let mut finite_d: Vec<f32> = d.iter().copied().filter(|v| v.is_finite()).collect();
                if finite_d.is_empty() {
                    return out;
                }
                quantile(&mut finite_d, q)
            }
            None => self.band(&d).1,
        };
        for (r, &i) in finite.iter().enumerate() {
            out[i as usize] = d[r] > cut;
        }
        out
    }

    /// Outside the cut on x or on y.
    fn range(&self, xy: &[[f32; 2]], finite: &[u32]) -> Vec<bool> {
        let mut out = vec![false; xy.len()];
        if finite.len() < 3 {
            return out;
        }
        for c in [0, 1] {
            let v: Vec<f32> = finite.iter().map(|&i| xy[i as usize][c]).collect();
            let (lo, hi) = match self.mad {
                Some(_) => self.band(&v),
                None => {
                    let q = self.quantile.unwrap_or(RANGE_QUANTILE);
                    let mut w = v.clone();
                    (quantile(&mut w, 1.0 - q), quantile(&mut w, q))
                }
            };
            for (r, &i) in finite.iter().enumerate() {
                out[i as usize] |= v[r] < lo || v[r] > hi;
            }
        }
        out
    }

    /// `median ± mad × MAD` of `v` (legume-numeric's robust band, the MAD
    /// scaled to a normal's σ); with no spread (over half the values one
    /// value) nothing is outside it.
    fn band(&self, v: &[f32]) -> (f32, f32) {
        legume_numeric::matrix::robust_band::robust_band(v, self.mad.unwrap_or(MAD), false, None)
            .map_or((f32::NEG_INFINITY, f32::INFINITY), |b| (b.lo, b.hi))
    }
}

/// `senna view`'s flags for outliers; `O` in the view cycles the mode.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct OutlierArgs {
    #[arg(
        long,
        value_enum,
        help = "Outlying points: fit the map to the rest (outliers pinned to its edge), hide them too, or off [default: off; O in the view]"
    )]
    pub outliers: Option<Mode>,

    #[arg(
        long,
        value_enum,
        help = "How outliers are found: isolated (k-th neighbour distance far above the rest) or range (outside a quantile range of x or y) [default: range on diffusion and other spectral maps, else isolated]"
    )]
    pub outlier_rule: Option<Rule>,

    #[arg(long, help = "The neighbour isolated measures to [default: 10]")]
    pub outlier_k: Option<usize>,

    #[arg(
        long,
        conflicts_with = "outlier_quantile",
        help = "Outliers past median + this many MADs [default: 5]"
    )]
    pub outlier_mad: Option<f32>,

    #[arg(
        long,
        help = "Outliers past this quantile, as 0.995 [range's default: 0.995]"
    )]
    pub outlier_quantile: Option<f32>,
}

impl OutlierArgs {
    /// The setting the flags ask for, checked.
    pub fn outliers(&self) -> Result<Outliers, String> {
        let o = Outliers {
            mode: self.outliers.unwrap_or_default(),
            rule: self.outlier_rule,
            k: self.outlier_k,
            mad: self.outlier_mad,
            quantile: self.outlier_quantile,
        };
        o.check()?;
        Ok(o)
    }
}

/// The outliers last asked for, by the layout's space, size and bounds
/// and the setting: their background load, and whether its answer has been
/// taken in.
pub(super) type Found = std::cell::RefCell<Option<(FoundKey, Arc<Loading<Flags>>, bool)>>;
type FoundKey = (usize, usize, [f32; 4], Outliers);

/// The outliers found on one layout.
#[derive(Clone, Debug)]
pub struct Flags {
    /// Per point: an outlier.
    pub out: Arc<[bool]>,
    /// How many of how many finite points.
    pub n: usize,
    pub of: usize,
    /// The bounds of the rest.
    pub bounds: [f32; 4],
    pub rule: Rule,
    pub mode: Mode,
}

impl Flags {
    /// What the status line and a reply say: `fit (range, 37 of 14,601)`.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "{} ({}, {} of {})",
            self.mode.name(),
            self.rule.name(),
            self.n,
            self.of
        )
    }
}

impl super::Scene {
    /// The outliers of the layout shown, unless off. They are found in the
    /// background once per layout and setting; until then (a viewer that
    /// does not wait) `None`, and every point counts.
    pub(super) fn current_outliers(&self) -> Option<Flags> {
        if self.outliers.mode == Mode::Off {
            return None;
        }
        let sp = self.current();
        let key = (
            self.space,
            sp.points.xy.len(),
            sp.points.bounds,
            self.outliers.clone(),
        );
        let mut cache = self.outlier_flags.borrow_mut();
        if cache.as_ref().is_none_or(|(k, ..)| *k != key) {
            // Only what the search reads, not the points' names.
            let (xy, order, bounds) = (
                sp.points.xy.clone(),
                sp.points.order.clone(),
                sp.points.bounds,
            );
            let (method, o) = (sp.method.clone(), self.outliers.clone());
            let load = Loading::new(move || Ok(o.find(&xy, &order, bounds, &method)));
            *cache = Some((key, Arc::new(load), false));
        }
        let (_, load, _) = cache.as_ref()?;
        load.answer(self.patient).ok().cloned()
    }

    /// Whether outliers being found have come in since the last look: the
    /// camera fits again.
    pub(super) fn outliers_arrived(&self) -> bool {
        let mut cache = self.outlier_flags.borrow_mut();
        match cache.as_mut() {
            Some((_, load, seen)) if !*seen && load.try_get().is_some() => {
                *seen = true;
                true
            }
            _ => false,
        }
    }

    /// The bounds the camera fits: the layout's, or with outliers on,
    /// those of the rest.
    pub(super) fn fit_bounds(&self) -> [f32; 4] {
        self.current_outliers()
            .map_or(self.current().points.bounds, |f| f.bounds)
    }

    /// Find and treat outliers as `o` says from now on.
    pub(super) fn set_outliers(&mut self, o: Outliers) {
        if o != self.outliers {
            self.outliers = o;
            *self.outlier_flags.borrow_mut() = None;
        }
    }

    /// `O`: off, fit, hide, off, the rest of the setting kept.
    pub(super) fn cycle_outliers(&mut self) {
        let mut o = self.outliers.clone();
        o.mode = o.mode.next();
        self.set_outliers(o);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(o: &Outliers, p: &Points, method: &str) -> Flags {
        o.find(&p.xy, &p.order, p.bounds, method)
    }

    /// A grid of points, a tight clique far out on x, and one stray.
    fn layout() -> Points {
        // A jittered grid: its spacing varies, as a layout's does.
        let jitter = |i: usize, k: usize| ((i * k) % 97) as f32 / 97.0 * 0.6;
        let mut xy: Vec<[f32; 2]> = (0..400)
            .map(|i| {
                [
                    (i % 20) as f32 + jitter(i, 31),
                    (i / 20) as f32 + jitter(i, 57),
                ]
            })
            .collect();
        // A localised component: 15 cells tight together, far out.
        xy.extend((0..15).map(|i| [500.0 + 0.01 * i as f32, 10.0]));
        // A lone stray between.
        xy.push([60.0, 60.0]);
        let names = (0..xy.len()).map(|i| i.to_string().into()).collect();
        Points::new(names, xy)
    }

    #[test]
    fn isolation_catches_the_stray_and_keeps_the_clique() {
        let p = layout();
        let o = Outliers {
            mode: Mode::Fit,
            rule: Some(Rule::Isolated),
            ..Default::default()
        };
        let f = find(&o, &p, "tsne");
        assert!(f.out[415], "the stray");
        assert!(!f.out[400], "the clique is tight, so kept");
        assert!(f.bounds[2] >= 500.0);
    }

    #[test]
    fn range_catches_the_clique_a_spectral_axis_spikes_on() {
        let p = layout();
        let o = Outliers {
            mode: Mode::Hide,
            quantile: Some(0.95),
            ..Default::default()
        };
        // Diffusion takes the range rule by default.
        let f = find(&o, &p, "diffusion");
        assert_eq!(f.rule, Rule::Range);
        assert!((400..415).all(|i| f.out[i]));
        assert!(f.bounds[2] < 100.0);
        assert!(f.describe().starts_with("hide (range, "));
    }

    #[test]
    fn a_stack_of_duplicates_does_not_make_the_rest_outliers() {
        // 60% of points at one place: the MAD of x is zero.
        let mut xy = vec![[0.0f32, 0.0]; 60];
        xy.extend((0..40).map(|i| [1.0 + i as f32 * 0.1, i as f32 * 0.1]));
        let names = (0..xy.len()).map(|i| i.to_string().into()).collect();
        let p = Points::new(names, xy);
        let o = Outliers {
            mode: Mode::Fit,
            rule: Some(Rule::Range),
            mad: Some(5.0),
            ..Default::default()
        };
        let f = find(&o, &p, "umap");
        assert!(f.n < 10, "{} flagged", f.n);
    }

    #[test]
    fn bad_settings_are_refused() {
        let bad = |o: Outliers| o.check().is_err();
        assert!(bad(Outliers {
            k: Some(0),
            ..Default::default()
        }));
        assert!(bad(Outliers {
            quantile: Some(0.4),
            ..Default::default()
        }));
        assert!(bad(Outliers {
            mad: Some(3.0),
            quantile: Some(0.99),
            ..Default::default()
        }));
    }
}
