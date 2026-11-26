//! How a number per cell is coloured: the scale its values are put on
//! (linear, log10, log1p, sqrt), the range the ramp spans (quantiles to
//! clip at, or a fixed min and max, so several maps share one), the
//! palette (sequential or diverging about a midpoint) and its direction,
//! how cells with no value are drawn, which points are drawn on top, and
//! the key: where it goes, how long its bar is, its title.
//!
//! One [`ColourScale`] holds for whatever is coloured (a feature, a cell
//! statistic, a table's values, a stage-2 map); `view_map` sets it and
//! every map and export after draws with it.

use super::activity::Levels;
use super::color::Rgb;
pub use legume_tui::map::legend::{KeyPlace, RampKey};
pub use legume_tui::map::palette::Palette;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The scale values are put on before the ramp spans them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transform {
    #[default]
    Linear,
    /// `log10(v)`; values at or below zero have no colour.
    Log10,
    /// `ln(1 + v)`; values below zero are taken as zero.
    Log1p,
    /// `√v`; values below zero are taken as zero.
    Sqrt,
}

impl Transform {
    #[inline]
    #[must_use]
    pub fn apply(self, v: f32) -> f32 {
        match self {
            Self::Linear => v,
            Self::Log10 if v > 0.0 => v.log10(),
            Self::Log10 => f32::NAN,
            Self::Log1p => v.max(0.0).ln_1p(),
            Self::Sqrt => v.max(0.0).sqrt(),
        }
    }

    /// The value whose transform is `t`.
    #[must_use]
    pub fn invert(self, t: f32) -> f32 {
        match self {
            Self::Linear => t,
            Self::Log10 => 10f32.powf(t),
            Self::Log1p => t.exp_m1(),
            Self::Sqrt => t * t,
        }
    }

    /// The scale a source's values take by default: counts on
    /// `ln(1 + count)`, the rest linear.
    #[must_use]
    pub fn for_counts(counts: bool) -> Self {
        if counts {
            Self::Log1p
        } else {
            Self::Linear
        }
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Log10 => "log10",
            Self::Log1p => "log1p",
            Self::Sqrt => "sqrt",
        }
    }
}

/// How points with no value (unmatched, not a number) are drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Missing {
    /// Muted grey, under the rest.
    #[default]
    Grey,
    /// Not drawn.
    Hidden,
}

/// Which points are drawn last, on top.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Order {
    /// High values on top (on a diverging ramp: the largest departures).
    #[default]
    High,
    /// Low values on top.
    Low,
    /// In the map's fixed random order, so no level hides another.
    Random,
}

/// The key of a number per cell.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyStyle {
    /// Where it goes [default: top_left]; in a figure panel `right` and
    /// `below` put it outside the map.
    #[serde(default)]
    pub place: KeyPlace,
    /// The bar's length in line heights [default: 8].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<f32>,
    /// The key's title [default: what is coloured].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// No key at all.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
}

/// How a number per cell is coloured; every field unset is the default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColourScale {
    /// [default: what is coloured says: log1p for counts, else linear]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<Transform>,
    /// The ramp spans these percentiles of the values (on the scale),
    /// `[low, high]` [default: 2 to 99, from 0 for counts].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<[f32; 2]>,
    /// The ramp's ends in the values' own units, before the scale; either
    /// overrides the percentile on its side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette: Option<Palette>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reverse: bool,
    /// The centre of a diverging ramp, in the values' units [default: 0].
    /// Given alone, it asks for the diverging ramp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub midpoint: Option<f32>,
    #[serde(default)]
    pub missing: Missing,
    #[serde(default)]
    pub order: Order,
    #[serde(default)]
    pub key: KeyStyle,
}

impl ColourScale {
    /// Refused: percentiles out of order or outside 0 to 100, a min not
    /// below its max, a non-finite end, a midpoint with a sequential
    /// palette, a key bar of no length.
    pub fn check(&self) -> Result<(), String> {
        if let Some([a, b]) = self.clip {
            if !(0.0..=100.0).contains(&a) || !(0.0..=100.0).contains(&b) || a >= b {
                return Err("clip is [low, high] percentiles, 0 ≤ low < high ≤ 100".into());
            }
        }
        for v in [self.min, self.max, self.midpoint].into_iter().flatten() {
            if !v.is_finite() {
                return Err("min, max and midpoint are finite numbers".into());
            }
        }
        if let (Some(lo), Some(hi)) = (self.min, self.max) {
            if lo >= hi {
                return Err("min is below max".into());
            }
        }
        let t = self.scale.unwrap_or_default();
        for v in [self.min, self.max, self.midpoint].into_iter().flatten() {
            if !t.apply(v).is_finite() {
                return Err(format!("{v} has no place on a {} scale", t.name()));
            }
        }
        if self.midpoint.is_some() && self.palette.is_some_and(|p| !p.is_diverging()) {
            return Err("a midpoint is for a diverging palette (blue_red, purple_green)".into());
        }
        if self.key.length.is_some_and(|l| !(l.is_finite() && l > 0.0)) {
            return Err("key length is a positive number of line heights".into());
        }
        Ok(())
    }

    /// Whether `levels` are drawn on a diverging ramp: the palette's
    /// kind, else a midpoint asks for one, else signed levels are.
    #[must_use]
    pub fn diverging(&self, signed: bool) -> bool {
        self.palette
            .map_or(signed || self.midpoint.is_some(), Palette::is_diverging)
    }

    /// The ramp drawn, 256 colours low to high.
    #[must_use]
    pub fn ramp(&self, diverging: bool) -> Vec<Rgb> {
        let p = self.palette.unwrap_or(if diverging {
            Palette::BlueRed
        } else {
            Palette::Activity
        });
        p.lin(256, self.reverse)
    }
}

/// Up to about six round values between `lo` and `hi` (the values' own
/// units) on `transform`'s scale, each with where it falls on the ramp,
/// 0 to 1, and its label; the ends are labelled too when no round value
/// is near them.
#[must_use]
pub fn ticks(transform: Transform, lo: f32, hi: f32) -> Vec<(f32, String)> {
    use legume_tui::hist::Axis;
    let (a, b) = (transform.apply(lo), transform.apply(hi));
    if !(a.is_finite() && b.is_finite() && b > a) {
        return Vec::new();
    }
    let at = |v: f64| (transform.apply(v as f32) - a) / (b - a);
    let axis = match transform {
        Transform::Linear | Transform::Sqrt => Axis::linear(lo as f64, hi as f64),
        Transform::Log1p => Axis::log1p(lo.max(0.0) as f64, hi as f64),
        Transform::Log10 => Axis {
            lo: lo as f64,
            hi: hi as f64,
            log: true,
            shift: 0.0,
        },
    };
    let mut out: Vec<(f32, String)> = axis
        .ticks()
        .into_iter()
        .map(|v| (at(v), label(v)))
        .filter(|(t, _)| (-1e-4..=1.0 + 1e-4).contains(t))
        .collect();
    // On a square root the low round values crowd: keep a tick only a
    // tenth of the bar past the last.
    let mut kept: Vec<(f32, String)> = Vec::new();
    for t in out.drain(..) {
        if kept.last().is_none_or(|l| t.0 - l.0 >= 0.1) {
            kept.push(t);
        }
    }
    let near = |t: f32, ticks: &[(f32, String)]| ticks.iter().any(|(u, _)| (u - t).abs() < 0.12);
    if !near(0.0, &kept) {
        kept.insert(0, (0.0, label(lo as f64)));
    }
    if !near(1.0, &kept) {
        kept.push((1.0, label(hi as f64)));
    }
    kept
}

/// A number as a key labels it: three significant figures at most,
/// no trailing zeros, thousands and up as `12k`.
#[must_use]
pub fn label(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return "0".into();
    }
    let a = v.abs();
    if a >= 1e4 {
        let (d, unit) = if a >= 1e6 { (1e6, "M") } else { (1e3, "k") };
        return format!("{}{unit}", label(v / d));
    }
    if a < 1e-3 {
        return format!("{v:.0e}");
    }
    let digits = (2 - a.log10().floor() as i32).max(0) as usize;
    let s = format!("{v:.digits$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

impl Levels {
    /// The ramp's ends in the values' own units.
    #[must_use]
    pub fn range(&self) -> [f32; 2] {
        [
            self.transform.invert(self.lo),
            self.transform.invert(self.hi),
        ]
    }

    /// The key's ticks for these levels; on a diverging ramp its middle
    /// is always one.
    pub(super) fn key_ticks(&self) -> Vec<(f32, String)> {
        let [lo, hi] = self.range();
        let mut t = ticks(self.transform, lo, hi);
        if self.diverging {
            let mid = self.transform.invert(0.5 * (self.lo + self.hi));
            t.retain(|(u, _)| (u - 0.5).abs() >= 0.08);
            t.push((0.5, label(f64::from(mid))));
            t.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        t
    }
}

/// What a ramp key shows, owned: its title, ramp, ticks, bar length and
/// the swatch for points with no value; borrowed as a [`RampKey`] to draw.
pub struct KeyParts {
    pub title: String,
    pub ramp: Arc<[Rgb]>,
    pub ticks: Vec<(f32, String)>,
    pub length: Option<f32>,
    pub missing: Option<Rgb>,
    pub place: KeyPlace,
}

impl KeyParts {
    #[must_use]
    pub fn key(&self) -> RampKey<'_> {
        RampKey {
            title: &self.title,
            ramp: &self.ramp,
            ticks: &self.ticks,
            length: self.length,
            missing: self.missing.map(|c| ("NA", c)),
        }
    }
}

impl super::Scene {
    /// The key of the number per cell drawn, unless there is none or the
    /// colour scale hides it.
    pub(super) fn key_parts(&self) -> Option<KeyParts> {
        let s = self.current_shown()?;
        let k = &self.colour_scale.key;
        if k.hidden {
            return None;
        }
        let l = &s.levels;
        Some(KeyParts {
            title: k.title.clone().unwrap_or_else(|| s.title.clone()),
            ramp: l.ramp_arc(),
            ticks: l.ticks().to_vec(),
            length: k.length,
            missing: (!l.hide_missing && l.any_missing())
                .then(|| super::color::linear_rgb(super::color::MUTED)),
            place: k.place,
        })
    }

    /// Colour every number per cell as `scale` says from now on.
    pub(super) fn set_colour_scale(&mut self, scale: ColourScale) {
        if scale == self.colour_scale {
            return;
        }
        self.colour_scale = scale;
        if let Some(s) = self.shown.as_mut() {
            s.levels.fit(&self.colour_scale);
            // A new id: cached draw orders were for the old levels.
            self.shown_ids += 1;
            s.id = self.shown_ids;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_invert_and_refuse_what_has_no_place() {
        for t in [
            Transform::Linear,
            Transform::Log10,
            Transform::Log1p,
            Transform::Sqrt,
        ] {
            let v = 37.5;
            assert!((t.invert(t.apply(v)) - v).abs() < 1e-3, "{}", t.name());
        }
        assert!(Transform::Log10.apply(0.0).is_nan());
        let bad = ColourScale {
            scale: Some(Transform::Log10),
            min: Some(0.0),
            ..Default::default()
        };
        assert!(bad.check().is_err());
        let seq_mid = ColourScale {
            palette: Some(Palette::Viridis),
            midpoint: Some(0.0),
            ..Default::default()
        };
        assert!(seq_mid.check().is_err());
        assert!(ColourScale {
            clip: Some([99.0, 1.0]),
            ..Default::default()
        }
        .check()
        .is_err());
    }

    #[test]
    fn ticks_are_round_and_the_ends_labelled() {
        let t = ticks(Transform::Linear, 0.0, 10.0);
        let labels: Vec<&str> = t.iter().map(|(_, l)| l.as_str()).collect();
        assert_eq!(labels, ["0", "2", "4", "6", "8", "10"]);
        assert!((t[1].0 - 0.2).abs() < 1e-6);
        // A clipped end that is not round is labelled as it is.
        let t = ticks(Transform::Linear, 0.0, 9.37);
        assert_eq!(t.last().unwrap(), &(1.0, "9.37".to_string()));
        // Decades on a log scale.
        let t = ticks(Transform::Log10, 1.0, 1000.0);
        let labels: Vec<&str> = t.iter().map(|(_, l)| l.as_str()).collect();
        assert!(labels.contains(&"10") && labels.contains(&"100"));
        assert_eq!(label(12_500.0), "12.5k");
        assert_eq!(label(0.15), "0.15");
        assert_eq!(label(-3.0), "-3");
    }
}
