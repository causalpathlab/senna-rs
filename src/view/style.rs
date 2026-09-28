//! Per-group point styles: colour, shape, opacity, size, visibility.
//!
//! Styles are keyed by grouping and group name (`annotation` / `CT1`), so a
//! choice made for a cell type holds in every layout and space, and are saved
//! to `{prefix}.view_style.json` beside the run so they carry over between
//! sessions and into saved images.

use super::color::{self, Rgb};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    #[default]
    Circle,
    Square,
    Diamond,
    Triangle,
    Cross,
}

impl Shape {
    pub const ALL: [Shape; 5] = [
        Shape::Circle,
        Shape::Square,
        Shape::Diamond,
        Shape::Triangle,
        Shape::Cross,
    ];

    #[must_use]
    pub fn glyph(self) -> &'static str {
        match self {
            Shape::Circle => "●",
            Shape::Square => "■",
            Shape::Diamond => "◆",
            Shape::Triangle => "▲",
            Shape::Cross => "✚",
        }
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Shape::Circle => "circle",
            Shape::Square => "square",
            Shape::Diamond => "diamond",
            Shape::Triangle => "triangle",
            Shape::Cross => "cross",
        }
    }

    /// Coverage of pixel offset `(dx, dy)` by this shape at radius `r`, with a
    /// one-pixel anti-aliased edge. Shapes are scaled to about the area of the
    /// circle of radius `r`, so switching shape does not change visual weight.
    #[inline]
    #[must_use]
    pub fn coverage(self, dx: f32, dy: f32, r: f32) -> f32 {
        let (ax, ay) = (dx.abs(), dy.abs());
        let d = match self {
            Shape::Circle => (dx * dx + dy * dy).sqrt() - r,
            Shape::Square => ax.max(ay) - 0.886 * r,
            Shape::Diamond => (ax + ay) * std::f32::consts::FRAC_1_SQRT_2 - 0.886 * r,
            Shape::Triangle => {
                // Upward equilateral triangle (screen y grows downward).
                let s = 1.35 * r;
                let k = 3f32.sqrt();
                let (mut px, mut py) = (ax - s, -dy + s / k);
                if px + k * py > 0.0 {
                    (px, py) = ((px - k * py) * 0.5, (-k * px - py) * 0.5);
                }
                px -= px.clamp(-2.0 * s, 0.0);
                -(px * px + py * py).sqrt() * py.signum()
            }
            Shape::Cross => {
                let (a, w) = (1.15 * r, 0.42 * r);
                (ax - a).max(ay - w).min((ax - w).max(ay - a))
            }
        };
        (0.5 - d).clamp(0.0, 1.0)
    }

    /// How far past `r` the shape reaches.
    #[must_use]
    pub fn reach(self, r: f32) -> f32 {
        1.4 * r + 1.0
    }
}

/// A user's choices for one group; unset fields fall back to the defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Style {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colour: Option<[u8; 3]>,
    #[serde(default)]
    pub shape: Shape,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub alpha: f32,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub size: f32,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
}

fn one() -> f32 {
    1.0
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_one(v: &f32) -> bool {
    (*v - 1.0).abs() < 1e-6
}

impl Style {
    #[must_use]
    pub fn plain() -> Self {
        Self {
            alpha: 1.0,
            size: 1.0,
            ..Default::default()
        }
    }
}

/// A group's style resolved for drawing.
#[derive(Clone, Copy, Debug)]
pub struct Resolved {
    pub colour: Rgb,
    pub ink: Rgb,
    pub shape: Shape,
    pub alpha: f32,
    pub size: f32,
    pub hidden: bool,
}

/// Colours offered in the style menu: the categorical palette's hues at two
/// lightnesses, then grays and near-black. Built once.
#[must_use]
pub fn swatches() -> &'static [[u8; 3]] {
    static SWATCHES: std::sync::LazyLock<Vec<[u8; 3]>> = std::sync::LazyLock::new(|| {
        let enc = color::encoder();
        let to8 = |c: Rgb| c.map(|v| enc.encode(v));
        let mut out: Vec<[u8; 3]> = (0..12).map(|i| to8(color::category(i, 12))).collect();
        out.extend((0..12).map(|i| to8(color::ink_for(color::category(i, 12)))));
        out.extend([[200, 199, 196], [150, 150, 150], [90, 90, 90], [30, 30, 30]]);
        out
    });
    &SWATCHES
}

/// Every saved style, by grouping then group name.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Book {
    #[serde(flatten)]
    pub groupings: BTreeMap<String, BTreeMap<String, Style>>,
}

impl Book {
    #[must_use]
    pub fn path(prefix: &str) -> String {
        format!("{prefix}.view_style.json")
    }

    /// The saved book, or an empty one when there is none or it is unreadable.
    #[must_use]
    pub fn load(prefix: &str) -> Self {
        super::files::read_json(std::path::Path::new(&Self::path(prefix))).unwrap_or_default()
    }

    pub fn save(&self, prefix: &str) -> anyhow::Result<()> {
        std::fs::write(Self::path(prefix), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    #[must_use]
    pub fn get(&self, grouping: &str, group: &str) -> Style {
        self.groupings
            .get(grouping)
            .and_then(|m| m.get(group))
            .cloned()
            .unwrap_or_else(Style::plain)
    }

    /// Store `style`, dropping it when it is all defaults.
    pub fn set(&mut self, grouping: &str, group: &str, style: Style) {
        let m = self.groupings.entry(grouping.to_string()).or_default();
        if style == Style::plain() {
            m.remove(group);
        } else {
            m.insert(group.to_string(), style);
        }
        if m.is_empty() {
            self.groupings.remove(grouping);
        }
    }

    /// Resolve every group of a grouping for drawing, `n` groups in all.
    #[must_use]
    pub fn resolve(&self, grouping: &str, levels: &[Box<str>]) -> Vec<Resolved> {
        let n = levels.len();
        levels
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let s = self.get(grouping, name);
                let colour = s
                    .colour
                    .map_or_else(|| color::category(i, n), color::linear_rgb);
                let ink = color::ink_for(colour);
                Resolved {
                    colour,
                    ink,
                    shape: s.shape,
                    alpha: s.alpha.clamp(0.05, 1.0),
                    size: s.size.clamp(0.25, 4.0),
                    hidden: s.hidden,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_styles_are_not_stored_and_others_round_trip() {
        let mut b = Book::default();
        b.set("annotation", "CT1", Style::plain());
        assert!(b.groupings.is_empty());
        let s = Style {
            colour: Some([10, 20, 30]),
            shape: Shape::Diamond,
            alpha: 0.5,
            size: 2.0,
            hidden: false,
        };
        b.set("annotation", "CT1", s.clone());
        let back: Book = serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
        assert_eq!(back.get("annotation", "CT1"), s);
        assert_eq!(back.get("annotation", "CT2"), Style::plain());
        b.set("annotation", "CT1", Style::plain());
        assert!(b.groupings.is_empty());
    }

    #[test]
    fn every_shape_covers_its_centre_and_not_far_outside() {
        for s in Shape::ALL {
            assert!(s.coverage(0.0, 0.0, 3.0) > 0.99, "{s:?} centre");
            assert!(s.coverage(6.0, 6.0, 3.0) < 0.01, "{s:?} outside");
        }
    }
}
