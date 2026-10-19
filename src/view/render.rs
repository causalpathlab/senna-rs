//! senna's points on legume-tui's map: how a scene's layer is ranked and
//! marked, and which labels win a collision.
//!
//! Points are drawn as opaque, anti-aliased marks straight onto a
//! linear-light canvas by the map's progressive `Job`; only a group given
//! an opacity below one in the style menu is translucent. With opaque marks
//! the draw order says what is seen, so each layer is drawn in rank order:
//! muted and unlabelled points first, then labelled ones, then the focused
//! group last so it pops on top; activity is drawn low to high.

use super::activity::Levels;
use super::color::{self, Rgb};
use super::data::{Points, NONE};
use super::style::{Resolved, Shape};
use legume_tui::map::points::{by_level, ranked, Mark, PointLayer};
pub use legume_tui::map::points::{Job, Medians};
pub use legume_tui::map::raster::Frame;
pub use legume_tui::map::viewport::Viewport;
use legume_tui::map::viewport::{Limits, YAxis};
use std::sync::Arc;

/// How much of the frame a fitted layout fills on its tighter axis.
const MARGIN: f32 = 0.92;

/// `bounds` fitted into `w × h` with a small margin, data y up.
#[must_use]
pub fn fit(bounds: [f32; 4], w: usize, h: usize) -> Viewport {
    Viewport::fit(bounds, w, h, MARGIN, YAxis::Up)
}

/// No limits: senna zooms and pans anywhere.
pub const FREE: Limits = Limits {
    scale: None,
    keep_inside: None,
};

/// How one layer of points is drawn.
pub struct Paint<'a> {
    pub points: &'a Points,
    /// Group id per point (`NONE` = unlabelled), or `None` for a flat colour.
    pub groups: Option<&'a [u32]>,
    /// Style per group id.
    pub styles: &'a [Resolved],
    pub focus: Option<u32>,
    /// Groups picked out (by group id), drawn in colour and on top; the rest
    /// muted. Overrides `focus`.
    pub selected: Option<&'a [bool]>,
    /// The group under the merge cursor: drawn in ink, on top of the rest,
    /// so where it lies shows whether or not it is chosen.
    pub cursor: Option<u32>,
    /// Draw everything muted (a backdrop layer).
    pub muted: bool,
    /// Radius multiplier.
    pub size: f32,
    /// Feature activity per point, drawn on `ramp` instead of group colours.
    pub levels: Option<(&'a Levels, &'a [Rgb])>,
    /// The draw order, when the caller has it cached; else it is computed.
    pub order: Option<Arc<Vec<u32>>>,
}

impl Paint<'_> {
    /// The mark for point `i`, muted and in ink as given, or `None` when
    /// its group is hidden.
    #[inline]
    fn mark_in(&self, i: usize, muted: Rgb, ink: Rgb) -> Option<Mark> {
        let plain = |colour| Mark {
            colour,
            alpha: 1.0,
            shape: Shape::Circle,
            size: 1.0,
        };
        if self.muted {
            return Some(plain(muted));
        }
        if let Some((levels, ramp)) = self.levels {
            let t = levels.t(i);
            if t < 0.0 {
                return Some(plain(muted));
            }
            let k = ((ramp.len() - 1) as f32 * t).round() as usize;
            return Some(plain(ramp[k]));
        }
        let g = self.groups.map_or(NONE, |g| g[i]);
        if g == NONE {
            return Some(plain(muted));
        }
        let st = &self.styles[g as usize];
        if st.hidden {
            return None;
        }
        if self.cursor == Some(g) {
            return Some(Mark {
                colour: ink,
                alpha: 1.0,
                shape: st.shape,
                size: st.size,
            });
        }
        let picked = match self.selected {
            Some(sel) => sel.get(g as usize).copied().unwrap_or(false),
            None => self.focus.is_none_or(|f| f == g),
        };
        let colour = if picked { st.colour } else { muted };
        Some(Mark {
            colour,
            alpha: st.alpha,
            shape: st.shape,
            size: st.size,
        })
    }

    /// Draw rank: higher is drawn later, on top.
    #[inline]
    fn rank(&self, i: usize) -> u8 {
        if self.muted || self.levels.is_some() {
            return 0;
        }
        let g = self.groups.map_or(NONE, |g| g[i]);
        if g != NONE && self.cursor == Some(g) {
            return 3;
        }
        if let (Some(sel), true) = (self.selected, g != NONE) {
            return if sel.get(g as usize).copied().unwrap_or(false) {
                2
            } else {
                0
            };
        }
        match (g, self.focus) {
            (NONE, _) => 0,
            (g, Some(f)) if g == f => 2,
            (_, Some(_)) => 0,
            (_, None) => 1,
        }
    }

    /// The order to draw in: by rank (activity: by level), keeping the
    /// shuffled order within a rank.
    #[must_use]
    pub fn draw_order(&self) -> Vec<u32> {
        let order = &self.points.order;
        if let Some((levels, _)) = self.levels {
            return by_level(order, |i| levels.t(i));
        }
        ranked(order, |i| self.rank(i), 4)
    }
}

impl PointLayer for Paint<'_> {
    fn xy(&self, i: usize) -> [f32; 2] {
        self.points.xy[i]
    }

    fn shuffled(&self) -> &[u32] {
        &self.points.order
    }

    fn order(&self) -> Arc<Vec<u32>> {
        self.order
            .clone()
            .unwrap_or_else(|| Arc::new(self.draw_order()))
    }

    #[inline]
    fn mark(&self, i: usize) -> Option<Mark> {
        self.mark_in(
            i,
            color::linear_rgb(color::MUTED),
            color::linear_rgb(color::TEXT),
        )
    }

    fn size(&self) -> f32 {
        self.size
    }
}

/// Which labels win a collision, before `priority` is compared: whatever
/// was just asked for over the map's own labels.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    /// Group and point names.
    Map,
    /// Cells near a clicked feature.
    NearCell,
    /// Features near a clicked cell or feature.
    NearFeature,
    /// The clicked cell or feature.
    Centre,
    /// The picked feature.
    Picked,
}

impl From<Tier> for u8 {
    /// The map's tier: the order kept, `Map` lowest.
    fn from(t: Tier) -> u8 {
        t as u8
    }
}

/// A label candidate in pixel space, keyed by the group it names on a cell
/// map, so a click can find it.
pub type Label = legume_tui::map::labels::Label<u32>;

/// Median position of each group, from a prefix of the shuffled order (a
/// uniform subsample, so the estimate is unbiased), with group sizes.
#[must_use]
pub fn group_medians(points: &Points, groups: &[u32], n_groups: usize) -> Medians {
    legume_tui::map::points::group_medians(
        |i| points.xy[i],
        &points.order,
        |i| Some(groups[i]).filter(|&g| g != NONE),
        n_groups,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::data::Points;
    use crate::view::style::Resolved;

    #[test]
    fn the_merge_cursor_group_is_drawn_last_in_ink() {
        let xy: Vec<[f32; 2]> = (0..6).map(|i| [i as f32, 0.0]).collect();
        let names = (0..6).map(|i| i.to_string().into()).collect();
        let points = Points::new(names, xy);
        let groups = [0, 1, 2, 0, 1, 2];
        let style = |c: f32| Resolved {
            colour: [c; 3],
            ink: [0.0; 3],
            shape: Shape::Circle,
            alpha: 1.0,
            size: 1.0,
            hidden: false,
        };
        let styles = [style(0.1), style(0.2), style(0.3)];
        // Group 0 chosen; the cursor on group 2, not chosen.
        let selected = [true, false, false];
        let paint = Paint {
            points: &points,
            groups: Some(&groups),
            styles: &styles,
            focus: None,
            selected: Some(&selected),
            cursor: Some(2),
            muted: false,
            size: 1.0,
            levels: None,
            order: None,
        };
        let order = paint.draw_order();
        let last: Vec<u32> = order[4..].iter().map(|&i| groups[i as usize]).collect();
        assert_eq!(last, [2, 2]);
        let (muted, ink) = ([0.9; 3], [0.05; 3]);
        assert_eq!(paint.mark_in(2, muted, ink).unwrap().colour, ink);
        assert_eq!(paint.mark_in(0, muted, ink).unwrap().colour, [0.1; 3]);
        assert_eq!(paint.mark_in(1, muted, ink).unwrap().colour, muted);
    }

    #[test]
    fn tiers_keep_their_order_as_the_map_ranks_them() {
        let tiers = [
            Tier::Map,
            Tier::NearCell,
            Tier::NearFeature,
            Tier::Centre,
            Tier::Picked,
        ];
        assert_eq!(tiers.map(u8::from), [0, 1, 2, 3, 4]);
    }
}
