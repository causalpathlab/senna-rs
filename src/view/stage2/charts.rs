//! Stage-2 charts, drawn in place of the map like the others (`H`):
//!
//! - **gain heatmap**: the gene shown, its log gain in each group on screen
//!   (rows) × donor (columns), at the mean state of the donor's cells
//!   there, faded toward 0 where the jackknife `|z|` is under
//!   [`gains::Z_FULL`]; blank where a donor has too few cells.
//! - **volcano**: every gene's log gain (x) against its `|z|` (y) for each
//!   donor at the state of its cells in the focused group (in the whole
//!   view, with none focused), a dot per (gene, donor) in the donor's
//!   colour; the gene shown ringed, the strongest named. It is made again
//!   when the state changes, not when another gene is shown.

use super::gains;
use super::Stage2;
use crate::view::color;
use crate::view::render::Frame;
use legume_tui::chart::heatmap::{self, Heat, HeatStyle};
use legume_tui::map::raster::{edge, ring, stamp, Shape};
use legume_tui::map::text::{self, Font};
use rayon::prelude::*;

/// Genes named on a volcano, the largest `|z|` first.
const NAMED: usize = 12;

/// `|z|` the volcano's y axis stops at; larger are drawn on its top edge.
const Z_TOP: f32 = 20.0;

/// The gene shown, its log gain by group × donor.
pub(crate) struct GainHeat {
    gene: String,
    grouping: String,
    groups: Vec<String>,
    donors: Vec<String>,
    /// Group-major, `groups × donors`: the gain, faded by its support;
    /// `NaN` where the donor has too few cells in the group.
    value: Vec<f32>,
    clip: f32,
}

/// Every gene's gain and `|z|` per donor at one state.
pub(crate) struct Volcano {
    /// Where the state is: the focused group, or the view.
    at: String,
    donors: Vec<String>,
    /// `(gene, donor, log gain, z)`.
    dots: Vec<(usize, usize, f32, f32)>,
    /// The strongest dots (by index into `dots`), named.
    named: Vec<(usize, String)>,
}

impl Volcano {
    /// Where its state is.
    pub(crate) fn at(&self) -> &str {
        &self.at
    }

    /// Its `top` dots of largest `|z|` (`donor`'s only, when given), the
    /// largest first: `(gene, donor, log gain, z)`.
    pub(crate) fn strongest(
        &self,
        donor: Option<usize>,
        top: usize,
    ) -> Vec<(usize, usize, f32, f32)> {
        strongest_of(&self.dots, donor, top)
            .into_iter()
            .map(|i| self.dots[i])
            .collect()
    }
}

/// The indices of `dots`' `top` of largest `|z|` (`donor`'s only, when
/// given), the largest first, found by a selection, not a sort of them all.
fn strongest_of(dots: &[(usize, usize, f32, f32)], donor: Option<usize>, top: usize) -> Vec<usize> {
    // A z without a spread (all refits alike) is no evidence either way.
    let mut order: Vec<usize> = (0..dots.len())
        .filter(|&i| dots[i].3.is_finite() && donor.is_none_or(|want| dots[i].1 == want))
        .collect();
    let by_z = |&a: &usize, &b: &usize| dots[b].3.abs().total_cmp(&dots[a].3.abs());
    if order.len() > top {
        order.select_nth_unstable_by(top, by_z);
        order.truncate(top);
    }
    order.sort_by(by_z);
    order
}

/// The gain heatmap of `gene` over the groups of `groups` (each cell's in
/// `levels`, beside its stage-2 row in `rows`).
pub(crate) fn gain_heat(
    s: &Stage2,
    gene: &str,
    rows: &[Option<u32>],
    groups: &[u32],
    levels: &[Box<str>],
    grouping: String,
) -> Result<GainHeat, String> {
    let g = s.gene(gene)?;
    let states = gains::donor_group_states(s, rows, Some(groups), levels.len());
    let mut shown: Vec<u32> = states.iter().map(|st| st.group).collect();
    shown.sort_unstable();
    shown.dedup();
    if shown.is_empty() {
        return Err(format!(
            "no donor has {} cells in any group",
            gains::MIN_CELLS_IN_GROUP
        ));
    }
    let d_n = s.donors.len();
    let mut value = vec![f32::NAN; shown.len() * d_n];
    for st in &states {
        let row = shown.binary_search(&st.group).expect("a group shown");
        let gain = s.gain(g, st.donor, &st.mean);
        value[row * d_n + st.donor] = gain.log_gain * gains::weight(gain.z);
    }
    let clip = value
        .iter()
        .filter(|v| v.is_finite())
        .fold(0f32, |m, v| m.max(v.abs()))
        .max(1e-3);
    Ok(GainHeat {
        gene: s.genes[g].to_string(),
        grouping,
        groups: shown
            .iter()
            .map(|&g| levels[g as usize].to_string())
            .collect(),
        donors: s.donors.iter().map(ToString::to_string).collect(),
        value,
        clip,
    })
}

/// The volcano at the mean state of each donor's cells among `rows` (the
/// cells of the focused group, or of the view).
pub(crate) fn volcano(s: &Stage2, rows: &[Option<u32>], at: String) -> Result<Volcano, String> {
    if !s.has_jackknife() {
        return Err("stage 2 has one batch: no jackknife, so no |z| to plot".into());
    }
    let states = gains::donor_group_states(s, rows, None, 1);
    if states.is_empty() {
        return Err(format!(
            "no donor has {} cells {at}",
            gains::MIN_CELLS_IN_GROUP
        ));
    }
    let dots: Vec<(usize, usize, f32, f32)> = (0..s.genes.len())
        .into_par_iter()
        .flat_map_iter(|g| {
            states.iter().map(move |st| {
                let gain = s.gain(g, st.donor, &st.mean);
                (g, st.donor, gain.log_gain, gain.z)
            })
        })
        // A deviation the fit set to 0 sits at the origin: left out.
        .filter(|&(_, _, gain, _)| gain != 0.0)
        .collect();
    let named = strongest_of(&dots, None, NAMED)
        .into_iter()
        .map(|i| {
            (
                i,
                format!("{} · {}", s.genes[dots[i].0], s.donors[dots[i].1]),
            )
        })
        .collect();
    Ok(Volcano {
        at,
        donors: s.donors.iter().map(ToString::to_string).collect(),
        dots,
        named,
    })
}

pub(crate) fn draw_gain_heat(frame: &mut Frame, h: &GainHeat, font: Font, bold: Font) {
    let title = format!(
        "{} log gain by {} × donor · at each donor's cells there · faded where |z| < {} · ±{:.2}",
        h.gene,
        h.grouping,
        gains::Z_FULL,
        h.clip
    );
    // A heatmap that cannot be drawn says why in its place.
    let _ = heatmap::draw_heat(
        frame,
        &Heat {
            title: &title,
            rows: &h.groups,
            cols: &h.donors,
            col_swatch: None,
            z: &h.value,
            clip: h.clip,
            bold_row: None,
        },
        font,
        bold,
        &color::THEME,
        &HeatStyle::for_theme(&color::THEME),
    );
}

/// The volcano, the dots of gene `ring_gene` (the one shown) ringed.
pub(crate) fn draw_volcano(
    frame: &mut Frame,
    v: &Volcano,
    ring_gene: Option<usize>,
    font: Font,
    bold: Font,
) {
    let ink = color::linear_rgb(color::INK);
    let muted = color::linear_rgb(color::MUTED);
    let bg = frame.background();
    let lh = font.line_height() as f32;
    let title = format!(
        "every gene's log gain × |z| per donor, {} · line at |z| = {}",
        v.at,
        gains::Z_FULL
    );
    text::draw(frame, bold, &title, lh as i32 / 2, lh as i32 / 2, ink, bg);
    // The plot: below the title and the donors' key, left of nothing.
    let (x0, y0) = (4.0 * lh, 3.5 * lh);
    let (x1, y1) = (frame.w as f32 - lh, frame.h as f32 - 2.5 * lh);
    if x1 - x0 < 4.0 * lh || y1 - y0 < 4.0 * lh {
        return;
    }
    let d_n = v.donors.len();
    let mut x = 1.5 * lh;
    for (d, name) in v.donors.iter().enumerate() {
        let c = color::category(d, d_n);
        stamp(frame, x, 2.2 * lh, lh / 3.0, Shape::Circle, c, 1.0);
        text::draw(
            frame,
            font,
            name,
            (x + lh / 2.0) as i32,
            (1.7 * lh) as i32,
            ink,
            bg,
        );
        x += lh + font.width(name) as f32 + lh;
    }
    let reach = v
        .dots
        .iter()
        .map(|d| d.2.abs())
        .fold(0f32, f32::max)
        .max(1e-3);
    let top = v
        .dots
        .iter()
        .map(|d| d.3.abs())
        .filter(|z| z.is_finite())
        .fold(gains::Z_FULL * 1.5, f32::max)
        .min(Z_TOP);
    let px = |gain: f32| x0 + (gain / reach + 1.0) / 2.0 * (x1 - x0);
    let py = |z: f32| {
        let z = if z.is_finite() { z.abs().min(top) } else { top };
        y1 - z / top * (y1 - y0)
    };
    // Axes: |z| = 0 and the support line, and 0 gain.
    edge(frame, (x0, y1), (x1, y1), (0.0, 0.0), ink, 0.8);
    edge(frame, (px(0.0), y0), (px(0.0), y1), (0.0, 0.0), muted, 0.8);
    let line = py(gains::Z_FULL);
    edge(frame, (x0, line), (x1, line), (0.0, 0.0), muted, 0.8);
    let ticks = [
        (format!("{:+.2}", -reach), x0),
        ("0".to_string(), px(0.0)),
        (format!("{:+.2}", reach), x1),
    ];
    for (t, at) in ticks {
        let w = font.width(&t) as f32;
        let left = (at - w / 2.0).clamp(0.0, frame.w as f32 - w);
        text::draw(
            frame,
            font,
            &t,
            left as i32,
            (y1 + lh / 3.0) as i32,
            ink,
            bg,
        );
    }
    text::draw(
        frame,
        font,
        "log gain",
        (x1 - 6.0 * lh) as i32,
        (y1 + 1.2 * lh) as i32,
        ink,
        bg,
    );
    text::draw(
        frame,
        font,
        &format!("{top:.0}"),
        (lh / 2.0) as i32,
        y0 as i32,
        ink,
        bg,
    );
    text::draw(
        frame,
        font,
        "|z|",
        (lh / 2.0) as i32,
        (y0 + lh) as i32,
        ink,
        bg,
    );
    let r = (lh / 6.0).max(1.5);
    for &(_, d, gain, z) in &v.dots {
        let c = color::category(d, d_n);
        stamp(
            frame,
            px(gain),
            py(z),
            r,
            Shape::Circle,
            c,
            0.35 + 0.65 * gains::weight(z),
        );
    }
    // The strongest dots, named; the gene shown, ringed.
    for (i, name) in &v.named {
        let (_, _, gain, z) = v.dots[*i];
        let (x, y) = (px(gain) + r + 2.0, py(z) - lh / 2.0);
        text::draw(frame, font, name, x as i32, y as i32, ink, bg);
    }
    for &(_, _, gain, z) in v.dots.iter().filter(|d| Some(d.0) == ring_gene) {
        ring(frame, px(gain), py(z), 2.0 * r, ink);
    }
}
