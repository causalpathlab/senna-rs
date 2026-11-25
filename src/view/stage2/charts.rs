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
use legume_tui::chart::scatter::{self, Point, Scatter, ScatterStyle};
use legume_tui::map::text::Font;
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
    /// `(gene, donor, log gain, z)`, the faintest (least supported) first.
    dots: Vec<(usize, usize, f32, f32)>,
    /// The dots as drawn, beside them: log gain × |z|, by donor.
    points: Vec<Point>,
    /// The axes: log gain over `±reach`, |z| over `0..top`.
    reach: f32,
    top: f32,
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
    let mut dots: Vec<(usize, usize, f32, f32)> = (0..s.genes.len())
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
    // Drawn in order: the supported dots last, on top of the faint ones.
    dots.par_sort_unstable_by(|a, b| gains::weight(a.3).total_cmp(&gains::weight(b.3)));
    let points = dots
        .iter()
        .map(|&(_, d, gain, z)| Point::new(gain, z.abs(), d, gains::weight(z)))
        .collect();
    let reach = dots
        .iter()
        .map(|d| d.2.abs())
        .fold(0f32, f32::max)
        .max(1e-3);
    let top = dots
        .iter()
        .map(|d| d.3.abs())
        .filter(|z| z.is_finite())
        .fold(gains::Z_FULL * 1.5, f32::max)
        .min(Z_TOP);
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
        points,
        reach,
        top,
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

/// The volcano, the dots of gene `ring_gene` (the one shown) ringed:
/// legume-tui's scatter of log gain × |z|, a series per donor, each dot as
/// strong as its support.
pub(crate) fn draw_volcano(
    frame: &mut Frame,
    v: &Volcano,
    ring_gene: Option<usize>,
    font: Font,
    bold: Font,
) {
    let ringed: Vec<usize> = v
        .dots
        .iter()
        .enumerate()
        .filter(|(_, d)| Some(d.0) == ring_gene)
        .map(|(i, _)| i)
        .collect();
    let title = format!(
        "every gene's log gain × |z| per donor, {} · line at |z| = {}",
        v.at,
        gains::Z_FULL
    );
    let chart = Scatter::new(&v.points, &v.donors)
        .with_title(&title)
        .with_labels("log gain", "|z|")
        .with_ranges(Some((-v.reach, v.reach)), Some((0.0, v.top)))
        .with_lines(&[gains::Z_FULL], &[0.0])
        .with_named(&v.named)
        .with_ringed(&ringed);
    // A volcano that cannot be drawn says why in its place.
    let _ = scatter::draw_scatter(
        frame,
        &chart,
        font,
        bold,
        &color::THEME,
        &ScatterStyle::default(),
    );
}
