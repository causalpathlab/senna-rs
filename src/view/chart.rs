//! Charts drawn in place of the map, for the cells in view and the grouping
//! on screen (`c` changes it, as on the map):
//!
//! - **structure**: each cell's topic mixture as a thin stacked bar, cells
//!   side by side in one panel per group — the admixture plot of population
//!   genetics, as `lupin plot-topic` draws it. For runs with topics: topic
//!   runs, and `bge` runs (topics resolved from the cell embedding). Topics keep
//!   one colour each and one order everywhere (most prevalent first); inside
//!   a panel, cells go by their dominant topic, then by how dominant it is.
//! - **heatmap**: the top features of each group × the groups. A group's
//!   features are those whose level there most exceeds the next-highest
//!   group's, so each peaks in its own group and the rows fall into a
//!   diagonal: the model proposes candidates, the values shown choose. Each cell of the heatmap is the
//!   group's mean `ln(1 + count)` from the data files (the model's expected
//!   level when they cannot be read), z-scored per feature across the groups
//!   and clipped to `±CLIP`.

use super::activity::Source;
use super::color::{self, Rgb};
use super::data::NONE;
use super::render::Frame;
use super::text::{self, Font};
use super::Axis;
use super::Scene;

/// z-scores are clipped to this, so one extreme group cannot wash out the
/// rest of the colours.
pub const CLIP: f32 = 2.5;
/// Candidates the model proposes per feature shown, for the counts to
/// choose from.
const CANDIDATES: usize = 5;
/// Top features per group to begin with, and the range `+` / `-` allow.
pub const TOP_START: usize = 10;
const TOP_RANGE: (usize, usize) = (1, 100);

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Kind {
    Structure,
    Heatmap,
}

impl Kind {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Kind::Structure => "structure plot",
            Kind::Heatmap => "heatmap",
        }
    }
}

/// What a chart was made from: the space, the grouping, and the features
/// per group; made again when any changes.
type ChartKey = (usize, Option<usize>, usize);

pub struct Chart {
    pub kind: Kind,
    /// Top features per group, for the heatmap.
    pub top: usize,
    made: Option<(ChartKey, Result<Data, String>)>,
}

impl Chart {
    /// The same chart for another view of the run, made afresh there.
    #[must_use]
    pub fn like(&self) -> Self {
        Self {
            top: self.top,
            ..Self::new(self.kind)
        }
    }

    fn new(kind: Kind) -> Self {
        Self {
            kind,
            top: TOP_START,
            made: None,
        }
    }
}

enum Data {
    Structure(Structure),
    Heatmap(Heatmap),
}

/// Cells' topic mixtures, grouped into panels.
pub struct Structure {
    grouping: String,
    /// Per panel, its name and its cells' mixtures in drawing order, each
    /// `k` values in the display order of the topics.
    panels: Vec<(String, Vec<f32>)>,
    k: usize,
    /// Topic ids in display order: most prevalent first.
    order: Vec<usize>,
}

/// Top features × groups, z-scored per feature.
pub struct Heatmap {
    grouping: String,
    features: Vec<Box<str>>,
    groups: Vec<String>,
    /// Feature-major, `features.len() × groups.len()`; `NaN`: no value.
    z: Vec<f32>,
    source: Source,
    top: usize,
}

impl Scene {
    /// Map → structure plot (runs with topics) → heatmap → map.
    pub fn cycle_chart(&mut self) {
        if self.current().axis() != Axis::Cells {
            self.note = Some("charts are drawn from a cell view (tab)".into());
            return;
        }
        let topic = self.is_topic_run();
        let next = match self.chart.as_ref().map(|c| c.kind) {
            None if topic => Some(Kind::Structure),
            None | Some(Kind::Structure) => Some(Kind::Heatmap),
            Some(Kind::Heatmap) => None,
        };
        let top = self.chart.as_ref().map_or(TOP_START, |c| c.top);
        self.chart = next.map(|kind| Chart {
            top,
            ..Chart::new(kind)
        });
        self.note = Some(match next {
            Some(k) => format!("{} · H next · c groups by another grouping", k.name()),
            None => "back to the map".into(),
        });
    }

    /// Start on chart `kind` (`--chart`), made at once.
    pub fn start_chart(&mut self, kind: Kind) {
        self.chart = Some(Chart::new(kind));
        self.refresh_chart();
    }

    /// Whether the run records a latent (topics, or its own cell factors).
    #[must_use]
    pub fn run_has_latent(&self) -> bool {
        self.data
            .run
            .as_ref()
            .is_some_and(|(m, _)| m.outputs.latent.is_some())
    }

    /// Whether the run has topic mixtures: a topic run, or a `bge` run
    /// with its topics resolved.
    fn is_topic_run(&mut self) -> bool {
        self.activity()
            .is_some_and(super::activity::Activity::has_mixtures)
    }

    /// `+` / `-` on a heatmap: more or fewer features per group.
    pub fn change_top(&mut self, more: bool) {
        let Some(c) = self.chart.as_mut().filter(|c| c.kind == Kind::Heatmap) else {
            return;
        };
        c.top = if more {
            (c.top + c.top.div_ceil(4)).min(TOP_RANGE.1)
        } else {
            c.top.saturating_sub(c.top.div_ceil(5)).max(TOP_RANGE.0)
        };
        self.note = Some(format!("top {} features per group", c.top));
    }

    /// Make the chart again if the space, grouping or feature count changed.
    pub fn refresh_chart(&mut self) {
        let Some(c) = self.chart.as_ref() else { return };
        let key = (self.space, self.colour, c.top);
        if c.made.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let (kind, top) = (c.kind, c.top);
        let data = match kind {
            _ if self.current().axis() != Axis::Cells => {
                Err("charts are drawn from a cell view (tab back to the cells)".into())
            }
            Kind::Structure => self.structure().map(Data::Structure),
            Kind::Heatmap => self.heatmap(top).map(Data::Heatmap),
        };
        // Not read yet: drawn again once it is, not kept as a failure.
        if crate::view::features::is_loading(&data) {
            return;
        }
        if let Some(c) = self.chart.as_mut() {
            c.made = Some((key, data));
        }
    }

    fn grouping_name(&self) -> Option<String> {
        self.colour
            .map(|i| self.data.labels[i].kind.title().to_string())
    }

    fn structure(&mut self) -> Result<Structure, String> {
        let grouping = self
            .grouping_name()
            .ok_or("colour the cells by a grouping first (c)")?;
        let groups = self.groups().ok_or("no grouping on screen")?.to_vec();
        let levels = self.levels().to_vec();
        let space = self.space;
        let (activity, data) = self
            .activity_and_data()
            .ok_or("no manifest to read the model from")?;
        let names = &data.spaces[space].points.names;
        let (theta, k) = activity.mixtures(space, names)?;
        Ok(build_structure(grouping, &theta, k, &groups, &levels))
    }

    fn heatmap(&mut self, top: usize) -> Result<Heatmap, String> {
        let grouping = self
            .grouping_name()
            .ok_or("colour the cells by a grouping first (c)")?;
        let groups = self.groups().ok_or("no grouping on screen")?.to_vec();
        let levels = self.levels().to_vec();
        let n = levels.len();
        let space = self.space;
        let (activity, data) = self
            .activity_and_data()
            .ok_or("no manifest to read the model from")?;
        let names = &data.spaces[space].points.names;
        let sums = activity.group_sums(space, names, &groups, n)?;
        // The model proposes candidates cheaply, for every feature; the
        // values shown (the counts, when there are any) choose among them,
        // so a row peaks in its group in what is drawn, and a feature the
        // data files lack is never picked.
        let (levels_fg, all) = activity.group_levels(&sums)?;
        let proposed = pick_top(&own_peaks(margins(&levels_fg, n)), top * CANDIDATES);
        let pool: Vec<Box<str>> = proposed.iter().map(|&(f, _)| all[f].clone()).collect();
        let (pool_means, source) = activity.group_means(space, names, &groups, n, &pool)?;
        let picked = pick_top(&own_peaks(margins(&pool_means, n)), top);
        let features: Vec<Box<str>> = picked.iter().map(|&(i, _)| pool[i].clone()).collect();
        let means: Vec<f32> = picked
            .iter()
            .flat_map(|&(i, _)| pool_means[i * n..(i + 1) * n].iter().copied())
            .collect();
        // Only groups with cells in view get a column.
        let mut size = vec![0usize; n];
        for &g in &groups {
            if let Some(s) = size.get_mut(g as usize) {
                *s += 1;
            }
        }
        let cols: Vec<usize> = (0..n).filter(|&g| size[g] > 0).collect();
        let mut z: Vec<f32> = features
            .iter()
            .enumerate()
            .flat_map(|(i, _)| cols.iter().map(move |&g| (i, g)))
            .map(|(i, g)| means[i * n + g])
            .collect();
        zscore_rows(&mut z, cols.len(), CLIP);
        Ok(Heatmap {
            grouping,
            features,
            groups: cols.iter().map(|&g| levels[g].to_string()).collect(),
            z,
            source,
            top,
        })
    }

    /// The chart on screen, drawn `w × h` with text sized for `cell_px`;
    /// `None` on the map. With `keep_text`, text is kept for a PDF.
    #[must_use]
    pub fn chart_frame(&self, w: usize, h: usize, cell_px: f32, keep_text: bool) -> Option<Frame> {
        let c = self.chart.as_ref()?;
        let mut frame = Frame::blank(w, h);
        if keep_text {
            frame.keep_text();
        }
        let px = self.text_px(cell_px);
        let font = Font::for_cell_height(px, false);
        let bold = Font::for_cell_height(px, true);
        match c.made.as_ref().map(|(_, d)| d) {
            Some(Ok(Data::Structure(s))) => draw_structure(&mut frame, s, font, bold),
            Some(Ok(Data::Heatmap(hm))) => draw_heatmap(&mut frame, hm, font, bold),
            Some(Err(e)) => note(&mut frame, &format!("no {}: {e}", c.kind.name()), bold),
            None => note(&mut frame, &format!("drawing the {}…", c.kind.name()), bold),
        }
        Some(frame)
    }
}

/// Panels by group in the grouping's order (cells without a group last, as
/// "unassigned"), cells in each by dominant topic then its weight; topics
/// by prevalence. Cells with no mixture (`NaN`) are left out.
fn build_structure(
    grouping: String,
    theta: &[f32],
    k: usize,
    groups: &[u32],
    levels: &[Box<str>],
) -> Structure {
    let valid = |p: usize| theta[p * k].is_finite();
    let mut total = vec![0f64; k];
    for p in (0..groups.len()).filter(|&p| valid(p)) {
        for (t, v) in total.iter_mut().zip(&theta[p * k..(p + 1) * k]) {
            *t += f64::from(*v);
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| total[b].total_cmp(&total[a]).then(a.cmp(&b)));
    let mut rank = vec![0usize; k];
    for (r, &t) in order.iter().enumerate() {
        rank[t] = r;
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); levels.len() + 1];
    for (p, &g) in groups.iter().enumerate() {
        if valid(p) {
            let at = if g == NONE {
                levels.len()
            } else {
                (g as usize).min(levels.len())
            };
            members[at].push(p);
        }
    }
    let panels = members
        .into_iter()
        .enumerate()
        .filter(|(_, cells)| !cells.is_empty())
        .map(|(g, mut cells)| {
            let top = |p: usize| {
                let row = &theta[p * k..(p + 1) * k];
                let (t, v) = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or((0, 0.0), |(t, &v)| (t, v));
                (rank[t], v)
            };
            cells.sort_by(|&a, &b| {
                let ((ra, va), (rb, vb)) = (top(a), top(b));
                ra.cmp(&rb).then(vb.total_cmp(&va))
            });
            let name = levels
                .get(g)
                .map_or_else(|| "unassigned".to_string(), ToString::to_string);
            let mix: Vec<f32> = cells
                .iter()
                .flat_map(|&p| order.iter().map(move |&t| theta[p * k + t]))
                .collect();
            (name, mix)
        })
        .collect();
    Structure {
        grouping,
        panels,
        k,
        order,
    }
}

/// Per group, each feature's margin over the next-highest group (levels
/// feature-major, `n` groups a feature): a feature scores where it peaks,
/// so a group's top features are its own. `NaN` where there is no level.
fn margins(levels: &[f32], n: usize) -> Vec<Vec<f32>> {
    let d = levels.len().checked_div(n).unwrap_or(0);
    let mut out = vec![vec![f32::NAN; d]; n];
    for f in 0..d {
        let row = &levels[f * n..(f + 1) * n];
        let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in row.iter().filter(|v| v.is_finite()) {
            if v > best {
                second = best;
                best = v;
            } else if v > second {
                second = v;
            }
        }
        for (g, &v) in row.iter().enumerate() {
            if v.is_finite() && second.is_finite() {
                let other = if v >= best { second } else { best };
                out[g][f] = v - other;
            }
        }
    }
    out
}

/// Margins with those at or below zero dropped: a feature counts only for
/// the group it peaks in.
fn own_peaks(mut m: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    for v in m.iter_mut().flatten() {
        if v.is_nan() || *v <= 0.0 {
            *v = f32::NAN;
        }
    }
    m
}

/// Each group's top `top` features by score, groups in order, a feature
/// taken by an earlier group not repeated: (feature, group) pairs.
fn pick_top(per_group: &[Vec<f32>], top: usize) -> Vec<(usize, usize)> {
    let mut taken = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (g, scores) in per_group.iter().enumerate() {
        let mut ranked: Vec<usize> = (0..scores.len())
            .filter(|&f| scores[f].is_finite())
            .collect();
        ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let mine: Vec<usize> = ranked
            .into_iter()
            .filter(|f| !taken.contains(f))
            .take(top)
            .collect();
        for f in mine {
            taken.insert(f);
            out.push((f, g));
        }
    }
    out
}

/// z-score each row of `ncols` values across its finite entries, clipped
/// to `±clip`; a row with no spread is all zeros.
fn zscore_rows(v: &mut [f32], ncols: usize, clip: f32) {
    if ncols == 0 {
        return;
    }
    for row in v.chunks_mut(ncols) {
        let finite: Vec<f32> = row.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.is_empty() {
            continue;
        }
        let n = finite.len() as f32;
        let mean = finite.iter().sum::<f32>() / n;
        let sd = (finite.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        for x in row.iter_mut().filter(|x| x.is_finite()) {
            *x = if sd > 1e-9 {
                ((*x - mean) / sd).clamp(-clip, clip)
            } else {
                0.0
            };
        }
    }
}

/// A diverging colour for `t` in `[-1, 1]`: blue below zero, red above,
/// near-white at zero.
fn diverging(t: f32) -> Rgb {
    let t = t.clamp(-1.0, 1.0);
    let a = t.abs();
    let hue = if t < 0.0 { 255.0 } else { 27.0 };
    color::oklch(0.97 - 0.45 * a, 0.16 * a, hue)
}

fn ink() -> Rgb {
    color::linear_rgb(color::TEXT)
}

/// One line of text where the chart would be.
fn note(frame: &mut Frame, s: &str, font: Font) {
    let bg = frame.background();
    let lh = font.line_height() as i32;
    text::draw(frame, font, s, lh, lh, ink(), bg);
}

fn draw_structure(frame: &mut Frame, s: &Structure, font: Font, bold: Font) {
    let bg = frame.background();
    let lh = font.line_height();
    let adv = font.advance();
    let title = format!(
        "topic mixture per cell · one panel per {} · {} topics",
        s.grouping, s.k
    );
    text::draw(frame, bold, &title, lh as i32 / 2, lh as i32 / 2, ink(), bg);
    // Right: one line per topic; bottom: panel names, turned when long.
    let names: Vec<String> = s.order.iter().map(|t| format!("T{t}")).collect();
    let key_w = (names.iter().map(String::len).max().unwrap_or(2) + 3) * adv;
    let widest = s
        .panels
        .iter()
        .map(|(n, _)| n.chars().count())
        .max()
        .unwrap_or(0);
    let (x0, y0) = (lh / 2, 2 * lh);
    let x1 = frame.w.saturating_sub(key_w + lh);
    let n_cells: usize = s.panels.iter().map(|(_, m)| m.len() / s.k.max(1)).sum();
    let gap = 2;
    let room = x1.saturating_sub(x0 + gap * s.panels.len().saturating_sub(1));
    let widths: Vec<usize> = s
        .panels
        .iter()
        .map(|(_, m)| ((room * (m.len() / s.k.max(1))) / n_cells.max(1)).max(1))
        .collect();
    let turned = s
        .panels
        .iter()
        .zip(&widths)
        .any(|((n, _), &w)| font.width(n) + adv > w);
    let label_h = if turned {
        (widest.min(18) * adv + lh / 2).min(frame.h / 3)
    } else {
        lh + lh / 2
    };
    let y1 = frame.h.saturating_sub(label_h);
    if y1 <= y0 + 4 || x1 <= x0 + 4 {
        return note(frame, "too small to draw", font);
    }
    let colours: Vec<Rgb> = s.order.iter().map(|&t| color::category(t, s.k)).collect();
    let height = (y1 - y0) as f32;
    let mut x = x0;
    for ((name, mix), &w) in s.panels.iter().zip(&widths) {
        let n = mix.len() / s.k.max(1);
        for col in 0..w {
            // The cells this pixel column stands for, averaged.
            let (a, b) = (col * n / w, ((col + 1) * n / w).max(col * n / w + 1).min(n));
            let mut avg = vec![0f32; s.k];
            for c in a..b {
                for (m, v) in avg.iter_mut().zip(&mix[c * s.k..(c + 1) * s.k]) {
                    *m += v / (b - a) as f32;
                }
            }
            let mut top = y0 as f32;
            for (t, v) in avg.iter().enumerate() {
                let bottom = top + v * height;
                let (ya, yb) = (top.round() as usize, bottom.round() as usize);
                frame.fill([x + col, ya, x + col + 1, yb.min(y1)], colours[t]);
                top = bottom;
            }
        }
        // The panel's name under it.
        let shown: String = name.chars().take(18).collect();
        if turned {
            let cx = (x + w / 2).saturating_sub(lh / 2) as i32;
            if w + gap >= lh / 2 {
                text::draw_vertical(
                    frame,
                    font,
                    &shown,
                    (cx, (y1 + label_h - lh / 4) as i32),
                    ink(),
                    bg,
                );
            }
        } else {
            let tx = (x + w / 2).saturating_sub(font.width(&shown) / 2) as i32;
            text::draw(frame, font, &shown, tx, (y1 + lh / 4) as i32, ink(), bg);
        }
        x += w + gap;
    }
    // The key: one swatch and name per topic, in the bars' order.
    let kx = frame.w.saturating_sub(key_w);
    for (i, (name, c)) in names.iter().zip(&colours).enumerate() {
        let y = y0 + i * lh;
        if y + lh > frame.h {
            break;
        }
        frame.fill([kx, y + lh / 4, kx + lh / 2, y + 3 * lh / 4], *c);
        text::draw(frame, font, name, (kx + lh) as i32, y as i32, ink(), bg);
    }
}

fn draw_heatmap(frame: &mut Frame, hm: &Heatmap, font: Font, bold: Font) {
    let bg = frame.background();
    let lh = font.line_height();
    let adv = font.advance();
    let what = match hm.source {
        Source::Observed => "mean ln(1 + count)",
        Source::Expected => "model's expected level (no count files)",
    };
    let title = format!(
        "top {} features per {} · {what}, z-scored per feature, clipped ±{CLIP}",
        hm.top, hm.grouping
    );
    text::draw(frame, bold, &title, lh as i32 / 2, lh as i32 / 2, ink(), bg);
    let (rows, cols) = (hm.features.len(), hm.groups.len());
    if rows == 0 || cols == 0 {
        return note(frame, "no features stand out in these groups", font);
    }
    let name_w = hm
        .features
        .iter()
        .map(|f| f.chars().count())
        .max()
        .unwrap_or(0)
        .min(16)
        * adv;
    let col_label = hm
        .groups
        .iter()
        .map(|g| g.chars().count())
        .max()
        .unwrap_or(0)
        .min(18)
        * adv;
    let key_w = 6 * adv + lh;
    let (x0, y0) = (lh / 2 + name_w + adv, 2 * lh);
    let x1 = frame.w.saturating_sub(key_w + lh);
    let y1 = frame
        .h
        .saturating_sub((col_label + lh / 2).min(frame.h / 3));
    if x1 <= x0 + cols || y1 <= y0 + 4 {
        return note(frame, "too small to draw", font);
    }
    let (cw, rh) = (
        (x1 - x0) as f32 / cols as f32,
        (y1 - y0) as f32 / rows as f32,
    );
    let muted = color::linear_rgb(color::MUTED);
    for r in 0..rows {
        let (ya, yb) = (
            (y0 as f32 + r as f32 * rh) as usize,
            (y0 as f32 + (r + 1) as f32 * rh) as usize,
        );
        for c in 0..cols {
            let xa = (x0 as f32 + c as f32 * cw) as usize;
            let xb = (x0 as f32 + (c + 1) as f32 * cw) as usize;
            let v = hm.z[r * cols + c];
            let colour = if v.is_finite() {
                diverging(v / CLIP)
            } else {
                muted
            };
            frame.fill([xa, ya, xb, yb.max(ya + 1)], colour);
        }
    }
    // Feature names down the left, as many as fit a line each.
    let mut next = 0.0;
    for (r, f) in hm.features.iter().enumerate() {
        let y = y0 as f32 + r as f32 * rh;
        if y + 0.5 * rh < next {
            continue;
        }
        let shown: String = f.chars().take(16).collect();
        let ty = (y + 0.5 * rh - lh as f32 / 2.0) as i32;
        text::draw(
            frame,
            font,
            &shown,
            (x0 - adv - font.width(&shown)) as i32,
            ty,
            ink(),
            bg,
        );
        next = y + 0.5 * rh + lh as f32;
    }
    // Group names under the columns, turned, as many as fit.
    let mut next = 0.0;
    for (c, g) in hm.groups.iter().enumerate() {
        let x = x0 as f32 + (c as f32 + 0.5) * cw;
        if x < next {
            continue;
        }
        let shown: String = g.chars().take(18).collect();
        let lx = (x - lh as f32 / 2.0) as i32;
        let bottom = (y1 + lh / 4 + font.width(&shown)) as i32;
        text::draw_vertical(frame, font, &shown, (lx, bottom), ink(), bg);
        next = x + lh as f32;
    }
    // The key: the ramp top (+CLIP) to bottom (−CLIP).
    let kx = frame.w.saturating_sub(key_w);
    let (ka, kb) = (y0 + lh, y1.saturating_sub(lh).max(y0 + lh + 2));
    for y in ka..kb {
        let t = 1.0 - 2.0 * (y - ka) as f32 / (kb - ka) as f32;
        frame.fill([kx, y, kx + lh / 2, y + 1], diverging(t));
    }
    let label_x = (kx + lh / 2 + adv / 2) as i32;
    text::draw(
        frame,
        font,
        &format!("+{CLIP}"),
        label_x,
        ka as i32 - lh as i32 / 2,
        ink(),
        bg,
    );
    text::draw(
        frame,
        font,
        "0",
        label_x,
        ((ka + kb) / 2) as i32 - lh as i32 / 2,
        ink(),
        bg,
    );
    text::draw(
        frame,
        font,
        &format!("-{CLIP}"),
        label_x,
        kb as i32 - lh as i32 / 2,
        ink(),
        bg,
    );
    text::draw(
        frame,
        font,
        "z",
        label_x,
        y0 as i32 - lh as i32 / 2,
        ink(),
        bg,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_panels_follow_groups_and_cells_their_dominant_topic() {
        // Three topics; topic 2 most prevalent, then 0, then 1.
        let theta = [
            0.1,
            0.1,
            0.8, // p0 group 0, dominant 2
            0.6,
            0.1,
            0.3, // p1 group 0, dominant 0
            0.2,
            0.0,
            0.8, // p2 group 1
            0.0,
            0.0,
            0.9, // p3 no group
            f32::NAN,
            f32::NAN,
            f32::NAN, // p4 no mixture
            0.3,
            0.1,
            0.6, // p5 group 0, dominant 2, weaker than p0
        ];
        let groups = [0, 0, 1, NONE, 0, 0];
        let levels: Vec<Box<str>> = vec!["A".into(), "B".into()];
        let s = build_structure("cluster".into(), &theta, 3, &groups, &levels);
        assert_eq!(s.order, [2, 0, 1]);
        let names: Vec<&str> = s.panels.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["A", "B", "unassigned"]);
        // A: p0 (2 at 0.8), p5 (2 at 0.6), then p1 (dominant 0); each in
        // the display order 2, 0, 1.
        assert_eq!(s.panels[0].1, [0.8, 0.1, 0.1, 0.6, 0.3, 0.1, 0.3, 0.6, 0.1]);
    }

    #[test]
    fn each_group_takes_its_own_top_features() {
        let per = vec![vec![3.0, 2.0, 1.0, f32::NAN], vec![3.5, 0.5, 2.5, 4.0]];
        // Group 1's best (3) and then 0 — but 0 went to group 0, so 2.
        assert_eq!(pick_top(&per, 2), [(0, 0), (1, 0), (3, 1), (2, 1)]);
    }

    #[test]
    fn a_feature_scores_by_its_margin_over_the_next_group() {
        // Two features × three groups.
        let levels = [1.0, 4.0, 2.0, 5.0, 5.0, f32::NAN];
        let m = margins(&levels, 3);
        assert_eq!(m[1][0], 2.0); // peaks in group 1, 2 over group 2
        assert_eq!(m[0][0], -3.0);
        assert_eq!(m[0][1], 0.0); // a tie peaks nowhere
        assert!(m[2][1].is_nan());
    }

    #[test]
    fn rows_are_z_scored_and_clipped() {
        let mut v = [1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 0.0, f32::NAN, 10.0];
        zscore_rows(&mut v, 3, 1.0);
        let r = 1.0 / (2.0f32 / 3.0).sqrt();
        assert!((v[0] + r.min(1.0)).abs() < 1e-5 && v[1].abs() < 1e-6);
        assert_eq!(&v[3..6], [0.0, 0.0, 0.0]);
        assert!(v[7].is_nan() && v[6] == -1.0 && v[8] == 1.0);
    }
}
