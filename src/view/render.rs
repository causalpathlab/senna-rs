//! Progressive point rasterizer.
//!
//! Points are drawn as opaque, anti-aliased marks (circle, square, diamond,
//! triangle, cross) straight onto a linear-light canvas; only a group given
//! an opacity below one in the style menu is translucent. With opaque marks
//! the draw order says what is seen, so each layer is drawn in rank order:
//! muted and unlabelled points first, then labelled ones, then the focused
//! group last so it pops on top; activity is drawn low to high.
//!
//! Within a rank points keep a fixed shuffled order, so every prefix is a
//! uniform subsample. A job draws in chunks of doubling size and the viewer
//! shows the canvas after each chunk: the picture appears at once, then fills
//! in.

use super::activity::Levels;
use super::color::{self, Rgb};
use super::data::{Points, NONE};
use super::style::{Resolved, Shape};
use super::text::{self, Canvas, Font};
use image::RgbaImage;
use rayon::prelude::*;
use std::sync::Arc;

/// First progressive chunk; each later chunk doubles.
const FIRST_CHUNK: usize = 1 << 15;

/// Maps data coordinates to pixels. `y` grows upward in data, downward on
/// screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub cx: f32,
    pub cy: f32,
    /// Pixels per data unit.
    pub scale: f32,
    pub w: usize,
    pub h: usize,
}

impl Viewport {
    /// Fit `bounds` into `w × h` with a small margin.
    #[must_use]
    pub fn fit(bounds: [f32; 4], w: usize, h: usize) -> Self {
        let [x0, y0, x1, y1] = bounds;
        let (dw, dh) = ((x1 - x0).max(1e-6), (y1 - y0).max(1e-6));
        let scale = 0.92 * (w as f32 / dw).min(h as f32 / dh);
        Self {
            cx: 0.5 * (x0 + x1),
            cy: 0.5 * (y0 + y1),
            scale,
            w,
            h,
        }
    }

    #[inline]
    #[must_use]
    pub fn to_px(self, [x, y]: [f32; 2]) -> (f32, f32) {
        (
            (x - self.cx) * self.scale + 0.5 * self.w as f32,
            (self.cy - y) * self.scale + 0.5 * self.h as f32,
        )
    }

    #[must_use]
    pub fn to_data(self, px: f32, py: f32) -> [f32; 2] {
        [
            (px - 0.5 * self.w as f32) / self.scale + self.cx,
            self.cy - (py - 0.5 * self.h as f32) / self.scale,
        ]
    }

    /// Zoom by `factor` keeping the data point under `(px, py)` fixed.
    pub fn zoom_at(&mut self, factor: f32, px: f32, py: f32) {
        let anchor = self.to_data(px, py);
        self.scale *= factor;
        let moved = self.to_data(px, py);
        self.cx += anchor[0] - moved[0];
        self.cy += anchor[1] - moved[1];
    }

    pub fn pan_px(&mut self, dx: f32, dy: f32) {
        self.cx -= dx / self.scale;
        self.cy += dy / self.scale;
    }
}

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

/// One point's mark.
struct Mark {
    colour: Rgb,
    alpha: f32,
    shape: Shape,
    size: f32,
}

impl Paint<'_> {
    /// The mark for point `i`, or `None` when its group is hidden.
    #[inline]
    fn mark(&self, i: usize, muted: Rgb, ink: Rgb) -> Option<Mark> {
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
            let t: Vec<f32> = (0..levels.values.len()).map(|i| levels.t(i)).collect();
            let mut o = order.clone();
            o.sort_by(|&a, &b| t[a as usize].total_cmp(&t[b as usize]));
            return o;
        }
        let mut buckets: [Vec<u32>; 4] = Default::default();
        for &i in order {
            buckets[self.rank(i as usize) as usize].push(i);
        }
        buckets.concat()
    }
}

/// A point radius for `n_visible` points spread over the viewport: small
/// marks when crowded, a little larger when sparse.
#[must_use]
pub fn auto_radius(n_visible: usize, vp: &Viewport) -> f32 {
    let per_px = n_visible as f32 / (vp.w * vp.h).max(1) as f32;
    (0.35 / per_px.max(1e-9).sqrt()).clamp(0.6, 2.4)
}

/// Estimated count of `points` inside the viewport, from a prefix of the
/// shuffled order.
#[must_use]
pub fn visible_count(points: &Points, vp: &Viewport) -> usize {
    let probe = points.order.len().min(4096);
    if probe == 0 {
        return 0;
    }
    let inside = points.order[..probe]
        .iter()
        .filter(|&&i| {
            let (x, y) = vp.to_px(points.xy[i as usize]);
            x >= 0.0 && y >= 0.0 && x < vp.w as f32 && y < vp.h as f32
        })
        .count();
    inside * points.order.len() / probe
}

/// One progressive rendering of a scene.
pub struct Job {
    pub vp: Viewport,
    canvas: Vec<Rgb>,
    orders: Vec<Arc<Vec<u32>>>,
    radius: Vec<f32>,
    /// Layer being drawn and position within its order.
    layer: usize,
    cursor: usize,
    chunk: usize,
}

impl Job {
    #[must_use]
    pub fn new(vp: Viewport, layers: &[Paint<'_>]) -> Self {
        let radius = layers
            .iter()
            .map(|l| l.size * auto_radius(visible_count(l.points, &vp), &vp))
            .collect();
        Self {
            vp,
            canvas: vec![color::linear_rgb(color::BACKGROUND); vp.w * vp.h],
            orders: layers
                .iter()
                .map(|l| l.order.clone().unwrap_or_else(|| Arc::new(l.draw_order())))
                .collect(),
            radius,
            layer: 0,
            cursor: 0,
            chunk: FIRST_CHUNK,
        }
    }

    fn done(&self) -> bool {
        self.layer >= self.orders.len()
    }

    #[inline]
    fn stamp(&mut self, x: f32, y: f32, r: f32, m: &Mark) {
        let (w, h) = (self.vp.w, self.vp.h);
        let reach = m.shape.reach(r);
        let (x0, x1) = ((x - reach).floor().max(0.0), (x + reach).ceil());
        let (y0, y1) = ((y - reach).floor().max(0.0), (y + reach).ceil());
        let (x1, y1) = (x1.min(w as f32 - 1.0), y1.min(h as f32 - 1.0));
        if x0 > x1 || y0 > y1 {
            return;
        }
        for py in y0 as usize..=y1 as usize {
            let dy = py as f32 + 0.5 - y;
            for px in x0 as usize..=x1 as usize {
                let a = m.alpha * m.shape.coverage(px as f32 + 0.5 - x, dy, r);
                if a > 0.0 {
                    let p = &mut self.canvas[py * w + px];
                    for (v, c) in p.iter_mut().zip(m.colour) {
                        *v += (c - *v) * a;
                    }
                }
            }
        }
    }

    /// Draw the next chunk. Returns `true` once every layer is drawn.
    pub fn step(&mut self, layers: &[Paint<'_>]) -> bool {
        let muted = color::linear_rgb(color::MUTED);
        let ink = color::linear_rgb(color::TEXT);
        let mut budget = self.chunk;
        while budget > 0 && self.layer < layers.len() {
            let paint = &layers[self.layer];
            let order = self.orders[self.layer].clone();
            let end = (self.cursor + budget).min(order.len());
            let base = self.radius[self.layer];
            let (w, h) = (self.vp.w as f32, self.vp.h as f32);
            for &i in &order[self.cursor..end] {
                let i = i as usize;
                let (x, y) = self.vp.to_px(paint.points.xy[i]);
                let pad = 4.0 * base + 2.0;
                if x < -pad || y < -pad || x > w + pad || y > h + pad {
                    continue;
                }
                if let Some(m) = paint.mark(i, muted, ink) {
                    self.stamp(x, y, base * m.size, &m);
                }
            }
            budget -= end - self.cursor;
            self.cursor = end;
            let finished = self.cursor >= order.len();
            if finished {
                self.layer += 1;
                self.cursor = 0;
            }
        }
        self.chunk *= 2;
        self.done()
    }

    /// The canvas so far, encoded for display (intermediate frames).
    #[must_use]
    pub fn image(&self) -> RgbaImage {
        encode(self.vp.w, self.vp.h, &self.canvas)
    }

    /// The finished canvas, handed over for labels and encoding.
    #[must_use]
    pub fn finish(self) -> Frame {
        Frame {
            w: self.vp.w,
            h: self.vp.h,
            px: self.canvas,
            bg: color::linear_rgb(color::BACKGROUND),
            text: None,
        }
    }
}

/// A composited linear-light image, ready for labels and encoding.
#[derive(Clone)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    px: Vec<Rgb>,
    bg: Rgb,
    /// Text kept as text, when recording; see `keep_text`.
    text: Option<Vec<text::TextRun>>,
}

impl Frame {
    /// A `w × h` page in the background colour, for a chart drawn from
    /// scratch rather than from points.
    #[must_use]
    pub fn blank(w: usize, h: usize) -> Self {
        let bg = color::linear_rgb(color::BACKGROUND);
        Self {
            w,
            h,
            px: vec![bg; w * h],
            bg,
            text: None,
        }
    }

    /// Fill the pixels `x0..x1 × y0..y1` (clipped to the frame) with `c`.
    pub fn fill(&mut self, [x0, y0, x1, y1]: [usize; 4], c: Rgb) {
        let (x1, y1) = (x1.min(self.w), y1.min(self.h));
        for y in y0.min(y1)..y1 {
            self.px[y * self.w + x0.min(x1)..y * self.w + x1].fill(c);
        }
    }

    #[must_use]
    pub fn background(&self) -> Rgb {
        self.bg
    }

    /// From now on keep text as runs instead of drawing it, for a vector
    /// export that sets it in a real font over the raster.
    pub fn keep_text(&mut self) {
        self.text.get_or_insert_with(Vec::new);
    }

    /// The text kept so far.
    pub fn take_text(&mut self) -> Vec<text::TextRun> {
        self.text.as_mut().map(std::mem::take).unwrap_or_default()
    }

    #[must_use]
    pub fn to_image(&self) -> RgbaImage {
        encode(self.w, self.h, &self.px)
    }
}

/// Linear-light pixels → 8-bit sRGBA, in parallel.
fn encode(w: usize, h: usize, px: &[Rgb]) -> RgbaImage {
    let enc = color::encoder();
    let buf: Vec<u8> = px
        .par_iter()
        .flat_map_iter(|p| [enc.encode(p[0]), enc.encode(p[1]), enc.encode(p[2]), 255])
        .collect();
    RgbaImage::from_raw(w as u32, h as u32, buf).expect("w × h × 4 bytes")
}

impl Canvas for Frame {
    fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
    #[inline]
    fn blend(&mut self, x: usize, y: usize, c: Rgb, a: f32) {
        let p = &mut self.px[y * self.w + x];
        for k in 0..3 {
            p[k] += (c[k] - p[k]) * a;
        }
    }
    fn text_sink(&mut self) -> Option<&mut Vec<text::TextRun>> {
        self.text.as_mut()
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

/// A label candidate in pixel space.
pub struct Label {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub ink: Rgb,
    pub tier: Tier,
    /// Within a tier, larger goes first when labels collide.
    pub priority: f32,
    pub font: Font,
    /// The group a label names on a cell map, so a click can find it.
    pub group: Option<u32>,
}

/// Place labels centred on their anchors, dropping any that would overlap one
/// already placed or a `reserved` rectangle, or run off the frame. Higher
/// priority wins. Returns where each group label was drawn.
pub fn draw_labels(
    frame: &mut Frame,
    mut labels: Vec<Label>,
    reserved: &[[f32; 4]],
) -> Vec<(u32, [f32; 4])> {
    let mut drawn = Vec::new();
    labels.sort_by(|a, b| b.tier.cmp(&a.tier).then(b.priority.total_cmp(&a.priority)));
    let bg = frame.background();
    let mut placed: Vec<[f32; 4]> = reserved.to_vec();
    for l in labels {
        let (tw, th) = (l.font.width(&l.text) as f32, l.font.line_height() as f32);
        let (x0, y0) = (l.x - 0.5 * tw, l.y - 0.5 * th);
        let rect = [x0 - 3.0, y0 - 1.0, x0 + tw + 3.0, y0 + th + 1.0];
        if rect[0] < 0.0 || rect[1] < 0.0 || rect[2] > frame.w as f32 || rect[3] > frame.h as f32 {
            continue;
        }
        let hit = placed
            .iter()
            .any(|p| rect[0] < p[2] && p[0] < rect[2] && rect[1] < p[3] && p[1] < rect[3]);
        if hit {
            continue;
        }
        placed.push(rect);
        if let Some(g) = l.group {
            drawn.push((g, rect));
        }
        text::draw_aligned(
            frame,
            l.font,
            &l.text,
            (x0 as i32, y0 as i32),
            true,
            l.ink,
            bg,
        );
    }
    drawn
}

/// A key in the top-left corner: a dot in the point colour and the name in
/// the label ink, one per line, no frame. Returns the area it covers, so map
/// labels can keep clear of it.
pub fn draw_legend(
    frame: &mut Frame,
    entries: &[(String, Rgb, Rgb)],
    font: Font,
) -> Option<[f32; 4]> {
    if entries.is_empty() {
        return None;
    }
    let bg = frame.background();
    let lh = font.line_height() as i32;
    let r = lh as f32 * 0.22;
    let (x0, mut y) = (lh, lh / 2);
    let text_x = x0 + lh / 2 + 2;
    let widest = entries
        .iter()
        .map(|(n, ..)| font.width(n))
        .max()
        .unwrap_or(0) as i32;
    let area = [
        0.0,
        0.0,
        (text_x + widest + lh / 2) as f32,
        (y + lh * entries.len() as i32 + lh / 2) as f32,
    ];
    for (name, dot, ink) in entries {
        if y + lh > frame.h as i32 {
            break;
        }
        let (cx, cy) = (x0 as f32, y as f32 + 0.5 * lh as f32);
        blend_round(frame, cx, cy, r + 1.0, *dot, |d| r + 0.5 - d);
        text::draw(frame, font, name, text_x, y, *ink, bg);
        y += lh;
    }
    Some(area)
}

/// A key for an activity view in the top-left corner: the title, then the
/// ramp as a thin bar. Returns the area it covers.
pub fn draw_ramp_key(frame: &mut Frame, title: &str, ramp: &[Rgb], font: Font) -> [f32; 4] {
    let bg = frame.background();
    let lh = font.line_height() as i32;
    let (x0, y0) = (lh / 2, lh / 2);
    text::draw(
        frame,
        font,
        title,
        x0,
        y0,
        color::linear_rgb(color::INK),
        bg,
    );
    let (bw, bh) = ((8 * lh).max(font.width(title) as i32 / 2), (lh / 4).max(3));
    let by = y0 + lh + 2;
    for dx in 0..bw {
        let k = (dx as usize * (ramp.len() - 1)) / (bw as usize - 1).max(1);
        for dy in 0..bh {
            let (x, y) = ((x0 + dx) as usize, (by + dy) as usize);
            if x < frame.w && y < frame.h {
                frame.blend(x, y, ramp[k], 1.0);
            }
        }
    }
    [
        0.0,
        0.0,
        (x0 + bw.max(font.width(title) as i32) + lh / 2) as f32,
        (by + bh + lh / 2) as f32,
    ]
}

/// A ring around `(x, y)`, to mark one point.
pub fn draw_ring(frame: &mut Frame, x: f32, y: f32, r: f32, ink: Rgb) {
    blend_round(frame, x, y, r + 2.0, ink, |d| 1.2 - (d - r).abs());
}

/// A thin line from `a` to `b` at opacity `alpha`, stopping `trim_a` and
/// `trim_b` short of its ends so it meets the rings drawn there.
pub fn draw_edge(
    frame: &mut Frame,
    a: (f32, f32),
    b: (f32, f32),
    trim_a: f32,
    trim_b: f32,
    ink: Rgb,
    alpha: f32,
) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = dx.hypot(dy);
    if len <= trim_a + trim_b + 1.0 {
        return;
    }
    let (ux, uy) = (dx / len, dy / len);
    // Pixel centres at integer coordinates, the major axis first.
    let steep = dy.abs() > dx.abs();
    let ends = [
        (a.0 + ux * trim_a - 0.5, a.1 + uy * trim_a - 0.5),
        (b.0 - ux * trim_b - 0.5, b.1 - uy * trim_b - 0.5),
    ]
    .map(|(x, y)| if steep { (y, x) } else { (x, y) });
    let [p, q] = if ends[0].0 <= ends[1].0 {
        ends
    } else {
        [ends[1], ends[0]]
    };
    let slope = (q.1 - p.1) / (q.0 - p.0);
    // One pixel column per step; the line's minor coordinate is shared
    // between the two pixels it falls between.
    for m in p.0.ceil() as i64..=q.0.floor() as i64 {
        let minor = p.1 + (m as f32 - p.0) * slope;
        let lo = minor.floor();
        for (o, w) in [(lo as i64, 1.0 - (minor - lo)), (lo as i64 + 1, minor - lo)] {
            let (x, y) = if steep { (o, m) } else { (m, o) };
            if x >= 0 && y >= 0 && (x as usize) < frame.w && (y as usize) < frame.h {
                frame.blend(x as usize, y as usize, ink, alpha * w);
            }
        }
    }
}

/// Blend `c` over the pixels within `reach` of `(x, y)`, each at the opacity
/// `cov(distance)` clamped to `[0, 1]`: a disc, a ring, whatever `cov` draws.
fn blend_round(frame: &mut Frame, x: f32, y: f32, reach: f32, c: Rgb, cov: impl Fn(f32) -> f32) {
    for py in (y - reach).floor() as i32..=(y + reach).ceil() as i32 {
        for px in (x - reach).floor() as i32..=(x + reach).ceil() as i32 {
            if px < 0 || py < 0 || px as usize >= frame.w || py as usize >= frame.h {
                continue;
            }
            let d = ((px as f32 + 0.5 - x).powi(2) + (py as f32 + 0.5 - y).powi(2)).sqrt();
            let a = cov(d).clamp(0.0, 1.0);
            if a > 0.0 {
                frame.blend(px as usize, py as usize, c, a);
            }
        }
    }
}

/// Each group's median position and size, `None` for an empty group.
pub type Medians = Vec<Option<([f32; 2], usize)>>;

/// Median position of each group, from a prefix of the shuffled order (a
/// uniform subsample, so the estimate is unbiased), with group sizes.
#[must_use]
pub fn group_medians(points: &Points, groups: &[u32], n_groups: usize) -> Medians {
    const SAMPLE: usize = 50_000;
    let mut xs: Vec<Vec<f32>> = vec![Vec::new(); n_groups];
    let mut ys: Vec<Vec<f32>> = vec![Vec::new(); n_groups];
    let take = points.order.len().min(SAMPLE);
    for &i in &points.order[..take] {
        let g = groups[i as usize];
        if g == NONE {
            continue;
        }
        let [x, y] = points.xy[i as usize];
        xs[g as usize].push(x);
        ys[g as usize].push(y);
    }
    let scale = points.order.len() as f32 / take.max(1) as f32;
    let median = |v: &mut Vec<f32>| {
        let m = v.len() / 2;
        *v.select_nth_unstable_by(m, f32::total_cmp).1
    };
    xs.iter_mut()
        .zip(ys.iter_mut())
        .map(|(x, y)| {
            if x.is_empty() {
                None
            } else {
                let n = (x.len() as f32 * scale).round() as usize;
                Some(([median(x), median(y)], n))
            }
        })
        .collect()
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
        assert_eq!(paint.mark(2, muted, ink).unwrap().colour, ink);
        assert_eq!(paint.mark(0, muted, ink).unwrap().colour, [0.1; 3]);
        assert_eq!(paint.mark(1, muted, ink).unwrap().colour, muted);
    }
}
