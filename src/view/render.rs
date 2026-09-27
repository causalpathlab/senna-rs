//! Progressive point rasterizer.
//!
//! Points are splatted as anti-aliased discs into a linear-light accumulator
//! that keeps, per pixel, a weighted colour sum and the total weight. The
//! composite shows the mean colour at an opacity that saturates with weight,
//! `a = 1 − exp(−k·w)`: a lone point reads as a solid dot, a crowd reads as
//! density, and overlapping groups mix instead of the last one drawn winning.
//!
//! Points are visited in a fixed shuffled order, so every prefix is a
//! uniform subsample. A job draws that order in chunks of doubling size and
//! the viewer shows the composite after each chunk: the full picture appears
//! at once, then fills in.

use super::color::{self, Encoder, Rgb};
use super::data::{Points, NONE};
use super::text::{self, Canvas, Font};
use image::RgbaImage;

/// Opacity gain: one fully covered point reaches `1 − e^−k` ≈ 0.9.
const OPACITY_GAIN: f32 = 2.3;
/// Weight of a focused point relative to the rest, so a focused group stays
/// saturated where it overlaps muted points.
const FOCUS_WEIGHT: f32 = 6.0;
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

/// How one layer of points is coloured.
pub struct Paint<'a> {
    pub points: &'a Points,
    /// Group id per point (`NONE` = unlabelled), or `None` for a flat colour.
    pub groups: Option<&'a [u32]>,
    pub palette: &'a [Rgb],
    pub focus: Option<u32>,
    /// Draw everything muted (a backdrop layer).
    pub muted: bool,
    /// Radius multiplier.
    pub size: f32,
    /// Weight of a labelled point relative to an unlabelled one. Above 1 when
    /// few points carry a label, so a sparse grouping (markers among all
    /// features) is not buried under the gray majority.
    pub boost: f32,
}

/// Weight of unlabelled, unfocused and backdrop points.
const RECEDE_WEIGHT: f32 = 0.4;

impl Paint<'_> {
    #[inline]
    fn colour(&self, i: usize, muted: Rgb) -> (Rgb, f32) {
        if self.muted {
            return (muted, RECEDE_WEIGHT);
        }
        let g = self.groups.map_or(NONE, |g| g[i]);
        match self.focus {
            Some(f) if g == f => (self.palette[g as usize], FOCUS_WEIGHT),
            Some(_) => (muted, RECEDE_WEIGHT),
            None if g == NONE => (muted, RECEDE_WEIGHT),
            None => (self.palette[g as usize], self.boost),
        }
    }
}

/// Labelled-point weight for a grouping covering `frac` of the points.
#[must_use]
pub fn sparsity_boost(frac: f32) -> f32 {
    (0.5 / frac.max(1e-6)).clamp(1.0, FOCUS_WEIGHT)
}

pub struct Accum {
    w: usize,
    h: usize,
    /// Per pixel: linear r, g, b sums and the total weight.
    buf: Vec<[f32; 4]>,
}

impl Accum {
    #[must_use]
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            buf: vec![[0.0; 4]; w * h],
        }
    }

    #[inline]
    fn add(&mut self, x: usize, y: usize, c: Rgb, wt: f32) {
        let p = &mut self.buf[y * self.w + x];
        p[0] += c[0] * wt;
        p[1] += c[1] * wt;
        p[2] += c[2] * wt;
        p[3] += wt;
    }

    /// Anti-aliased disc of radius `r` at `(x, y)`.
    #[inline]
    fn disc(&mut self, x: f32, y: f32, r: f32, c: Rgb, wt: f32) {
        let reach = r + 0.5;
        let (x0, x1) = ((x - reach).floor().max(0.0), (x + reach).ceil());
        let (y0, y1) = ((y - reach).floor().max(0.0), (y + reach).ceil());
        let (x1, y1) = (x1.min(self.w as f32 - 1.0), y1.min(self.h as f32 - 1.0));
        if x0 > x1 || y0 > y1 {
            return;
        }
        for py in y0 as usize..=y1 as usize {
            let dy = py as f32 + 0.5 - y;
            for px in x0 as usize..=x1 as usize {
                let dx = px as f32 + 0.5 - x;
                let cov = (reach - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.add(px, py, c, wt * cov);
                }
            }
        }
    }
}

/// A point radius for `n_visible` points spread over the viewport: large
/// discs when sparse, sub-pixel when crowded.
#[must_use]
pub fn auto_radius(n_visible: usize, vp: &Viewport) -> f32 {
    let per_px = n_visible as f32 / (vp.w * vp.h).max(1) as f32;
    (0.55 / per_px.max(1e-9).sqrt()).clamp(0.6, 3.2)
}

/// Weight per point so the *average* covered pixel sits at about a quarter
/// of saturation: sparse views keep solid dots, crowded ones keep a density
/// gradient instead of flooding to a flat colour.
#[must_use]
pub fn point_weight(n_visible: usize, radius: f32, vp: &Viewport) -> f32 {
    let per_px = n_visible as f32 / (vp.w * vp.h).max(1) as f32;
    let coverage = per_px * std::f32::consts::PI * (radius + 0.5).powi(2);
    (0.25 / coverage.max(1e-9)).clamp(0.03, 1.0)
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
    accum: Accum,
    /// Layer being drawn and position within its order.
    layer: usize,
    cursor: usize,
    chunk: usize,
    radius: Vec<f32>,
    weight: Vec<f32>,
}

impl Job {
    #[must_use]
    pub fn new(vp: Viewport, layers: &[Paint<'_>]) -> Self {
        let (radius, weight) = layers
            .iter()
            .map(|l| {
                let n = visible_count(l.points, &vp);
                let r = l.size * auto_radius(n, &vp);
                (r, point_weight(n, r, &vp))
            })
            .unzip();
        Self {
            vp,
            accum: Accum::new(vp.w, vp.h),
            layer: 0,
            cursor: 0,
            chunk: FIRST_CHUNK,
            radius,
            weight,
        }
    }

    #[must_use]
    pub fn done(&self, layers: &[Paint<'_>]) -> bool {
        self.layer >= layers.len()
    }

    /// Draw the next chunk. Returns `true` once every layer is drawn.
    pub fn step(&mut self, layers: &[Paint<'_>]) -> bool {
        let muted = color::linear_rgb(color::MUTED);
        let mut budget = self.chunk;
        while budget > 0 && self.layer < layers.len() {
            let paint = &layers[self.layer];
            let order = &paint.points.order;
            let end = (self.cursor + budget).min(order.len());
            let (r, pw) = (self.radius[self.layer], self.weight[self.layer]);
            let (w, h) = (self.vp.w as f32, self.vp.h as f32);
            for &i in &order[self.cursor..end] {
                let i = i as usize;
                let (x, y) = self.vp.to_px(paint.points.xy[i]);
                if x < -r || y < -r || x > w + r || y > h + r {
                    continue;
                }
                let (c, wt) = paint.colour(i, muted);
                self.accum.disc(x, y, r, c, wt * pw);
            }
            budget -= end - self.cursor;
            self.cursor = end;
            if self.cursor >= order.len() {
                self.layer += 1;
                self.cursor = 0;
            }
        }
        self.chunk *= 2;
        self.done(layers)
    }

    /// The current accumulation on the page background.
    #[must_use]
    pub fn composite(&self) -> Frame {
        let bg = color::linear_rgb(color::BACKGROUND);
        let px = self
            .accum
            .buf
            .iter()
            .map(|&[r, g, b, wt]| {
                if wt <= 0.0 {
                    return bg;
                }
                let a = 1.0 - (-OPACITY_GAIN * wt).exp();
                let inv = 1.0 / wt;
                [
                    bg[0] + (r * inv - bg[0]) * a,
                    bg[1] + (g * inv - bg[1]) * a,
                    bg[2] + (b * inv - bg[2]) * a,
                ]
            })
            .collect();
        Frame {
            w: self.vp.w,
            h: self.vp.h,
            px,
            bg,
        }
    }
}

/// A composited linear-light image, ready for labels and encoding.
pub struct Frame {
    pub w: usize,
    pub h: usize,
    px: Vec<Rgb>,
    bg: Rgb,
}

impl Frame {
    #[must_use]
    pub fn background(&self) -> Rgb {
        self.bg
    }

    #[must_use]
    pub fn to_image(&self, enc: &Encoder) -> RgbaImage {
        let mut img = RgbaImage::new(self.w as u32, self.h as u32);
        for (dst, src) in img.pixels_mut().zip(&self.px) {
            *dst = image::Rgba([
                enc.encode(src[0]),
                enc.encode(src[1]),
                enc.encode(src[2]),
                255,
            ]);
        }
        img
    }
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
}

/// A label candidate in pixel space.
pub struct Label {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub ink: Rgb,
    /// Larger goes first when labels collide.
    pub priority: f32,
    pub font: Font,
}

/// Place labels centred on their anchors, dropping any that would overlap one
/// already placed or a `reserved` rectangle. Higher priority wins.
pub fn draw_labels(frame: &mut Frame, mut labels: Vec<Label>, reserved: &[[f32; 4]]) {
    labels.sort_by(|a, b| b.priority.total_cmp(&a.priority));
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
        text::draw(frame, l.font, &l.text, x0 as i32, y0 as i32, l.ink, bg);
    }
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
        let reach = r + 1.0;
        for py in (cy - reach) as i32..=(cy + reach) as i32 {
            for px in (cx - reach) as i32..=(cx + reach) as i32 {
                if px < 0 || py < 0 || px as usize >= frame.w || py as usize >= frame.h {
                    continue;
                }
                let d = ((px as f32 + 0.5 - cx).powi(2) + (py as f32 + 0.5 - cy).powi(2)).sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0);
                if a > 0.0 {
                    frame.blend(px as usize, py as usize, *dot, a);
                }
            }
        }
        text::draw(frame, font, name, text_x, y, *ink, bg);
        y += lh;
    }
    Some(area)
}

/// Median position of each group, from a prefix of the shuffled order (a
/// uniform subsample, so the estimate is unbiased), with group sizes.
#[must_use]
pub fn group_medians(
    points: &Points,
    groups: &[u32],
    n_groups: usize,
) -> Vec<Option<([f32; 2], usize)>> {
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
