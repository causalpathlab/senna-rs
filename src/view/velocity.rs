//! A run's velocity field on a map, as scVelo draws it (Bergen et al.,
//! *Nature Biotechnology* (2020)): every cell's velocity, in the space its
//! map was laid out from, turned into transition weights over its nearest
//! cells there (`exp(κ cos(v_i, z_j − z_i))`, normalised), and the map's
//! displacement towards those cells averaged with the weights, less the
//! plain average (so a cell with no direction does not drift). Drawn as
//! arrows over a grid of the visible map.

use super::data::Points;
use super::render::{self, Frame, Viewport};
use super::Rgb;
use legume_numeric::matrix::knn::{ColumnDict, SearchScratch};
use rayon::prelude::*;
use senna::embed_common::*;
use senna::run_manifest::{self, RunManifest};
use std::collections::HashMap;
use std::path::Path;

/// Nearest cells a velocity is read against.
const NEIGHBOURS: usize = 30;
/// Sharpness of the transition weights in the cosine.
const KAPPA: f32 = 10.0;
/// Grid bins across the visible map.
const BINS: f32 = 24.0;
/// Cells a bin needs before it draws an arrow.
const LEAST: usize = 3;

/// An arrow on the map, from and to, in pixels.
pub type Arrow = ((f32, f32), (f32, f32));

/// A grid bin's cells: summed places and displacements, and how many.
#[derive(Default)]
struct Bin {
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    n: usize,
}

/// Every cell's nearest cells and its transition weight to each.
pub struct Field {
    index: HashMap<Box<str>, usize>,
    names: Vec<Box<str>>,
    near: Vec<Vec<(u32, f32)>>,
}

impl Field {
    /// The field of a run with a `cell_velocity` beside its `cell_embedding`
    /// (row for row in the same coordinates); `None` for a run without.
    pub fn load(m: &RunManifest, dir: &Path) -> anyhow::Result<Option<Self>> {
        let (Some(z), Some(v)) = (&m.outputs.cell_embedding, &m.outputs.cell_velocity) else {
            return Ok(None);
        };
        let read = |rel: &str| {
            let path = run_manifest::resolve(dir, rel);
            Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
        };
        let (z, v) = (read(z)?, read(v)?);
        Self::new(&z.rows, &z.mat, &v.rows, &v.mat).map(Some)
    }

    /// The field of cells `z_names` at `z` (`[N × K]`) with velocities `v`
    /// (`[M × K]`, rows `v_names`): the cells in both.
    pub fn new(
        z_names: &[Box<str>],
        z: &Mat,
        v_names: &[Box<str>],
        v: &Mat,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            z.ncols() == v.ncols(),
            "cell_velocity has {} columns, cell_embedding {}",
            v.ncols(),
            z.ncols()
        );
        let row_of: HashMap<&str, usize> = v_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i))
            .collect();
        // Cells in both tables, in the embedding's order.
        let keep: Vec<(usize, usize)> = (0..z_names.len())
            .filter_map(|i| row_of.get(z_names[i].as_ref()).map(|&j| (i, j)))
            .collect();
        let n = keep.len();
        let k = z.ncols();
        let cols = Mat::from_fn(k, n, |d, c| z[(keep[c].0, d)]);
        let names: Vec<Box<str>> = keep.iter().map(|&(i, _)| z_names[i].clone()).collect();
        let dict = ColumnDict::from_dvector_views(cols.column_iter().collect(), (0..n).collect());
        let near: Vec<Vec<(u32, f32)>> = (0..n)
            .into_par_iter()
            .map_init(SearchScratch::default, |scratch, c| {
                let here: Vec<f32> = cols.column(c).iter().copied().collect();
                let velocity: Vec<f32> = (0..k).map(|d| v[(keep[c].1, d)]).collect();
                let speed = velocity.iter().map(|x| x * x).sum::<f32>().sqrt();
                let want = NEIGHBOURS.min(n.saturating_sub(1)) + 1;
                let (found, _) = dict.search_by_query_data_reuse(&here, want, scratch)?;
                let mut out: Vec<(u32, f32)> = found
                    .into_iter()
                    .filter(|&j| j != c)
                    .map(|j| {
                        let mut dot = 0f32;
                        let mut len = 0f32;
                        for d in 0..k {
                            let step = cols[(d, j)] - here[d];
                            dot += step * velocity[d];
                            len += step * step;
                        }
                        let cos = dot / (len.sqrt() * speed).max(1e-12);
                        (j as u32, KAPPA * cos)
                    })
                    .collect();
                let top = out.iter().map(|w| w.1).fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = out.iter().map(|w| (w.1 - top).exp()).sum();
                for w in &mut out {
                    w.1 = (w.1 - top).exp() / total.max(1e-12);
                }
                Ok(out)
            })
            .collect::<anyhow::Result<_>>()?;
        let index = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        Ok(Self { index, names, near })
    }

    /// Every point's displacement on the map `points` (NaN for a point the
    /// field does not hold): its neighbours' unit directions on the map,
    /// averaged with the transition weights, less their plain average.
    #[must_use]
    pub fn embed(&self, points: &Points) -> Vec<[f32; 2]> {
        let at: HashMap<&str, usize> = points
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i))
            .collect();
        // Every field cell's point on this map.
        let point: Vec<Option<usize>> = self
            .names
            .iter()
            .map(|n| at.get(n.as_ref()).copied())
            .collect();
        points
            .names
            .par_iter()
            .enumerate()
            .map(|(p, name)| {
                let Some(&c) = self.index.get(name) else {
                    return [f32::NAN; 2];
                };
                let [x, y] = points.xy[p];
                let (mut wx, mut wy, mut ux, mut uy, mut m) = (0f32, 0f32, 0f32, 0f32, 0f32);
                for &(j, w) in &self.near[c] {
                    let Some(q) = point[j as usize] else { continue };
                    let (dx, dy) = (points.xy[q][0] - x, points.xy[q][1] - y);
                    let len = dx.hypot(dy);
                    if len.is_nan() || len <= 0.0 {
                        continue;
                    }
                    let (dx, dy) = (dx / len, dy / len);
                    wx += w * dx;
                    wy += w * dy;
                    ux += dx;
                    uy += dy;
                    m += 1.0;
                }
                if m == 0.0 {
                    return [f32::NAN; 2];
                }
                [wx - ux / m, wy - uy / m]
            })
            .collect()
    }
}

/// Arrows over a grid of the visible map, in pixels: every bin with
/// [`LEAST`] cells or more, from its cells' mean place, along their mean
/// displacement, the longest arrow most of a bin long.
#[must_use]
pub fn arrows(points: &Points, moved: &[[f32; 2]], vp: &Viewport) -> Vec<Arrow> {
    let bin = (vp.w.max(vp.h) as f32 / BINS).max(4.0);
    let mut bins: HashMap<(i64, i64), Bin> = HashMap::new();
    for (i, d) in moved.iter().enumerate() {
        if !(d[0].is_finite() && d[1].is_finite()) {
            continue;
        }
        let at = vp.to_px(points.xy[i]);
        if at.0 < 0.0 || at.1 < 0.0 || at.0 >= vp.w as f32 || at.1 >= vp.h as f32 {
            continue;
        }
        // The displacement in pixels' directions (the map's y may run up).
        let ahead = vp.to_px([points.xy[i][0] + d[0], points.xy[i][1] + d[1]]);
        let (dx, dy) = ((ahead.0 - at.0) / vp.scale, (ahead.1 - at.1) / vp.scale);
        let e = bins
            .entry(((at.0 / bin) as i64, (at.1 / bin) as i64))
            .or_default();
        e.x += at.0;
        e.y += at.1;
        e.dx += dx;
        e.dy += dy;
        e.n += 1;
    }
    let means: Vec<Arrow> = bins
        .into_values()
        .filter(|b| b.n >= LEAST)
        .map(|b| {
            let n = b.n as f32;
            ((b.x / n, b.y / n), (b.dx / n, b.dy / n))
        })
        .collect();
    let longest = means
        .iter()
        .map(|(_, d)| d.0.hypot(d.1))
        .fold(0f32, f32::max)
        .max(1e-12);
    means
        .into_iter()
        .map(|(at, d)| {
            let s = 0.9 * bin / longest;
            (at, (at.0 + d.0 * s, at.1 + d.1 * s))
        })
        .collect()
}

/// Every arrow drawn: a shaft and a two-stroke head, stroked wide enough to
/// read at the frame's size (a few offset lines), and at least `floor`
/// pixels (a frame shown in coarser pixels than it is drawn in).
pub fn draw(frame: &mut Frame, arrows: &[Arrow], ink: Rgb, floor: f32) {
    let width = (frame.w.max(frame.h) as f32 / 450.0).max(floor);
    let offsets: Vec<f32> = {
        let n = width.ceil() as i32;
        (-n..=n)
            .map(|i| i as f32 * 0.5)
            .filter(|o| o.abs() <= width * 0.5 + 0.01)
            .collect()
    };
    let stroke = |frame: &mut Frame, a: (f32, f32), b: (f32, f32)| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = dx.hypot(dy).max(1e-6);
        let (nx, ny) = (-dy / len, dx / len);
        for &o in &offsets {
            render::draw_edge(
                frame,
                (a.0 + nx * o, a.1 + ny * o),
                (b.0 + nx * o, b.1 + ny * o),
                0.0,
                0.0,
                ink,
                0.85,
            );
        }
    };
    for &(a, b) in arrows {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = dx.hypot(dy);
        if len < 2.0 * width {
            continue;
        }
        stroke(frame, a, b);
        let head = (0.35 * len).clamp(3.0 * width, 9.0 * width);
        let (ux, uy) = (dx / len, dy / len);
        for side in [-1.0f32, 1.0] {
            // 25° either side of the way back.
            let (c, s) = (0.906f32, 0.423 * side);
            let back = (-(ux * c - uy * s), -(ux * s + uy * c));
            stroke(frame, b, (b.0 + back.0 * head, b.1 + back.1 * head));
        }
    }
}

/// Streamlines' grid nodes across the visible map.
const NODES: f32 = 48.0;
/// Occupancy cells across the visible map by default: the spacing between
/// lines.
pub const LANES: f32 = 40.0;
/// The densities `{` and `}` step between.
pub const LANES_RANGE: (f32, f32) = (10.0, 120.0);
/// A grid node is in the field when its cells' kernel weight reaches this.
const LEAST_WEIGHT: f32 = 1.0;

/// The field smoothed onto a regular grid of the visible map: every node's
/// mean displacement in pixels' directions, `None` where too few cells are
/// near (a Gaussian kernel one node spacing wide).
struct Grid {
    nx: usize,
    ny: usize,
    step: f32,
    at: Vec<Option<(f32, f32)>>,
}

impl Grid {
    fn new(points: &Points, moved: &[[f32; 2]], vp: &Viewport) -> Self {
        let step = (vp.w.max(vp.h) as f32 / NODES).max(2.0);
        let (nx, ny) = (
            (vp.w as f32 / step).ceil() as usize + 1,
            (vp.h as f32 / step).ceil() as usize + 1,
        );
        let mut sum = vec![(0f32, 0f32, 0f32); nx * ny];
        let reach = 3i64;
        for (i, d) in moved.iter().enumerate() {
            if !(d[0].is_finite() && d[1].is_finite()) {
                continue;
            }
            let at = vp.to_px(points.xy[i]);
            let ahead = vp.to_px([points.xy[i][0] + d[0], points.xy[i][1] + d[1]]);
            let (dx, dy) = ((ahead.0 - at.0) / vp.scale, (ahead.1 - at.1) / vp.scale);
            let (gx, gy) = ((at.0 / step).round() as i64, (at.1 / step).round() as i64);
            for oy in -reach..=reach {
                for ox in -reach..=reach {
                    let (x, y) = (gx + ox, gy + oy);
                    if x < 0 || y < 0 || x >= nx as i64 || y >= ny as i64 {
                        continue;
                    }
                    let (px, py) = (x as f32 * step - at.0, y as f32 * step - at.1);
                    let w = (-(px * px + py * py) / (2.0 * step * step)).exp();
                    let e = &mut sum[y as usize * nx + x as usize];
                    *e = (e.0 + w * dx, e.1 + w * dy, e.2 + w);
                }
            }
        }
        let at = sum
            .into_iter()
            .map(|(x, y, w)| (w >= LEAST_WEIGHT).then(|| (x / w, y / w)))
            .collect();
        Self { nx, ny, step, at }
    }

    /// The field at pixel `p`, bilinear between the four nodes around it;
    /// `None` outside the map or where a node is out of the field.
    fn sample(&self, p: (f32, f32)) -> Option<(f32, f32)> {
        let (fx, fy) = (p.0 / self.step, p.1 / self.step);
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        if x0 + 1 >= self.nx || y0 + 1 >= self.ny {
            return None;
        }
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let node = |x: usize, y: usize| self.at[y * self.nx + x];
        let (a, b, c, d) = (
            node(x0, y0)?,
            node(x0 + 1, y0)?,
            node(x0, y0 + 1)?,
            node(x0 + 1, y0 + 1)?,
        );
        let mix = |a: f32, b: f32, c: f32, d: f32| {
            (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty
        };
        Some((mix(a.0, b.0, c.0, d.0), mix(a.1, b.1, c.1, d.1)))
    }
}

/// Streamlines of the field over the visible map, in pixels, as matplotlib
/// draws them (and scVelo with it): seeded over an occupancy grid `lanes`
/// cells across (the lines' density), each
/// traced forward and backward at unit speed (second-order Runge–Kutta)
/// until it leaves the map or the field, stalls, or runs into a lane another
/// line holds; lines too short to read are dropped.
#[must_use]
pub fn streamlines(
    points: &Points,
    moved: &[[f32; 2]],
    vp: &Viewport,
    lanes: f32,
) -> Vec<Vec<(f32, f32)>> {
    let grid = Grid::new(points, moved, vp);
    let fastest = grid
        .at
        .iter()
        .flatten()
        .map(|v| v.0.hypot(v.1))
        .fold(0f32, f32::max);
    if fastest <= 0.0 {
        return Vec::new();
    }
    let lane = (vp.w.max(vp.h) as f32 / lanes.max(1.0)).max(3.0);
    let (lx, ly) = (
        (vp.w as f32 / lane).ceil() as usize,
        (vp.h as f32 / lane).ceil() as usize,
    );
    let mut taken = vec![false; lx * ly];
    let lane_of = |p: (f32, f32)| -> Option<usize> {
        let (x, y) = ((p.0 / lane) as i64, (p.1 / lane) as i64);
        (x >= 0 && y >= 0 && (x as usize) < lx && (y as usize) < ly)
            .then(|| y as usize * lx + x as usize)
    };
    let step = 0.25 * grid.step;
    let most = (4.0 * (vp.w + vp.h) as f32 / step) as usize;
    // Unit direction at `p`, if the field there moves.
    let ahead = |p: (f32, f32)| -> Option<(f32, f32)> {
        let v = grid.sample(p)?;
        let speed = v.0.hypot(v.1);
        (speed > 0.05 * fastest).then(|| (v.0 / speed, v.1 / speed))
    };
    let mut lines = Vec::new();
    for seed in 0..lx * ly {
        if taken[seed] {
            continue;
        }
        let start = (
            ((seed % lx) as f32 + 0.5) * lane,
            ((seed / lx) as f32 + 0.5) * lane,
        );
        if ahead(start).is_none() {
            continue;
        }
        let mut mine: Vec<usize> = vec![seed];
        let trace = |sign: f32, mine: &mut Vec<usize>| -> Vec<(f32, f32)> {
            let mut path = Vec::new();
            let mut p = start;
            for _ in 0..most {
                let Some(d) = ahead(p) else { break };
                let mid = (p.0 + sign * 0.5 * step * d.0, p.1 + sign * 0.5 * step * d.1);
                let Some(d) = ahead(mid) else { break };
                let next = (p.0 + sign * step * d.0, p.1 + sign * step * d.1);
                let Some(cell) = lane_of(next) else { break };
                if taken[cell] && !mine.contains(&cell) {
                    break;
                }
                if !mine.contains(&cell) {
                    mine.push(cell);
                }
                path.push(next);
                p = next;
            }
            path
        };
        let back = trace(-1.0, &mut mine);
        let fore = trace(1.0, &mut mine);
        if mine.len() < 3 {
            continue;
        }
        for &c in &mine {
            taken[c] = true;
        }
        let mut line: Vec<(f32, f32)> = back.into_iter().rev().collect();
        line.push(start);
        line.extend(fore);
        lines.push(line);
    }
    lines
}

/// Every streamline drawn, with an arrowhead halfway along it, at least
/// `floor` pixels wide (see [`draw`]).
pub fn draw_lines(frame: &mut Frame, lines: &[Vec<(f32, f32)>], ink: Rgb, floor: f32) {
    let width = (frame.w.max(frame.h) as f32 / 700.0).max(floor);
    for line in lines {
        for pair in line.windows(2) {
            stroke(frame, pair[0], pair[1], width, ink, 0.75);
        }
        if line.len() >= 3 {
            let m = line.len() / 2;
            let (a, b) = (line[m - 1], line[m]);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let len = dx.hypot(dy).max(1e-6);
            let (ux, uy) = (dx / len, dy / len);
            let head = 6.0 * width.max(1.5);
            for side in [-1.0f32, 1.0] {
                let (c, s) = (0.906f32, 0.423 * side);
                let back = (-(ux * c - uy * s), -(ux * s + uy * c));
                stroke(
                    frame,
                    b,
                    (b.0 + back.0 * head, b.1 + back.1 * head),
                    width,
                    ink,
                    0.9,
                );
            }
        }
    }
}

/// A line `width` pixels wide, as offset one-pixel lines.
fn stroke(frame: &mut Frame, a: (f32, f32), b: (f32, f32), width: f32, ink: Rgb, alpha: f32) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = dx.hypot(dy).max(1e-6);
    let (nx, ny) = (-dy / len, dx / len);
    let n = width.ceil() as i32;
    for i in -n..=n {
        let o = i as f32 * 0.5;
        if o.abs() > width * 0.5 + 0.01 {
            continue;
        }
        render::draw_edge(
            frame,
            (a.0 + nx * o, a.1 + ny * o),
            (b.0 + nx * o, b.1 + ny * o),
            0.0,
            0.0,
            ink,
            alpha,
        );
    }
}

#[cfg(test)]
#[path = "tests/velocity.rs"]
mod tests;
