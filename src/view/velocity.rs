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
/// read at the frame's size (a few offset lines).
pub fn draw(frame: &mut Frame, arrows: &[Arrow], ink: Rgb) {
    let width = (frame.w.max(frame.h) as f32 / 450.0).max(1.0);
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

#[cfg(test)]
#[path = "tests/velocity.rs"]
mod tests;
