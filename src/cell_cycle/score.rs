//! Each cell's S and G2/M scores, phase and position around the cycle.
//!
//! The scores are Tirosh et al.'s module score (Science 352:189, 2016), as
//! Seurat's `AddModuleScore`/`CellCycleScoring` compute it: a set's mean
//! log-normalised expression in the cell minus that of control genes drawn
//! from the same expression bins (24 equal-size bins of the genes' mean
//! expression, 100 controls per set gene). The phase is G1 when both scores
//! are below zero, else the higher one (`Undecided` on a tie, as Seurat).
//!
//! The angle is the cell's direction in the plane of the top two principal
//! components of the cycle genes' standardised expression, a cycle position
//! in the spirit of Revelio (Schwabe et al. 2020) and tricycle (Zheng et al.
//! 2022) but fitted on these cells, not projected onto tricycle's
//! reference. It is turned so G1 cells sit near 0 and runs S before G2/M,
//! in `[0, 2π)`.
//!
//! The counts are streamed in column blocks, three passes, so memory holds
//! a few numbers per cell and not the cells' genes: each gene's mean
//! expression (for the control bins and `--learn`), the scores and the
//! cycle genes' covariance, then the angle and each gene's correlation with
//! the scores. Work per cell is over its nonzero counts.

use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::cell_cycle::{CycleGenes, CyclePhase};
use genomic_data::name_matching::GeneIndex;
use legume_numeric::matrix::covariate::score::SCALE;
use legume_numeric::matrix::covariate::ExpressionBins;
use nalgebra::{DMatrix, SymmetricEigen};
use rand::SeedableRng;
use std::sync::{Arc, Mutex};

/// Fewest matched genes of either phase to score at all.
pub const MIN_MATCHED: usize = 10;

/// A cell's phase as the scores call it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    G1,
    S,
    G2M,
    /// The two scores tie, not both below zero (Seurat's `Undecided`).
    Undecided,
}

impl Phase {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::G1 => "G1",
            Self::S => "S",
            Self::G2M => "G2M",
            Self::Undecided => "Undecided",
        }
    }

    /// Seurat's call: G1 when both scores are negative, else the higher,
    /// and `Undecided` on a tie.
    #[must_use]
    pub fn call(s: f32, g2m: f32) -> Self {
        if s < 0.0 && g2m < 0.0 {
            Self::G1
        } else if s > g2m {
            Self::S
        } else if g2m > s {
            Self::G2M
        } else {
            Self::Undecided
        }
    }
}

/// Per-cell scores, and what was matched.
pub struct Scores {
    pub s: Vec<f32>,
    pub g2m: Vec<f32>,
    pub phase: Vec<Phase>,
    pub angle: Vec<f32>,
    /// Rows of the S and G2/M genes found.
    pub s_rows: Vec<usize>,
    pub g2m_rows: Vec<usize>,
    /// With `learn`: each gene's correlation with the S and G2/M scores.
    pub learned: Option<(Vec<f32>, Vec<f32>)>,
}

/// Why no scores were made: too few of the genes among the features.
#[derive(Debug, PartialEq, Eq)]
pub struct Skipped {
    pub s: usize,
    pub g2m: usize,
}

impl std::fmt::Display for Skipped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "only {} S and {} G2/M genes match the features (need {MIN_MATCHED} of each)",
            self.s, self.g2m
        )
    }
}

/// Score every column of `data`, `block` columns at a time; `Err(Skipped)`
/// when too few of the genes match its features.
///
/// # Errors
/// No columns, or a column block that cannot be read.
pub fn score(
    data: &SparseIoVec,
    genes: &CycleGenes,
    block: usize,
    learn: bool,
    seed: u64,
) -> anyhow::Result<Result<Scores, Skipped>> {
    let index = GeneIndex::build(&data.row_names()?);
    let s_rows = genes.matched_rows(&index, CyclePhase::S);
    let g2m_rows = genes.matched_rows(&index, CyclePhase::G2M);
    if s_rows.len() < MIN_MATCHED || g2m_rows.len() < MIN_MATCHED {
        return Ok(Err(Skipped {
            s: s_rows.len(),
            g2m: g2m_rows.len(),
        }));
    }
    let (n, d) = (data.num_columns(), data.num_rows());
    anyhow::ensure!(n > 0, "no cells to score");
    let bs = Some(block.max(1));

    // Pass 1: each gene's sum and sum of squares of log expression.
    let mut sums = (vec![0f64; d], vec![0f64; d]);
    data.visit_columns_by_block(
        &|(lb, ub), data: &SparseIoVec, _: &(), out: Arc<Mutex<&mut (Vec<f64>, Vec<f64>)>>| {
            let mut local = (vec![0f64; d], vec![0f64; d]);
            visit(data, lb, ub, |_, rows, vals| {
                for (&g, &v) in rows.iter().zip(vals) {
                    local.0[g] += f64::from(v);
                    local.1[g] += f64::from(v) * f64::from(v);
                }
            })?;
            let mut out = lock(&out);
            add(&mut out.0, &local.0);
            add(&mut out.1, &local.1);
            Ok(())
        },
        &(),
        &mut sums,
        bs,
    )?;
    let (sum, sumsq) = sums;
    let mean: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();

    // The module scores' weights over every gene, controls drawn from one
    // set of expression bins.
    let bins = ExpressionBins::new(&mean, &(0..d).collect::<Vec<_>>());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let dense = |w: Vec<(usize, f32)>| {
        let mut out = vec![0f64; d];
        for (g, x) in w {
            out[g] = f64::from(x);
        }
        out
    };
    let w_s = dense(bins.weights(&s_rows, &mut rng));
    let w_g2m = dense(bins.weights(&g2m_rows, &mut rng));

    // The cycle genes, each once, standardised: a cell's value is its
    // zero-level `base[k]` plus `x / sd[k]` where the gene is expressed.
    let mut pos: Vec<Option<usize>> = vec![None; d];
    let mut cycle: Vec<usize> = Vec::new();
    for &g in s_rows.iter().chain(&g2m_rows) {
        if pos[g].is_none() {
            pos[g] = Some(cycle.len());
            cycle.push(g);
        }
    }
    let m = cycle.len();
    let sd: Vec<f64> = cycle
        .iter()
        .map(|&g| (sumsq[g] / n as f64 - mean[g] * mean[g]).max(1e-12).sqrt())
        .collect();
    let base: Vec<f64> = cycle.iter().zip(&sd).map(|(&g, s)| -mean[g] / s).collect();

    // Pass 2: the scores, and the cycle genes' departures from their
    // zero-level: their sum and cross products over expressed pairs.
    let mut pass2 = Pass2 {
        s: vec![0f32; n],
        g2m: vec![0f32; n],
        cross: Cross::zeros(m),
    };
    data.visit_columns_by_block(
        &|(lb, ub), data: &SparseIoVec, _: &(), out: Arc<Mutex<&mut Pass2>>| {
            let (mut s_blk, mut g_blk) = (Vec::with_capacity(ub - lb), Vec::with_capacity(ub - lb));
            let mut cross = Cross::zeros(m);
            visit(data, lb, ub, |_, rows, vals| {
                let (mut a, mut b) = (0f64, 0f64);
                cross.nz.clear();
                for (&g, &v) in rows.iter().zip(vals) {
                    a += w_s[g] * f64::from(v);
                    b += w_g2m[g] * f64::from(v);
                    if let Some(k) = pos[g] {
                        cross.nz.push((k, f64::from(v) / sd[k]));
                    }
                }
                s_blk.push(a as f32);
                g_blk.push(b as f32);
                cross.add_cell();
            })?;
            let mut out = lock(&out);
            out.s[lb..ub].copy_from_slice(&s_blk);
            out.g2m[lb..ub].copy_from_slice(&g_blk);
            out.cross.add(&cross);
            Ok(())
        },
        &(),
        &mut pass2,
        bs,
    )?;
    let Pass2 { s, g2m, cross } = pass2;
    let phase: Vec<Phase> = s
        .iter()
        .zip(&g2m)
        .map(|(&a, &b)| Phase::call(a, b))
        .collect();

    // The plane of the top two components of the standardised genes.
    let eig = SymmetricEigen::new(cross.covariance(&base, n));
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let u: Vec<f64> = eig.eigenvectors.column(order[0]).iter().copied().collect();
    let v: Vec<f64> = eig.eigenvectors.column(order[1]).iter().copied().collect();
    let (u0, v0) = (dot(&u, &base), dot(&v, &base));

    // Pass 3: each cell's angle in that plane, and with `learn` each gene's
    // sums of expression times either score.
    let dl = if learn { d } else { 0 };
    let mut pass3 = Pass3 {
        angle: vec![0f32; n],
        xs: vec![0f64; dl],
        xg: vec![0f64; dl],
    };
    data.visit_columns_by_block(
        &|(lb, ub), data: &SparseIoVec, _: &(), out: Arc<Mutex<&mut Pass3>>| {
            let mut a_blk = Vec::with_capacity(ub - lb);
            let (mut xs, mut xg) = (vec![0f64; dl], vec![0f64; dl]);
            visit(data, lb, ub, |j, rows, vals| {
                let (mut p, mut q) = (u0, v0);
                for (&g, &x) in rows.iter().zip(vals) {
                    let x = f64::from(x);
                    if let Some(k) = pos[g] {
                        p += u[k] * x / sd[k];
                        q += v[k] * x / sd[k];
                    }
                    if learn {
                        xs[g] += x * f64::from(s[j]);
                        xg[g] += x * f64::from(g2m[j]);
                    }
                }
                a_blk.push(q.atan2(p) as f32);
            })?;
            let mut out = lock(&out);
            out.angle[lb..ub].copy_from_slice(&a_blk);
            add(&mut out.xs, &xs);
            add(&mut out.xg, &xg);
            Ok(())
        },
        &(),
        &mut pass3,
        bs,
    )?;
    let Pass3 { mut angle, xs, xg } = pass3;
    orient(&mut angle, &phase);
    let learned = learn.then(|| correlations(&mean, &sumsq, &s, &g2m, &xs, &xg));

    Ok(Ok(Scores {
        s,
        g2m,
        phase,
        angle,
        s_rows,
        g2m_rows,
        learned,
    }))
}

/// Read columns `lb..ub` and hand each to `each` with its index, its
/// nonzero rows and their log-normalised values (`ln(1 + 10⁴ x / total)`),
/// one buffer reused across the block.
fn visit(
    data: &SparseIoVec,
    lb: usize,
    ub: usize,
    mut each: impl FnMut(usize, &[usize], &[f32]),
) -> anyhow::Result<()> {
    let csc = data.read_columns_csc(lb..ub)?;
    let (offsets, rows, values) = (csc.col_offsets(), csc.row_indices(), csc.values());
    let mut buf: Vec<f32> = Vec::new();
    for c in 0..csc.ncols() {
        let (lo, hi) = (offsets[c], offsets[c + 1]);
        let x = &values[lo..hi];
        let total: f32 = x.iter().sum();
        let k = if total > 0.0 { SCALE / total } else { 0.0 };
        buf.clear();
        buf.extend(x.iter().map(|v| (v * k).ln_1p()));
        each(lb + c, &rows[lo..hi], &buf);
    }
    Ok(())
}

/// The shared output a visitor adds its block into, its lock taken; a
/// poisoned lock (a visitor that panicked) is taken all the same, since the
/// panic fails the whole visit.
fn lock<'a, 'b, T>(out: &'a Mutex<&'b mut T>) -> std::sync::MutexGuard<'a, &'b mut T> {
    out.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn add(into: &mut [f64], from: &[f64]) {
    for (x, y) in into.iter_mut().zip(from) {
        *x += y;
    }
}

/// Pass 2's shared output: the scores and the cycle genes' sums.
struct Pass2 {
    s: Vec<f32>,
    g2m: Vec<f32>,
    cross: Cross,
}

/// Pass 3's shared output: the angles, and with `learn` the genes' sums.
struct Pass3 {
    angle: Vec<f32>,
    xs: Vec<f64>,
    xg: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// The cycle genes' departures from their zero-level, `δ = x / sd`, summed
/// over cells: `Σδ`, `Σδδᵀ` (over each cell's expressed pairs only) and the
/// cell's departures being added (`nz`, a reused buffer).
struct Cross {
    sum: Vec<f64>,
    outer: DMatrix<f64>,
    nz: Vec<(usize, f64)>,
}

impl Cross {
    fn zeros(m: usize) -> Self {
        Self {
            sum: vec![0.0; m],
            outer: DMatrix::zeros(m, m),
            nz: Vec::new(),
        }
    }

    fn add_cell(&mut self) {
        for &(k, a) in &self.nz {
            self.sum[k] += a;
            for &(l, b) in &self.nz {
                self.outer[(k, l)] += a * b;
            }
        }
    }

    fn add(&mut self, other: &Self) {
        add(&mut self.sum, &other.sum);
        self.outer += &other.outer;
    }

    /// The covariance of the standardised genes `z = base + δ` over `n`
    /// cells: `(n·base·baseᵀ + base·Σδᵀ + Σδ·baseᵀ + Σδδᵀ) / (n − 1)`.
    fn covariance(&self, base: &[f64], n: usize) -> DMatrix<f64> {
        let m = base.len();
        DMatrix::from_fn(m, m, |k, l| {
            (n as f64 * base[k] * base[l]
                + base[k] * self.sum[l]
                + self.sum[k] * base[l]
                + self.outer[(k, l)])
                / (n.max(2) - 1) as f64
        })
    }
}

/// Turn the angles so G1 cells' circular mean is 0, and run S before G2/M.
fn orient(angle: &mut [f32], phase: &[Phase]) {
    let circ_mean = |p: Phase, a: &[f32]| {
        let (s, c) = a
            .iter()
            .zip(phase)
            .filter(|(_, &q)| q == p)
            .fold((0f64, 0f64), |(s, c), (&x, _)| {
                (s + f64::from(x).sin(), c + f64::from(x).cos())
            });
        (s != 0.0 || c != 0.0).then(|| s.atan2(c))
    };
    let wrap = |x: f64| x.rem_euclid(std::f64::consts::TAU);
    let g1 = circ_mean(Phase::G1, angle).unwrap_or(0.0);
    for a in angle.iter_mut() {
        *a = wrap(f64::from(*a) - g1) as f32;
    }
    if let (Some(s), Some(g)) = (circ_mean(Phase::S, angle), circ_mean(Phase::G2M, angle)) {
        if wrap(s) > wrap(g) {
            for a in angle.iter_mut() {
                *a = wrap(-f64::from(*a)) as f32;
            }
        }
    }
}

/// Each gene's Pearson correlation, over cells, with the S and G2/M
/// scores, from its sums `xs`, `xg` of expression times score.
fn correlations(
    mean: &[f64],
    sumsq: &[f64],
    s: &[f32],
    g2m: &[f32],
    xs: &[f64],
    xg: &[f64],
) -> (Vec<f32>, Vec<f32>) {
    let n = s.len() as f64;
    let moments = |y: &[f32]| {
        let m = y.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let v = y.iter().map(|&v| (f64::from(v) - m).powi(2)).sum::<f64>() / n;
        (m, v.sqrt())
    };
    let ((ms, ss), (mg, sg)) = (moments(s), moments(g2m));
    let r = |sum_xy: &[f64], my: f64, sy: f64| -> Vec<f32> {
        (0..mean.len())
            .map(|g| {
                let sx = (sumsq[g] / n - mean[g] * mean[g]).max(0.0).sqrt();
                if sx <= 0.0 || sy <= 0.0 {
                    return 0.0;
                }
                ((sum_xy[g] / n - mean[g] * my) / (sx * sy)) as f32
            })
            .collect()
    };
    (r(xs, ms, ss), r(xg, mg, sg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seurat_s_phase_call() {
        assert_eq!(Phase::call(-0.1, -0.2), Phase::G1);
        assert_eq!(Phase::call(0.3, 0.1), Phase::S);
        assert_eq!(Phase::call(-0.1, 0.2), Phase::G2M);
        assert_eq!(Phase::call(0.1, 0.1), Phase::Undecided);
    }

    #[test]
    fn angles_put_g1_at_zero_and_s_before_g2m() {
        let phase = [Phase::G1, Phase::G1, Phase::S, Phase::G2M];
        let mut a = [2.0f32, 2.2, 1.0, 3.0];
        orient(&mut a, &phase);
        assert!(a[0] < 0.3 || a[0] > 6.0);
        assert!(a[2] < a[3]);
    }
}
