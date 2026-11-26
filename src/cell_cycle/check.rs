//! What the cycle scores say about a run: how much of each latent dimension
//! they explain, and each cluster's phase make-up.
//!
//! - `{out}.cell_cycle.latent.parquet`: per latent dimension, the R² of a
//!   least-squares fit on the S score, on the G2/M score, and on both.
//! - `{out}.cell_cycle.clusters.parquet`: per cluster, its cells and the
//!   share in each phase; `one_phase` marks a cluster at or past
//!   [`ONE_PHASE`] in one phase, where a cycle covariate is not used.

use super::{Phase, Scores};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use nalgebra::{DMatrix, DVector};
use senna::cell_labels::{label_composition, tolerant_align};
use senna::embed_common::*;
use senna::run_manifest::{resolve, RunManifest};
use std::path::Path;

/// A cluster this much in one phase ignores the cycle covariate.
pub const ONE_PHASE: f32 = 0.95;

/// The phases a cluster's make-up counts, in its columns' order.
const PHASES: [Phase; 3] = [Phase::G1, Phase::S, Phase::G2M];

/// Write the run's diagnostics beside `out`; each is skipped, with a note,
/// when the run has nothing to read for it.
///
/// # Errors
/// A manifest or table that cannot be read or written.
pub fn check(manifest: &str, cells: &[Box<str>], scores: &Scores, out: &str) -> anyhow::Result<()> {
    let (m, dir) = RunManifest::load(Path::new(manifest))?;
    match m.read_cell_latent(&dir)? {
        Some((_, latent)) => {
            let at = tolerant_align(&latent.rows, cells)?;
            let dst = format!("{out}.cell_cycle.latent.parquet");
            match write_latent(&dst, &latent.mat, &at, scores)? {
                Some((k, r2)) => {
                    info!("wrote {dst}: most cycle-explained dimension {k} (R² {r2:.2})")
                }
                None => info!("wrote {dst}: the cycle explains none of the latent"),
            }
        }
        None => info!("{}: no cell latent to check against the cycle", m.prefix),
    }
    match m.cluster.clusters.as_deref() {
        Some(rel) => {
            let path = resolve(&dir, rel).to_string_lossy().into_owned();
            let named = Mat::from_parquet_with_row_names(&path, Some(0))?;
            let at = tolerant_align(&named.rows, cells)?;
            let dst = format!("{out}.cell_cycle.clusters.parquet");
            let one = write_clusters(&dst, &named.mat, &at, &scores.phase)?;
            info!(
                "wrote {dst}: {one} cluster(s) at least {:.0}% one phase",
                100.0 * ONE_PHASE
            );
        }
        None => info!("{}: no clusters to check against the cycle", m.prefix),
    }
    Ok(())
}

/// R² of every latent column (row `r` is cell `at[r]` of the scores) on S,
/// on G2/M and on both; returns the dimension the two explain most and its
/// R², if any explains anything.
fn write_latent(
    path: &str,
    z: &Mat,
    at: &[usize],
    s: &Scores,
) -> anyhow::Result<Option<(usize, f32)>> {
    let rows: Vec<(usize, usize)> = at
        .iter()
        .enumerate()
        .filter(|&(_, &j)| j != usize::MAX)
        .map(|(r, &j)| (r, j))
        .collect();
    let covariate = |v: &[f32]| {
        rows.iter()
            .map(|&(_, j)| f64::from(v[j]))
            .collect::<Vec<_>>()
    };
    let (x_s, x_g) = (covariate(&s.s), covariate(&s.g2m));
    let fits = [
        Ols::new(&[&x_s]),
        Ols::new(&[&x_g]),
        Ols::new(&[&x_s, &x_g]),
    ];
    let mut r2: [Vec<f32>; 3] = Default::default();
    for k in 0..z.ncols() {
        let y: Vec<f64> = rows.iter().map(|&(r, _)| f64::from(z[(r, k)])).collect();
        for (fit, col) in fits.iter().zip(r2.iter_mut()) {
            col.push(fit.r2(&y) as f32);
        }
    }
    let names: Vec<Box<str>> = (0..z.ncols()).map(|k| k.to_string().into()).collect();
    let [r_s, r_g, r_b] = &r2;
    write_named_table(
        path,
        "dimension",
        &names,
        &[
            ("r2_s".into(), Column::F32(r_s)),
            ("r2_g2m".into(), Column::F32(r_g)),
            ("r2_both".into(), Column::F32(r_b)),
        ],
    )?;
    Ok(r_b
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, v)| v > 0.0)
        .max_by(|a, b| a.1.total_cmp(&b.1)))
}

/// A least-squares fit on an intercept and some covariates, set up once
/// (the covariates centred and their Gram matrix inverted) to give the R²
/// of any number of responses.
struct Ols {
    centred: Vec<Vec<f64>>,
    gram_inv: Option<DMatrix<f64>>,
}

impl Ols {
    fn new(xs: &[&[f64]]) -> Self {
        let centred: Vec<Vec<f64>> = xs.iter().map(|x| centre(x)).collect();
        let p = centred.len();
        let gram = DMatrix::from_fn(p, p, |a, b| dot(&centred[a], &centred[b]));
        Self {
            gram_inv: gram.try_inverse(),
            centred,
        }
    }

    /// The share of `y`'s variance the covariates explain; 0 when there are
    /// too few cells, no variance or collinear covariates.
    fn r2(&self, y: &[f64]) -> f64 {
        let Some(inv) = &self.gram_inv else {
            return 0.0;
        };
        if y.len() < self.centred.len() + 2 {
            return 0.0;
        }
        let yc = centre(y);
        let tss = dot(&yc, &yc);
        if tss <= 0.0 {
            return 0.0;
        }
        let xy =
            DVector::from_iterator(self.centred.len(), self.centred.iter().map(|x| dot(x, &yc)));
        (xy.dot(&(inv * &xy)) / tss).clamp(0.0, 1.0)
    }
}

fn centre(v: &[f64]) -> Vec<f64> {
    let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
    v.iter().map(|x| x - m).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// Each cluster's cells and phase shares; returns how many are one phase.
/// A cell with a negative or missing label is in no cluster; an Undecided
/// one counts among its cluster's cells but in none of the phases.
fn write_clusters(path: &str, label: &Mat, at: &[usize], phase: &[Phase]) -> anyhow::Result<usize> {
    let mut ids: Vec<i64> = Vec::new();
    let mut group_of_cell = vec![usize::MAX; phase.len()];
    for (r, &j) in at.iter().enumerate() {
        let v = label[(r, 0)];
        if j == usize::MAX || !(v.is_finite() && v >= 0.0) {
            continue;
        }
        let id = v as i64;
        let g = ids.iter().position(|&c| c == id).unwrap_or_else(|| {
            ids.push(id);
            ids.len() - 1
        });
        group_of_cell[j] = g;
    }
    let phase_of_cell: Vec<Option<u32>> = phase
        .iter()
        .map(|p| PHASES.iter().position(|q| q == p).map(|k| k as u32))
        .collect();
    let comp = label_composition(&group_of_cell, &phase_of_cell, ids.len(), PHASES.len());
    let mut cells = vec![0i32; ids.len()];
    for &g in group_of_cell.iter().filter(|&&g| g != usize::MAX) {
        cells[g] += 1;
    }
    let share = |k: usize| -> Vec<f32> {
        comp.iter()
            .map(|c| c.as_ref().map_or(0.0, |c| c[k]))
            .collect()
    };
    let (g1, s, g2m) = (share(0), share(1), share(2));
    let one: Vec<i32> = (0..ids.len())
        .map(|i| i32::from(g1[i].max(s[i]).max(g2m[i]) >= ONE_PHASE))
        .collect();
    let names: Vec<Box<str>> = ids.iter().map(|c| c.to_string().into()).collect();
    write_named_table(
        path,
        "cluster",
        &names,
        &[
            ("cells".into(), Column::I32(&cells)),
            ("g1".into(), Column::F32(&g1)),
            ("s".into(), Column::F32(&s)),
            ("g2m".into(), Column::F32(&g2m)),
            ("one_phase".into(), Column::I32(&one)),
        ],
    )?;
    Ok(one.iter().filter(|&&o| o == 1).count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2_is_one_for_an_exact_fit_and_zero_for_none() {
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 1.0).collect();
        assert!((Ols::new(&[&x]).r2(&y) - 1.0).abs() < 1e-9);
        let noise: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let flat: Vec<f64> = (0..20).map(|i| f64::from(i / 2)).collect();
        assert!(Ols::new(&[&flat]).r2(&noise) < 0.1);
        // Two covariates that together give y exactly.
        let z: Vec<f64> = (0..20).map(|i| f64::from((i * 7) % 5)).collect();
        let y2: Vec<f64> = x.iter().zip(&z).map(|(a, b)| a - 2.0 * b).collect();
        assert!((Ols::new(&[&x, &z]).r2(&y2) - 1.0).abs() < 1e-9);
        // Collinear covariates give no fit rather than nonsense.
        assert_eq!(Ols::new(&[&x, &x]).r2(&y), 0.0);
    }
}
