//! Step 3: every lineage's direction from the splicing alone, as the
//! posterior of one binary `d` (forward, reversed) shared by every gene.
//!
//! Given the lineage's steady fit (blind to direction), every pseudobulk's
//! grid time is drawn from its posterior and every gene's unspliced and
//! spliced reads are pooled per grid time, `U_kg`, `S_kg`. Read with the
//! velocity forward or reversed (`d x_kg`, `x` the curve's `d log s/dt`), the
//! unspliced share is `μ_kg = σ(log m⁺(γ_g + d x_kg) − log β_g)`, and
//! `U_kg ~ BetaBinomial(U + S, μ κ_g, (1 − μ) κ_g)`. Every gene's evidence
//! for each `d` integrates its `(log γ/β, log γ, log κ)` exactly on a grid,
//! in f64 on the host, less the same on the reads with no dynamics (every
//! grid time at the gene's overall unspliced share): the prior and the floor
//! alone favour one way on a skewed curve, and that bias, summed over
//! thousands of genes, would decide. Given `d` the genes are independent, so
//! `log BF = Σ_g log BF_g` and `p(forward) = σ(log BF)`, averaged over the
//! time draws.
//!
//! Wakefield's approximate Bayes factor (*Am J Hum Genet* (2007)) checks it:
//! linearised, the unspliced log-odds is `log γ/β + (d/γ) x`, so the sign of
//! a quasi-binomial logistic slope on `x` is the direction; its sign Bayes
//! factor under a normal prior is `Φ(z′)/Φ(−z′)`.

use super::TrainConfig;
use crate::td_rbm::curve::Spline;
use crate::td_rbm::model::{Energy, Tables, FLOOR_SOFTNESS};
use crate::td_rbm::rbm::Reads;
use crate::td_rbm::start::log_sum_exp;
use legume_numeric::candle::candle_core::{Device, Result, Tensor, D};
use log::info;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// What a lineage's direction rests on.
#[derive(Clone, Debug)]
pub struct Direction {
    pub lineage: usize,
    /// `log BF`, the lineage as it runs (from its tree's root) over
    /// reversed, its mean and sd over the time draws; `p(forward)`, the
    /// draws' mean. A report: the root is chosen per tree.
    pub log_bf: f64,
    pub log_bf_sd: f64,
    pub p_forward: f64,
    /// Wakefield's sign Bayes factor, `Σ_g log ABF_g`, forward over reversed.
    pub log_abf: f64,
    /// `[G]` every gene's `log BF_g`, and its logistic slope's `z`, both
    /// signed for the kept orientation.
    pub gene_votes: Vec<f32>,
    pub gene_z: Vec<f32>,
}

/// A tree's root is called at this posterior probability.
pub(crate) const P_CALL: f64 = 0.95;
/// The grid each gene's evidence integrates over: the log rest ratio
/// `log γ/β` about the gene's overall `log U/S`, `log γ` against the gene's
/// largest `|x|`, and `log κ`.
const R_SPAN: (f64, f64, usize) = (-3.0, 3.0, 17);
const GAMMA_SPAN: (f64, f64, usize) = (0.25, 64.0, 16);
const KAPPA_SPAN: (f64, f64, usize) = (2.0, 2000.0, 6);
/// Wakefield's prior variance on the standardised slope.
const ABF_W: f64 = 1.0;

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64)
        .collect()
}

/// Every lineage's direction from the splicing alone ([`orient`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn orient_all(
    energy: &Energy,
    spline: &Spline,
    finest: &[f32],
    f: usize,
    lineage: &[usize],
    cfg: &TrainConfig,
    dev: &Device,
    rng: &mut StdRng,
) -> anyhow::Result<Vec<Direction>> {
    let steady = energy.steady_tables(spline)?;
    let mut out = Vec::new();
    let mut rate_sum: Option<(Tensor, Tensor, f64)> = None;
    for lin in 0..cfg.lineages {
        let rows: Vec<usize> = (0..lineage.len()).filter(|&r| lineage[r] == lin).collect();
        if rows.len() < 2 {
            continue;
        }
        let x: Vec<f32> = rows
            .iter()
            .flat_map(|&r| finest[r * f..(r + 1) * f].to_vec())
            .collect();
        let (d, (lb, lg)) = orient(energy, spline, &steady, &x, f, lin, cfg, dev, rng)?;
        let w = rows.len() as f64;
        rate_sum = Some(match rate_sum {
            None => ((&lb * w)?, (&lg * w)?, w),
            Some((sb, sg, sw)) => ((sb + (&lb * w)?)?, (sg + (&lg * w)?)?, sw + w),
        });
        info!(
            "td-rbm lineage L{} ({} pseudobulks), from its tree's root: p(forward) {:.4}, log BF {:+.1} (sd {:.1} over time draws), Wakefield log ABF {:+.1}; genes for {} against {}",
            lin + 1,
            rows.len(),
            d.p_forward,
            d.log_bf,
            d.log_bf_sd,
            d.log_abf,
            d.gene_votes.iter().filter(|&&v| v > 0.0).count(),
            d.gene_votes.iter().filter(|&&v| v < 0.0).count(),
        );
        out.push(d);
    }
    // The model carries on with every gene's rates' posterior means under
    // the kept directions, averaged over the lineages by their pseudobulks.
    if let Some((sb, sg, sw)) = rate_sum {
        energy.set_log_rates(&(sb / sw)?, &(sg / sw)?)?;
    }
    Ok(out)
}

/// Lineage `lin`'s splicing evidence (module docs) on the finest
/// pseudobulks `x` (`[P × F]` counts) it is read on, forward meaning its
/// curve as it runs now.
pub(crate) struct Bayes {
    /// `Σ_g log BF_g` per time draw, and every gene's mean over the draws.
    pub lbf: Vec<f64>,
    pub genes: Vec<f64>,
    /// Every gene's posterior mean `log β`, `log γ` as it runs.
    pub rates: (Vec<f64>, Vec<f64>),
    /// Wakefield's check, when asked for: every gene's slope `z`, and
    /// `Σ_g log ABF_g`.
    pub z: Vec<f64>,
    pub log_abf: f64,
}

impl Bayes {
    pub fn log_bf(&self) -> f64 {
        self.lbf.iter().sum::<f64>() / self.lbf.len().max(1) as f64
    }
    pub fn log_bf_sd(&self) -> f64 {
        let (m, n) = (self.log_bf(), self.lbf.len());
        (self.lbf.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64).sqrt()
    }
    pub fn p_forward(&self) -> f64 {
        self.lbf
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .sum::<f64>()
            / self.lbf.len().max(1) as f64
    }
}

/// [`Bayes`] for lineage `lin` on `x`, its time posteriors from the steady
/// tables `t`; Wakefield's check only with `wakefield`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bayes(
    energy: &Energy,
    spline: &Spline,
    t: &Tables,
    x: &[f32],
    f: usize,
    lin: usize,
    wakefield: bool,
    cfg: &TrainConfig,
    dev: &Device,
    rng: &mut StdRng,
) -> anyhow::Result<Bayes> {
    let g = f / 2;
    let p = x.len() / f;
    let mut fixed = x.to_vec();
    crate::td_rbm::start::scale_rows(&mut fixed, f, cfg.depth as f32);
    let rows = Reads::from_features(Tensor::from_vec(fixed, (p, f), dev)?)?;
    let k = t.grid_len()?;
    let logits = rows.x.matmul(&t.psi.narrow(0, lin * k, k)?.t()?)?; // [P × K]
    let q = legume_numeric::candle::candle_nn::ops::softmax(&logits, D::Minus1)?;
    let readout = energy.velocity_readout(spline, lin)?.detach(); // [K × G]
    let soft = q.t()?.matmul(&rows.x)?.to_vec2::<f32>()?; // [K × F]
    let xv = readout.to_vec2::<f32>()?;

    let wake: Vec<(f64, f64)> = if wakefield {
        (0..g)
            .into_par_iter()
            .map(|j| {
                let u: Vec<f64> = (0..k).map(|kk| f64::from(soft[kk][j])).collect();
                let n: Vec<f64> = (0..k)
                    .map(|kk| f64::from(soft[kk][j] + soft[kk][g + j]))
                    .collect();
                let xs: Vec<f64> = (0..k).map(|kk| f64::from(xv[kk][j])).collect();
                wakefield_sign(&u, &n, &xs)
            })
            .collect()
    } else {
        Vec::new()
    };

    // Every gene's overall log U/S and largest |x| place its grid.
    let (mut tot_u, mut tot_s) = (vec![0f64; g], vec![0f64; g]);
    for row in &soft {
        for j in 0..g {
            tot_u[j] += f64::from(row[j]);
            tot_s[j] += f64::from(row[g + j]);
        }
    }
    let rest: Vec<f32> = (0..g)
        .map(|j| ((tot_u[j] + 0.5) / (tot_s[j] + 0.5)).ln() as f32)
        .collect();
    let reach: Vec<f32> = (0..g)
        .map(|j| {
            (0..k)
                .map(|kk| xv[kk][j].abs())
                .fold(1e-6f32, f32::max)
                .ln()
        })
        .collect();
    let unspliced_share = Tensor::from_vec(
        (0..g)
            .map(|j| (tot_u[j] / (tot_u[j] + tot_s[j]).max(1e-12)) as f32)
            .collect::<Vec<_>>(),
        (1, g),
        dev,
    )?;

    // Time draws: every pseudobulk at one grid time from its posterior, and
    // the same reads with no dynamics (every grid time at the gene's overall
    // unspliced share), whose `log BF` is the bias of the prior and the floor
    // on this curve, what the data's is read against.
    let qv = q.to_vec2::<f32>()?;
    let draws = cfg.direction_draws.max(1);
    let mut tables: Vec<(Tensor, Tensor)> = Vec::with_capacity(2 * draws);
    for _ in 0..draws {
        let mut hot = vec![0f32; p * k];
        for (r, row) in qv.iter().enumerate() {
            hot[r * k + crate::td_rbm::rbm::categorical(row, rng)] = 1.0;
        }
        let pooled = Tensor::from_vec(hot, (p, k), dev)?.t()?.matmul(&rows.x)?; // [K × F]
        let (u, s) = (pooled.narrow(1, 0, g)?, pooled.narrow(1, g, g)?);
        let n = (&u + &s)?;
        let u0 = n.broadcast_mul(&unspliced_share)?;
        let s0 = (&n - &u0)?;
        tables.push((u, s));
        tables.push((u0, s0));
    }
    // Scored on the host whatever the device: every gene on its own thread
    // (a grid of thousands of tiny per-gene sums is far slower as kernels).
    let host: Vec<HostTable> = tables
        .iter()
        .map(|(u, s)| Ok((u.to_vec2::<f32>()?, s.to_vec2::<f32>()?)))
        .collect::<Result<_>>()?;
    let (fwd, bwd) = evidence_host(&host, &xv, &rest, &reach);
    let share = 1.0 / draws as f64;
    let mut out = Bayes {
        lbf: Vec::with_capacity(draws),
        genes: vec![0.0; g],
        rates: (vec![0.0; g], vec![0.0; g]),
        z: wake.iter().map(|w| w.0).collect(),
        log_abf: wake.iter().map(|w| w.1).sum(),
    };
    for i in 0..draws {
        let (data, null) = (2 * i, 2 * i + 1);
        let mut total = 0f64;
        for j in 0..g {
            let v = (fwd[data].log_e[j] - bwd[data].log_e[j])
                - (fwd[null].log_e[j] - bwd[null].log_e[j]);
            out.genes[j] += v * share;
            total += v;
            out.rates.0[j] += fwd[data].log_beta[j] * share;
            out.rates.1[j] += fwd[data].log_gamma[j] * share;
        }
        out.lbf.push(total);
    }
    Ok(out)
}

/// Lineage `lin`'s splicing evidence read out ([`bayes`], with Wakefield's
/// check) on its finest pseudobulks `x`, the steady tables `t`; its
/// direction is its tree's. Returns the report and every gene's posterior
/// mean `log β`, `log γ` (`[1 × G]`) as it runs.
#[allow(clippy::too_many_arguments)]
pub(super) fn orient(
    energy: &Energy,
    spline: &Spline,
    t: &Tables,
    x: &[f32],
    f: usize,
    lin: usize,
    cfg: &TrainConfig,
    dev: &Device,
    rng: &mut StdRng,
) -> anyhow::Result<(Direction, (Tensor, Tensor))> {
    let g = f / 2;
    let b = bayes(energy, spline, t, x, f, lin, true, cfg, dev, rng)?;
    let row =
        |v: &[f64]| Tensor::from_vec(v.iter().map(|&x| x as f32).collect::<Vec<_>>(), (1, g), dev);
    let rates = (row(&b.rates.0)?, row(&b.rates.1)?);
    let d = Direction {
        lineage: lin,
        log_bf: b.log_bf(),
        log_bf_sd: b.log_bf_sd(),
        p_forward: b.p_forward(),
        log_abf: b.log_abf,
        gene_votes: b.genes.iter().map(|&v| v as f32).collect(),
        gene_z: b.z.iter().map(|&v| v as f32).collect(),
    };
    Ok((d, rates))
}

/// One table of pooled reads on the host: unspliced and spliced, `[K][G]`.
type HostTable = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Every gene's log evidence for one orientation, with its rates' posterior
/// means.
struct Evidence {
    log_e: Vec<f64>,
    log_beta: Vec<f64>,
    log_gamma: Vec<f64>,
}

/// The grid evidence (module docs) for every table of `tables` (`u`, `s`, each
/// `[K][G]`) at once, the velocity `x` (`[K][G]`) read forward and reversed:
/// every gene on its own thread, in f64 with exact log-gammas. The share,
/// `a`, `b` and `ln Γ(a) + ln Γ(b)` at every grid point are the tables'
/// common part, and `ln Γ(κ) − ln Γ(n + κ)` the signs', so each is worked
/// out once. Returns every table's evidence forward, then reversed.
fn evidence_host(
    tables: &[HostTable],
    x: &[Vec<f32>],
    rest: &[f32],
    reach: &[f32],
) -> (Vec<Evidence>, Vec<Evidence>) {
    let lgam = |z: f64| special::Gamma::ln_gamma(z).0;
    let k = x.len();
    let g = rest.len();
    let n_tab = tables.len();
    let (r_off, g_off, k_log) = grid();
    let n_grid = r_off.len() * g_off.len() * k_log.len();
    // Every gene: per sign, per table, (log evidence, mean log β, mean log γ).
    let per_gene: Vec<[Vec<(f64, f64, f64)>; 2]> = (0..g)
        .into_par_iter()
        .map(|j| {
            let uj: Vec<Vec<f64>> = tables
                .iter()
                .map(|(u, _)| (0..k).map(|kk| f64::from(u[kk][j])).collect())
                .collect();
            let sj: Vec<Vec<f64>> = tables
                .iter()
                .map(|(_, s)| (0..k).map(|kk| f64::from(s[kk][j])).collect())
                .collect();
            // ln Γ(κ) − ln Γ(n + κ) for every table and κ: no rate in it.
            let by_kappa: Vec<Vec<f64>> = (0..n_tab)
                .map(|t| {
                    k_log
                        .iter()
                        .map(|&lk| {
                            let kappa = lk.exp();
                            (0..k)
                                .map(|kk| lgam(kappa) - lgam(uj[t][kk] + sj[t][kk] + kappa))
                                .sum()
                        })
                        .collect()
                })
                .collect();
            [1.0, -1.0].map(|sign| {
                let xj: Vec<f64> = (0..k).map(|kk| sign * f64::from(x[kk][j])).collect();
                let mut ll = vec![Vec::with_capacity(n_grid); n_tab];
                let mut at = Vec::with_capacity(n_grid);
                for &ro in &r_off {
                    let lr = f64::from(rest[j]) + ro;
                    for &go in &g_off {
                        let lg = f64::from(reach[j]) + go;
                        let mu: Vec<f64> =
                            xj.iter().map(|&xv| unspliced_share(xv, lg, lr)).collect();
                        for (ik, &lk) in k_log.iter().enumerate() {
                            let kappa = lk.exp();
                            let ab: Vec<(f64, f64, f64)> = mu
                                .iter()
                                .map(|&m| {
                                    let (a, b) = (m * kappa, (1.0 - m) * kappa);
                                    (a, b, lgam(a) + lgam(b))
                                })
                                .collect();
                            for t in 0..n_tab {
                                let mut v = by_kappa[t][ik];
                                for (kk, &(a, b, base)) in ab.iter().enumerate() {
                                    v += lgam(uj[t][kk] + a) + lgam(sj[t][kk] + b) - base;
                                }
                                ll[t].push(v);
                            }
                            at.push((lg, lg - lr));
                        }
                    }
                }
                ll.iter()
                    .map(|ll| {
                        let log_e = log_sum_exp(ll) - (n_grid as f64).ln();
                        let w: Vec<f64> = ll.iter().map(|v| (v - log_e).exp()).collect();
                        let total: f64 = w.iter().sum();
                        let mg = w.iter().zip(&at).map(|(w, a)| w * a.0).sum::<f64>() / total;
                        let mb = w.iter().zip(&at).map(|(w, a)| w * a.1).sum::<f64>() / total;
                        (log_e, mb, mg)
                    })
                    .collect()
            })
        })
        .collect();
    let gather = |side: usize| -> Vec<Evidence> {
        (0..n_tab)
            .map(|t| Evidence {
                log_e: per_gene.iter().map(|v| v[side][t].0).collect(),
                log_beta: per_gene.iter().map(|v| v[side][t].1).collect(),
                log_gamma: per_gene.iter().map(|v| v[side][t].2).collect(),
            })
            .collect()
    };
    (gather(0), gather(1))
}

/// The evidence grid's offsets: the log rest ratio about the gene's log
/// `U/S`, `log γ` about its log largest `|x|`, and `log κ`.
fn grid() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        linspace(R_SPAN.0, R_SPAN.1, R_SPAN.2),
        linspace(GAMMA_SPAN.0.ln(), GAMMA_SPAN.1.ln(), GAMMA_SPAN.2),
        linspace(KAPPA_SPAN.0.ln(), KAPPA_SPAN.1.ln(), KAPPA_SPAN.2),
    )
}

/// The model's unspliced share at velocity `x` with `log γ = lg` and
/// `log γ/β = lr`: `σ(log m⁺ − log β)`, `m⁺` the softly floored `γ + x`
/// ([`FLOOR_SOFTNESS`]), kept inside `(10⁻⁶, 1 − 10⁻⁶)`.
fn unspliced_share(x: f64, lg: f64, lr: f64) -> f64 {
    let gamma = lg.exp();
    let floor = FLOOR_SOFTNESS * gamma;
    let z = (gamma + x) / floor;
    let sp = if z > 30.0 { z } else { z.exp().ln_1p() };
    let eta = (floor * sp).ln() - (lg - lr);
    (1.0 / (1.0 + (-eta).exp())).clamp(1e-6, 1.0 - 1e-6)
}

/// Wakefield's sign check for one gene: a quasi-binomial logistic fit of the
/// unspliced share `u/n` on the velocity `x` (standardised) over the grid
/// times; its slope's `z`, and `log Φ(z′) − log Φ(−z′)` with
/// `z′ = θ̂ √(W / (V (V + W)))`, `V` the slope's variance inflated by the
/// Pearson dispersion. `(0, 0)` when the gene has too little to fit.
fn wakefield_sign(u: &[f64], n: &[f64], x: &[f64]) -> (f64, f64) {
    let k = x.len() as f64;
    let mx = x.iter().sum::<f64>() / k;
    let sx = (x.iter().map(|v| (v - mx).powi(2)).sum::<f64>() / k).sqrt();
    let total: f64 = n.iter().sum();
    if sx < 1e-9 || total < 1.0 || x.len() < 3 {
        return (0.0, 0.0);
    }
    let xs: Vec<f64> = x.iter().map(|v| (v - mx) / sx).collect();
    let share = (u.iter().sum::<f64>() + 0.5) / (total + 1.0);
    let (mut a, mut b) = ((share / (1.0 - share)).ln(), 0.0);
    let mut info = [0.0; 3];
    for _ in 0..25 {
        let (mut g0, mut g1) = (0.0, 0.0);
        info = [0.0; 3];
        for i in 0..xs.len() {
            let mu = 1.0 / (1.0 + (-(a + b * xs[i])).exp());
            let w = n[i] * mu * (1.0 - mu);
            let r = u[i] - n[i] * mu;
            g0 += r;
            g1 += r * xs[i];
            info[0] += w;
            info[1] += w * xs[i];
            info[2] += w * xs[i] * xs[i];
        }
        let det = info[0] * info[2] - info[1] * info[1];
        if det <= 1e-12 {
            return (0.0, 0.0);
        }
        let (da, db) = (
            (info[2] * g0 - info[1] * g1) / det,
            (info[0] * g1 - info[1] * g0) / det,
        );
        a += da;
        b += db;
        if da.abs() + db.abs() < 1e-8 {
            break;
        }
    }
    let det = info[0] * info[2] - info[1] * info[1];
    let mut pearson = 0.0;
    for i in 0..xs.len() {
        let mu = 1.0 / (1.0 + (-(a + b * xs[i])).exp());
        let var = n[i] * mu * (1.0 - mu);
        if var > 1e-12 {
            pearson += (u[i] - n[i] * mu).powi(2) / var;
        }
    }
    let dispersion = (pearson / (k - 2.0)).max(1.0);
    let v = info[0] / det * dispersion;
    if !(v.is_finite() && v > 0.0) {
        return (0.0, 0.0);
    }
    let z = b / v.sqrt();
    let zp = b * (ABF_W / (v * (v + ABF_W))).sqrt();
    (z, ln_phi(zp) - ln_phi(-zp))
}

/// `log Φ(z)`, the standard normal's log cdf, stable in both tails.
fn ln_phi(z: f64) -> f64 {
    if z > -5.0 {
        (0.5 * special::Error::compl_error(-z / std::f64::consts::SQRT_2)).ln()
    } else {
        // Mills' ratio: Φ(z) ≈ φ(z)/|z| (1 − 1/z² + 3/z⁴).
        let z2 = z * z;
        -0.5 * z2 - (-z).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            + (1.0 - 1.0 / z2 + 3.0 / (z2 * z2)).ln()
    }
}

#[cfg(test)]
#[path = "../tests/direction.rs"]
mod tests;
