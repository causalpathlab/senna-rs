//! The time context of `senna tde`: which finest pseudobulks each phase-1
//! unit shares its time with.
//!
//! Every unit carries a distribution over time bins, the mix of its labelled
//! cells (`p_u`). Its observed context over the finest pseudobulks `v` is
//!
//! ```text
//! w_uv = Σ_t p_u(t) · p_v(t) · N_v        q_uv = w_uv / Σ_v w_uv
//! ```
//!
//! the chance the two sit in the same bin, in `v`'s reads. No width, no
//! weight: the bins and the counts set it.

use graph_embedding_util as ge;

/// Each unit's distribution over the time bins, `p_u`: the mix of its
/// labelled cells (all zero when it has none).
pub fn profiles(
    level: &[u8],
    source: &[u32],
    cell_to_pb_per_level: &[Vec<usize>],
    cell_bin: &[Option<u32>],
    n_bins: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let n_levels = cell_to_pb_per_level.len();
    anyhow::ensure!(n_levels > 0, "no pseudobulk level");
    anyhow::ensure!(level.len() == source.len(), "one level and source per unit");
    // Bin counts per pseudobulk, per level.
    let comp: Vec<Vec<Vec<f32>>> = cell_to_pb_per_level
        .iter()
        .map(|c2p| {
            anyhow::ensure!(c2p.len() == cell_bin.len(), "one time bin per cell");
            let n_pb = c2p.iter().max().map_or(0, |&m| m + 1);
            let mut comp = vec![vec![0f32; n_bins]; n_pb];
            for (&p, b) in c2p.iter().zip(cell_bin) {
                if let Some(t) = b {
                    comp[p][*t as usize] += 1.0;
                }
            }
            Ok(comp)
        })
        .collect::<anyhow::Result<_>>()?;
    Ok((0..level.len())
        .map(|u| {
            let (l, s) = (level[u] as usize, source[u] as usize);
            let mut p = if l < n_levels {
                comp[l].get(s).cloned().unwrap_or_else(|| vec![0.0; n_bins])
            } else {
                let mut p = vec![0f32; n_bins];
                if let Some(t) = cell_bin[s] {
                    p[t as usize] = 1.0;
                }
                p
            };
            let z: f32 = p.iter().sum();
            if z > 0.0 {
                p.iter_mut().for_each(|x| *x /= z);
            }
            p
        })
        .collect())
}

/// `q_uv ∝ Σ_t p_u(t) · p_v(t) · N_v` over the finest pseudobulks `v` (the
/// units at level `n_levels − 1`); an empty row for a unit with no time.
pub fn context_from_profiles(
    profile: &[Vec<f32>],
    level: &[u8],
    source: &[u32],
    n_levels: usize,
    total: &[f32],
) -> anyhow::Result<ge::UnitContext> {
    anyhow::ensure!(
        profile.len() == level.len() && total.len() == level.len(),
        "one profile and total per unit"
    );
    let finest = (n_levels - 1) as u8;
    let targets: Vec<usize> = (0..level.len()).filter(|&u| level[u] == finest).collect();
    let n_targets = targets
        .iter()
        .map(|&u| source[u] as usize + 1)
        .max()
        .unwrap_or(0);
    let n_bins = profile.first().map_or(0, Vec::len);
    // a_v(t) = p_v(t) · N_v.
    let mut a = vec![vec![0f32; n_bins]; n_targets];
    for &u in &targets {
        for (t, x) in profile[u].iter().enumerate() {
            a[source[u] as usize][t] = x * total[u];
        }
    }
    let rows = profile
        .iter()
        .map(|p| {
            let w: Vec<(u32, f32)> = a
                .iter()
                .enumerate()
                .map(|(v, av)| (v as u32, p.iter().zip(av).map(|(x, y)| x * y).sum::<f32>()))
                .filter(|&(_, w)| w > 0.0)
                .collect();
            let z: f32 = w.iter().map(|&(_, x)| x).sum();
            w.into_iter().map(|(v, x)| (v, x / z)).collect()
        })
        .collect();
    Ok(ge::UnitContext { n_targets, rows })
}

/// The kernel width for continuous time: the pooled within-pseudobulk
/// standard deviation of τ over the finest pseudobulks `cell_to_pb`, i.e. how
/// precisely a cell's expression neighbourhood pins its time.
pub fn kernel_width(cell_to_pb: &[usize], tau: &[Option<f32>]) -> anyhow::Result<f32> {
    anyhow::ensure!(cell_to_pb.len() == tau.len(), "one τ per cell");
    let n_pb = cell_to_pb.iter().max().map_or(0, |&m| m + 1);
    let (mut sum, mut n) = (vec![0f64; n_pb], vec![0usize; n_pb]);
    for (&p, t) in cell_to_pb.iter().zip(tau) {
        if let Some(t) = t {
            sum[p] += f64::from(*t);
            n[p] += 1;
        }
    }
    let ss: f64 = cell_to_pb
        .iter()
        .zip(tau)
        .filter_map(|(&p, t)| t.map(|t| (f64::from(t) - sum[p] / n[p] as f64).powi(2)))
        .sum();
    let dof = n.iter().sum::<usize>() as f64 - n.iter().filter(|&&k| k > 0).count() as f64;
    anyhow::ensure!(
        dof > 0.0 && ss > 0.0,
        "no spread of time within any pseudobulk"
    );
    Ok((ss / dof).sqrt() as f32)
}

/// The context for continuous time: a unit's time is the mean τ of its timed
/// cells, and `q_uv ∝ exp(−(τ_u − τ_v)²/2h²) · N_v` over the finest
/// pseudobulks, with `h` from [`kernel_width`]. Units are given as in
/// `time_context` (in the tests).
pub fn continuous_context(
    level: &[u8],
    source: &[u32],
    cell_to_pb_per_level: &[Vec<usize>],
    tau: &[Option<f32>],
    total: &[f32],
) -> anyhow::Result<ge::UnitContext> {
    let n_levels = cell_to_pb_per_level.len();
    anyhow::ensure!(n_levels > 0, "no pseudobulk level");
    let h = kernel_width(&cell_to_pb_per_level[n_levels - 1], tau)?;
    let mean_tau: Vec<Option<f32>> = unit_mean_tau(level, source, cell_to_pb_per_level, tau);
    let finest = (n_levels - 1) as u8;
    let targets: Vec<(usize, f32, f32)> = (0..level.len())
        .filter(|&u| level[u] == finest)
        .filter_map(|u| mean_tau[u].map(|t| (source[u] as usize, t, total[u])))
        .collect();
    let n_targets = cell_to_pb_per_level[n_levels - 1]
        .iter()
        .max()
        .map_or(0, |&m| m + 1);
    let rows = mean_tau
        .iter()
        .map(|t| {
            let Some(t) = t else { return Vec::new() };
            let w: Vec<(u32, f32)> = targets
                .iter()
                .map(|&(v, tv, n)| (v as u32, (-(t - tv).powi(2) / (2.0 * h * h)).exp() * n))
                .filter(|&(_, w)| w > 0.0)
                .collect();
            let z: f32 = w.iter().map(|&(_, x)| x).sum();
            w.into_iter().map(|(v, x)| (v, x / z)).collect()
        })
        .collect();
    Ok(ge::UnitContext { n_targets, rows })
}

/// Each unit's mean τ over its timed cells; `None` with none.
pub fn unit_mean_tau(
    level: &[u8],
    source: &[u32],
    cell_to_pb_per_level: &[Vec<usize>],
    tau: &[Option<f32>],
) -> Vec<Option<f32>> {
    let n_levels = cell_to_pb_per_level.len();
    let means: Vec<Vec<Option<f32>>> = cell_to_pb_per_level
        .iter()
        .map(|c2p| {
            let n_pb = c2p.iter().max().map_or(0, |&m| m + 1);
            let (mut s, mut n) = (vec![0f32; n_pb], vec![0f32; n_pb]);
            for (&p, t) in c2p.iter().zip(tau) {
                if let Some(t) = t {
                    s[p] += t;
                    n[p] += 1.0;
                }
            }
            s.iter()
                .zip(&n)
                .map(|(s, &n)| (n > 0.0).then(|| s / n))
                .collect()
        })
        .collect();
    (0..level.len())
        .map(|u| {
            let (l, i) = (level[u] as usize, source[u] as usize);
            if l < n_levels {
                means[l].get(i).copied().flatten()
            } else {
                tau[i]
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "tests/context.rs"]
mod tests;
