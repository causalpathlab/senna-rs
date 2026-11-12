#![allow(
    clippy::items_after_statements,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

use legume_numeric::candle::candle_core::*;

/// t-SNE with an Rtsne-aligned update rule: per-parameter adaptive gains
/// (Jacobs 1988), momentum switching, strong early exaggeration, and init
/// rescaling to std = 1e-4, on a graph's affinities. Also supports a
/// densMAP-style density-preservation auxiliary loss.
/// Built on candle for automatic differentiation.
///
/// References:
/// - van der Maaten & Hinton, *JMLR* 2008 — t-SNE.
/// - Jacobs, *Neural Networks* 1988 — increased rate of convergence through
///   learning rate adaptation (delta-bar-delta / adaptive gains).
/// - Narayan, Berger & Cho, *Nat Biotechnol* 2021 — densMAP / den-SNE
///   density-preserving extension.
pub struct TSne {
    learning_rate: f32,
    initial_momentum: f32,
    final_momentum: f32,
    n_iter: usize,
    early_exaggeration: f32,
    early_exaggeration_iter: usize,
    /// Optional per-point weights (PB cell counts) for the densMAP target.
    weights: Option<Vec<f32>>,
    /// densMAP-style density-preservation strength. When `> 0` and
    /// `weights` is set, an auxiliary loss pins each point's soft low-D
    /// local radius `R_i = Σ_j P_ij · d_ij(Y)` to `∝ 1/√n_i`, so dense
    /// regions (big PBs) shrink and sparse regions (small PBs) spread.
    density_lambda: f32,
}

impl Default for TSne {
    fn default() -> Self {
        Self {
            learning_rate: 200.0,
            initial_momentum: 0.5,
            final_momentum: 0.8,
            n_iter: 1000,
            early_exaggeration: 12.0,
            early_exaggeration_iter: 250,
            weights: None,
            density_lambda: 0.0,
        }
    }
}

impl TSne {
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Attach per-point weights (PB cell counts), the density target of
    /// [`TSne::density_lambda`]. A graph's summed edge weights already grow
    /// with its nodes' sizes, so P itself is not weighted again.
    pub fn weights(mut self, w: &[usize]) -> Self {
        self.weights = Some(w.iter().map(|&x| x as f32).collect());
        self
    }

    /// Enable densMAP-style density preservation with strength `lambda`.
    /// Requires `weights` to be set — the density target is `1/√w_i`.
    /// A good starting value is ~0.1. Set to 0 to disable.
    pub fn density_lambda(mut self, lambda: f32) -> Self {
        self.density_lambda = lambda.max(0.0);
        self
    }

    /// t-SNE of the `n` nodes of the undirected graph `edges`
    /// (`(i, j, weight)`), from `init` (`n × 2`, row-major). Each node's edge
    /// weights, normalized, are its `P(j|i)`: the graph is taken as it is
    /// rather than through a perplexity-calibrated Gaussian of distances,
    /// which on a kNN graph has too little contrast between linked and
    /// unlinked nodes to keep groups apart. Returns a flat n×2 embedding.
    ///
    /// The update rule follows Rtsne / van der Maaten's reference
    /// implementation: adaptive per-parameter gains, momentum switch at
    /// `early_exaggeration_iter`, init rescaled to std = 1e-4 so early
    /// exaggeration has room to separate clusters, and Y centered every
    /// iteration to prevent drift.
    pub fn fit_graph(
        &self,
        n: usize,
        edges: &[(usize, usize, f32)],
        init: &[f32],
    ) -> Result<Vec<f32>> {
        let device = Device::Cpu;

        let p_tensor = Tensor::from_vec(joint_probabilities(n, edges), (n, n), &device)?;

        // Pre-build the centered log-target tensor for density preservation:
        //   target_i ∝ 1/√w_i   =>   log target_i = -½ log w_i
        let density_target = if self.density_lambda > 0.0 {
            self.weights.as_ref().and_then(|w| {
                if w.len() != n {
                    return None;
                }
                let mut log_t: Vec<f32> = w.iter().map(|&wi| -0.5 * wi.max(1.0).ln()).collect();
                let mean = log_t.iter().sum::<f32>() / n as f32;
                for v in &mut log_t {
                    *v -= mean;
                }
                Tensor::from_vec(log_t, (n,), &device).ok()
            })
        } else {
            None
        };

        let mut y_flat: Vec<f32> = init.to_vec();

        // Rescale the init to std = 1e-4 (Rtsne / openTSNE convention).
        // Keeps the *shape* of the init while shrinking the scale so that
        // early exaggeration has room to push clusters apart.
        {
            let mean_x: f32 = (0..n).map(|i| y_flat[i * 2]).sum::<f32>() / n as f32;
            let mean_y: f32 = (0..n).map(|i| y_flat[i * 2 + 1]).sum::<f32>() / n as f32;
            for i in 0..n {
                y_flat[i * 2] -= mean_x;
                y_flat[i * 2 + 1] -= mean_y;
            }
            let var: f32 =
                y_flat.iter().map(|v| v * v).sum::<f32>() / (y_flat.len() as f32).max(1.0);
            let std = var.sqrt().max(1e-12);
            let scale = 1e-4 / std;
            for v in &mut y_flat {
                *v *= scale;
            }
        }

        let y = Var::from_tensor(&Tensor::from_vec(y_flat, (n, 2), &device)?)?;

        // Auto-scale learning rate with n (Kobak & Berens 2019). The fixed
        // lr=200 Rtsne convention diverges for small n: early-exaggeration
        // gradients grow faster than the Student-t tail can damp them,
        // blowing Y up to overflow within ~200 iters at n ≈ 70.
        let lr_auto = (n as f32 / self.early_exaggeration).max(50.0);
        let learning_rate = self.learning_rate.min(lr_auto);

        // Per-parameter velocity and adaptive gains (Jacobs 1988).
        let mut velocity: Vec<f32> = vec![0.0; n * 2];
        let mut gains: Vec<f32> = vec![1.0; n * 2];
        const MIN_GAIN: f32 = 0.01;

        for iter in 0..self.n_iter {
            let p_mult = if iter < self.early_exaggeration_iter {
                self.early_exaggeration
            } else {
                1.0
            };
            let p_scaled = (&p_tensor * f64::from(p_mult))?;

            // KL(+ density). Density term disabled during early exaggeration
            // so gradients through log R don't blow up at tiny distances.
            let kl = self.kl_divergence(&y, &p_scaled, n)?;
            let loss = if iter >= self.early_exaggeration_iter {
                if let Some(log_target) = &density_target {
                    let dens = self.density_loss(&y, &p_tensor, log_target, n)?;
                    (&kl + (dens * f64::from(self.density_lambda))?)?
                } else {
                    kl
                }
            } else {
                kl
            };

            let grad = loss.backward()?;
            let y_grad = grad.get(&y).unwrap();
            let g_flat: Vec<f32> = y_grad.flatten_all()?.to_vec1::<f32>()?;

            // Adaptive gains (van der Maaten reference):
            //   sign(grad) ≠ sign(velocity) (oscillating) → gain += 0.2
            //   sign(grad) == sign(velocity) (steady)      → gain *= 0.8
            for i in 0..gains.len() {
                if (velocity[i] > 0.0) == (g_flat[i] > 0.0) {
                    gains[i] *= 0.8;
                } else {
                    gains[i] += 0.2;
                }
                if gains[i] < MIN_GAIN {
                    gains[i] = MIN_GAIN;
                }
            }

            // Momentum switch at the end of early exaggeration.
            let momentum = if iter < self.early_exaggeration_iter {
                self.initial_momentum
            } else {
                self.final_momentum
            };

            // v ← momentum · v − lr · gains · grad
            for i in 0..velocity.len() {
                velocity[i] = momentum * velocity[i] - learning_rate * gains[i] * g_flat[i];
            }

            // Y ← Y + v, then center so the mean is at the origin.
            let y_vec: Vec<f32> = y.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            let mut y_new: Vec<f32> = y_vec
                .iter()
                .zip(velocity.iter())
                .map(|(yv, vv)| yv + vv)
                .collect();
            let mean_x: f32 = (0..n).map(|i| y_new[i * 2]).sum::<f32>() / n as f32;
            let mean_y: f32 = (0..n).map(|i| y_new[i * 2 + 1]).sum::<f32>() / n as f32;
            for i in 0..n {
                y_new[i * 2] -= mean_x;
                y_new[i * 2 + 1] -= mean_y;
            }
            y.set(&Tensor::from_vec(y_new, (n, 2), &device)?)?;
        }

        let result = y.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        Ok(result)
    }

    /// Compute KL divergence between P and Q (Student-t in low-dim)
    fn kl_divergence(&self, y: &Var, p: &Tensor, n: usize) -> Result<Tensor> {
        // Compute pairwise squared distances in low-dim
        // ||y_i - y_j||^2 = ||y_i||^2 + ||y_j||^2 - 2 * y_i · y_j
        let y_sq = y.as_tensor().sqr()?.sum(D::Minus1)?; // (n,)
        let y_sq_row = y_sq.reshape((n, 1))?; // (n, 1)
        let y_sq_col = y_sq.reshape((1, n))?; // (1, n)
        let dot = y.as_tensor().matmul(&y.as_tensor().t()?)?; // (n, n)
        let dist_sq = (y_sq_row.broadcast_add(&y_sq_col)? - (&dot * 2.0)?)?;

        // Student-t: q_ij = (1 + ||y_i - y_j||^2)^-1
        let q_unnorm = (&dist_sq + 1.0)?.recip()?;

        // Zero out diagonal
        let mask = Tensor::eye(n, DType::F32, y.as_tensor().device())?;
        let q_unnorm = (&q_unnorm * (1.0 - &mask)?)?;

        // Normalize
        let q_sum = q_unnorm.sum_all()?;
        let q = q_unnorm.broadcast_div(&q_sum)?.clamp(1e-12, 1.0)?;

        // KL divergence: sum(P * log(P/Q))
        let log_pq = p.broadcast_div(&q)?.log()?;
        let kl = (p * log_pq)?.sum_all()?;

        Ok(kl)
    }

    /// densMAP-style density-preservation loss (Narayan, Berger & Cho,
    /// *Nat Biotechnol* 2021). For each point i, compute a soft local radius
    /// in 2D as a P-weighted mean distance,
    ///     `R_i` = `Σ_j` `P_ij` · `d_ij(Y)`,
    /// and penalize the centered log-radius against a centered target
    /// `log_target_i = -½ log(w_i)` (i.e. target radius ∝ `1/√w_i`, so dense
    /// high-D regions stay dense in 2D). Both sides are mean-centered, so no
    /// absolute scale offset needs to be fit.
    fn density_loss(
        &self,
        y: &Var,
        p: &Tensor,
        log_target_centered: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        // Pairwise Euclidean distances in 2D.
        let y_sq = y.as_tensor().sqr()?.sum(D::Minus1)?;
        let y_sq_row = y_sq.reshape((n, 1))?;
        let y_sq_col = y_sq.reshape((1, n))?;
        let dot = y.as_tensor().matmul(&y.as_tensor().t()?)?;
        let dist_sq = (y_sq_row.broadcast_add(&y_sq_col)? - (&dot * 2.0)?)?;
        let dist = dist_sq.clamp(1e-12f32, f32::INFINITY)?.sqrt()?;

        // R_i = Σ_j P_ij · d_ij, row-wise weighted sum of 2D distances.
        // Floor with 1e-3 so the gradient of log R stays bounded when the
        // layout is still collapsed near the origin.
        let r = (p * &dist)?.sum(D::Minus1)?;
        let log_r = r.clamp(1e-3f32, f32::INFINITY)?.log()?;

        // Center log R so only the pattern (not the absolute scale) is fit.
        let mean_log_r = log_r.mean_all()?;
        let log_r_centered = log_r.broadcast_sub(&mean_log_r)?;

        // Loss = mean squared deviation between centered low-D log-radii
        // and centered target.
        let diff = (&log_r_centered - log_target_centered)?;
        let loss = diff.sqr()?.mean_all()?;
        Ok(loss)
    }
}

/// The joint `P` (row-major `n × n`) of the graph `edges`:
/// `P_ij = (P(j|i) + P(i|j)) / 2n` with `P(j|i) = w_ij / Σ_k w_ik`, floored
/// at 1e-12 for numerical stability.
fn joint_probabilities(n: usize, edges: &[(usize, usize, f32)]) -> Vec<f32> {
    let mut total = vec![0.0_f32; n];
    for &(a, b, w) in edges {
        if a != b {
            total[a] += w;
            total[b] += w;
        }
    }
    let mut p = vec![1e-12_f32; n * n];
    let half = 1.0 / (2.0 * n as f32);
    for &(a, b, w) in edges {
        if a != b {
            let v = half * (w / total[a] + w / total[b]);
            p[a * n + b] += v;
            p[b * n + a] += v;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// t-SNE on a graph's affinities: three groups of 100, each node linked
    /// to its 16 nearest along a ring (as a kNN graph of PBs is), joined by a
    /// few weak links, stay apart.
    #[test]
    fn three_linked_groups_on_a_graph_stay_apart() {
        let (per, n) = (100, 300);
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        for g in 0..3 {
            for k in 0..per {
                for s in 1..=8 {
                    edges.push((g * per + k, g * per + (k + s) % per, 4.0));
                }
            }
        }
        for g in 0..3 {
            edges.push((g * per, ((g + 1) % 3) * per + 5, 0.2));
        }
        let init = legume_numeric::matrix::pca::random_init_2d(n, 1);
        let y = TSne::default()
            .n_iter(500)
            .fit_graph(n, &edges, &init)
            .unwrap();
        let centre = |g: usize| {
            (0..per).fold([0.0_f32; 2], |[a, b], k| {
                let i = g * per + k;
                [a + y[2 * i] / per as f32, b + y[2 * i + 1] / per as f32]
            })
        };
        let c: Vec<[f32; 2]> = (0..3).map(centre).collect();
        let spread = (0..n)
            .map(|i| {
                let g = &c[i / per];
                (y[2 * i] - g[0]).hypot(y[2 * i + 1] - g[1])
            })
            .sum::<f32>()
            / n as f32;
        for a in 0..3 {
            for b in 0..a {
                let apart = (c[a][0] - c[b][0]).hypot(c[a][1] - c[b][1]);
                assert!(
                    apart > 2.0 * spread,
                    "groups {a}, {b}: {apart} apart, spread {spread}"
                );
            }
        }
    }
}
