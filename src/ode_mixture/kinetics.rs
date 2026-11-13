//! The splicing ODE on a time grid, solved exactly.
//!
//! ```text
//! du/dt = α(t) − β u,    ds/dt = β u − γ s
//! ```
//!
//! from steady state at the first grid point, `α` held at `α_k` over the
//! step `[t_k, t_k + h)`. Over one step, exactly:
//!
//! ```text
//! u_{k+1} = α_k/β + (u_k − α_k/β) e^{−βh}
//! s_{k+1} = s_k e^{−γh} + (α_k/γ)(1 − e^{−γh}) + (β u_k − α_k) E(h)
//! E(h)    = (e^{−βh} − e^{−γh})/(γ − β) = h e^{−γh} expm1(dh)/(dh),  d = γ − β
//! ```
//!
//! the first form where `dh` is away from `0` (both exponentials at most 1,
//! so it cannot overflow however far apart the rates are), the second by
//! its series near `0`, where the first cancels (`E(h) = h e^{−γh}` at
//! `β = γ`).

/// Below this `|dh|`, `E(h)` by its series.
const SERIES: f64 = 1e-3;

/// `E(h)` of a step, from `e^{−βh}`, `e^{−γh}` and `dh`.
fn splice_factor(eb: f64, eg: f64, beta: f64, gamma: f64, delta: f64) -> f64 {
    let x = (gamma - beta) * delta;
    if x.abs() < SERIES {
        delta * eg * (1.0 + x / 2.0 + x * x / 6.0)
    } else {
        (eb - eg) / (gamma - beta)
    }
}

/// `(u, s)` a time `delta` after `(u0, s0)`, transcription held at `alpha`.
pub fn step(u0: f64, s0: f64, alpha: f64, beta: f64, gamma: f64, delta: f64) -> (f64, f64) {
    let (eb, eg) = ((-beta * delta).exp(), (-gamma * delta).exp());
    let e = splice_factor(eb, eg, beta, gamma, delta);
    let s = s0 * eg + alpha / gamma * (1.0 - eg) + (beta * u0 - alpha) * e;
    let u = alpha / beta + (u0 - alpha / beta) * eb;
    (u, s)
}

/// One gene's unspliced and spliced levels at every grid point, the grid
/// step `h`, transcription `alpha[k]` over step `k` (the last value only
/// sets nothing past the end), from steady state `α_0/β`, `α_0/γ`.
#[cfg(test)]
pub fn trajectory(alpha: &[f64], beta: f64, gamma: f64, h: f64) -> (Vec<f64>, Vec<f64>) {
    fine_trajectory(alpha, beta, gamma, h, 1)
}

/// [`trajectory`] at `m` evenly spaced points of every step: positions
/// `t_k + j h/m` (`j < m`) and the last node, `(n − 1) m + 1` of them, each
/// the exact partial step from its node (`α` held over the step). The
/// levels between nodes, so a pseudobulk's time need not sit on a node.
pub fn fine_trajectory(
    alpha: &[f64],
    beta: f64,
    gamma: f64,
    h: f64,
    m: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = alpha.len();
    let m = m.max(1);
    let (mut u, mut s) = (Vec::with_capacity(n.saturating_sub(1) * m + 1), Vec::new());
    let Some(&a0) = alpha.first() else {
        return (u, s);
    };
    let (mut uk, mut sk) = (a0 / beta, a0 / gamma);
    for &a in &alpha[..n - 1] {
        u.push(uk);
        s.push(sk);
        for j in 1..m {
            let (uj, sj) = step(uk, sk, a, beta, gamma, j as f64 * h / m as f64);
            u.push(uj);
            s.push(sj);
        }
        (uk, sk) = step(uk, sk, a, beta, gamma, h);
    }
    u.push(uk);
    s.push(sk);
    (u, s)
}
