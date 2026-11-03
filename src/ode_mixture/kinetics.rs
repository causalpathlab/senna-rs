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
//! the last form finite at `β = γ`, where `E(h) = h e^{−γh}`.

/// `expm1(x)/x`, `1` at `x = 0`.
fn expm1_ratio(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        x.exp_m1() / x
    }
}

/// One gene's unspliced and spliced levels at every grid point, the grid
/// step `h`, transcription `alpha[k]` over step `k` (the last value only
/// sets nothing past the end), from steady state `α_0/β`, `α_0/γ`.
pub fn trajectory(alpha: &[f64], beta: f64, gamma: f64, h: f64) -> (Vec<f64>, Vec<f64>) {
    let n = alpha.len();
    let (mut u, mut s) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let Some(&a0) = alpha.first() else {
        return (u, s);
    };
    let (eb, eg) = ((-beta * h).exp(), (-gamma * h).exp());
    let e = h * eg * expm1_ratio((gamma - beta) * h);
    let (mut uk, mut sk) = (a0 / beta, a0 / gamma);
    u.push(uk);
    s.push(sk);
    for &a in &alpha[..n - 1] {
        let next_s = sk * eg + a / gamma * (1.0 - eg) + (beta * uk - a) * e;
        uk = a / beta + (uk - a / beta) * eb;
        sk = next_s;
        u.push(uk);
        s.push(sk);
    }
    (u, s)
}
