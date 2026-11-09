//! Symmetric precisions, banded, and their guarded Cholesky factors: a
//! gene's conditional's negative Hessian in its coordinates (dense, so the
//! band is full) and the prior's diagonal.
//!
//! Factoring is guarded ([`Factor::new`]): the matrix Jacobi-scaled
//! (`D^{−½} P D^{−½}`, `D = |diag P|`), a ridge `λ I` added on failure
//! (from `1e-8`, ×10 up to [`MAX_RIDGE`]), the fallback's taken last; it
//! never panics.

use std::sync::atomic::{AtomicBool, Ordering};

/// The largest ridge on a Jacobi-scaled matrix (unit diagonal in size)
/// before the fallback is taken: past it the factor is the scaled
/// diagonal's, near enough.
const MAX_RIDGE: f64 = 1e2;

/// A symmetric `n × n` matrix of half-bandwidth `p`, its lower band by
/// rows, `(i, j)`, `j ≤ i ≤ j + p`, at `i (p + 1) + i − j`.
#[derive(Clone, Debug)]
pub struct Band {
    n: usize,
    p: usize,
    a: Vec<f64>,
}

impl Band {
    pub fn zeros(n: usize, p: usize) -> Self {
        Self {
            n,
            p,
            a: vec![0.0; n * (p + 1)],
        }
    }

    /// A small dense symmetric `m` as a full band (`p = n − 1`),
    /// symmetrized: `(m + mᵀ)/2`.
    pub fn from_dense(m: &nalgebra::DMatrix<f64>) -> Self {
        let n = m.nrows();
        let mut b = Self::zeros(n, n.saturating_sub(1));
        for i in 0..n {
            for j in 0..=i {
                b.add(i, j, 0.5 * (m[(i, j)] + m[(j, i)]));
            }
        }
        b
    }

    /// A diagonal `d` as a band of width zero.
    pub fn diagonal_of(d: &[f64]) -> Self {
        Self {
            n: d.len(),
            p: 0,
            a: d.to_vec(),
        }
    }

    fn at(&self, i: usize, j: usize) -> Option<usize> {
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        (i - j <= self.p).then(|| i * (self.p + 1) + i - j)
    }

    /// Adds `v` at `(i, j)` (and so at `(j, i)`); off the band, nothing.
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        if let Some(k) = self.at(i, j) {
            self.a[k] += v;
        }
    }

    fn diagonal(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.a[i * (self.p + 1)]).collect()
    }

    /// The plain Cholesky factor `L` (same band), `None` on a pivot that is
    /// not positive and finite.
    fn cholesky(&self) -> Option<Band> {
        let (n, p) = (self.n, self.p);
        let mut l = self.clone();
        for j in 0..n {
            let lo = j.saturating_sub(p);
            let mut d = l.a[j * (p + 1)];
            for k in lo..j {
                d -= l.a[j * (p + 1) + j - k].powi(2);
            }
            if !(d > 0.0 && d.is_finite()) {
                return None;
            }
            let d = d.sqrt();
            l.a[j * (p + 1)] = d;
            for i in j + 1..(j + p + 1).min(n) {
                let mut v = l.a[i * (p + 1) + i - j];
                for k in i.saturating_sub(p).max(lo)..j {
                    v -= l.a[i * (p + 1) + i - k] * l.a[j * (p + 1) + j - k];
                }
                l.a[i * (p + 1) + i - j] = v / d;
            }
        }
        Some(l)
    }

    /// `L⁻¹ b`, `self` a lower factor.
    fn lower_solve(&self, b: &[f64]) -> Vec<f64> {
        let (n, p) = (self.n, self.p);
        let mut x = b.to_vec();
        for i in 0..n {
            for k in i.saturating_sub(p)..i {
                x[i] -= self.a[i * (p + 1) + i - k] * x[k];
            }
            x[i] /= self.a[i * (p + 1)];
        }
        x
    }

    /// `L⁻ᵀ b`, `self` a lower factor.
    fn upper_solve(&self, b: &[f64]) -> Vec<f64> {
        let (n, p) = (self.n, self.p);
        let mut x = b.to_vec();
        for i in (0..n).rev() {
            for k in i + 1..(i + p + 1).min(n) {
                x[i] -= self.a[k * (p + 1) + k - i] * x[k];
            }
            x[i] /= self.a[i * (p + 1)];
        }
        x
    }
}

/// Logs once a process, at debug, each kind of guard that acted: a ridge,
/// a fallback, a branch time's coupling dropped.
fn note(seen: &AtomicBool, what: &str) {
    if !seen.swap(true, Ordering::Relaxed) {
        log::debug!("ode-mixture: {what}");
    }
}

static RIDGED: AtomicBool = AtomicBool::new(false);
static FELL_BACK: AtomicBool = AtomicBool::new(false);

/// The guards' ridge: `factor(ridge)` at none, then (when `ridged`) from
/// `1e-8` up by tens to [`MAX_RIDGE`], the first that factors.
fn with_ridge<T>(ridged: bool, factor: impl Fn(f64) -> Option<T>) -> Option<T> {
    let mut ridge = 0.0;
    loop {
        if let Some(f) = factor(ridge) {
            if ridge > 0.0 {
                note(&RIDGED, "a proposal precision needed a ridge to factor");
            }
            return Some(f);
        }
        if !ridged || ridge >= MAX_RIDGE {
            return None;
        }
        ridge = if ridge == 0.0 { 1e-8 } else { ridge * 10.0 };
    }
}

/// `P = D^{½} L Lᵀ D^{½}`, `L` the Cholesky factor of the Jacobi-scaled
/// `P` (plus the ridge it needed), `D` its diagonal.
#[derive(Clone, Debug)]
pub struct Factor {
    l: Band,
    /// `D^{½}`.
    root: Vec<f64>,
}

impl Factor {
    /// `p` factored, guarded as the module says; `fallback` (itself
    /// factored plainly, then as the identity) where `p` will not.
    pub fn new(p: &Band, fallback: impl Fn() -> Band) -> Self {
        Self::scaled(p, true).unwrap_or_else(|| {
            note(&FELL_BACK, "a proposal precision fell back to its prior's");
            let f = fallback();
            Self::scaled(&f, false).unwrap_or_else(|| Self {
                l: Band::diagonal_of(&vec![1.0; f.n]),
                root: vec![1.0; f.n],
            })
        })
    }

    fn scaled(p: &Band, ridged: bool) -> Option<Self> {
        if !p.a.iter().all(|v| v.is_finite()) {
            return None;
        }
        let root: Vec<f64> = p.diagonal().iter().map(|&d| scale_of(d)).collect();
        let mut s = p.clone();
        for i in 0..p.n {
            for j in i.saturating_sub(p.p)..=i {
                s.a[i * (p.p + 1) + i - j] /= root[i] * root[j];
            }
        }
        let l = with_ridge(ridged, |ridge| {
            let mut t = s.clone();
            for i in 0..p.n {
                t.a[i * (p.p + 1)] += ridge;
            }
            t.cholesky()
        })?;
        Some(Self { l, root })
    }

    /// `P⁻¹ b`.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        self.upper_solve(&self.lower_solve(b))
    }

    /// `F⁻¹ b`, `F = D^{½} L`.
    fn lower_solve(&self, b: &[f64]) -> Vec<f64> {
        let y: Vec<f64> = b.iter().zip(&self.root).map(|(b, r)| b / r).collect();
        self.l.lower_solve(&y)
    }

    /// `F⁻ᵀ b`.
    fn upper_solve(&self, b: &[f64]) -> Vec<f64> {
        self.l
            .upper_solve(b)
            .iter()
            .zip(&self.root)
            .map(|(x, r)| x / r)
            .collect()
    }
}

/// A Jacobi scale from a diagonal entry: `√|d|`, one where that is zero
/// (an indefinite matrix's diagonal may not be positive; the ridge then
/// does the rest).
fn scale_of(d: f64) -> f64 {
    let r = d.abs().sqrt();
    if r > 0.0 {
        r
    } else {
        1.0
    }
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
