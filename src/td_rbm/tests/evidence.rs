use super::*;
use legume_numeric::candle::candle_core::Device;

fn f64_ln_gamma(z: f64) -> f64 {
    special::Gamma::ln_gamma(z).0
}

fn grid() -> Vec<f64> {
    let mut v = Vec::new();
    let mut z = 1e-7;
    while z < 5000.0 {
        v.push(z);
        z *= 1.37;
    }
    v.extend([0.5, 1.0, 2.0, 7.99, 8.0, 8.01, 100.0]);
    v
}

#[test]
fn ln_gamma_matches_f64_everywhere_it_is_used() {
    let z = grid();
    let t = Tensor::from_vec(
        z.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        z.len(),
        &Device::Cpu,
    )
    .unwrap();
    let got = ln_gamma(&t).unwrap().to_vec1::<f32>().unwrap();
    for (&zi, &g) in z.iter().zip(&got) {
        let want = f64_ln_gamma(f64::from(zi as f32));
        let tol = 2e-6 * want.abs().max(1.0);
        assert!(
            (f64::from(g) - want).abs() < tol,
            "ln Γ({zi}) = {g}, want {want}"
        );
    }
}

/// The beta function to a small absolute error even where the log-gammas it
/// is made of are large: what the evidence sums over thousands of terms.
#[test]
fn ln_beta_matches_f64_to_a_small_absolute_error() {
    let z = grid();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for &a in &z {
        for &b in &z {
            xs.push(a as f32);
            ys.push(b as f32);
        }
    }
    let n = xs.len();
    let x = Tensor::from_vec(xs.clone(), n, &Device::Cpu).unwrap();
    let y = Tensor::from_vec(ys.clone(), n, &Device::Cpu).unwrap();
    let got = ln_beta(&x, &y).unwrap().to_vec1::<f32>().unwrap();
    let mut worst = 0f64;
    for i in 0..n {
        let (a, b) = (f64::from(xs[i]), f64::from(ys[i]));
        let want = f64_ln_gamma(a) + f64_ln_gamma(b) - f64_ln_gamma(a + b);
        // f32 can hold the answer itself only to its own relative precision.
        let err = (f64::from(got[i]) - want).abs() / want.abs().max(1.0);
        worst = worst.max(err);
        assert!(err < 2e-5, "ln B({a}, {b}) = {}, want {want}", got[i]);
    }
    eprintln!("worst relative error {worst:e}");
    let mut by_abs: Vec<(f64, f64, f64, f64)> = (0..n)
        .map(|i| {
            let (a, b) = (f64::from(xs[i]), f64::from(ys[i]));
            let want = f64_ln_gamma(a) + f64_ln_gamma(b) - f64_ln_gamma(a + b);
            ((f64::from(got[i]) - want).abs(), a, b, want)
        })
        .collect();
    by_abs.sort_by(|p, q| q.0.total_cmp(&p.0));
    for w in by_abs.iter().take(6) {
        eprintln!(
            "abs error {:.2e} at ({:.3e}, {:.3e}), value {:.1}",
            w.0, w.1, w.2, w.3
        );
    }
}
