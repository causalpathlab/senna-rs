use super::*;
use legume_numeric::candle::candle_core::Device;

const F: usize = 4;
const K: usize = 3;

/// A small machine with fixed tables and a one-dimensional state.
fn tables() -> Tables {
    let dev = Device::Cpu;
    let m = |v: &[f32], s: (usize, usize)| Tensor::from_slice(v, s, &dev).unwrap();
    let raw = m(
        &[
            0.5, -0.2, 0.1, 0.0, -0.3, 0.4, 0.2, -0.1, 0.1, 0.1, -0.5, 0.6,
        ],
        (K, F),
    );
    Tables {
        psi: legume_numeric::candle::candle_nn::ops::log_softmax(&raw, 1).unwrap(),
        bias: m(&[0.0; K], (1, K)),
        w: m(&[1.5, -2.0, -1.0, 0.5], (F, 1)),
        curve: m(&[0.0; K], (K, 1)),
        velocity: m(&[0.0; K], (K, 1)),
        size: m(&[0.0; K], (K, 1)),
        lineages: 1,
    }
}

fn reads(rows: &[Vec<f32>]) -> Reads {
    let x: Vec<f32> = rows.iter().flatten().copied().collect();
    Reads::from_features(Tensor::from_vec(x, (rows.len(), F), &Device::Cpu).unwrap()).unwrap()
}

/// `log Σ_k ∫ e^{−E(v,k,h)} dh` by quadrature over the one-dimensional state.
fn log_marginal(t: &Tables, v: &[f32]) -> f64 {
    let psi = t.psi.to_vec2::<f32>().unwrap();
    let w: Vec<f32> = t.w.flatten_all().unwrap().to_vec1().unwrap();
    let n: f64 = v.iter().map(|&x| f64::from(x)).sum();
    let drive: f64 = v.iter().zip(&w).map(|(x, y)| f64::from(x * y)).sum::<f64>() / n;
    let dh = 1e-3;
    let integral: f64 = (-12000..12000)
        .map(|i| {
            let h = i as f64 * dh;
            (drive * h - 0.5 * h * h).exp() * dh
        })
        .sum();
    let time: f64 = (0..K)
        .map(|k| {
            v.iter()
                .zip(&psi[k])
                .map(|(x, p)| f64::from(x * p))
                .sum::<f64>()
                .exp()
        })
        .sum();
    time.ln() + integral.ln()
}

#[test]
fn the_free_energy_and_posterior_are_the_exact_marginals() {
    let t = tables();
    let v = [3.0f32, 0.0, 5.0, 2.0];
    let r = reads(&[v.to_vec()]);
    let fe = f64::from(free_energy(&r, &t).unwrap().to_vec1::<f32>().unwrap()[0]);
    // F omits the Gaussian's constant log √(2π).
    let exact = -(log_marginal(&t, &v) - 0.5 * (2.0 * std::f64::consts::PI).ln());
    assert!((fe - exact).abs() < 1e-4, "{fe} vs {exact}");
    let (pk, mean) = posterior(&r, &t).unwrap();
    let pk = pk.to_vec2::<f32>().unwrap()[0].clone();
    let psi = t.psi.to_vec2::<f32>().unwrap();
    let logits: Vec<f32> = (0..K)
        .map(|k| v.iter().zip(&psi[k]).map(|(x, p)| x * p).sum())
        .collect();
    let z: f32 = logits.iter().map(|l| l.exp()).sum();
    for k in 0..K {
        assert!((pk[k] - logits[k].exp() / z).abs() < 1e-5);
    }
    let m = mean.to_vec2::<f32>().unwrap()[0][0];
    assert!(
        (m - (3.0 * 1.5 + -5.0 + 2.0 * 0.5) / 10.0).abs() < 1e-5,
        "{m}"
    );
}

/// Every read vector of `n` reads over `F` features.
fn compositions(n: usize) -> Vec<Vec<f32>> {
    let mut out = vec![];
    for a in 0..=n {
        for b in 0..=n - a {
            for c in 0..=n - a - b {
                out.push(vec![a as f32, b as f32, c as f32, (n - a - b - c) as f32]);
            }
        }
    }
    out
}

fn ln_factorial(n: f32) -> f64 {
    (1..=n as u32).map(|i| f64::from(i).ln()).sum()
}

#[test]
fn gibbs_chains_reach_the_machines_time_marginal() {
    // At depth n, p(v) ∝ (n! / Π v_f!) e^{−F(v)}: the time unit's marginal is
    // that average of p(k | v); compare the chains' with it.
    let t = tables();
    let n = 6usize;
    let all = compositions(n);
    let r = reads(&all);
    let fe = free_energy(&r, &t).unwrap().to_vec1::<f32>().unwrap();
    let (pk, _) = posterior(&r, &t).unwrap();
    let pk = pk.to_vec2::<f32>().unwrap();
    let logw: Vec<f64> = all
        .iter()
        .zip(&fe)
        .map(|(v, &f)| {
            ln_factorial(n as f32) - v.iter().map(|&x| ln_factorial(x)).sum::<f64>() - f64::from(f)
        })
        .collect();
    let top = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logw.iter().map(|l| (l - top).exp()).collect();
    let z: f64 = w.iter().sum();
    let mut exact = [0f64; K];
    for (wi, p) in w.iter().zip(&pk) {
        for k in 0..K {
            exact[k] += wi / z * f64::from(p[k]);
        }
    }
    let s = 2000;
    let start: Vec<f32> = (0..s).flat_map(|_| [n as f32, 0.0, 0.0, 0.0]).collect();
    let mut chains = Chains::new(start, F, 11);
    let mut seen = [0f64; K];
    for sweep in 0..60 {
        chains.sweep(&t).unwrap();
        if sweep >= 10 {
            let (p, _) = posterior(&chains.reads(&Device::Cpu).unwrap(), &t).unwrap();
            for row in p.to_vec2::<f32>().unwrap() {
                for k in 0..K {
                    seen[k] += f64::from(row[k]);
                }
            }
        }
    }
    let total: f64 = seen.iter().sum();
    for k in 0..K {
        let m = seen[k] / total;
        assert!(
            (m - exact[k]).abs() < 0.02,
            "time {k}: chains {m:.3}, exact {:.3}",
            exact[k]
        );
    }
}

#[test]
fn multinomial_draws_keep_the_depth() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut out = [0f32; 4];
    for n in [0u64, 1, 17, 1000] {
        multinomial(n, &[0.1, 0.2, 0.3, 0.4], &mut out, &mut rng);
        assert_eq!(out.iter().sum::<f32>() as u64, n);
    }
}

#[test]
#[ignore = "diagnostic"]
fn diagnostic_gibbs_with_and_without_state() {
    for scale in [0.0f32, 1.0] {
        let mut t = tables();
        t.w = (t.w * f64::from(scale)).unwrap();
        let n = 6usize;
        let all = compositions(n);
        let r = reads(&all);
        let fe = free_energy(&r, &t).unwrap().to_vec1::<f32>().unwrap();
        let (pk, _) = posterior(&r, &t).unwrap();
        let pk = pk.to_vec2::<f32>().unwrap();
        let logw: Vec<f64> = all
            .iter()
            .zip(&fe)
            .map(|(v, &f)| {
                ln_factorial(n as f32)
                    - v.iter().map(|&x| ln_factorial(x)).sum::<f64>()
                    - f64::from(f)
            })
            .collect();
        let top = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = logw.iter().map(|l| (l - top).exp()).collect();
        let z: f64 = w.iter().sum();
        let mut exact = [0f64; K];
        for (wi, p) in w.iter().zip(&pk) {
            for k in 0..K {
                exact[k] += wi / z * f64::from(p[k]);
            }
        }
        let start: Vec<f32> = (0..2000).flat_map(|_| [n as f32, 0.0, 0.0, 0.0]).collect();
        let mut chains = Chains::new(start, F, 11);
        let mut seen = [0f64; K];
        for sweep in 0..1000 {
            chains.sweep(&t).unwrap();
            if sweep >= 200 {
                let (p, _) = posterior(&chains.reads(&Device::Cpu).unwrap(), &t).unwrap();
                for row in p.to_vec2::<f32>().unwrap() {
                    for k in 0..K {
                        seen[k] += f64::from(row[k]);
                    }
                }
            }
        }
        let tot: f64 = seen.iter().sum();
        eprintln!(
            "scale {scale}: exact {:?} chains {:?}",
            exact,
            seen.map(|s| s / tot)
        );
    }
}

/// `−E(v, k, h)` for one read vector and a one-dimensional state.
fn neg_energy(t: &Tables, v: &[f32], k: usize, h: f64) -> f64 {
    let psi = t.psi.to_vec2::<f32>().unwrap();
    let w: Vec<f32> = t.w.flatten_all().unwrap().to_vec1().unwrap();
    let n: f64 = v.iter().map(|&x| f64::from(x)).sum();
    let time: f64 = v.iter().zip(&psi[k]).map(|(x, p)| f64::from(x * p)).sum();
    let state: f64 = v.iter().zip(&w).map(|(x, y)| f64::from(x * y)).sum::<f64>() * h / n;
    time + state - 0.5 * h * h
}

#[test]
fn reads_given_time_and_state_are_the_samplers_multinomial() {
    // p(v | k, h) ∝ (n!/Π v_f!) e^{−E} over every read vector of depth n
    // equals Multinomial(n, softmax(ψ_k + W h / n)).
    let t = tables();
    let n = 5usize;
    let all = compositions(n);
    let psi = t.psi.to_vec2::<f32>().unwrap();
    let w: Vec<f32> = t.w.flatten_all().unwrap().to_vec1().unwrap();
    for (k, h) in [(0usize, -1.3f64), (2, 0.7)] {
        let logw: Vec<f64> = all
            .iter()
            .map(|v| {
                ln_factorial(n as f32) - v.iter().map(|&x| ln_factorial(x)).sum::<f64>()
                    + neg_energy(&t, v, k, h)
            })
            .collect();
        let z: f64 = logw.iter().map(|l| l.exp()).sum();
        let logits: Vec<f64> = (0..F)
            .map(|f| f64::from(psi[k][f]) + f64::from(w[f]) * h / n as f64)
            .collect();
        let lz = logits.iter().map(|l| l.exp()).sum::<f64>().ln();
        for (v, lw) in all.iter().zip(&logw) {
            let pmf = (ln_factorial(n as f32) - v.iter().map(|&x| ln_factorial(x)).sum::<f64>()
                + v.iter()
                    .zip(&logits)
                    .map(|(x, l)| f64::from(*x) * (l - lz))
                    .sum::<f64>())
            .exp();
            assert!(
                (lw.exp() / z - pmf).abs() < 1e-6 * pmf,
                "{v:?}: {} vs {pmf}",
                lw.exp() / z
            );
        }
    }
}

#[test]
fn time_and_state_given_reads_factorise_into_the_closed_forms() {
    // p(k, h | v) ∝ e^{−E(v,k,h)} on a grid of h: its time marginal, the
    // state's mean and variance, and the factorisation.
    let t = tables();
    let v = [3.0f32, 0.0, 5.0, 2.0];
    let dh = 1e-3;
    let hs: Vec<f64> = (-10000..10000).map(|i| i as f64 * dh).collect();
    let joint: Vec<Vec<f64>> = (0..K)
        .map(|k| {
            hs.iter()
                .map(|&h| neg_energy(&t, &v, k, h).exp() * dh)
                .collect()
        })
        .collect();
    let z: f64 = joint.iter().flatten().sum();
    let (pk, mean) = posterior(&reads(&[v.to_vec()]), &t).unwrap();
    let pk = pk.to_vec2::<f32>().unwrap()[0].clone();
    let mean = f64::from(mean.to_vec2::<f32>().unwrap()[0][0]);
    for k in 0..K {
        let mk: f64 = joint[k].iter().sum::<f64>() / z;
        assert!(
            (mk - f64::from(pk[k])).abs() < 1e-5,
            "p(k={k}|v) {mk} vs {}",
            pk[k]
        );
        // Given k too, the state is the same Gaussian: time and state are
        // independent given the reads.
        let zk: f64 = joint[k].iter().sum();
        let m: f64 = joint[k].iter().zip(&hs).map(|(p, h)| p * h).sum::<f64>() / zk;
        let var: f64 = joint[k]
            .iter()
            .zip(&hs)
            .map(|(p, h)| p * (h - m).powi(2))
            .sum::<f64>()
            / zk;
        assert!(
            (m - mean).abs() < 1e-4 && (var - 1.0).abs() < 1e-3,
            "k={k}: mean {m} vs {mean}, var {var}"
        );
    }
}

/// What the machine believes about time before it sees any read, at depth
/// `n`: `p(k | N = n) ∝ ∫ e^{−½h² + n·LSE_f(ψ_k + W h/n)} dh`. Printed, not
/// asserted: the state makes it uneven over the grid.
#[test]
#[ignore = "diagnostic"]
fn diagnostic_the_time_prior_at_a_depth() {
    let t = tables();
    let psi = t.psi.to_vec2::<f32>().unwrap();
    let w: Vec<f32> = t.w.flatten_all().unwrap().to_vec1().unwrap();
    for n in [5.0f64, 1000.0] {
        let dh = 1e-3;
        let prior: Vec<f64> = (0..K)
            .map(|k| {
                (-10000..10000)
                    .map(|i| {
                        let h = i as f64 * dh;
                        let lse = (0..F)
                            .map(|f| (f64::from(psi[k][f]) + f64::from(w[f]) * h / n).exp())
                            .sum::<f64>()
                            .ln();
                        (-0.5 * h * h + n * lse).exp() * dh
                    })
                    .sum()
            })
            .collect();
        let z: f64 = prior.iter().sum();
        let approx: Vec<f64> = (0..K)
            .map(|k| {
                let m: f64 = (0..F)
                    .map(|f| f64::from(psi[k][f]).exp() * f64::from(w[f]))
                    .sum();
                (0.5 * m * m).exp()
            })
            .collect();
        let za: f64 = approx.iter().sum();
        eprintln!(
            "depth {n}: p(k) {:?}; ½‖Wᵀπ_k‖² approximation {:?}",
            prior.iter().map(|p| p / z).collect::<Vec<_>>(),
            approx.iter().map(|p| p / za).collect::<Vec<_>>()
        );
    }
}
