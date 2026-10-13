// Blocks are ranges; one block is a one-range list.
#![allow(clippy::single_range_in_vec_init)]

use super::*;
use legume_numeric::candle::candle_core::Var;
use rand::SeedableRng;

const K: usize = 2;
const N: usize = 3;

fn kinds() -> [Hidden; 4] {
    [
        Hidden::Bernoulli,
        Hidden::Binomial { trials: 2 },
        Hidden::Poisson { max: 3 },
        Hidden::Gaussian,
    ]
}

/// One block of three features, or two blocks of two.
fn machine(hidden: Hidden, ranges: Vec<Range<usize>>) -> Machine {
    let dev = Device::Cpu;
    let f = ranges.last().unwrap().end;
    let wv: Vec<f32> = (0..f * K).map(|i| 0.4 * ((i as f32 * 1.7).sin())).collect();
    let bv: Vec<f32> = (0..f).map(|i| 0.3 * (i as f32 - 1.0)).collect();
    let depth = vec![N as f32; ranges.len()];
    let mu: Vec<f32> = (0..f).map(|i| 0.5 + 0.2 * i as f32).collect();
    Machine::new(
        Tensor::from_vec(bv, (1, f), &dev).unwrap(),
        Tensor::from_slice(&[0.2f32, -0.3], (1, K), &dev).unwrap(),
        Tensor::from_vec(wv, (f, K), &dev).unwrap(),
        hidden,
        ranges,
        depth,
    )
    .unwrap()
    .centred(
        Tensor::from_vec(mu, (1, f), &dev).unwrap(),
        Tensor::from_slice(&[0.4f32, 0.7], (1, K), &dev).unwrap(),
    )
}

/// Every composition of `n` into `f` parts.
fn compositions(n: usize, f: usize) -> Vec<Vec<f32>> {
    if f == 1 {
        return vec![vec![n as f32]];
    }
    (0..=n)
        .flat_map(|first| {
            compositions(n - first, f - 1)
                .into_iter()
                .map(move |mut rest| {
                    rest.insert(0, first as f32);
                    rest
                })
        })
        .collect()
}

/// Every row of reads with `N` reads in each block.
fn rows(ranges: &[Range<usize>]) -> Vec<Vec<f32>> {
    let mut out = vec![vec![]];
    for r in ranges {
        let parts = compositions(N, r.len());
        out = out
            .into_iter()
            .flat_map(|head| {
                parts.iter().map(move |p| {
                    let mut v = head.clone();
                    v.extend(p);
                    v
                })
            })
            .collect();
    }
    out
}

fn ln_choose(n: u32, k: u32) -> f64 {
    let lf = |x: u32| (1..=x).map(|i| f64::from(i).ln()).sum::<f64>();
    lf(n) - lf(k) - lf(n - k)
}

/// The hidden support with its log base measure (a grid for the Gaussian).
fn support(hidden: Hidden) -> Vec<(f64, f64)> {
    match hidden {
        Hidden::Bernoulli => vec![(0.0, 0.0), (1.0, 0.0)],
        Hidden::Binomial { trials } => (0..=trials)
            .map(|h| (f64::from(h), ln_choose(trials, h)))
            .collect(),
        Hidden::Poisson { max } => (0..=max)
            .map(|h| {
                (
                    f64::from(h),
                    -(1..=h).map(|i| f64::from(i).ln()).sum::<f64>(),
                )
            })
            .collect(),
        Hidden::Gaussian => {
            let dh = 0.02;
            (0..800)
                .map(|i| {
                    let h = -8.0 + dh * f64::from(i);
                    (h, -0.5 * h * h + dh.ln())
                })
                .collect()
        }
    }
}

/// `log Σ_h e^{−E(v, h)}` and `E[h | v]`, by brute force over the hidden
/// units.
fn brute(m: &Machine, v: &[f32]) -> (f64, Vec<f64>) {
    let w = m.w.to_vec2::<f32>().unwrap();
    let b: Vec<f32> = m.b.flatten_all().unwrap().to_vec1().unwrap();
    let c: Vec<f32> = m.c.flatten_all().unwrap().to_vec1().unwrap();
    let mu: Vec<f32> = m.mu.flatten_all().unwrap().to_vec1().unwrap();
    let lambda: Vec<f32> = m.lambda.flatten_all().unwrap().to_vec1().unwrap();
    let depth: Vec<f32> = m.depth_t.flatten_all().unwrap().to_vec1().unwrap();
    let mut s = vec![0f64; v.len()];
    for (bi, r) in m.ranges.iter().enumerate() {
        let n: f64 = v[r.clone()].iter().map(|&x| f64::from(x)).sum();
        for i in r.clone() {
            s[i] = if n > 0.0 {
                f64::from(depth[bi]) / n
            } else {
                0.0
            };
        }
    }
    let vb: f64 = v.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
    // Hidden units are independent given v: one factor each.
    let mut log_z = vb;
    let mut mean = Vec::new();
    for k in 0..K {
        // `(s ⊙ v − μ)ᵀ W` on the blocks the row has reads in.
        let r: f64 = v
            .iter()
            .enumerate()
            .filter(|(i, _)| s[*i] > 0.0)
            .map(|(i, &x)| (s[i] * f64::from(x) - f64::from(mu[i])) * f64::from(w[i][k]))
            .sum();
        let a = f64::from(c[k]) + r;
        log_z -= r * f64::from(lambda[k]);
        let terms: Vec<(f64, f64)> = support(m.hidden)
            .into_iter()
            .map(|(h, lm)| (h, a * h + lm))
            .collect();
        let top = terms.iter().map(|t| t.1).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = terms.iter().map(|t| (t.1 - top).exp()).sum();
        log_z += top + z.ln();
        mean.push(terms.iter().map(|t| t.0 * (t.1 - top).exp()).sum::<f64>() / z);
    }
    (log_z, mean)
}

fn tensor(v: &[Vec<f32>]) -> Tensor {
    let flat: Vec<f32> = v.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (v.len(), v[0].len()), &Device::Cpu).unwrap()
}

fn ranges_cases() -> [Vec<Range<usize>>; 2] {
    [vec![0..3], vec![0..2, 2..4]]
}

#[test]
fn free_energy_and_posterior_match_enumeration() {
    for ranges in ranges_cases() {
        for hidden in kinds() {
            let m = machine(hidden, ranges.clone());
            let vs = rows(&ranges);
            let fe: Vec<f32> = m.free_energy(&tensor(&vs)).unwrap().to_vec1().unwrap();
            let mean = m
                .posterior_mean(&tensor(&vs))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let (z0, _) = brute(&m, &vs[0]);
            for (j, v) in vs.iter().enumerate() {
                let (z, mb) = brute(&m, v);
                // F is −log Σ_h e^{−E} up to a constant: compare differences.
                let got = f64::from(fe[j] - fe[0]);
                assert!(
                    (got + (z - z0)).abs() < 2e-3,
                    "{hidden:?} {ranges:?} v={v:?}: ΔF {got} vs {}",
                    -(z - z0)
                );
                for k in 0..K {
                    assert!(
                        (f64::from(mean[j][k]) - mb[k]).abs() < 2e-3,
                        "{hidden:?} mean {} vs {}",
                        mean[j][k],
                        mb[k]
                    );
                }
            }
        }
    }
}

#[test]
fn a_block_without_reads_drops_out() {
    for hidden in kinds() {
        let two = machine(hidden, vec![0..2, 2..4]);
        // The second block's half of the machine, alone.
        let one = Machine::new(
            two.b.narrow(1, 0, 2).unwrap(),
            two.c.clone(),
            two.w.narrow(0, 0, 2).unwrap(),
            hidden,
            vec![0..2],
            vec![N as f32],
        )
        .unwrap()
        .centred(two.mu.narrow(1, 0, 2).unwrap(), two.lambda.clone());
        let x2 = tensor(&[vec![2.0, 1.0, 0.0, 0.0], vec![0.0, 3.0, 0.0, 0.0]]);
        let x1 = tensor(&[vec![2.0, 1.0], vec![0.0, 3.0]]);
        let a: Vec<f32> = two.free_energy(&x2).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = one.free_energy(&x1).unwrap().to_vec1().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{hidden:?}: {x} vs {y}");
        }
    }
}

/// The log multinomial coefficient of every block of `v`.
fn ln_coef(v: &[f32], ranges: &[Range<usize>]) -> f64 {
    let lf = |x: f32| (1..=x as u32).map(|i| f64::from(i).ln()).sum::<f64>();
    ranges
        .iter()
        .map(|r| {
            let n: f32 = v[r.clone()].iter().sum();
            lf(n) - v[r.clone()].iter().map(|&x| lf(x)).sum::<f64>()
        })
        .sum()
}

/// The exact `p(v)` over every row at depth `N` per block.
fn exact(m: &Machine, vs: &[Vec<f32>]) -> Vec<f64> {
    let fe: Vec<f32> = m.free_energy(&tensor(vs)).unwrap().to_vec1().unwrap();
    let lp: Vec<f64> = vs
        .iter()
        .zip(&fe)
        .map(|(v, &f)| ln_coef(v, &m.ranges) - f64::from(f))
        .collect();
    let top = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = lp.iter().map(|l| (l - top).exp()).sum();
    lp.iter().map(|l| (l - top).exp() / z).collect()
}

#[test]
fn gibbs_chains_reach_the_exact_marginal() {
    for ranges in ranges_cases() {
        for hidden in kinds() {
            let m = machine(hidden, ranges.clone());
            let vs = rows(&ranges);
            let p = exact(&m, &vs);
            let f = m.features().unwrap();
            let s = 3000;
            let x0: Vec<f32> = (0..s).flat_map(|j| vs[j % vs.len()].clone()).collect();
            let mut chains = Chains::new(x0, f, &ranges, 7, &Device::Cpu).unwrap();
            for _ in 0..30 {
                chains.sweep(&m).unwrap();
            }
            let mut counts = vec![0f64; vs.len()];
            for _ in 0..20 {
                chains.sweep(&m).unwrap();
                for row in chains.reads().to_vec2::<f32>().unwrap() {
                    let j = vs.iter().position(|v| *v == row).expect("a valid row");
                    counts[j] += 1.0;
                }
            }
            let total: f64 = counts.iter().sum();
            let tv: f64 = counts
                .iter()
                .zip(&p)
                .map(|(c, q)| (c / total - q).abs())
                .sum::<f64>()
                / 2.0;
            assert!(tv < 0.03, "{hidden:?} {ranges:?}: total variation {tv}");
        }
    }
}

#[test]
fn backward_of_the_contrast_is_the_likelihood_gradient() {
    for hidden in kinds() {
        let ranges = vec![0..3];
        let base = machine(hidden, ranges.clone());
        let vs = rows(&ranges);
        let data = vs[4].clone();
        let w = Var::from_tensor(&base.w).unwrap();
        let m = Machine {
            w: w.as_tensor().clone(),
            ..base.clone()
        };
        // Exact model expectation in place of the fantasies.
        let p = exact(&base, &vs);
        let pt = Tensor::from_vec(
            p.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            vs.len(),
            &Device::Cpu,
        )
        .unwrap();
        let loss = (m
            .free_energy(&tensor(std::slice::from_ref(&data)))
            .unwrap()
            .sum_all()
            .unwrap()
            - (m.free_energy(&tensor(&vs)).unwrap() * pt)
                .unwrap()
                .sum_all()
                .unwrap())
        .unwrap();
        let grads = loss.backward().unwrap();
        let g = grads.get(w.as_tensor()).unwrap().to_vec2::<f32>().unwrap();
        // −log p(data) by finite differences on W.
        let nll = |wv: &Tensor| {
            let mm = Machine {
                w: wv.clone(),
                ..base.clone()
            };
            let j = vs.iter().position(|v| *v == data).unwrap();
            -exact(&mm, &vs)[j].ln()
        };
        let eps = 1e-2f32;
        let w0 = base.w.to_vec2::<f32>().unwrap();
        for i in 0..3 {
            for k in 0..K {
                let mut up = w0.clone();
                up[i][k] += eps;
                let mut dn = w0.clone();
                dn[i][k] -= eps;
                let fd = (nll(&tensor(&up)) - nll(&tensor(&dn))) / (2.0 * f64::from(eps));
                assert!(
                    (f64::from(g[i][k]) - fd).abs() < 2e-3,
                    "{hidden:?} W[{i},{k}]: autograd {} vs finite difference {fd}",
                    g[i][k]
                );
            }
        }
    }
}

#[test]
fn device_multinomial_keeps_every_rows_depth_and_shares() {
    let mut rng = StdRng::seed_from_u64(3);
    let p = [0.5f32, 0.3, 0.15, 0.05];
    let rows = 4000;
    let probs: Vec<f32> = (0..rows).flat_map(|_| p).collect();
    let cum = Tensor::from_vec(probs, (rows, 4), &Device::Cpu)
        .unwrap()
        .cumsum(1)
        .unwrap();
    // Every other row draws fewer reads, padding the uniforms it does not use.
    let n: Vec<u32> = (0..rows).map(|j| if j % 2 == 0 { 20 } else { 7 }).collect();
    let x = multinomial_rows(&cum, &n, &mut rng)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let mut total = [0f64; 4];
    for (row, &nj) in x.iter().zip(&n) {
        assert_eq!(row.iter().sum::<f32>(), nj as f32);
        assert!(row.iter().all(|&c| c >= 0.0));
        for (t, &c) in total.iter_mut().zip(row) {
            *t += f64::from(c);
        }
    }
    let all: f64 = n.iter().map(|&v| f64::from(v)).sum();
    for (t, &pi) in total.iter().zip(&p) {
        assert!((t / all - f64::from(pi)).abs() < 0.01, "{t} vs {pi}");
    }
}
