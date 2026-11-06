use crate::embed_ode::curves::{CurveSpec, Curves};
use crate::embed_ode::model::{Model, Parts, Pseudobulks, QMass};
use crate::embed_ode::place::expect_on_grid;
use crate::ode_mixture::sim::{simulate, SimConfig};
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::traits::RandomizedAlgs;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};

/// Pseudobulks from the simulator, two lineages on a trunk, its spliced
/// reads every gene's, then a block of genes driven by one factor `ν_p`
/// alone, no trajectory in it. `q` sharp at every pseudobulk's lineage and
/// nearest position.
struct Planted {
    data: Pseudobulks,
    spec: CurveSpec,
    lineage: Vec<usize>,
    time: Vec<f64>,
    nu: Vec<f64>,
}

const POINTS: usize = 17;
const SUB: usize = 4;
const BRANCH: f64 = 0.3;

fn planted(nuisance_genes: usize, nuisance_sd: f64, seed: u64) -> Planted {
    let cfg = SimConfig {
        n_genes: 100,
        n_pb: 400,
        dim: 3,
        lineages: 2,
        branch_at: Some(BRANCH),
        seed,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let (p_n, g0) = (sim.n_pb, sim.n_genes);
    let g = g0 + nuisance_genes;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
    let unit = Normal::new(0.0, 1.0).unwrap();
    let span = Uniform::new(0.5f64, 1.0).unwrap();
    let nu: Vec<f64> = (0..p_n).map(|_| unit.sample(&mut rng)).collect();
    let load: Vec<f64> = (0..nuisance_genes)
        .map(|j| span.sample(&mut rng) * if j % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut y = vec![0f32; p_n * g];
    for p in 0..p_n {
        let spliced: f64 = (0..g0).map(|j| sim.s[p * g0 + j]).sum();
        for j in 0..g0 {
            y[p * g + j] = sim.s[p * g0 + j] as f32;
        }
        // The block's mean share per gene is a sim gene's.
        let each = spliced / g0 as f64;
        for (j, l) in load.iter().enumerate() {
            let rate = each * (nuisance_sd * nu[p] * l).exp();
            y[p * g + g0 + j] = Poisson::new(rate.max(1e-9)).unwrap().sample(&mut rng) as f32;
        }
    }
    let spec = CurveSpec {
        lineages: 2,
        points: POINTS,
        sub: SUB,
        kappa: cfg.kappa,
        branch: Some(vec![BRANCH]),
    };
    let n = spec.positions();
    let q = (0..p_n)
        .map(|p| QMass {
            pb: p,
            lineage: sim.truth.lineage[p],
            position: (sim.truth.time[p] / cfg.t_len * (n - 1) as f64).round() as usize,
            q: 1.0,
        })
        .collect();
    Planted {
        data: Pseudobulks::new(y, p_n, g, q).unwrap(),
        spec,
        lineage: sim.truth.lineage.clone(),
        time: sim.truth.time.clone(),
        nu,
    }
}

/// `[P × H]` the leading `h` components of log1p CPM.
fn pca_log_cpm(data: &Pseudobulks, h: usize) -> Vec<f32> {
    let (p_n, g) = (data.n_pb, data.n_genes);
    let mut m = DMatrix::<f64>::from_fn(p_n, g, |p, j| {
        let total: f32 = data.y[p * g..(p + 1) * g].iter().sum();
        (f64::from(data.y[p * g + j]) / f64::from(total.max(1.0)) * 1e6).ln_1p()
    });
    for j in 0..g {
        let mean = m.column(j).mean();
        m.column_mut(j).add_scalar_mut(-mean);
    }
    let (u, d, _) = m.rsvd(h).unwrap();
    (0..p_n)
        .flat_map(|p| (0..h).map(move |k| (p, k)))
        .map(|(p, k)| (u[(p, k)] * d[k]) as f32)
        .collect()
}

/// `[P × H]` every pseudobulk's prior mean `M_p = Σ q_p(ℓ,k) μ_ℓ(k)`; zero
/// where it has none.
pub(super) fn prior_means(data: &Pseudobulks, model: &Model) -> Vec<f32> {
    let (mu, n, h) = (model.mu().unwrap(), model.curves().positions(), model.dim());
    (0..data.n_pb)
        .flat_map(|p| expect_on_grid(data.of(p).iter().map(QMass::on_grid), &mu, n, h))
        .collect()
}

/// Every pseudobulk.
fn all(data: &Pseudobulks) -> Vec<usize> {
    (0..data.n_pb).collect()
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The mean share of every row's `k` nearest neighbours (among `rows`) of
/// its own label.
fn knn_purity(x: &[f32], h: usize, rows: &[usize], label: &[usize], k: usize) -> f64 {
    let mut total = 0.0;
    for &i in rows {
        let mut near: Vec<(f32, usize)> = rows
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| (dist2(&x[i * h..(i + 1) * h], &x[j * h..(j + 1) * h]), j))
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let same = near[..k]
            .iter()
            .filter(|(_, j)| label[*j] == label[i])
            .count();
        total += same as f64 / k as f64;
    }
    total / rows.len() as f64
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        for &t in &idx[i..=j] {
            r[t] = (i + j) as f64 / 2.0;
        }
        i = j + 1;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&ranks(a), &ranks(b))
}

/// The correlation of `target` with its least-squares fit on the columns of
/// `r [P × H]` and an intercept.
fn multiple_correlation(r: &[f32], h: usize, target: &[f64]) -> f64 {
    let p_n = target.len();
    let a = DMatrix::<f64>::from_fn(p_n, h + 1, |p, k| {
        if k == h {
            1.0
        } else {
            f64::from(r[p * h + k])
        }
    });
    let t = nalgebra::DVector::from_column_slice(target);
    let coef = (a.transpose() * &a)
        .cholesky()
        .unwrap()
        .solve(&(a.transpose() * &t));
    let fit: Vec<f64> = (a * coef).iter().copied().collect();
    pearson(&fit, target)
}

#[test]
fn q_is_renormalised_within_every_pseudobulk_and_dropped_where_empty() {
    let y = vec![1f32; 3 * 2];
    let q = vec![
        QMass {
            pb: 0,
            lineage: 0,
            position: 1,
            q: 0.2,
        },
        QMass {
            pb: 2,
            lineage: 1,
            position: 0,
            q: 0.3,
        },
        QMass {
            pb: 0,
            lineage: 1,
            position: 2,
            q: 0.6,
        },
        QMass {
            pb: 1,
            lineage: 0,
            position: 0,
            q: 0.0,
        },
    ];
    let data = Pseudobulks::new(y, 3, 2, q).unwrap();
    assert_eq!(data.has_prior(), vec![true, false, true]);
    let mass = |pb: usize| -> f32 { data.q().iter().filter(|m| m.pb == pb).map(|m| m.q).sum() };
    assert!((mass(0) - 1.0).abs() < 1e-6);
    assert!((mass(2) - 1.0).abs() < 1e-6);
    let first = data
        .q()
        .iter()
        .find(|m| m.pb == 0 && m.position == 1)
        .unwrap();
    assert!((first.q - 0.25).abs() < 1e-6);
    assert!(data.q().iter().all(|m| m.pb != 1));
}

#[test]
fn a_q_off_the_grid_is_an_error() {
    let q = vec![QMass {
        pb: 4,
        lineage: 0,
        position: 0,
        q: 1.0,
    }];
    assert!(Pseudobulks::new(vec![1f32; 6], 3, 2, q).is_err());
}

/// The loss, term by term, on a case small enough to write out.
#[test]
fn the_loss_is_the_softmax_the_prior_and_the_gauge_terms() {
    let dev = Device::Cpu;
    let spec = CurveSpec {
        lineages: 1,
        points: 3,
        sub: 1,
        kappa: 0.5,
        branch: None,
    };
    let curves = Curves::new(&spec, &dev).unwrap();
    let (p_n, g, h, n) = (2, 3, 2, 3);
    let y = vec![3f32, 0.0, 1.0, 2.0, 5.0, 4.0];
    let q = vec![
        QMass {
            pb: 0,
            lineage: 0,
            position: 0,
            q: 0.5,
        },
        QMass {
            pb: 0,
            lineage: 0,
            position: 2,
            q: 0.5,
        },
    ];
    let data = Pseudobulks::new(y.clone(), p_n, g, q).unwrap();
    let parts = Parts {
        x: vec![0.3, -0.2, 1.1, 0.4],
        w: vec![0.5, -0.1, 0.2, 0.7, -0.3, 0.4],
        b: vec![0.1, -0.4, 0.6],
        eta: vec![0.2, -0.5, 0.8, 0.1, -0.3, 0.4],
        log_a: 0.2,
        log_sigma: -0.3,
    };
    let model = Model::from_parts(&data, curves, &parts, &dev).unwrap();
    let got = model.loss(&all(&data)).unwrap().to_vec0::<f32>().unwrap();
    assert_eq!(model.w().unwrap(), parts.w);
    assert_eq!(model.b().unwrap(), parts.b);
    assert!((model.amplitude().unwrap() - 0.2f32.exp()).abs() < 1e-6);

    let f = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let (x, w, b, eta) = (f(&parts.x), f(&parts.w), f(&parts.b), f(&parts.eta));
    let mut want = 0.0;
    for p in 0..p_n {
        let s: Vec<f64> = (0..g)
            .map(|j| (0..h).map(|d| x[p * h + d] * w[j * h + d]).sum::<f64>() + b[j])
            .collect();
        let lse = s.iter().map(|v| v.exp()).sum::<f64>().ln();
        want -= (0..g)
            .map(|j| f64::from(y[p * g + j]) * (s[j] - lse))
            .sum::<f64>();
    }
    let prior = crate::ode_mixture::kernel::GridPrior::new(n, 1.0, 0.5).unwrap();
    let a = 0.2f64.exp();
    let mu: Vec<Vec<f64>> = {
        let cols: Vec<_> = (0..h)
            .map(|d| prior.colour(&nalgebra::DVector::from_fn(n, |k, _| eta[k * h + d])))
            .collect();
        (0..n)
            .map(|k| (0..h).map(|d| a * cols[d][k]).collect())
            .collect()
    };
    let sigma = (-0.3f64).exp();
    for k in [0, 2] {
        let d2: f64 = (0..h).map(|d| (x[d] - mu[k][d]).powi(2)).sum();
        want += 0.5 * d2 / (2.0 * sigma * sigma);
    }
    want += h as f64 * sigma.ln(); // one pseudobulk with a prior
    want += 0.5 * w.iter().map(|v| v * v).sum::<f64>();
    want += 0.5 * eta.iter().map(|v| v * v).sum::<f64>();
    want += 0.5 * 0.2 * 0.2;
    let b_mean = b.iter().sum::<f64>() / g as f64;
    want += 0.5 * g as f64 * b_mean * b_mean;
    assert!(
        (f64::from(got) - want).abs() < 1e-4 * want.abs().max(1.0),
        "{got} vs {want}"
    );
}

/// On a subset of the pseudobulks the loss is their terms scaled up to the
/// whole, the terms on the shared parameters as they are: its mean over a
/// partition is the full loss.
#[test]
fn the_loss_on_a_partition_averages_to_the_full_loss() {
    let dev = Device::Cpu;
    let pl = planted(10, 1.0, 7);
    let curves = Curves::new(&pl.spec, &dev).unwrap();
    let model = Model::init(&pl.data, curves, 4, 3, &dev).unwrap();
    let full = model
        .loss(&all(&pl.data))
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();
    let p_n = pl.data.n_pb;
    let parts: Vec<Vec<usize>> = (0..4)
        .map(|c| (0..p_n).filter(|p| p % 4 == c).collect())
        .collect();
    let mean = parts
        .iter()
        .map(|s| f64::from(model.loss(s).unwrap().to_vec0::<f32>().unwrap()))
        .sum::<f64>()
        / parts.len() as f64;
    assert!(
        (mean - f64::from(full)).abs() < 1e-4 * f64::from(full).abs(),
        "{mean} vs {full}"
    );
}

/// Fit on pseudobulks with `q` from the truth: the lineages apart off the
/// trunk, past the branch by a length scale (more so than in a PCA of the
/// reads, which the nuisance block muddles), the nearest curve point's
/// phase in step with the true time, and the nuisance kept in what the
/// curves do not explain.
#[test]
fn a_fit_with_a_sharp_q_follows_the_truth_and_keeps_the_nuisance() {
    let dev = Device::Cpu;
    let pl = planted(40, 1.5, 11);
    let h = 5;
    let curves = Curves::new(&pl.spec, &dev).unwrap();
    let model = Model::init(&pl.data, curves, h, 5, &dev).unwrap();
    let trace = model.fit(1500, 0.02, None, 5).unwrap();
    assert!(trace.last().unwrap() < trace.first().unwrap());
    let x = model.x().unwrap();
    let p_n = pl.data.n_pb;

    // Off the trunk by a length scale at least: right past the branch the
    // lineages are still one, the truth's own curves no purer there.
    let off: Vec<usize> = (0..p_n)
        .filter(|&p| pl.time[p] > BRANCH + pl.spec.kappa)
        .collect();
    let purity = knn_purity(&x, h, &off, &pl.lineage, 10);
    let pca = pca_log_cpm(&pl.data, h);
    let purity_pca = knn_purity(&pca, h, &off, &pl.lineage, 10);
    eprintln!("purity: fit {purity:.3}, pca {purity_pca:.3}");
    assert!(purity > 0.9, "{purity}");
    assert!(purity > purity_pca, "{purity} vs {purity_pca}");

    let mu = model.mu().unwrap();
    let (l_n, n) = (pl.spec.lineages, pl.spec.positions());
    let phase: Vec<f64> = (0..p_n)
        .map(|p| {
            let xp = &x[p * h..(p + 1) * h];
            let best = (0..l_n * n)
                .min_by(|&a, &b| {
                    dist2(xp, &mu[a * h..(a + 1) * h])
                        .total_cmp(&dist2(xp, &mu[b * h..(b + 1) * h]))
                })
                .unwrap();
            model.curves().phase(best % n)
        })
        .collect();
    let rho = spearman(&phase, &pl.time);
    eprintln!("spearman(nearest phase, time) {rho:.3}");
    assert!(rho > 0.9, "{rho}");

    let m = prior_means(&pl.data, &model);
    let resid: Vec<f32> = x.iter().zip(&m).map(|(a, b)| a - b).collect();
    let r = multiple_correlation(&resid, h, &pl.nu);
    eprintln!(
        "residual vs nuisance {r:.3}; sigma {:.3}",
        model.sigma().unwrap()
    );
    assert!(r > 0.5, "{r}");
}

/// Fitting in steps over subsets of the pseudobulks lowers the full loss
/// about as far as fitting on all of them at once.
#[test]
fn a_fit_in_batches_lowers_the_loss_as_a_full_one_does() {
    let dev = Device::Cpu;
    let pl = planted(10, 1.0, 13);
    let start = {
        let curves = Curves::new(&pl.spec, &dev).unwrap();
        let model = Model::init(&pl.data, curves, 4, 3, &dev).unwrap();
        f64::from(
            model
                .loss(&all(&pl.data))
                .unwrap()
                .to_vec0::<f32>()
                .unwrap(),
        )
    };
    let end = |batch: Option<usize>, epochs: usize| {
        let curves = Curves::new(&pl.spec, &dev).unwrap();
        let model = Model::init(&pl.data, curves, 4, 3, &dev).unwrap();
        model.fit(epochs, 0.02, batch, 3).unwrap();
        f64::from(
            model
                .loss(&all(&pl.data))
                .unwrap()
                .to_vec0::<f32>()
                .unwrap(),
        )
    };
    let full = end(None, 400);
    let batched = end(Some(100), 100);
    eprintln!("start {start:.1}, full {full:.1}, batched {batched:.1}");
    assert!(full < start && batched < start);
    assert!(
        (batched - full) < 0.05 * (start - full),
        "{batched} vs {full}"
    );
}
