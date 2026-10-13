use super::*;
use crate::rbm_core::multinomial;
use rand_distr::{Distribution, Gamma};

const TYPES: usize = 3;
const PER_PB: usize = 5;
static NEVER: AtomicBool = AtomicBool::new(false);

/// Planted cell types over `blocks`: every type a profile per block, every
/// cell a multinomial draw from its type's profile at its own depth.
struct Sim {
    blocks: Blocks,
    /// `[C × F]` row-major, machine order.
    cells: Vec<f32>,
    label: Vec<usize>,
    /// Every type's shares, `[TYPES × F]`.
    profile: Vec<Vec<f32>>,
}

fn sim(names: &[Box<str>], cells_per_type: usize, seed: u64) -> Sim {
    let blocks = Blocks::from_names(names);
    let f = blocks.features();
    let mut rng = StdRng::seed_from_u64(seed);
    let gamma = Gamma::new(0.3, 1.0).unwrap();
    let profile: Vec<Vec<f32>> = (0..TYPES)
        .map(|_| {
            let mut p: Vec<f32> = (0..f)
                .map(|_| gamma.sample(&mut rng) as f32 + 1e-3)
                .collect();
            for r in &blocks.ranges {
                let s: f32 = p[r.clone()].iter().sum();
                p[r.clone()].iter_mut().for_each(|x| *x /= s);
            }
            p
        })
        .collect();
    let mut cells = Vec::new();
    let mut label = Vec::new();
    for (t, prof) in profile.iter().enumerate() {
        for j in 0..cells_per_type {
            let mut row = vec![0f32; f];
            for r in &blocks.ranges {
                let depth = 150 + (j * 37) % 400;
                multinomial(
                    depth as u64,
                    &prof[r.clone()],
                    &mut row[r.clone()],
                    &mut rng,
                );
            }
            cells.extend(row);
            label.push(t);
        }
    }
    Sim {
        blocks,
        cells,
        label,
        profile,
    }
}

impl Sim {
    /// Pseudobulks of `PER_PB` cells of one type each, as a level.
    fn level(&self) -> Level {
        let f = self.blocks.features();
        let mut rates = Vec::new();
        let mut pbs = 0;
        for t in 0..TYPES {
            let members: Vec<usize> = (0..self.label.len())
                .filter(|&c| self.label[c] == t)
                .collect();
            for group in members.chunks(PER_PB) {
                let mut sum = vec![0f32; f];
                for &c in group {
                    for (s, &x) in sum.iter_mut().zip(&self.cells[c * f..(c + 1) * f]) {
                        *s += x;
                    }
                }
                // Back to the data's row order.
                let mut data = vec![0f32; f];
                for (i, &p) in self.blocks.perm.iter().enumerate() {
                    data[p] = sum[i];
                }
                rates.extend(data);
                pbs += 1;
            }
        }
        Level::from_rates(
            &nalgebra::DMatrix::from_column_slice(f, pbs, &rates),
            &self.blocks,
        )
    }
}

fn names(genes: usize, modalities: &[&str]) -> Vec<Box<str>> {
    if modalities.is_empty() {
        return (0..genes).map(|g| format!("GENE{g}").into()).collect();
    }
    modalities
        .iter()
        .flat_map(|m| (0..genes).map(move |g| format!("GENE{g}/{m}").into()))
        .collect()
}

fn config(epochs: usize) -> TrainConfig<'static> {
    TrainConfig {
        epochs,
        batch: 6,
        chains: 60,
        sweeps: 1,
        lr: 0.005,
        lora_lr_ratio: 4.0,
        lora_ridge: 0.05,
        embedding_l2: 0.01,
        draw_data: false,
        sparsity_target: 0.1,
        sparsity_cost: 0.0,
        module_balance: 0.0,
        module_ridge: 0.0,
        module_warmup: 0,
        holdout: 0.0,
        seed: 5,
        stop: &NEVER,
    }
}

fn train(s: &Sim, hidden: Hidden, k: usize, h: usize, tie: bool, epochs: usize) -> Fitted {
    let dev = Device::Cpu;
    let row: Vec<u32> = if tie {
        s.blocks.symbol.clone()
    } else {
        (0..s.blocks.features() as u32).collect()
    };
    let w = Weights::new(&row, k, h, s.blocks.len(), None, 3, &dev).unwrap();
    let depth = vec![20.0; s.blocks.len()];
    fit(
        &[s.level()],
        &s.blocks,
        w,
        hidden,
        depth,
        &config(epochs),
        &dev,
    )
    .unwrap()
}

fn latent(fitted: &Fitted, blocks: &Blocks, x: &[f32]) -> Vec<Vec<f32>> {
    let f = blocks.features();
    let m = fitted.machine(blocks).unwrap();
    let t = Tensor::from_slice(x, (x.len() / f, f), &Device::Cpu).unwrap();
    m.posterior_mean(&t).unwrap().to_vec2().unwrap()
}

/// Leave-one-out nearest-centroid accuracy of `z` against `label`.
fn centroid_accuracy(z: &[Vec<f32>], label: &[usize]) -> f64 {
    let k = z[0].len();
    let mut sum = vec![vec![0f64; k]; TYPES];
    let mut n = [0f64; TYPES];
    for (zi, &l) in z.iter().zip(label) {
        for d in 0..k {
            sum[l][d] += f64::from(zi[d]);
        }
        n[l] += 1.0;
    }
    let hits = z
        .iter()
        .zip(label)
        .filter(|(zi, &l)| {
            let best = (0..TYPES)
                .min_by(|&a, &b| {
                    let dist = |t: usize| {
                        let own = if t == l { 1.0 } else { 0.0 };
                        (0..k)
                            .map(|d| {
                                let c = (sum[t][d] - own * f64::from(zi[d])) / (n[t] - own);
                                (f64::from(zi[d]) - c).powi(2)
                            })
                            .sum::<f64>()
                    };
                    dist(a).total_cmp(&dist(b))
                })
                .unwrap();
            best == l
        })
        .count();
    hits as f64 / z.len() as f64
}

#[test]
fn trained_on_pseudobulks_the_cells_separate_by_type() {
    let s = sim(&names(30, &[]), 60, 1);
    for hidden in [
        Hidden::Bernoulli,
        Hidden::Binomial { trials: 3 },
        Hidden::Poisson { max: 6 },
        Hidden::Gaussian,
    ] {
        let fitted = train(&s, hidden, 4, 6, false, 100);
        let z = latent(&fitted, &s.blocks, &s.cells);
        let acc = centroid_accuracy(&z, &s.label);
        assert!(acc > 0.95, "{hidden:?}: nearest-centroid accuracy {acc}");
        let gap = fitted.trace.gap.last().copied().unwrap();
        assert!(gap.is_finite(), "{hidden:?}: gap {gap}");
    }
}

#[test]
fn a_cells_latent_does_not_depend_on_its_depth() {
    let s = sim(&names(30, &[]), 60, 2);
    let fitted = train(&s, Hidden::Bernoulli, 4, 6, false, 60);
    let f = s.blocks.features();
    // Every type's profile at a shallow and a deep depth, in expectation.
    for p in &s.profile {
        let shallow: Vec<f32> = p.iter().map(|x| x * 200.0).collect();
        let deep: Vec<f32> = p.iter().map(|x| x * 5000.0).collect();
        let a = latent(&fitted, &s.blocks, &shallow);
        let b = latent(&fitted, &s.blocks, &deep);
        for (x, y) in a[0].iter().zip(&b[0]) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y} (F = {f})");
        }
    }
}

#[test]
fn a_cell_missing_a_modality_keeps_its_place() {
    let s = sim(&names(20, &["rna", "atac"]), 60, 3);
    assert_eq!(s.blocks.len(), 2);
    let fitted = train(&s, Hidden::Bernoulli, 4, 6, true, 100);
    let f = s.blocks.features();
    // The same cells with their second block emptied.
    let mut one = s.cells.clone();
    let r = s.blocks.ranges[1].clone();
    for row in one.chunks_mut(f) {
        row[r.clone()].iter_mut().for_each(|x| *x = 0.0);
    }
    let z_one = latent(&fitted, &s.blocks, &one);
    let acc = centroid_accuracy(&z_one, &s.label);
    assert!(acc > 0.9, "one modality: nearest-centroid accuracy {acc}");
}

#[test]
fn tied_symbols_hold_one_row_per_name() {
    let s = sim(&names(20, &["rna", "atac"]), 30, 4);
    let fitted = train(&s, Hidden::Bernoulli, 3, 4, true, 2);
    assert_eq!(fitted.weights.rho().unwrap().dim(0).unwrap(), 20);
    let untied = train(&s, Hidden::Bernoulli, 3, 4, false, 2);
    assert_eq!(untied.weights.rho().unwrap().dim(0).unwrap(), 40);
}

#[test]
#[ignore = "diagnostic"]
fn diagnostic_every_kind() {
    let dev = Device::Cpu;
    for hidden in [
        Hidden::Bernoulli,
        Hidden::Binomial { trials: 3 },
        Hidden::Poisson { max: 6 },
        Hidden::Gaussian,
    ] {
        for (d, lr, l2) in [(20.0, 0.01, 1e-2), (20.0, 0.02, 1e-2), (20.0, 0.005, 1e-2)] {
            let mut out = Vec::new();
            for seed in 0..5u64 {
                let s = sim(&names(30, &[]), 60, 10 + seed);
                let row: Vec<u32> = (0..s.blocks.features() as u32).collect();
                let w = Weights::new(&row, 8, 6, 1, None, seed, &dev).unwrap();
                let cfg = TrainConfig {
                    lr,
                    embedding_l2: l2,
                    seed,
                    ..config(150)
                };
                let fitted = fit(&[s.level()], &s.blocks, w, hidden, vec![d], &cfg, &dev).unwrap();
                let z = latent(&fitted, &s.blocks, &s.cells);
                let wn = fitted
                    .weights
                    .w(&s.blocks.ranges)
                    .unwrap()
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    .sqrt();
                out.push(format!("{:.2}/|W|{wn:.0}", centroid_accuracy(&z, &s.label)));
            }
            eprintln!("{hidden:?} D {d} lr {lr} l2 {l2}: {}", out.join("  "));
        }
    }
}

#[test]
fn modules_and_the_sparsity_target_keep_every_unit_in_use() {
    let s = sim(&names(30, &[]), 60, 7);
    let dev = Device::Cpu;
    let row: Vec<u32> = (0..s.blocks.features() as u32).collect();
    // Modules from the planted profiles: every gene on its most-expressed type.
    let labels: Vec<u32> = (0..s.blocks.features())
        .map(|g| {
            (0..TYPES)
                .max_by(|&a, &b| s.profile[a][g].total_cmp(&s.profile[b][g]))
                .unwrap() as u32
        })
        .collect();
    let w = Weights::new(&row, 8, 6, 1, None, 3, &dev)
        .unwrap()
        .with_modules(labels, TYPES, 0.9, 4)
        .unwrap();
    let cfg = TrainConfig {
        sparsity_cost: 0.01,
        module_balance: 1.0,
        module_ridge: 0.1,
        module_warmup: 25,
        holdout: 0.1,
        ..config(100)
    };
    let fitted = fit(
        &[s.level()],
        &s.blocks,
        w,
        Hidden::Bernoulli,
        vec![20.0],
        &cfg,
        &dev,
    )
    .unwrap();
    let z = latent(&fitted, &s.blocks, &s.cells);
    let acc = centroid_accuracy(&z, &s.label);
    assert!(acc > 0.95, "nearest-centroid accuracy {acc}");
    let live = *fitted.trace.live_units.last().unwrap();
    assert!(live >= 6.0, "{live}/8 units in use");
    assert!(fitted.trace.heldout.last().unwrap().is_finite());
}

#[test]
fn moving_the_hidden_centre_with_the_bias_leaves_the_machine_unchanged() {
    // `b += W Δλ` keeps the free energy of rows at the training depth fixed up
    // to one constant, which is what the training loop relies on.
    let s = sim(&names(10, &[]), 4, 11);
    let dev = Device::Cpu;
    let fitted = train(&s, Hidden::Bernoulli, 3, 4, false, 2);
    let m = fitted.machine(&s.blocks).unwrap();
    let w = m.w.clone();
    let delta = Tensor::from_slice(&[0.3f32, -0.2, 0.1], (1, 3), &dev).unwrap();
    let moved = m
        .with(
            (&m.b + delta.matmul(&w.t().unwrap()).unwrap()).unwrap(),
            m.c.clone(),
            w,
        )
        .centred(m.mu.clone(), (&m.lambda + &delta).unwrap());
    let f = s.blocks.features();
    // Rows at the training depth: the level's expected reads.
    let rows = s.level().expected(&s.blocks, &[20.0]);
    let x = Tensor::from_vec(rows, (rows_len(&s), f), &dev).unwrap();
    let a: Vec<f32> = m.free_energy(&x).unwrap().to_vec1().unwrap();
    let b: Vec<f32> = moved.free_energy(&x).unwrap().to_vec1().unwrap();
    let shift = a[0] - b[0];
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y - shift).abs() < 1e-3, "{x} vs {y} (shift {shift})");
    }
}

fn rows_len(s: &Sim) -> usize {
    s.level().rows
}
