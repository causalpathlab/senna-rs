use crate::embed_ode::curves::{CurveSpec, Curves};
use crate::embed_ode::model::{Model, Pseudobulks, QMass};
use crate::embed_ode::place::{expect_on_grid, grid_prior, Placer};
use crate::ode_mixture::sim::{simulate, SimConfig};
use legume_numeric::candle::candle_core::Device;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Normal, Poisson};

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Random genes `w [G × H]`, `b [G]` and a point `x`.
fn genes(g: usize, h: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Normal::new(0.0, 1.0).unwrap();
    let w = (0..g * h)
        .map(|_| 0.5 * unit.sample(&mut rng) as f32)
        .collect();
    let b = (0..g).map(|_| unit.sample(&mut rng) as f32).collect();
    let x = (0..h).map(|_| unit.sample(&mut rng) as f32).collect();
    (w, b, x)
}

/// `[1 × G]` reads at depth `n` from `softmax(x w + b)`.
fn reads_at(w: &[f32], b: &[f32], x: &[f32], n: f64, seed: u64) -> DMatrix<f32> {
    let (g, h) = (b.len(), x.len());
    let s: Vec<f64> = (0..g)
        .map(|j| f64::from(b[j] + (0..h).map(|d| x[d] * w[j * h + d]).sum::<f32>()))
        .collect();
    let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = s.iter().map(|v| (v - top).exp()).sum();
    let mut rng = StdRng::seed_from_u64(seed);
    DMatrix::from_fn(1, g, |_, j| {
        Poisson::new((n * (s[j] - top).exp() / z).max(1e-9))
            .unwrap()
            .sample(&mut rng) as f32
    })
}

/// With `σ → 0` a cell sits at its centre; with `σ → ∞` it follows its
/// reads, at the softmax's own optimum (its score gradient zero) and near
/// the point they were drawn at.
#[test]
fn with_sigma_to_zero_a_cell_sits_at_its_centre_and_to_infinity_follows_its_reads() {
    let (g, h) = (300, 3);
    let (w, b, x) = genes(g, h, 3);
    let placer = Placer::new(&w, &b, h, &Device::Cpu).unwrap();
    let reads = reads_at(&w, &b, &x, 2e5, 4);
    let centre = vec![4f32, -4.0, 4.0];

    let pinned = placer.place(&reads, &centre, 1e12).unwrap();
    assert!(dist2(&pinned, &centre) < 1e-6, "{pinned:?}");

    let free = placer.place(&reads, &centre, 0.0).unwrap();
    assert!(dist2(&free, &x) < 0.01, "{free:?} vs {x:?}");
    // The gradient of the reads' log likelihood at the placement.
    let s: Vec<f64> = (0..g)
        .map(|j| f64::from(b[j] + (0..h).map(|d| free[d] * w[j * h + d]).sum::<f32>()))
        .collect();
    let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = s.iter().map(|v| (v - top).exp()).sum();
    let n: f64 = reads.iter().map(|&v| f64::from(v)).sum();
    for d in 0..h {
        let grad: f64 = (0..g)
            .map(|j| {
                (f64::from(reads[(0, j)]) - n * (s[j] - top).exp() / z) * f64::from(w[j * h + d])
            })
            .sum();
        assert!(grad.abs() < 1e-2 * n.sqrt(), "dimension {d}: {grad}");
    }

    // In between, between the two.
    let mid = placer.place(&reads, &centre, 1.0).unwrap();
    assert!(dist2(&mid, &x) < dist2(&centre, &x));
    assert!(dist2(&mid, &centre) < dist2(&free, &centre));
}

/// A cell with no reads at all sits at its centre whatever `σ`.
#[test]
fn a_cell_with_no_reads_sits_at_its_centre() {
    let (w, b, _) = genes(50, 2, 5);
    let placer = Placer::new(&w, &b, 2, &Device::Cpu).unwrap();
    let centre = vec![0.5f32, -1.0];
    for inv_var in [0.0, 1.0] {
        let at = placer
            .place(&DMatrix::zeros(1, 50), &centre, inv_var)
            .unwrap();
        assert!(dist2(&at, &centre) < 1e-8, "{at:?}");
    }
}

/// A block is every cell on its own: the same as placing them one by one.
#[test]
fn a_block_of_cells_is_placed_cell_by_cell() {
    let (g, h) = (120, 3);
    let (w, b, x) = genes(g, h, 6);
    let placer = Placer::new(&w, &b, h, &Device::Cpu).unwrap();
    let r0 = reads_at(&w, &b, &x, 500.0, 7);
    let other = vec![-x[0], x[1], 0.5 * x[2]];
    let r1 = reads_at(&w, &b, &other, 3000.0, 8);
    let block = DMatrix::from_fn(2, g, |i, j| if i == 0 { r0[(0, j)] } else { r1[(0, j)] });
    let centre = vec![0f32, 0.1, 0.2, -0.3, 0.0, 0.4];
    let both = placer.place(&block, &centre, 0.5).unwrap();
    let one = placer.place(&r0, &centre[..h], 0.5).unwrap();
    let two = placer.place(&r1, &centre[h..], 0.5).unwrap();
    assert!(dist2(&both[..h], &one) < 1e-8);
    assert!(dist2(&both[h..], &two) < 1e-8);
}

/// A pseudobulk's own reads, placed with its prior mean and the fit's `σ`,
/// land at its fitted point (the same objective in it); a cell of a share
/// of them lands near it.
#[test]
fn a_cell_made_from_a_pseudobulks_reads_lands_near_its_point() {
    let dev = Device::Cpu;
    let cfg = SimConfig {
        n_genes: 60,
        n_pb: 120,
        lineages: 2,
        branch_at: Some(0.3),
        seed: 9,
        ..SimConfig::default()
    };
    let sim = simulate(&cfg).unwrap();
    let (p_n, g) = (sim.n_pb, sim.n_genes);
    let spec = CurveSpec {
        lineages: 2,
        points: 9,
        sub: 2,
        kappa: cfg.kappa,
        branch: Some(vec![0.3]),
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
    let y: Vec<f32> = sim.s.iter().map(|&v| v as f32).collect();
    let data = Pseudobulks::new(y.clone(), p_n, g, q).unwrap();
    let h = 3;
    let model = Model::init(&data, Curves::new(&spec, &dev).unwrap(), h, 2, &dev).unwrap();
    model.fit(3000, 0.02, None, 2).unwrap();
    let (x, m) = (model.x().unwrap(), super::model::prior_means(&data, &model));
    let sigma = model.sigma().unwrap();
    let placer = Placer::new(&model.w().unwrap(), &model.b().unwrap(), h, &dev).unwrap();
    let inv_var = 1.0 / (sigma * sigma);

    let all = DMatrix::from_fn(p_n, g, |p, j| y[p * g + j]);
    let back = placer.place(&all, &m, inv_var).unwrap();
    let spread: f32 = (0..p_n)
        .map(|p| {
            dist2(
                &x[p * h..(p + 1) * h],
                &x[((p + 1) % p_n) * h..((p + 1) % p_n + 1) * h],
            )
        })
        .sum::<f32>()
        / p_n as f32;
    let off: f32 = (0..p_n)
        .map(|p| dist2(&back[p * h..(p + 1) * h], &x[p * h..(p + 1) * h]))
        .sum::<f32>()
        / p_n as f32;
    eprintln!("placed back: mean square off {off:.5}, between neighbours {spread:.4}");
    assert!(off < 1e-3 * spread, "{off} vs {spread}");

    // A tenth of every pseudobulk's reads, as a cell.
    let mut rng = StdRng::seed_from_u64(10);
    let cells = DMatrix::from_fn(p_n, g, |p, j| {
        Binomial::new(y[p * g + j].round() as u64, 0.1)
            .unwrap()
            .sample(&mut rng) as f32
    });
    let placed = placer.place(&cells, &m, inv_var).unwrap();
    let near = (0..p_n)
        .filter(|&p| {
            let at = &placed[p * h..(p + 1) * h];
            let own = dist2(at, &x[p * h..(p + 1) * h]);
            let closer = (0..p_n)
                .filter(|&o| dist2(at, &x[o * h..(o + 1) * h]) < own)
                .count();
            closer < 10
        })
        .count();
    eprintln!("cells near their own pseudobulk: {near} of {p_n}");
    assert!(near * 10 >= p_n * 9, "{near} of {p_n}");
}

/// A cell's prior on the grid: every lineage's posterior spread on its
/// grid around the cell's time over the lineage's duration.
/// A time far past one lineage's end keeps that lineage's mass, on its
/// last position, and the prior sums to one.
#[test]
fn a_time_off_a_lineages_grid_keeps_its_mass_at_the_end() {
    let n = 41;
    let q = grid_prior(&[0.7, 0.3], 1.0, 0.0, &[1.0, 0.05], n).unwrap();
    let mass = |l: usize| -> f32 { q.iter().filter(|m| m.0 == l).map(|m| m.2).sum() };
    assert!((mass(1) - 0.3).abs() < 1e-5, "{}", mass(1));
    assert!((q.iter().map(|m| m.2).sum::<f32>() - 1.0).abs() < 1e-5);
    let last = q
        .iter()
        .filter(|m| m.0 == 1)
        .max_by(|a, b| a.2.total_cmp(&b.2));
    assert_eq!(last.unwrap().1, n - 1);
}

#[test]
fn a_cells_prior_puts_its_lineage_mass_around_its_phase() {
    let n = 11;
    let q = grid_prior(&[0.25, 0.75], 1.0, 0.1, &[2.0, 4.0], n).unwrap();
    let mass = |l: usize| -> f32 { q.iter().filter(|m| m.0 == l).map(|m| m.2).sum() };
    assert!((mass(0) - 0.25).abs() < 1e-5);
    assert!((mass(1) - 0.75).abs() < 1e-5);
    let mode = |l: usize| {
        q.iter()
            .filter(|m| m.0 == l)
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .unwrap()
            .1
    };
    // Phase 1/2 and 1/4 on ten steps.
    assert_eq!(mode(0), 5);
    assert!([2, 3].contains(&mode(1)));
    // A sharp time is a grid step wide at least.
    let sharp = grid_prior(&[1.0], 0.5, 0.0, &[1.0], n).unwrap();
    assert!(sharp.iter().filter(|m| m.2 > 0.01).count() >= 3);

    // Nothing known: no prior.
    assert!(grid_prior(&[f32::NAN], 0.5, 0.1, &[1.0], n).is_none());
    assert!(grid_prior(&[1.0], f32::NAN, 0.1, &[1.0], n).is_none());
}

#[test]
fn an_expectation_on_the_grid_weighs_every_row_by_its_mass() {
    // Two lineages of three positions, H = 2: row (l, k) is (l, k).
    let table: Vec<f32> = (0..2)
        .flat_map(|l| (0..3).flat_map(move |k| [l as f32, k as f32]))
        .collect();
    let at = expect_on_grid([(0, 2, 0.5), (1, 1, 0.5)], &table, 3, 2);
    assert_eq!(at, vec![0.5, 1.5]);
}
