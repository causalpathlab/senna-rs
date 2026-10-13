use super::direction::orient;
use super::*;
use crate::td_rbm::sim::{pearson, simulate, Sim, SimConfig};
use crate::td_rbm::start::{
    base_start, rho_start, row_states, scale_rows, start_along, STATE_DIMS,
};
use crate::td_rbm::tree::Forest;

/// Clusters of the simulated finest pseudobulks.
const CLUSTERS: usize = 6;
use crate::td_rbm::test_util::{noisy, spearman};

/// The cells `rows` as `[n × F]` row-major counts (unspliced genes first).
fn features(sim: &Sim, rows: &[usize]) -> Vec<f32> {
    let g = sim.n_genes;
    rows.iter()
        .flat_map(|&i| {
            sim.unspliced[i * g..(i + 1) * g]
                .iter()
                .chain(&sim.spliced[i * g..(i + 1) * g])
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The cells `rows` pooled in groups of `size` along `order`: `[P × F]`
/// counts and every pseudobulk's mean order.
fn pool(x: &[f32], f: usize, order: &[f32], size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut idx: Vec<usize> = (0..order.len()).collect();
    idx.sort_by(|&a, &b| order[a].total_cmp(&order[b]));
    let (mut out, mut mean) = (Vec::new(), Vec::new());
    for grp in idx.chunks(size) {
        let mut row = vec![0f32; f];
        for &i in grp {
            row.iter_mut()
                .zip(&x[i * f..(i + 1) * f])
                .for_each(|(a, b)| *a += b);
        }
        out.extend(row);
        mean.push(grp.iter().map(|&i| order[i]).sum::<f32>() / grp.len() as f32);
    }
    (out, mean)
}

/// Fit on the first `n_train` cells, pooled 50 and 10 a pseudobulk along a
/// start order within each true lineage (as expression clusters keep
/// lineages apart), with at most `cfg.lineages` lineages started by k-means;
/// read every cell. `reverse` starts every lineage the wrong way round.
fn fit_sim_from(
    sim: &Sim,
    n_train: usize,
    cfg: &TrainConfig,
    reverse: bool,
) -> (Model, Vec<f32>, Summary) {
    let (g, f) = (sim.n_genes, 2 * sim.n_genes);
    let train: Vec<usize> = (0..n_train).collect();
    let x = features(sim, &train);
    let noisy_start = noisy(&sim.tau, 0.2, 7);
    let start: Vec<f32> = noisy_start
        .iter()
        .map(|&t| if reverse { 1.0 - t } else { t })
        .collect();
    // Pool along (true lineage, start order), bystanders apart: a distinct
    // expression state never shares a pseudobulk with the trajectory, as the
    // collapse keeps clusters apart.
    let key: Vec<f32> = (0..n_train)
        .map(|i| {
            if sim.bystander[i] {
                100.0 + start[i]
            } else {
                sim.lineage[i] as f32 * 10.0 + start[i]
            }
        })
        .collect();
    let pooled: Vec<(Vec<f32>, Vec<f32>)> =
        [50, 10].iter().map(|&s| pool(&x, f, &key, s)).collect();
    let levels: Vec<Level> = pooled
        .iter()
        .zip([50f32, 10.0])
        .map(|((r, _), size)| Level {
            x: r.clone(),
            cells: vec![size; r.len() / f],
        })
        .collect();
    let (fin, keys) = &pooled[1];
    let p = keys.len();
    let pu: Vec<f32> = fin.chunks(f).flat_map(|r| r[..g].to_vec()).collect();
    let ps: Vec<f32> = fin.chunks(f).flat_map(|r| r[g..].to_vec()).collect();
    let rho = rho_start(&pu, &ps, p, g, cfg.embedding);
    let base = base_start(&ps, p, g);
    // The forest over the finest pseudobulks; every tree provisionally
    // rooted at the leaf earliest by the start order (latest when
    // `reverse`, though the start order itself already runs backwards then).
    let states = row_states(fin, f, STATE_DIMS);
    let forest = Forest::new(&states, CLUSTERS, cfg.knn, 3);
    let roots: Vec<usize> = (0..forest.trees.len())
        .map(|t| {
            let mean = |c: usize| {
                let v: Vec<f32> = (0..p)
                    .filter(|&r| forest.cluster[r] == c)
                    .map(|r| keys[r].fract())
                    .collect();
                v.iter().sum::<f32>() / v.len().max(1) as f32
            };
            *forest
                .leaves(t)
                .iter()
                .min_by(|&&x, &&y| mean(x).total_cmp(&mean(y)))
                .unwrap()
        })
        .collect();
    let values = Spline::new(cfg.knots, &grid_times(cfg.grid), &Device::Cpu)
        .unwrap()
        .values
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let begin = start_along(
        fin,
        f,
        &states,
        &forest,
        &roots,
        &rho,
        &values,
        (g, cfg.embedding, cfg.grid, cfg.knots),
        cfg.knn,
    );
    let cfg = TrainConfig {
        lineages: begin.paths.len(),
        ..cfg.clone()
    };
    let mut begin = begin;
    let curve = std::mem::take(&mut begin.curve);
    let model = fit(
        &levels,
        f,
        &Init0 {
            rho: &rho,
            base: &base,
            curve: &curve,
            beta: BETA0,
            gamma: GAMMA0,
        },
        begin,
        &forest,
        &states,
        &roots,
        &vec![false; forest.trees.len()],
        &cfg,
        &Device::Cpu,
    )
    .unwrap();
    let all: Vec<usize> = (0..sim.n_cells).collect();
    let cells = features(sim, &all);
    let out = model
        .summarise(&cells, f, cfg.depth as f32, &Device::Cpu)
        .unwrap();
    (model, start, out)
}

fn fit_sim(sim: &Sim, n_train: usize, cfg: &TrainConfig) -> (Model, Vec<f32>, Summary) {
    fit_sim_from(sim, n_train, cfg, false)
}

#[test]
fn the_fit_orders_held_out_cells_and_reads_their_velocity() {
    let sim = simulate(&SimConfig::default()).unwrap();
    let cfg = TrainConfig::default();
    let (model, start, out) = fit_sim(&sim, 1500, &cfg);
    let held: Vec<usize> = (1500..sim.n_cells).collect();
    let tau: Vec<f32> = held.iter().map(|&i| out.tau[i]).collect();
    let truth: Vec<f32> = held.iter().map(|&i| sim.tau[i]).collect();
    let begun: Vec<f32> = held.iter().map(|&i| start[i]).collect();
    let (r, r0) = (spearman(&tau, &truth), spearman(&begun, &truth));
    let emb = model.energy.rho().to_vec2::<f32>().unwrap();
    let vel: f64 = held
        .iter()
        .map(|&i| {
            let v = &out.velocity[i];
            let pred: Vec<f32> = emb
                .iter()
                .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
                .collect();
            pearson(&pred, &sim.log_slope[i])
        })
        .sum::<f64>()
        / held.len() as f64;
    eprintln!(
        "held-out Spearman {r:.3} (start {r0:.3}); gene velocity r {vel:.3}; roots {:?}",
        model
            .roots
            .iter()
            .map(|c| (c.provisional, c.chosen, c.determined))
            .collect::<Vec<_>>()
    );
    assert!(r > 0.95 && r > r0, "held-out Spearman {r}, start {r0}");
    assert!(vel > 0.7, "gene velocity correlation {vel}");
}

#[test]
fn bystanders_move_the_state_not_time() {
    let sim = simulate(&SimConfig {
        bystanders: 0.25,
        ..SimConfig::default()
    })
    .unwrap();
    // A distinct still state may take a lineage of its own.
    let cfg = TrainConfig {
        lineages: 2,
        ..TrainConfig::default()
    };
    let (_, _, out) = fit_sim(&sim, 1500, &cfg);
    let on: Vec<usize> = (1500..sim.n_cells).filter(|&i| !sim.bystander[i]).collect();
    let tau: Vec<f32> = on.iter().map(|&i| out.tau[i]).collect();
    let truth: Vec<f32> = on.iter().map(|&i| sim.tau[i]).collect();
    let r = spearman(&tau, &truth);
    // The state separates bystanders: the best single state axis.
    let d = out.state[0].len();
    let label: Vec<f32> = (0..sim.n_cells)
        .map(|i| f32::from(u8::from(sim.bystander[i])))
        .collect();
    let sep = (0..d)
        .map(|j| {
            let axis: Vec<f32> = out.state.iter().map(|h| h[j]).collect();
            pearson(&axis, &label).abs()
        })
        .fold(0.0, f64::max);
    eprintln!("trajectory cells' Spearman {r:.3}; best state axis vs bystander r {sep:.3}");
    assert!(r > 0.9, "trajectory Spearman {r}");
    assert!(sep > 0.8, "state separation {sep}");
}

#[test]
fn a_start_the_wrong_way_round_is_turned_by_the_splicing() {
    let sim = simulate(&SimConfig::default()).unwrap();
    let cfg = TrainConfig {
        ..TrainConfig::default()
    };
    let (model, _, out) = fit_sim_from(&sim, 1500, &cfg, true);
    let held: Vec<usize> = (1500..sim.n_cells).collect();
    let tau: Vec<f32> = held.iter().map(|&i| out.tau[i]).collect();
    let truth: Vec<f32> = held.iter().map(|&i| sim.tau[i]).collect();
    let r = spearman(&tau, &truth);
    let tree = model
        .roots
        .iter()
        .max_by_key(|c| c.candidates.len())
        .unwrap();
    eprintln!(
        "reversed start: held-out signed Spearman {r:.3}; root {} from {}, p {:?}",
        tree.chosen, tree.provisional, tree.p
    );
    assert!(
        tree.chosen != tree.provisional && tree.determined && r > 0.9,
        "signed Spearman {r}, root {} from {}",
        tree.chosen,
        tree.provisional
    );
}

#[test]
fn spare_lineages_empty_and_each_trajectory_gets_its_own() {
    let sim = simulate(&SimConfig {
        lineages: 2,
        ..SimConfig::default()
    })
    .unwrap();
    let cfg = TrainConfig {
        lineages: 4,
        ..TrainConfig::default()
    };
    let (model, _, out) = fit_sim(&sim, 1500, &cfg);
    let used = model.log_weight.iter().filter(|&&w| w > -1e3).count();
    let held: Vec<usize> = (1500..sim.n_cells).collect();
    // Purity: the share of held-out cells whose fitted lineage holds mostly
    // cells of their own true lineage.
    let mut table = vec![vec![0usize; 2]; cfg.lineages];
    for &i in &held {
        table[out.lineage[i]][sim.lineage[i]] += 1;
    }
    let pure: usize = table.iter().map(|r| *r.iter().max().unwrap()).sum();
    let purity = pure as f64 / held.len() as f64;
    let per: Vec<f64> = (0..2)
        .map(|lin| {
            let rows: Vec<usize> = held
                .iter()
                .copied()
                .filter(|&i| sim.lineage[i] == lin)
                .collect();
            let tau: Vec<f32> = rows.iter().map(|&i| out.tau[i]).collect();
            let truth: Vec<f32> = rows.iter().map(|&i| sim.tau[i]).collect();
            spearman(&tau, &truth)
        })
        .collect();
    eprintln!("lineages used {used}; purity {purity:.3}; per-lineage signed Spearman {per:?}; table {table:?}");
    assert!(purity > 0.95, "purity {purity}");
    assert!(
        per.iter().all(|r| r.abs() > 0.9),
        "per-lineage Spearman {per:?}"
    );
}

/// The truth against its mirror: the simulation's own curve, embedding and
/// rates, forward and with the curve run backwards (velocity reversed), by
/// the exact time-only likelihood of the cells.
#[test]
#[ignore = "diagnostic"]
fn diagnostic_truth_against_its_mirror() {
    use crate::td_rbm::model::time_tables;
    use crate::td_rbm::rbm::lineage_log_likelihood;
    use legume_numeric::candle::candle_nn::ops::log_softmax;
    let sim = simulate(&SimConfig::default()).unwrap();
    let dev = Device::Cpu;
    let k = 32;
    let spline = Spline::new(5, &grid_times(k), &dev).unwrap();
    let m = |v: &Vec<Vec<f64>>| {
        let (r, c) = (v.len(), v[0].len());
        Tensor::from_vec(
            v.iter().flatten().map(|&x| x as f32).collect::<Vec<_>>(),
            (r, c),
            &dev,
        )
        .unwrap()
    };
    let row = |v: &Vec<f64>| {
        Tensor::from_vec(
            v.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            (1, v.len()),
            &dev,
        )
        .unwrap()
    };
    let (rho, coef) = (m(&sim.rho), m(&sim.coef));
    let all: Vec<usize> = (0..sim.n_cells).collect();
    let mut cells = features(&sim, &all);
    scale_rows(&mut cells, 2 * sim.n_genes, 1000.0);
    let r = Reads::from_features(
        Tensor::from_vec(cells, (sim.n_cells, 2 * sim.n_genes), &dev).unwrap(),
    )
    .unwrap();
    for (name, c) in [
        ("forward", coef.clone()),
        (
            "mirrored",
            coef.index_select(
                &Tensor::from_vec((0..5u32).rev().collect::<Vec<_>>(), 5, &dev).unwrap(),
                0,
            )
            .unwrap(),
        ),
    ] {
        let curve = spline.values.matmul(&c).unwrap();
        let velocity = spline.slopes.matmul(&c).unwrap();
        let size = Tensor::zeros(
            (k, 1),
            legume_numeric::candle::candle_core::DType::F32,
            &dev,
        )
        .unwrap();
        let psi = log_softmax(
            &time_tables(
                &rho,
                &curve,
                &velocity,
                &size,
                &row(&sim.base),
                &row(&sim.beta),
                &row(&sim.gamma),
            )
            .unwrap(),
            1,
        )
        .unwrap();
        let t = Tables {
            psi,
            bias: Tensor::zeros(
                (1, k),
                legume_numeric::candle::candle_core::DType::F32,
                &dev,
            )
            .unwrap(),
            w: Tensor::zeros(
                (2 * sim.n_genes, 1),
                legume_numeric::candle::candle_core::DType::F32,
                &dev,
            )
            .unwrap(),
            curve,
            velocity,
            size,
            lineages: 1,
        };
        let ll = lineage_log_likelihood(&r, &t)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("{name}: exact log-likelihood of the cells {ll:.1}");
    }
}

/// `orient` at the simulation's truth with no training calls the direction
/// right from either orientation, and makes no call once the unspliced reads
/// carry no splicing.
#[test]
fn direction_called_only_with_splicing() {
    use crate::td_rbm::model::{unbounded, LOG_RATE};
    let sim = simulate(&SimConfig::default()).unwrap();
    let dev = Device::Cpu;
    let (g, f) = (sim.n_genes, 2 * sim.n_genes);
    let h = sim.rho[0].len();
    let rho: Vec<f32> = sim.rho.iter().flatten().map(|&x| x as f32).collect();
    let base: Vec<f32> = sim.base.iter().map(|&x| x as f32).collect();
    let curve: Vec<f32> = sim.coef.iter().flatten().map(|&x| x as f32).collect();
    let cfg = TrainConfig {
        knots: 5,
        embedding: h,
        ..TrainConfig::default()
    };
    let energy = Energy::new(
        Shape {
            genes: g,
            embedding: h,
            knots: 5,
            states: 1,
            lineages: 1,
        },
        &Init0 {
            rho: &rho,
            base: &base,
            curve: &curve,
            beta: 4.0,
            gamma: 1.0,
        },
        1,
        &dev,
    )
    .unwrap();
    let free = |v: &Vec<f64>| {
        Tensor::from_vec(
            v.iter()
                .map(|&r| unbounded(r.ln(), LOG_RATE) as f32)
                .collect::<Vec<_>>(),
            (1, g),
            &dev,
        )
        .unwrap()
    };
    let snap = energy.snapshot().unwrap();
    energy
        .restore(&[
            snap[0].clone(),
            snap[1].clone(),
            free(&sim.beta),
            free(&sim.gamma),
        ])
        .unwrap();
    let spline = Spline::new(5, &grid_times(cfg.grid), &dev).unwrap();
    let all: Vec<usize> = (0..sim.n_cells).collect();
    let x = features(&sim, &all);
    let mut rng = StdRng::seed_from_u64(1);
    let restore = |mirror: bool| {
        energy
            .restore(&[
                snap[0].clone(),
                snap[1].clone(),
                free(&sim.beta),
                free(&sim.gamma),
            ])
            .unwrap();
        if mirror {
            energy.mirror(0).unwrap();
        }
    };
    for (name, mirror) in [("truth", false), ("mirrored truth", true)] {
        restore(mirror);
        let (d, _) = orient(
            &energy,
            &spline,
            &energy.steady_tables(&spline).unwrap(),
            &x,
            f,
            0,
            &cfg,
            &dev,
            &mut rng,
        )
        .unwrap();
        eprintln!(
            "from the {name}: p(as started) {:.4}, log BF {:+.1} (sd {:.1}), log ABF {:+.1}",
            d.p_forward, d.log_bf, d.log_bf_sd, d.log_abf
        );
        let p = if mirror {
            1.0 - d.p_forward
        } else {
            d.p_forward
        };
        assert!(p > 0.99, "the truth's direction is called from the {name}");
    }
    // No dynamics: every cell's reads of a gene split between the tracks at
    // the gene's overall unspliced share, the same either way in time. (The
    // unspliced reads shuffled across cells are not that: a flat `u` is a
    // relaxation of `s` towards `βu/γ`, which runs one way.)
    let mut steady = sim.clone();
    let n = sim.n_cells;
    let mut draw = StdRng::seed_from_u64(7);
    for j in 0..g {
        let (u, t) = (0..n).fold((0f64, 0f64), |(u, t), i| {
            let (a, b) = (
                sim.unspliced[i * g + j] as f64,
                sim.spliced[i * g + j] as f64,
            );
            (u + a, t + a + b)
        });
        let share = u / t.max(1.0);
        for i in 0..n {
            let total = (sim.unspliced[i * g + j] + sim.spliced[i * g + j]).round() as u64;
            let a = rand_distr::Binomial::new(total, share)
                .map_or(0, |b| rand_distr::Distribution::sample(&b, &mut draw));
            steady.unspliced[i * g + j] = a as _;
            steady.spliced[i * g + j] = (total - a) as _;
        }
    }
    let x = features(&steady, &all);
    restore(false);
    let (d, _) = orient(
        &energy,
        &spline,
        &energy.steady_tables(&spline).unwrap(),
        &x,
        f,
        0,
        &cfg,
        &dev,
        &mut rng,
    )
    .unwrap();
    eprintln!(
        "no dynamics: p(as started) {:.4}, log BF {:+.1} (sd {:.1}), log ABF {:+.1}",
        d.p_forward, d.log_bf, d.log_bf_sd, d.log_abf
    );
    assert!(
        d.p_forward > 0.01 && d.p_forward < 0.99,
        "no call without splicing"
    );
}

/// How the gene velocity's accuracy depends on the start (the reseed's
/// neighbours) and on the curves' roughness prior.
#[test]
#[ignore = "diagnostic"]
fn diagnostic_velocity_against_start_and_smoothness() {
    let sim = simulate(&SimConfig::default()).unwrap();
    for smooth in [0.0, 1e2, 1e3, 1e4, 1e5] {
        for knn in [10usize, 15] {
            let cfg = TrainConfig {
                knn,
                smooth,
                ..TrainConfig::default()
            };
            let (model, _, out) = fit_sim(&sim, 1500, &cfg);
            let held: Vec<usize> = (1500..sim.n_cells).collect();
            let tau: Vec<f32> = held.iter().map(|&i| out.tau[i]).collect();
            let truth: Vec<f32> = held.iter().map(|&i| sim.tau[i]).collect();
            let emb = model.energy.rho().to_vec2::<f32>().unwrap();
            let vel: f64 = held
                .iter()
                .map(|&i| {
                    let v = &out.velocity[i];
                    let pred: Vec<f32> = emb
                        .iter()
                        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
                        .collect();
                    pearson(&pred, &sim.log_slope[i])
                })
                .sum::<f64>()
                / held.len() as f64;
            let rough = model
                .energy
                .roughness()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!(
                "smooth {smooth:>8}: knn {knn}: ordering {:.3}, gene velocity {vel:.3}, roughness {rough:.3}",
                spearman(&tau, &truth)
            );
        }
    }
}
