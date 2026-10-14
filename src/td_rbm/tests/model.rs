use super::*;
use crate::td_rbm::curve::Spline;
use legume_numeric::candle::candle_core::{DType, Device, Var};

const G: usize = 6;
const H: usize = 3;

fn rho_values() -> Vec<f32> {
    (0..G * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) / 4.0)
        .collect()
}

fn rho() -> Tensor {
    Tensor::from_vec(rho_values(), (G, H), &Device::Cpu).unwrap()
}

fn row(v: &[f32]) -> Tensor {
    Tensor::from_slice(v, (1, v.len()), &Device::Cpu).unwrap()
}

fn zeros(shape: (usize, usize)) -> Tensor {
    Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
}

/// The log odds of unspliced over spliced of every gene, at one state.
fn odds(curve: &[f32], velocity: &Tensor, gamma: f32) -> Vec<f32> {
    let psi = time_tables(
        &rho(),
        &row(curve),
        velocity,
        &zeros((1, 1)),
        &zeros((1, G)),
        &row(&[2.0; G]),
        &row(&[gamma; G]),
    )
    .unwrap()
    .squeeze(0)
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    (0..G).map(|g| psi[g] - psi[G + g]).collect()
}

#[test]
fn at_rest_every_gene_sits_at_its_steady_ratio() {
    for o in odds(&[0.3, -0.2, 0.5], &zeros((1, H)), 3.0) {
        // u/s = γ/β = 3/2.
        assert!((o - 1.5f32.ln()).abs() < 1e-4, "{o}");
    }
}

#[test]
fn unspliced_follows_the_splicing_identity() {
    // β u/s = γ + d log s/dτ along a straight curve z(τ) = z0 + τ v, s the
    // spliced softmax share.
    let (z0, v) = ([0.3f32, -0.2, 0.5], [1.0f32, -2.0, 0.5]);
    let gamma = 6.0f32;
    let share = |tau: f32| -> Vec<f32> {
        let z: Vec<f32> = z0.iter().zip(&v).map(|(a, b)| a + tau * b).collect();
        let l = row(&z).matmul(&rho().t().unwrap()).unwrap();
        log_softmax(&l, 1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let h = 1e-3f32;
    let (up, down) = (share(h), share(-h));
    let o = odds(&z0, &row(&v), gamma);
    for g in 0..G {
        let dlog = (up[g] - down[g]) / (2.0 * h);
        let lhs = 2.0 * o[g].exp(); // β u/s
        assert!(
            (lhs - (gamma + dlog)).abs() < 1e-2 * (1.0 + dlog.abs()),
            "gene {g}: {lhs} vs {}",
            gamma + dlog
        );
    }
}

#[test]
fn reversing_time_flips_every_genes_excess_unspliced() {
    let (z0, v) = ([0.3f32, -0.2, 0.5], [0.4f32, -0.6, 0.2]);
    let back: Vec<f32> = v.iter().map(|x| -x).collect();
    let (f, b) = (odds(&z0, &row(&v), 6.0), odds(&z0, &row(&back), 6.0));
    let steady = 3f32.ln();
    for g in 0..G {
        let (f, b) = (f[g] - steady, b[g] - steady);
        assert!(f * b <= 1e-6, "gene {g}: forward {f}, backward {b}");
    }
}

#[test]
fn a_gene_falling_faster_than_it_decays_keeps_a_finite_gradient() {
    let vel = Var::from_tensor(&row(&[50.0, -80.0, 30.0])).unwrap();
    let psi = time_tables(
        &rho(),
        &zeros((1, H)),
        vel.as_tensor(),
        &zeros((1, 1)),
        &zeros((1, G)),
        &row(&[2.0; G]),
        &row(&[1.0; G]),
    )
    .unwrap();
    let v = psi.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
    let o: Vec<f32> = (0..G).map(|g| v[g] - v[G + g]).collect();
    assert!(o.iter().all(|r| r.is_finite()));
    assert!(
        o.iter().any(|&r| r < -5.0),
        "some gene should fall too fast: {o:?}"
    );
    let grads = psi.sum_all().unwrap().backward().unwrap();
    let g = grads.get(&vel).unwrap().to_vec2::<f32>().unwrap();
    assert!(g[0].iter().all(|x| x.is_finite()));
}

#[test]
fn states_never_touch_the_unspliced_share_nor_the_curves_span() {
    let shape = Shape {
        genes: G,
        embedding: H,
        knots: 4,
        states: 2,
        lineages: 1,
    };
    let base = vec![0f32; G];
    let rho = rho_values();
    // A curve travelling along the first axis only.
    let curve: Vec<f32> = (0..4).flat_map(|m| [m as f32, 0.0, 0.0]).collect();
    let e = Energy::new(
        shape,
        &Init0 {
            rho: &rho,
            base: &base,
            curve: &curve,
            beta: 2.0,
            gamma: 1.0,
            given: None,
            modules: None,
        },
        3,
        &Device::Cpu,
    )
    .unwrap();
    let spline = Spline::new(4, &crate::td_rbm::curve::grid_times(8), &Device::Cpu).unwrap();
    let t = e.tables(&spline, true, &[0.0]).unwrap();
    let w = t.w.to_vec2::<f32>().unwrap();
    for g in 0..G {
        assert_eq!(w[g], w[G + g]);
    }
    let coef = Tensor::from_vec(curve.clone(), (4, H), &Device::Cpu).unwrap();
    let load = crate::td_rbm::model::without_span(e.load_var().as_tensor(), &coef, &[0.0])
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert!(
        load[0].iter().all(|x| x.abs() < 1e-6),
        "state along the curve: {:?}",
        load[0]
    );
    assert!(load[1].iter().chain(&load[2]).any(|x| x.abs() > 1e-3));
    assert_eq!(t.psi.dims2().unwrap(), (8, 2 * G));
    let b = t.bias.to_vec2::<f32>().unwrap()[0].clone();
    assert!(b.iter().all(|x| (x + 8f32.ln()).abs() < 1e-6));
}

/// Curves that between them span the whole embedding still leave the states
/// their `d` directions (the least travelled), and a dead lineage's curve
/// takes none.
#[test]
fn crowded_curves_leave_the_states_their_directions() {
    use crate::td_rbm::model::without_span;
    let h = 3;
    // Lineage 1 travels along axes 0 and 1 (most along 0); lineage 2 along
    // axis 2; lineage 3, dead, along axis 1 only.
    let rows: Vec<f32> = [
        [0.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 9.0, 0.0],
        [0.0, 18.0, 0.0],
    ]
    .concat();
    let coef = Tensor::from_vec(rows, (9, h), &Device::Cpu).unwrap();
    let load = Tensor::from_vec(vec![1f32, 0.5, 0.2, 1.0, 0.3, 0.7], (h, 2), &Device::Cpu).unwrap();
    let out = without_span(&load, &coef, &[0.0, 0.0, f32::NEG_INFINITY]).unwrap();
    let out = out.to_vec2::<f32>().unwrap();
    // One direction removed (H − d = 1): the most travelled, axis 0.
    assert!(out[0].iter().all(|x| x.abs() < 1e-5), "{out:?}");
    assert!(
        (out[1][0] - 0.2).abs() < 1e-5 && (out[2][1] - 0.7).abs() < 1e-5,
        "{out:?}"
    );
}

/// A lineage's speed scales its velocity in the shared rates' time, and
/// only the speeds' ratios count: a common shift of the log speeds leaves
/// every table as it was.
#[test]
fn a_lineages_speed_scales_its_velocity_and_only_ratios_count() {
    let shape = Shape {
        genes: G,
        embedding: H,
        knots: 5,
        states: 1,
        lineages: 2,
    };
    let base = vec![0f32; G];
    let rho = rho_values();
    // The same curve for both lineages.
    let one: Vec<f32> = (0..5)
        .flat_map(|m| [m as f32 * 0.3, (m * m) as f32 * 0.05, -(m as f32) * 0.1])
        .collect();
    let curve = [one.clone(), one].concat();
    let e = Energy::new(
        shape,
        &Init0 {
            rho: &rho,
            base: &base,
            curve: &curve,
            beta: 2.0,
            gamma: 1.0,
            given: None,
            modules: None,
        },
        3,
        &Device::Cpu,
    )
    .unwrap();
    let spline = Spline::new(5, &crate::td_rbm::curve::grid_times(8), &Device::Cpu).unwrap();
    let speed = e.vars().pop().unwrap();
    let set = |a: f32, b: f32| {
        speed
            .set(&Tensor::from_vec(vec![a, b], (2, 1), &Device::Cpu).unwrap())
            .unwrap()
    };
    set(2f32.ln(), 0.0);
    let (x0, x1) = (
        e.velocity_readout(&spline, 0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
        e.velocity_readout(&spline, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
    );
    for (a, b) in x0.iter().flatten().zip(x1.iter().flatten()) {
        assert!((a - 2.0 * b).abs() < 1e-5, "{a} against twice {b}");
    }
    let before = e.tables(&spline, false, &[0.0, 0.0]).unwrap().psi;
    set(2f32.ln() + 1.0, 1.0);
    let after = e.tables(&spline, false, &[0.0, 0.0]).unwrap().psi;
    let gap = (before - after).unwrap().abs().unwrap().max_all().unwrap();
    assert!(gap.to_scalar::<f32>().unwrap() < 1e-6);
}

#[test]
fn mirroring_a_lineage_runs_its_curve_backwards_exactly() {
    let shape = Shape {
        genes: G,
        embedding: H,
        knots: 5,
        states: 1,
        lineages: 2,
    };
    let base = vec![0f32; G];
    let rho = rho_values();
    let curve: Vec<f32> = (0..10)
        .flat_map(|m| [m as f32 * 0.3, (m * m) as f32 * 0.05, -(m as f32) * 0.1])
        .collect();
    let e = Energy::new(
        shape,
        &Init0 {
            rho: &rho,
            base: &base,
            curve: &curve,
            beta: 2.0,
            gamma: 1.0,
            given: None,
            modules: None,
        },
        3,
        &Device::Cpu,
    )
    .unwrap();
    let times = crate::td_rbm::curve::grid_times(16);
    let spline = Spline::new(5, &times, &Device::Cpu).unwrap();
    let before = e.tables(&spline, false, &[0.0, 0.0]).unwrap();
    e.mirror(1).unwrap();
    let after = e.tables(&spline, false, &[0.0, 0.0]).unwrap();
    let (c0, c1) = (
        before.curve.to_vec2::<f32>().unwrap(),
        after.curve.to_vec2::<f32>().unwrap(),
    );
    let (v0, v1) = (
        before.velocity.to_vec2::<f32>().unwrap(),
        after.velocity.to_vec2::<f32>().unwrap(),
    );
    for k in 0..16 {
        // Lineage 1 untouched; lineage 2 at τ is lineage 2 at 1 − τ, velocity reversed.
        assert_eq!(c0[k], c1[k]);
        let (a, b) = (16 + k, 16 + 15 - k);
        for d in 0..H {
            assert!((c1[a][d] - c0[b][d]).abs() < 1e-4);
            assert!((v1[a][d] + v0[b][d]).abs() < 1e-3);
        }
    }
    e.mirror(1).unwrap();
    let back = e
        .tables(&spline, false, &[0.0, 0.0])
        .unwrap()
        .curve
        .to_vec2::<f32>()
        .unwrap();
    assert!(back
        .iter()
        .flatten()
        .zip(c0.iter().flatten())
        .all(|(x, y)| (x - y).abs() < 1e-6));
}

/// A given table's rows under each mode, one optimiser step on a loss that
/// moves every row: `Freeze` holds them, `Lora` starts at them and moves
/// only through its residual, `Init` lets them train; free rows always do.
#[test]
fn a_given_tables_rows_are_held_as_its_mode_says() {
    use legume_numeric::candle::candle_nn::Optimizer;
    let shape = Shape {
        genes: G,
        embedding: H,
        knots: 4,
        states: 2,
        lineages: 1,
    };
    let (base, rho) = (vec![0f32; G], rho_values());
    let curve = vec![0.1f32; 4 * H];
    // Genes 0 and 2 come from the table.
    let mut mask = vec![0f32; G];
    mask[0] = 1.0;
    mask[2] = 1.0;
    let step = |pinned: bool, lora_rank: Option<usize>| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let e = Energy::new(
            shape,
            &Init0 {
                rho: &rho,
                base: &base,
                curve: &curve,
                beta: 2.0,
                gamma: 1.0,
                given: Some(GivenRho {
                    mask: &mask,
                    pinned,
                    lora_rank,
                }),
                modules: None,
            },
            3,
            &Device::Cpu,
        )
        .unwrap();
        let before = e.rho().unwrap().to_vec2::<f32>().unwrap();
        let mut vars = e.vars();
        vars.extend(e.lora_shared());
        let mut opt = crate::rbm_core::adam(vars, 0.1).unwrap();
        let loss = (e.rho().unwrap() - 10.0)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        opt.backward_step(&loss).unwrap();
        (before, e.rho().unwrap().to_vec2::<f32>().unwrap())
    };
    let moved = |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6);
    let given: Vec<Vec<f32>> = rho.chunks(H).map(<[f32]>::to_vec).collect();

    let (before, after) = step(true, None);
    assert_eq!(before, given, "starts at the table");
    assert!(
        !moved(&after[0], &given[0]) && !moved(&after[2], &given[2]),
        "frozen rows held"
    );
    assert!(moved(&after[1], &before[1]), "a free row trains");

    let (before, after) = step(true, Some(2));
    assert_eq!(before, given, "the residual starts at zero");
    assert!(moved(&after[0], &given[0]), "the residual moves a held row");

    let (_, after) = step(false, None);
    assert!(moved(&after[0], &given[0]), "an init row trains");
}

/// With gene modules a gene's row of ρ is its module's vector plus its
/// residual, and the ridge is the residuals' mean square norm.
#[test]
fn modules_compose_rho_from_their_vectors_and_residuals() {
    let shape = Shape {
        genes: G,
        embedding: H,
        knots: 4,
        states: 2,
        lineages: 1,
    };
    let (base, curve) = (vec![0f32; G], vec![0.1f32; 4 * H]);
    // Genes 0-2 in module 0, 3-5 in module 1; residuals 0 but on gene 4.
    let labels = [0u32, 0, 0, 1, 1, 1];
    let mu = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
    let mut residual = vec![0f32; G * H];
    residual[4 * H..5 * H].copy_from_slice(&[0.5, 0.5, 0.5]);
    let e = Energy::new(
        shape,
        &Init0 {
            rho: &residual,
            base: &base,
            curve: &curve,
            beta: 2.0,
            gamma: 1.0,
            given: None,
            modules: Some(ModulesInit {
                labels: &labels,
                mu: &mu,
            }),
        },
        3,
        &Device::Cpu,
    )
    .unwrap();
    let rho = e.rho().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(rho[1], vec![1.0, 2.0, 3.0]);
    assert_eq!(rho[4], vec![-0.5, -1.5, -2.5]);
    let pen = e
        .module_penalty()
        .unwrap()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!((pen - 0.75 / G as f32).abs() < 1e-6, "{pen}");
}
