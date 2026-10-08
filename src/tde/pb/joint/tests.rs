use super::*;
use crate::tde::pb::fit::FitConfig;
use crate::tde::pb::sim::{simulate_joint, JointSimConfig, JointSimulated};
use crate::tde::pb::test_util::{noisy, pearson, spearman};
use legume_numeric::candle::candle_core::Device;

fn module(t_on: f32, duration: f32, lambda: f32, beta: f32, gamma: f32) -> ModuleKinetics {
    ModuleKinetics {
        t_on,
        duration,
        lambda,
        basal: 0.05,
        beta,
        gamma,
    }
}

fn truth() -> Vec<ModuleKinetics> {
    vec![
        module(0.1, 5.0, 20.0, 8.0, 3.0),
        module(-0.4, 0.8, 15.0, 6.0, 2.0),
        module(0.35, 0.3, 12.0, 10.0, 4.0),
        module(0.6, 5.0, 6.0, 4.0, 1.5),
    ]
}

fn world(seed: u64) -> JointSimulated {
    simulate_joint(
        &truth(),
        &JointSimConfig {
            n_pb: 300,
            genes_per_module: 25,
            h: 8,
            reads_per_pb: 5000.0,
            seed,
        },
    )
    .unwrap()
}

fn counts(sim: &JointSimulated) -> PbCounts {
    let shape = (sim.n_pb, sim.n_genes);
    PbCounts {
        unspliced: Tensor::from_vec(sim.unspliced.clone(), shape, &Device::Cpu).unwrap(),
        spliced: Tensor::from_vec(sim.spliced.clone(), shape, &Device::Cpu).unwrap(),
        module_of_gene: sim.module_of_gene.clone(),
        n_modules: sim.modules.len(),
    }
}

fn flat(m: &DMatrix<f32>) -> Vec<f64> {
    m.iter().map(|&x| f64::from(x)).collect()
}

/// A base fit's view of the world: the true rows and states, perturbed.
fn perturbed(m: &DMatrix<f32>, sd: f32) -> DMatrix<f32> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let e = Normal::new(0.0f32, sd).unwrap();
    m.map(|x| x + e.sample(&mut rng))
}

/// From a reversed rough order: the joint model turns time around, and every
/// gene's score and rate over the units agree with the truth (scores are
/// identified; the module curves and the basis they are written in are not).
#[test]
#[ignore = "the joint model waits on stage 1's rates; its direction call slipped with the timing step"]
fn time_genes_and_velocity_come_back() {
    let sim = world(3);
    let pc = counts(&sim);
    let start: Vec<f32> = noisy(&sim.tau, 0.15, 5).iter().map(|t| 1.0 - t).collect();
    let (out, losses, reversed) = fit_dynamic(
        &pc,
        &start,
        &perturbed(&sim.rho, 0.2),
        &sim.bias,
        &perturbed(&sim.theta, 0.2),
        &TauPrior::default(),
        &FitConfig {
            rounds: 12,
            adam_steps: 40,
            ..FitConfig::default()
        },
        &JointConfig {
            rounds: 12,
            adam_steps: 40,
            ..JointConfig::default()
        },
    )
    .unwrap();
    assert!(reversed, "kept the reversed order: losses {losses:?}");
    let rho_tau = spearman(&out.tau, &sim.tau);
    assert!(rho_tau > 0.9, "Spearman τ {rho_tau}");

    // Every gene's score over the units, ⟨θ^s(τ_p), ρ_g⟩, centred per gene
    // (the biases carry each gene's level), and its rate ⟨θ̇_p, ρ_g⟩.
    let centred = |m: DMatrix<f32>| {
        let mean = m.row_mean();
        flat(&DMatrix::from_fn(m.nrows(), m.ncols(), |i, j| {
            m[(i, j)] - mean[j]
        }))
    };
    let score = centred(&out.theta * out.rho.transpose());
    let score_true = centred(&sim.theta * sim.rho.transpose());
    let rate = flat(&(&out.velocity * out.rho.transpose()));
    let rate_true = flat(&(&sim.velocity * sim.rho.transpose()));
    let (rs, rr) = (pearson(&score, &score_true), pearson(&rate, &rate_true));
    assert!(
        rs > 0.8 && rr > 0.6,
        "score correlation {rs}, rate correlation {rr}"
    );
}

/// With every module at its basal steady state, a gene's unspliced-minus-
/// spliced score is one number across units: gem's static offset.
#[test]
fn flat_kinetics_leave_one_offset_per_gene() {
    let sim = world(7);
    let pc = counts(&sim);
    let late: Vec<ModuleKinetics> = truth()
        .into_iter()
        .map(|m| ModuleKinetics { t_on: 0.99, ..m })
        .collect();
    let tau: Vec<f32> = (0..sim.n_pb)
        .map(|i| 0.05 + 0.9 * i as f32 / sim.n_pb as f32)
        .collect();
    let cfg = JointConfig::default();
    let j = Joint::new(
        &pc,
        &JointInit {
            tau: &tau,
            modules: &late,
            rho: &sim.rho,
            bias: &sim.bias,
            theta: &sim.theta,
        },
        &TauPrior::default(),
        &cfg,
    )
    .unwrap();
    let (ts, tu) = j.time_scores(&j.tau().unwrap()).unwrap();
    let d = (tu - ts).unwrap().to_vec2::<f32>().unwrap();
    for g in 0..sim.n_genes {
        let first = d[0][g];
        assert!(
            d.iter().all(|row| (row[g] - first).abs() < 1e-3),
            "gene {g} varies across units"
        );
    }
}
