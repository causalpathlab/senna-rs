use super::*;
use crate::tde::pb::sim::{simulate, SimConfig, Simulated};
use crate::tde::pb::test_util::{noisy, spearman};

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

/// Induction that stays on, repression of a module already on, a transient
/// pulse, and a late slow induction.
fn truth() -> Vec<ModuleKinetics> {
    vec![
        module(0.1, 5.0, 20.0, 8.0, 3.0),
        module(-0.4, 0.8, 15.0, 6.0, 2.0),
        module(0.35, 0.3, 12.0, 10.0, 4.0),
        module(0.6, 5.0, 6.0, 4.0, 1.5),
    ]
}

fn counts(sim: &Simulated, dev: &Device) -> PbCounts {
    let shape = (sim.n_pb, sim.n_genes);
    PbCounts {
        unspliced: Tensor::from_vec(sim.unspliced.clone(), shape, dev).unwrap(),
        spliced: Tensor::from_vec(sim.spliced.clone(), shape, dev).unwrap(),
        module_of_gene: sim.module_of_gene.clone(),
        n_modules: sim.modules.len(),
    }
}

fn small_sim(seed: u64) -> Simulated {
    simulate(
        &truth(),
        &SimConfig {
            n_pb: 300,
            genes_per_module: 25,
            seed,
            ..SimConfig::default()
        },
    )
    .unwrap()
}

fn quick() -> FitConfig {
    FitConfig {
        rounds: 12,
        adam_steps: 40,
        ..FitConfig::default()
    }
}

#[test]
fn grid_step_recovers_tau_at_the_true_kinetics() {
    let sim = small_sim(3);
    let pc = counts(&sim, &Device::Cpu);
    let cfg = quick();
    let mut run = Run::new(&pc, &vec![0.5; sim.n_pb], None, &TauPrior::default(), &cfg).unwrap();
    let truth = FreeKinetics::new(&sim.modules, &Device::Cpu).unwrap();
    for (old, new) in run.free.vars().iter().zip(truth.vars()) {
        old.set(new.as_tensor()).unwrap();
    }
    run.offset
        .set(&Tensor::from_vec(sim.offset.clone(), (1, sim.n_genes), &Device::Cpu).unwrap())
        .unwrap();
    // The modules' loadings where the truth puts them, then every time reset.
    let set_tau = |run: &mut Run, tau: &[f32]| {
        let z: Vec<f32> = tau.iter().map(|&t| logit(f64::from(t)) as f32).collect();
        run.z
            .set(&Tensor::from_vec(z, tau.len(), &Device::Cpu).unwrap())
            .unwrap();
    };
    set_tau(&mut run, &sim.tau);
    run.init_loadings().unwrap();
    set_tau(&mut run, &vec![0.5; sim.n_pb]);
    run.grid_step().unwrap();
    let tau = run.tau().unwrap().to_vec1::<f32>().unwrap();
    let rho = spearman(&tau, &sim.tau);
    assert!(rho > 0.95, "Spearman {rho}");
}

#[test]
fn fit_recovers_tau_from_a_rough_order() {
    let sim = small_sim(5);
    let pc = counts(&sim, &Device::Cpu);
    let out = fit(
        &pc,
        &noisy(&sim.tau, 0.15, 7),
        &TauPrior::default(),
        &quick(),
    )
    .unwrap();
    let rho = spearman(&out.tau, &sim.tau);
    assert!(
        !out.reversed,
        "orientation losses {:?}",
        out.orientation_loss
    );
    assert!(rho > 0.9, "Spearman {rho}");
    assert!(
        out.loss.last().unwrap() < out.loss.first().unwrap(),
        "loss {:?}",
        out.loss
    );
}

#[test]
fn fit_turns_a_reversed_order_around() {
    let sim = small_sim(11);
    let pc = counts(&sim, &Device::Cpu);
    let start: Vec<f32> = noisy(&sim.tau, 0.15, 13).iter().map(|t| 1.0 - t).collect();
    let out = fit(&pc, &start, &TauPrior::default(), &quick()).unwrap();
    assert!(
        out.reversed,
        "orientation losses {:?}",
        out.orientation_loss
    );
    let rho = spearman(&out.tau, &sim.tau);
    assert!(rho > 0.9, "Spearman {rho}");
}

/// The truth mapped onto the fitted time axis `τ̂ ≈ a + b τ`: times move
/// affinely, rates divide by `b`.
fn aligned(m: &ModuleKinetics, a: f32, b: f32) -> ModuleKinetics {
    ModuleKinetics {
        t_on: a + b * m.t_on,
        duration: b * m.duration,
        lambda: m.lambda / b,
        basal: m.basal,
        beta: m.beta / b,
        gamma: m.gamma / b,
    }
}

fn affine(x: &[f32], y: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let (mx, my) = (x.iter().sum::<f32>() / n, y.iter().sum::<f32>() / n);
    let sxy: f32 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f32 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let b = sxy / sxx;
    (my - b * mx, b)
}

/// A recovery table across read depths: `cargo test --release --bin senna
/// recovery_report -- --ignored --nocapture` (CUDA when built with it).
#[test]
#[ignore = "a report, not a check"]
fn recovery_report() {
    let dev = Device::cuda_if_available(0).unwrap();
    for mean_reads in [5.0, 20.0, 80.0] {
        let sim = simulate(
            &truth(),
            &SimConfig {
                n_pb: 800,
                genes_per_module: 40,
                mean_reads,
                ..SimConfig::default()
            },
        )
        .unwrap();
        let pc = counts(&sim, &dev);
        let t0 = std::time::Instant::now();
        let out = fit(
            &pc,
            &noisy(&sim.tau, 0.15, 7),
            &TauPrior::default(),
            &FitConfig::default(),
        )
        .unwrap();
        let (a, b) = affine(&sim.tau, &out.tau);
        let mae: f32 = sim
            .tau
            .iter()
            .zip(&out.tau)
            .map(|(t, e)| (a + b * t - e).abs())
            .sum::<f32>()
            / sim.n_pb as f32;
        println!(
            "\nreads/pb/gene {mean_reads}: Spearman τ {:.3}, τ̂ ≈ {a:.2} + {b:.2} τ, MAE {mae:.3}, \
             reversed {}, loss {:.4} → {:.4}, {:.1}s",
            spearman(&out.tau, &sim.tau),
            out.reversed,
            out.loss.first().unwrap(),
            out.loss.last().unwrap(),
            t0.elapsed().as_secs_f32()
        );
        println!(
            "  module   t_on        t_off       λ             β             γ            basal"
        );
        for (m, (tr, est)) in sim.modules.iter().zip(&out.modules).enumerate() {
            let tr = aligned(tr, a, b);
            println!(
                "  {m}  {:>5.2}/{:<5.2} {:>5.2}/{:<5.2} {:>6.1}/{:<6.1} {:>6.1}/{:<6.1} {:>6.1}/{:<6.1} {:.3}/{:.3}",
                tr.t_on,
                est.t_on,
                tr.t_on + tr.duration,
                est.t_on + est.duration,
                tr.lambda,
                est.lambda,
                tr.beta,
                est.beta,
                tr.gamma,
                est.gamma,
                tr.basal,
                est.basal
            );
        }
    }
}

/// Loss per read at the given kinetics and times, with every gene's offset
/// and every pseudobulk's κ profiled.
fn loss_at(run: &mut Run, modules: &[ModuleKinetics], tau: &[f32]) -> f64 {
    let dev = run.xu.device().clone();
    let fixed = FreeKinetics::new(modules, &dev).unwrap();
    for (old, new) in run.free.vars().iter().zip(fixed.vars()) {
        old.set(new.as_tensor()).unwrap();
    }
    let z: Vec<f32> = tau.iter().map(|&t| logit(f64::from(t)) as f32).collect();
    run.z
        .set(&Tensor::from_vec(z, tau.len(), &dev).unwrap())
        .unwrap();
    for _ in 0..20 {
        let a = run.logits().unwrap().0.detach();
        let curve = a.broadcast_sub(run.offset.as_tensor()).unwrap();
        let kappa = run.profiled_kappa(&a).unwrap();
        run.kappa.set(&kappa).unwrap();
        let b = profile_offset(
            &curve.broadcast_add(&kappa).unwrap(),
            &run.xu,
            &run.n,
            run.offset.as_tensor(),
            0,
        )
        .unwrap();
        run.offset.set(&b).unwrap();
    }
    run.evaluate().unwrap()
}

/// Why a start is turned the wrong way: the loss at the truth against both
/// orientations fitted to the end. `cargo test --release --bin senna
/// orientation_diagnosis -- --ignored --nocapture`.
#[test]
#[ignore = "a report, not a check"]
fn orientation_diagnosis() {
    let cfg = FitConfig {
        rounds: 30,
        ..quick()
    };
    for seed in [5u64, 11, 17] {
        let sim = small_sim(seed);
        let pc = counts(&sim, &Device::Cpu);
        let start = noisy(&sim.tau, 0.15, 7);
        let reverse: Vec<f32> = start.iter().map(|t| 1.0 - t).collect();
        let mut probe = Run::new(&pc, &start, None, &TauPrior::default(), &cfg).unwrap();
        let at_truth = loss_at(&mut probe, &sim.modules, &sim.tau);
        let mut line = format!("seed {seed}: truth {at_truth:.5}");
        for (name, t0) in [
            ("forward", &start),
            ("reverse", &reverse),
            ("from truth", &sim.tau),
        ] {
            let mut run = Run::new(&pc, t0, None, &TauPrior::default(), &cfg).unwrap();
            let mut trace = Vec::new();
            for r in 0..cfg.rounds {
                run.round().unwrap();
                if [3, 9, 29].contains(&r) {
                    trace.push(format!("{:.5}", run.loss[r]));
                }
            }
            let tau = run.tau().unwrap().to_vec1::<f32>().unwrap();
            line += &format!(
                " | {name} {} (ρ {:+.3})",
                trace.join("→"),
                spearman(&tau, &sim.tau)
            );
        }
        println!("{line}");
    }
}

/// Each pseudobulk joined to its `k` nearest by the true time, each edge once.
fn neighbour_edges(tau: &[f32], k: usize) -> Vec<(usize, usize, f32)> {
    let mut edges = std::collections::BTreeSet::new();
    for p in 0..tau.len() {
        let mut order: Vec<usize> = (0..tau.len()).filter(|&q| q != p).collect();
        order.sort_by(|&a, &b| (tau[a] - tau[p]).abs().total_cmp(&(tau[b] - tau[p]).abs()));
        for &q in &order[..k] {
            edges.insert((p.min(q), p.max(q)));
        }
    }
    edges.into_iter().map(|(p, q)| (p, q, 1.0)).collect()
}

#[test]
fn a_neighbour_graph_sharpens_tau_when_reads_are_thin() {
    let sim = simulate(
        &truth(),
        &SimConfig {
            n_pb: 300,
            genes_per_module: 25,
            mean_reads: 0.3,
            seed: 21,
            ..SimConfig::default()
        },
    )
    .unwrap();
    let pc = counts(&sim, &Device::Cpu);
    let start = noisy(&sim.tau, 0.15, 7);
    let plain = fit(&pc, &start, &TauPrior::default(), &quick()).unwrap();
    let prior = TauPrior {
        anchor: None,
        graph: Some((100.0, neighbour_edges(&sim.tau, 10))),
    };
    let smooth = fit(&pc, &start, &prior, &quick()).unwrap();
    let (a, b) = (
        spearman(&plain.tau, &sim.tau),
        spearman(&smooth.tau, &sim.tau),
    );
    assert!(
        1.0 - b < 0.7 * (1.0 - a),
        "Spearman without the graph {a}, with {b}: the graph should cut the rank error by 30%"
    );
}

#[test]
fn refine_keeps_the_order_it_starts_from() {
    let sim = small_sim(23);
    let pc = counts(&sim, &Device::Cpu);
    let start = noisy(&sim.tau, 0.1, 29);
    let prior = TauPrior {
        anchor: Some((50.0, start.clone())),
        graph: None,
    };
    let out = refine(&pc, &start, &sim.modules, &prior, &quick()).unwrap();
    assert!(!out.reversed);
    let rho = spearman(&out.tau, &sim.tau);
    assert!(rho > 0.95, "Spearman {rho}");
}

/// With the modules' levels in the likelihood, a reversed start is turned
/// around and the times recovered.
#[test]
fn a_reversed_start_is_turned_around() {
    let sim = small_sim(31);
    let pc = counts(&sim, &Device::Cpu);
    let start: Vec<f32> = noisy(&sim.tau, 0.15, 37).iter().map(|t| 1.0 - t).collect();
    let out = fit(&pc, &start, &TauPrior::default(), &quick()).unwrap();
    assert!(
        out.reversed,
        "orientation losses {:?}",
        out.orientation_loss
    );
    let rho = spearman(&out.tau, &sim.tau);
    assert!(rho > 0.9, "Spearman {rho}");
}

/// A repression module started as "never off" jumps back to its switch-off:
/// no gradient reaches a switch-off past the last pseudobulk, the timing step
/// does.
#[test]
fn the_timing_step_finds_a_switch_off_the_gradient_cannot() {
    let sim = small_sim(41);
    let pc = counts(&sim, &Device::Cpu);
    let mut start = sim.modules.clone();
    // Module 1 really switches off at ≈ 0.4.
    start[1].duration = 1.0 - start[1].t_on;
    let cfg = quick();
    let mut run = Run::new(&pc, &sim.tau, Some(&start), &TauPrior::default(), &cfg).unwrap();
    run.timing_step().unwrap();
    let k = run.free.kinetics().unwrap().to_modules().unwrap();
    let t_off = k[1].t_on + k[1].duration;
    let truth = sim.modules[1].t_on + sim.modules[1].duration;
    assert!(
        (t_off - truth).abs() < 0.15,
        "switch-off {t_off}, truth {truth}"
    );
}
