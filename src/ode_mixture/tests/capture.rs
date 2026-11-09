//! One global capture of the unspliced reads, `e^c` on every unspliced
//! level, each pseudobulk's jitter around it (the climb's gate is in
//! `gates.rs`).

use super::model::truth_params;
use crate::ode_mixture::sim::{simulate, Sim, SimConfig};
use std::sync::OnceLock;

const POINTS: usize = 45;
const KAPPA: f64 = 0.25;
const CAPTURE: f64 = -1.2;
const JITTER: f64 = 0.3;

/// One lineage, every unspliced read captured at `e^{CAPTURE}`, every
/// pseudobulk's share jittering by [`JITTER`] around it.
fn sim() -> &'static Sim {
    static SIM: OnceLock<Sim> = OnceLock::new();
    SIM.get_or_init(|| {
        simulate(&SimConfig {
            fine: 133,
            kappa: KAPPA,
            log_capture: CAPTURE,
            share_jitter: JITTER,
            ..SimConfig::default()
        })
        .unwrap()
    })
}

/// The sim's levels carry the capture: the truth's unspliced levels are
/// the uncaptured ones scaled by `e^c`.
#[test]
fn the_truths_unspliced_levels_carry_the_capture() {
    let truth = truth_params(sim(), 133, POINTS);
    assert_eq!(truth.log_capture, CAPTURE);
    let mut bare = truth.clone();
    bare.log_capture = 0.0;
    let (lv, lv0) = (truth.levels(), bare.levels());
    for (a, b) in lv.u.iter().zip(lv0.u.iter()) {
        assert!((a - b * CAPTURE.exp()).abs() <= 1e-12 * a.abs().max(1e-300));
    }
    assert_eq!(lv.s, lv0.s);
}
