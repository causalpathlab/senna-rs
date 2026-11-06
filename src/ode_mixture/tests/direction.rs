use crate::ode_mixture::direction::{call_of, Call, UNDETERMINED_PER_PB};

/// The call from the evidence over `n` pseudobulks: forward or reversed
/// past the threshold, undetermined within it, diverged when a held refit
/// was not finite.
#[test]
fn the_evidence_calls_the_direction() {
    let n = 100;
    let past = 2.0 * UNDETERMINED_PER_PB * n as f64;
    assert_eq!(call_of(past, n), Call::Forward);
    assert_eq!(call_of(-past, n), Call::Reversed);
    assert_eq!(call_of(0.5 * past / 2.0, n), Call::Undetermined);
    assert_eq!(call_of(f64::NAN, n), Call::Diverged);
    assert_eq!(call_of(f64::INFINITY, n), Call::Diverged);
    assert_eq!(Call::Reversed.as_str(), "reversed");
}
