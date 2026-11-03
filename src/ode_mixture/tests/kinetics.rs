use crate::ode_mixture::kinetics::trajectory;

/// The ODE integrated by fourth-order Runge–Kutta in `sub` steps per grid
/// step, `α` held at `alpha[k]` over step `k`, from steady state.
fn runge_kutta(alpha: &[f64], beta: f64, gamma: f64, h: f64, sub: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut u, mut s) = (alpha[0] / beta, alpha[0] / gamma);
    let (mut us, mut ss) = (vec![u], vec![s]);
    let dt = h / sub as f64;
    for &a in &alpha[..alpha.len() - 1] {
        let f = |u: f64, s: f64| (a - beta * u, beta * u - gamma * s);
        for _ in 0..sub {
            let (k1u, k1s) = f(u, s);
            let (k2u, k2s) = f(u + 0.5 * dt * k1u, s + 0.5 * dt * k1s);
            let (k3u, k3s) = f(u + 0.5 * dt * k2u, s + 0.5 * dt * k2s);
            let (k4u, k4s) = f(u + dt * k3u, s + dt * k3s);
            u += dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            s += dt / 6.0 * (k1s + 2.0 * k2s + 2.0 * k3s + k4s);
        }
        us.push(u);
        ss.push(s);
    }
    (us, ss)
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * y.abs().max(1e-12),
            "at {k}: {x} vs {y}"
        );
    }
}

#[test]
fn constant_transcription_stays_at_steady_state() {
    let alpha = vec![3.0; 11];
    let (u, s) = trajectory(&alpha, 2.0, 0.5, 0.1);
    assert_eq!(u.len(), 11);
    close(&u, &[1.5; 11], 1e-12);
    close(&s, &[6.0; 11], 1e-12);
}

#[test]
fn a_changing_transcription_matches_a_fine_integration() {
    let alpha: Vec<f64> = (0..21)
        .map(|k| 1.0 + 4.0 * (k as f64 / 4.0).sin().powi(2))
        .collect();
    for (beta, gamma) in [(2.0, 0.5), (0.3, 1.7), (5.0, 5.5)] {
        let (u, s) = trajectory(&alpha, beta, gamma, 0.25);
        let (ur, sr) = runge_kutta(&alpha, beta, gamma, 0.25, 400);
        close(&u, &ur, 1e-9);
        close(&s, &sr, 1e-9);
    }
}

#[test]
fn equal_splicing_and_decay_rates_take_the_limit() {
    let alpha: Vec<f64> = (0..11).map(|k| if k < 3 { 1.0 } else { 6.0 }).collect();
    let (u, s) = trajectory(&alpha, 1.3, 1.3, 0.2);
    assert!(s.iter().all(|x| x.is_finite()));
    let (ur, sr) = runge_kutta(&alpha, 1.3, 1.3, 0.2, 400);
    close(&u, &ur, 1e-9);
    close(&s, &sr, 1e-9);
    // And continuous on either side of the limit.
    let (_, near) = trajectory(&alpha, 1.3 * (1.0 + 1e-7), 1.3, 0.2);
    close(&near, &s, 1e-6);
}

#[test]
fn long_after_a_switch_the_new_steady_state_is_reached() {
    let mut alpha = vec![1.0; 200];
    alpha[1..].iter_mut().for_each(|a| *a = 8.0);
    let (u, s) = trajectory(&alpha, 2.0, 0.5, 0.5);
    assert!(
        (u[199] - 4.0).abs() < 1e-9 && (s[199] - 16.0).abs() < 1e-9,
        "{} {}",
        u[199],
        s[199]
    );
}

/// Between nodes, the levels at `m` evenly spaced points of every step are
/// the ODE's (α held over the step), and the nodes are the grid's.
#[test]
fn between_the_nodes_the_levels_are_the_odes() {
    use crate::ode_mixture::kinetics::fine_trajectory;
    let alpha: Vec<f64> = (0..11)
        .map(|k| 1.0 + 3.0 * (k as f64 / 2.0).cos().powi(2))
        .collect();
    let (beta, gamma, h, m) = (2.5, 0.8, 0.3, 4);
    let (u, s) = fine_trajectory(&alpha, beta, gamma, h, m);
    assert_eq!(u.len(), (alpha.len() - 1) * m + 1);
    // The same ODE on the sub-grid itself, α held at the node's value.
    let fine_alpha: Vec<f64> = (0..u.len())
        .map(|i| alpha[(i / m).min(alpha.len() - 1)])
        .collect();
    let (ur, sr) = runge_kutta(&fine_alpha, beta, gamma, h / m as f64, 400);
    close(&u, &ur, 1e-9);
    close(&s, &sr, 1e-9);
    // Every m-th point is a node.
    let (un, sn) = trajectory(&alpha, beta, gamma, h);
    for k in 0..alpha.len() {
        assert!((u[k * m] - un[k]).abs() < 1e-12 && (s[k * m] - sn[k]).abs() < 1e-12);
    }
    // m = 1 is the grid.
    let (u1, _) = fine_trajectory(&alpha, beta, gamma, h, 1);
    close(&u1, &un, 1e-12);
}
