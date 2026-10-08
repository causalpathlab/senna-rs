use super::*;
use legume_numeric::candle::candle_core::{DType, Device, Var};

/// One module's kinetics as plain numbers.
#[derive(Clone, Copy, Debug)]
struct One {
    t_on: f64,
    dur: f64,
    lambda: f64,
    basal: f64,
    beta: f64,
    gamma: f64,
}

impl One {
    fn transcription(&self, t: f64) -> f64 {
        let r = self.basal;
        let t_off = self.t_on + self.dur;
        if t < self.t_on {
            r
        } else if t < t_off {
            1.0 - (1.0 - r) * (-self.lambda * (t - self.t_on)).exp()
        } else {
            let a_off = 1.0 - (1.0 - r) * (-self.lambda * self.dur).exp();
            r + (a_off - r) * (-self.lambda * (t - t_off)).exp()
        }
    }

    /// RK4 from the basal steady state at `t0 < t_on`, stepping exactly onto
    /// both switch times so the input is smooth within every step.
    fn integrate(&self, t0: f64, t: f64) -> (f64, f64) {
        let (mut u, mut s) = (self.basal / self.beta, self.basal / self.gamma);
        let mut breaks = vec![t0];
        for b in [self.t_on, self.t_on + self.dur] {
            if b > t0 && b < t {
                breaks.push(b);
            }
        }
        breaks.push(t);
        for w in breaks.windows(2) {
            let n = (((w[1] - w[0]) / 1e-4).ceil() as usize).max(1);
            let h = (w[1] - w[0]) / n as f64;
            // Evaluate the input just inside the segment so a step that starts
            // on a switch reads the new piece.
            let a_at = |x: f64| self.transcription(x.clamp(w[0] + 1e-12, w[1] - 1e-12));
            let f =
                |x: f64, u: f64, s: f64| (a_at(x) - self.beta * u, self.beta * u - self.gamma * s);
            for i in 0..n {
                let x = w[0] + i as f64 * h;
                let k1 = f(x, u, s);
                let k2 = f(x + h / 2.0, u + h / 2.0 * k1.0, s + h / 2.0 * k1.1);
                let k3 = f(x + h / 2.0, u + h / 2.0 * k2.0, s + h / 2.0 * k2.1);
                let k4 = f(x + h, u + h * k3.0, s + h * k3.1);
                u += h / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
                s += h / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
            }
        }
        (u, s)
    }
}

fn kinetics(modules: &[One], dev: &Device) -> Kinetics {
    let col = |f: fn(&One) -> f64| {
        Tensor::from_vec(
            modules.iter().map(f).collect::<Vec<f64>>(),
            modules.len(),
            dev,
        )
        .unwrap()
    };
    Kinetics {
        t_on: col(|m| m.t_on),
        duration: col(|m| m.dur),
        lambda: col(|m| m.lambda),
        basal: col(|m| m.basal),
        beta: col(|m| m.beta),
        gamma: col(|m| m.gamma),
    }
}

fn grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

/// A spread of modules: induction only, repression only, transient, slow
/// and fast relaxation, and coincident rates.
fn modules() -> Vec<One> {
    let m = |t_on, dur, lambda, basal, beta, gamma| One {
        t_on,
        dur,
        lambda,
        basal,
        beta,
        gamma,
    };
    vec![
        m(0.2, 5.0, 20.0, 0.05, 8.0, 3.0),
        m(-0.4, 0.8, 30.0, 0.1, 5.0, 2.0),
        m(0.3, 0.3, 10.0, 0.02, 12.0, 4.0),
        m(0.1, 0.5, 2.0, 0.2, 3.0, 6.0),
        m(0.25, 0.4, 6.0, 0.05, 6.0, 6.0),
        m(0.25, 0.4, 6.0, 0.05, 6.0, 6.000_001),
        m(0.15, 0.6, 9.0, 0.1, 9.0, 9.0),
        m(0.4, 0.2, 400.0, 0.01, 15.0, 1.5),
    ]
}

#[test]
fn closed_form_matches_rk4() {
    let dev = Device::Cpu;
    let mods = modules();
    let taus = grid(41);
    let tau = Tensor::from_vec(taus.clone(), taus.len(), &dev).unwrap();
    let (u, s) = curves(&kinetics(&mods, &dev), &tau).unwrap();
    let (u, s) = (u.to_vec2::<f64>().unwrap(), s.to_vec2::<f64>().unwrap());
    for (j, one) in mods.iter().enumerate() {
        for (i, &t) in taus.iter().enumerate() {
            let (ru, rs) = one.integrate(-0.6, t);
            let eu = (u[i][j] - ru).abs() / ru;
            let es = (s[i][j] - rs).abs() / rs;
            assert!(
                eu < 1e-6 && es < 1e-6,
                "module {j} at τ={t}: u {} vs {ru}, s {} vs {rs}",
                u[i][j],
                s[i][j]
            );
        }
    }
}

#[test]
fn basal_steady_state_before_switch_on() {
    let dev = Device::Cpu;
    let one = One {
        t_on: 0.6,
        dur: 0.2,
        lambda: 10.0,
        basal: 0.1,
        beta: 4.0,
        gamma: 2.0,
    };
    let tau = Tensor::new(&[0.0f64, 0.3, 0.59], &dev).unwrap();
    let lr = log_ratio(&kinetics(&[one], &dev), &tau).unwrap();
    for v in lr.flatten_all().unwrap().to_vec1::<f64>().unwrap() {
        assert!((v - (2.0f64 / 4.0).ln()).abs() < 1e-12, "{v}");
    }
}

#[test]
fn induction_raises_and_repression_lowers_the_ratio() {
    let dev = Device::Cpu;
    let one = One {
        t_on: 0.2,
        dur: 0.4,
        lambda: 20.0,
        basal: 0.05,
        beta: 6.0,
        gamma: 2.0,
    };
    let steady = (2.0f64 / 6.0).ln();
    let tau = Tensor::new(&[0.25f64, 0.9], &dev).unwrap();
    let lr = log_ratio(&kinetics(&[one], &dev), &tau)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f64>()
        .unwrap();
    assert!(
        lr[0] > steady + 0.5,
        "induction {} vs steady {steady}",
        lr[0]
    );
    assert!(
        lr[1] < steady - 0.5,
        "repression {} vs steady {steady}",
        lr[1]
    );
}

#[test]
fn f32_curves_match_f64() {
    let dev = Device::Cpu;
    let mods = modules();
    let tau = Tensor::from_vec(grid(21), 21, &dev).unwrap();
    let k64 = kinetics(&mods, &dev);
    let to32 = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
    let k32 = Kinetics {
        t_on: to32(&k64.t_on),
        duration: to32(&k64.duration),
        lambda: to32(&k64.lambda),
        basal: to32(&k64.basal),
        beta: to32(&k64.beta),
        gamma: to32(&k64.gamma),
    };
    let a = log_ratio(&k64, &tau)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f64>()
        .unwrap();
    let b = log_ratio(&k32, &to32(&tau))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - f64::from(*y)).abs() < 1e-4, "{x} vs {y}");
    }
}

/// The log ratio's gradient in every kinetic parameter against central
/// differences, at a generic point and with all three rates coincident.
#[test]
fn gradients_are_finite_and_match_differences() {
    let dev = Device::Cpu;
    for one in [
        One {
            t_on: 0.2,
            dur: 0.4,
            lambda: 7.0,
            basal: 0.1,
            beta: 5.0,
            gamma: 3.0,
        },
        One {
            t_on: 0.2,
            dur: 0.4,
            lambda: 5.0,
            basal: 0.1,
            beta: 5.0,
            gamma: 5.0,
        },
    ] {
        let taus = [0.1f64, 0.3, 0.55, 0.8];
        let tau = Tensor::new(&taus, &dev).unwrap();
        let params = [
            one.t_on, one.dur, one.lambda, one.basal, one.beta, one.gamma,
        ];
        let vars: Vec<Var> = params
            .iter()
            .map(|&p| Var::from_tensor(&Tensor::new(&[p], &dev).unwrap()).unwrap())
            .collect();
        let k = Kinetics {
            t_on: vars[0].as_tensor().clone(),
            duration: vars[1].as_tensor().clone(),
            lambda: vars[2].as_tensor().clone(),
            basal: vars[3].as_tensor().clone(),
            beta: vars[4].as_tensor().clone(),
            gamma: vars[5].as_tensor().clone(),
        };
        let total = log_ratio(&k, &tau).unwrap().sum_all().unwrap();
        let grads = total.backward().unwrap();
        let eval = |p: [f64; 6]| -> f64 {
            let o = One {
                t_on: p[0],
                dur: p[1],
                lambda: p[2],
                basal: p[3],
                beta: p[4],
                gamma: p[5],
            };
            log_ratio(&kinetics(&[o], &dev), &tau)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        };
        for (i, var) in vars.iter().enumerate() {
            let g = grads
                .get(var.as_tensor())
                .unwrap()
                .to_vec1::<f64>()
                .unwrap()[0];
            assert!(g.is_finite(), "{one:?}: parameter {i} gradient {g}");
            let h = 1e-6;
            let (mut lo, mut hi) = (params, params);
            lo[i] -= h;
            hi[i] += h;
            let fd = (eval(hi) - eval(lo)) / (2.0 * h);
            assert!(
                (g - fd).abs() < 1e-4 * (1.0 + fd.abs()),
                "{one:?}: parameter {i} autograd {g} vs differences {fd}"
            );
        }
    }
}
