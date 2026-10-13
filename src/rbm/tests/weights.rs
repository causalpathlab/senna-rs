// Blocks are ranges; one block is a one-range list.
#![allow(clippy::single_range_in_vec_init)]

use super::*;
use graph_embedding_util::PresetMode;
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};

const H: usize = 3;
const K: usize = 2;

fn given_rows() -> Vec<f32> {
    vec![1.0, 2.0, 3.0, -1.0, 0.5, 0.0]
}

fn step(w: &Weights, ranges: &[Range<usize>]) {
    let mut vars = w.vars();
    vars.extend(w.module_vars());
    let mut opt = AdamW::new(
        vars,
        ParamsAdamW {
            lr: 0.1,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let loss = w.w(ranges).unwrap().sqr().unwrap().sum_all().unwrap();
    opt.backward_step(&loss).unwrap();
}

#[test]
fn a_frozen_row_never_moves_and_the_rest_train() {
    let dev = Device::Cpu;
    let row: Vec<u32> = (0..4).collect();
    let rows = given_rows();
    let w = Weights::new(
        &row,
        K,
        H,
        1,
        Some(Given {
            ids: &[0, 2],
            rows: &rows,
            mode: PresetMode::Freeze,
        }),
        1,
        &dev,
    )
    .unwrap();
    let before = w.rho().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(before[0], rows[..3]);
    assert_eq!(before[2], rows[3..]);
    for _ in 0..5 {
        step(&w, &[0..4]);
    }
    let after = w.rho().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(after[0], before[0]);
    assert_eq!(after[2], before[2]);
    assert_ne!(after[1], before[1]);
}

#[test]
fn an_init_row_starts_from_the_table_and_trains() {
    let dev = Device::Cpu;
    let row: Vec<u32> = (0..2).collect();
    let rows = given_rows();
    let w = Weights::new(
        &row,
        K,
        H,
        1,
        Some(Given {
            ids: &[0, 1],
            rows: &rows,
            mode: PresetMode::Init,
        }),
        1,
        &dev,
    )
    .unwrap();
    let before = w.rho().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(before.concat(), rows);
    step(&w, &[0..2]);
    assert_ne!(w.rho().unwrap().to_vec2::<f32>().unwrap(), before);
}

#[test]
fn tied_symbols_share_a_row_across_blocks() {
    let dev = Device::Cpu;
    // Two blocks over the same two symbols.
    let row = [0u32, 1, 0, 1];
    let w = Weights::new(&row, K, H, 2, None, 3, &dev).unwrap();
    let ranges = [0..2, 2..4];
    let full = w.w(&ranges).unwrap();
    let rho = w.rho().unwrap();
    assert_eq!(rho.dim(0).unwrap(), 2);
    // Block 1's row of symbol 0 is ρ₀ read by α₁.
    let want = rho
        .narrow(0, 0, 1)
        .unwrap()
        .matmul(&w.alpha[1].as_tensor().t().unwrap())
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let got = full.narrow(0, 2, 1).unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(got, want);
}

#[test]
fn modules_compose_rho_and_a_frozen_membership_holds() {
    let dev = Device::Cpu;
    let row: Vec<u32> = (0..4).collect();
    let w = Weights::new(&row, K, H, 1, None, 1, &dev)
        .unwrap()
        .with_modules(vec![0, 0, 1, 1], 2, 0.9, 2)
        .unwrap();
    let m = w.modules.as_ref().unwrap();
    let pi = m.membership().unwrap();
    // Every row on the simplex, mostly on its own module.
    for (i, r) in pi.to_vec2::<f32>().unwrap().iter().enumerate() {
        assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(r[[0, 0, 1, 1][i]] > 0.85, "row {i}: {r:?}");
    }
    let want = (pi.matmul(m.mu.as_tensor()).unwrap() + m.residual.as_tensor())
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(w.rho().unwrap().to_vec2::<f32>().unwrap(), want);

    w.set_frozen(true);
    let logits = m.logits.as_tensor().to_vec2::<f32>().unwrap();
    step(&w, &[0..4]);
    assert_eq!(m.logits.as_tensor().to_vec2::<f32>().unwrap(), logits);
    w.set_frozen(false);
    step(&w, &[0..4]);
    assert_ne!(m.logits.as_tensor().to_vec2::<f32>().unwrap(), logits);
}

#[test]
fn modules_refuse_a_given_table() {
    let dev = Device::Cpu;
    let row: Vec<u32> = (0..2).collect();
    let rows = given_rows();
    let w = Weights::new(
        &row,
        K,
        H,
        1,
        Some(Given {
            ids: &[0, 1],
            rows: &rows,
            mode: PresetMode::Init,
        }),
        1,
        &dev,
    )
    .unwrap();
    assert!(w.with_modules(vec![0, 1], 2, 0.9, 2).is_err());
}
