use super::*;
use rand::SeedableRng;

/// Every row is drawn at the depth asked for; the same seed draws the same
/// samples, however many threads draw them.
#[test]
fn resampled_rows_keep_the_depth_and_replay_on_any_threads() {
    let f = 50;
    let x: Vec<f32> = (0..40 * f).map(|i| ((i * 7) % 13) as f32).collect();
    let draw = || resample(&x, f, 1000, &mut StdRng::seed_from_u64(9));
    let many = draw();
    for row in many.chunks(f) {
        assert_eq!(row.iter().sum::<f32>() as u64, 1000);
    }
    assert_eq!(many, draw());
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(draw);
    assert_eq!(many, one);
}
