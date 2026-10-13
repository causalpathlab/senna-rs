use super::*;

/// Cells on a grid in two dimensions, laid out as themselves, all moving
/// along +x: every displacement on the map points along +x, and the arrows
/// over the grid do too.
#[test]
fn a_field_along_x_draws_arrows_along_x() {
    let side = 60;
    let names: Vec<Box<str>> = (0..side * side).map(|i| format!("c{i}").into()).collect();
    let z = Mat::from_fn(side * side, 2, |i, d| {
        if d == 0 {
            (i % side) as f32
        } else {
            (i / side) as f32
        }
    });
    let v = Mat::from_fn(side * side, 2, |_, d| if d == 0 { 1.0 } else { 0.0 });
    let field = Field::new(&names, &z, &names, &v).unwrap();
    let xy: Vec<[f32; 2]> = (0..side * side).map(|i| [z[(i, 0)], z[(i, 1)]]).collect();
    let points = Points::new(names.clone(), xy);
    let moved = field.embed(&points);
    // Away from the grid's edges, where neighbours lie on every side.
    let inner = (0..side * side).filter(|&i| {
        let (x, y) = (i % side, i / side);
        (6..side - 6).contains(&x) && (6..side - 6).contains(&y)
    });
    for i in inner {
        let [dx, dy] = moved[i];
        assert!(dx > 0.0 && dx > 3.0 * dy.abs(), "cell {i}: ({dx}, {dy})");
    }
    let vp = Viewport::fit(points.bounds, 400, 400);
    let arrows = arrows(&points, &moved, &vp);
    assert!(!arrows.is_empty());
    let right = arrows.iter().filter(|(a, b)| b.0 > a.0).count();
    assert!(
        right * 10 >= arrows.len() * 8,
        "{right} of {} point right",
        arrows.len()
    );
}

/// A cell the field does not hold has no displacement.
#[test]
fn a_cell_without_a_velocity_does_not_move() {
    let names: Vec<Box<str>> = ["a", "b", "c"].map(Into::into).to_vec();
    let z = Mat::from_fn(3, 2, |i, d| (i * 2 + d) as f32);
    let v = Mat::from_fn(2, 2, |_, d| d as f32);
    let field = Field::new(&names, &z, &names[..2], &v).unwrap();
    let points = Points::new(names, vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
    let moved = field.embed(&points);
    assert!(moved[2][0].is_nan());
}
