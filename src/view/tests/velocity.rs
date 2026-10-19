use super::*;
use crate::view::{color, render};

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
    let vp = render::fit(points.bounds, 400, 400);
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

/// A field along +x gives streamlines running left to right, spread over
/// the map, none of them crossing back.
#[test]
fn a_field_along_x_streams_along_x() {
    let side = 60;
    let names: Vec<Box<str>> = (0..side * side).map(|i| format!("c{i}").into()).collect();
    let xy: Vec<[f32; 2]> = (0..side * side)
        .map(|i| [(i % side) as f32, (i / side) as f32])
        .collect();
    let points = Points::new(names, xy);
    let moved = vec![[1.0f32, 0.0]; side * side];
    let vp = render::fit(points.bounds, 400, 400);
    let lines = streamlines(&points, &moved, &vp, LANES);
    // Denser lanes, more lines.
    assert!(streamlines(&points, &moved, &vp, 2.0 * LANES).len() > lines.len());
    assert!(lines.len() >= 10, "{} lines", lines.len());
    for line in &lines {
        for pair in line.windows(2) {
            assert!(pair[1].0 > pair[0].0, "a step back: {pair:?}");
        }
    }
    // Spread across the map, not bunched.
    let ys: Vec<f32> = lines.iter().map(|l| l[l.len() / 2].1).collect();
    let (lo, hi) = ys
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &y| (a.min(y), b.max(y)));
    assert!(hi - lo > 200.0, "lines span {lo}..{hi}");
}

/// Half blocks show every ten frame pixels as one: a streamline stroked at
/// the frame's own width averages away there, one stroked at least a screen
/// pixel wide still shows.
#[test]
fn streamlines_survive_a_screen_coarser_than_the_frame() {
    let line = vec![(20.0, 200.0), (200.0, 200.0), (380.0, 200.0)];
    let darkest = |floor: f32| {
        let mut frame = Frame::blank(400, 400, color::BACKGROUND);
        let bg = frame.to_image().get_pixel(0, 0)[0];
        draw_lines(
            &mut frame,
            std::slice::from_ref(&line),
            [0.0, 0.0, 0.0],
            floor,
        );
        let small = image::imageops::resize(
            &frame.to_image(),
            40,
            40,
            image::imageops::FilterType::Triangle,
        );
        let low = small.pixels().map(|p| p[0]).min().unwrap_or(bg);
        i32::from(bg) - i32::from(low)
    };
    let (thin, floored) = (darkest(1.0), darkest(10.0));
    assert!(floored > 3 * thin.max(1), "thin {thin}, floored {floored}");
    assert!(floored > 60, "floored {floored}");
}
