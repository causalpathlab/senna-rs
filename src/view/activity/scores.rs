//! Small helpers over scores and per-cell tables.

use super::*;

/// The `top` finite scores, best first, with their names.
pub(in crate::view) fn best(
    scores: &[f32],
    names: &[Box<str>],
    top: usize,
) -> Vec<(Box<str>, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .collect();
    // Only the top few are wanted: select them, then sort just those.
    let by_score = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if top < ranked.len() {
        ranked.select_nth_unstable_by(top, by_score);
        ranked.truncate(top);
    }
    ranked.sort_by(by_score);
    ranked
        .into_iter()
        .map(|(g, v)| (names[g].clone(), v))
        .collect()
}

/// The upper median of the finite values.
pub(super) fn median_of(v: &[f32]) -> Option<f32> {
    let mut finite: Vec<f32> = v.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let k = finite.len() / 2;
    Some(*finite.select_nth_unstable_by(k, f32::total_cmp).1)
}

/// Set scores of features whose baseline is below its median to -inf.
pub(super) fn drop_below_median(scores: &mut [f32], baseline: Option<(&[f32], f32)>) {
    let Some((b, median)) = baseline.filter(|(b, _)| b.len() == scores.len()) else {
        return;
    };
    for (s, &v) in scores.iter_mut().zip(b) {
        if v.is_nan() || v < median {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Sum of the given rows of `m`, as a column vector.
pub(super) fn row_sum(m: &Mat, rows: &[usize]) -> Vector {
    let mut acc = Vector::zeros(m.ncols());
    for &i in rows {
        acc += m.row(i).transpose();
    }
    acc
}

/// The `top` rows of `m` pointing nearest `at`'s way, with their `names`:
/// each scaled to unit length, as an embedding run's cells and features are
/// laid out, and scored by minus the distance between them; `skip` left
/// out. None when `at` is not of `m`'s width.
pub(in crate::view) fn nearest_directions(
    m: &Mat,
    at: &Vector,
    names: &[Box<str>],
    skip: Option<usize>,
    top: usize,
) -> Option<Vec<(Box<str>, f32)>> {
    if m.ncols() != at.len() {
        return None;
    }
    let at = unit(at.clone());
    let mut scores: Vec<f32> = (0..m.nrows())
        .into_par_iter()
        .map(|i| {
            let row = unit(m.row(i).transpose());
            -(row - &at).norm()
        })
        .collect();
    if let Some(i) = skip {
        scores[i] = f32::NAN;
    }
    Some(best(&scores, names, top))
}

/// `v` scaled to unit length; a zero vector stays zero.
pub(super) fn unit(v: Vector) -> Vector {
    let n = v.norm();
    if n > 1e-9 {
        v / n
    } else {
        v
    }
}

/// Mean of the given rows of `m`, as a column vector.
pub(super) fn row_mean(m: &Mat, rows: &[usize]) -> Vector {
    row_sum(m, rows) / rows.len().max(1) as f32
}

/// Covariance of the given rows of `m` (columns are variables), summed over
/// chunks of rows in parallel.
pub(super) fn row_cov(m: &Mat, rows: &[usize]) -> Mat {
    let mu = row_mean(m, rows);
    let h = m.ncols();
    let c = rows
        .par_chunks(4096)
        .map(|chunk| {
            let mut c = Mat::zeros(h, h);
            for &i in chunk {
                let d = m.row(i).transpose() - &mu;
                c.ger(1.0, &d, &d, 1.0);
            }
            c
        })
        .reduce(|| Mat::zeros(h, h), |a, b| a + b);
    c / (rows.len().max(2) - 1) as f32
}

/// `x_gᵀ C x_g` for every row `x_g` of `x`.
pub(super) fn quadratic_forms(x: &Mat, c: &Mat) -> Vec<f32> {
    let xc = x * c;
    xc.row_iter()
        .zip(x.row_iter())
        .map(|(a, b)| a.dot(&b))
        .collect()
}
