//! Where the fit starts — the genes' embedding from the leading components of
//! rows' log expression, every gene's spliced baseline from its mean spliced
//! share, the curve from rows binned by a starting order — and the rows the
//! fit reads: pseudobulks drawn again at a cell's depth every epoch.

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma};

/// Reads a row is scaled to before the log.
pub const SCALE: f32 = 1e4;
/// Principal components of the expression states the orders are taken in.
pub const STATE_DIMS: usize = 30;

/// `[G × H]` row-major: the top `h` right singular vectors of rows' centred
/// `log1p` expression per 10⁴ (unspliced plus spliced, `[N × G]` row-major),
/// each column scaled to unit sd over the genes; columns past the data's
/// rank are zero.
pub fn rho_start(xu: &[f32], xs: &[f32], n: usize, g: usize, h: usize) -> Vec<f32> {
    let mut m = DMatrix::<f32>::zeros(n, g);
    for i in 0..n {
        let row = |x: &[f32]| x[i * g..(i + 1) * g].to_vec();
        let (u, s) = (row(xu), row(xs));
        let total: f32 = u.iter().chain(&s).sum::<f32>().max(1.0);
        for j in 0..g {
            m[(i, j)] = ((u[j] + s[j]) / total * SCALE).ln_1p();
        }
    }
    for j in 0..g {
        let mean = m.column(j).mean();
        m.column_mut(j).add_scalar_mut(-mean);
    }
    let svd = m.svd(false, true);
    let vt = svd.v_t.expect("asked for V");
    let mut rho = vec![0f32; g * h];
    for d in 0..h.min(vt.nrows()) {
        let col: Vec<f32> = (0..g).map(|j| vt[(d, j)]).collect();
        let sd = (col.iter().map(|x| x * x).sum::<f32>() / g as f32)
            .sqrt()
            .max(1e-12);
        for j in 0..g {
            rho[j * h + d] = col[j] / sd;
        }
    }
    rho
}

/// `[G]` every gene's log mean spliced share over rows (`[N × G]` row-major).
pub fn base_start(xs: &[f32], n: usize, g: usize) -> Vec<f32> {
    let mut sum = vec![0.5f64; g];
    for i in 0..n {
        for j in 0..g {
            sum[j] += f64::from(xs[i * g + j]);
        }
    }
    let total: f64 = sum.iter().sum();
    sum.iter().map(|s| (s / total).ln() as f32).collect()
}

/// Ridge of the least-squares fits below.
const RIDGE: f64 = 1e-3;

/// `[M × H]` row-major: the curve's coefficients so that the curve at every
/// grid time explains the rows binned there by their starting `order`, as
/// best a line through `ρ` can: per grid time `k`, `z_k` fits the bin's log
/// shares of every gene's reads (`x` `[N × G]` row-major, both tracks summed)
/// about their mean, on `ρ` `[G × H]` centred over genes; then the
/// coefficients fit the `z_k` of the non-empty bins through the basis
/// `values` `[K × M]` row-major.
pub fn curve_start(
    x: &[f32],
    order: &[f32],
    rho: &[f32],
    values: &[f32],
    (g, h, k, m): (usize, usize, usize, usize),
) -> Vec<f32> {
    let n = order.len();
    let mut rho_c = DMatrix::<f64>::from_fn(g, h, |j, d| f64::from(rho[j * h + d]));
    for d in 0..h {
        let mean = rho_c.column(d).mean();
        rho_c.column_mut(d).add_scalar_mut(-mean);
    }
    let gram = rho_c.transpose() * &rho_c + DMatrix::identity(h, h) * RIDGE;
    let solve_z = gram.cholesky().expect("a ridge keeps it positive definite");
    let mut sums = vec![vec![0f64; g]; k];
    for i in 0..n {
        let bin = ((order[i] * k as f32) as usize).min(k - 1);
        for j in 0..g {
            sums[bin][j] += f64::from(x[i * g + j]);
        }
    }
    let (mut rows_b, mut rows_z) = (Vec::new(), Vec::new());
    for (bin, s) in sums.iter().enumerate() {
        let total: f64 = s.iter().sum();
        if total <= 0.0 {
            continue;
        }
        let y: Vec<f64> = s.iter().map(|v| ((v + 0.5) / total).ln()).collect();
        let mean = y.iter().sum::<f64>() / g as f64;
        let t = DVector::from_iterator(g, y.iter().map(|v| v - mean));
        rows_z.push(solve_z.solve(&(rho_c.transpose() * t)));
        rows_b.push(bin);
    }
    let r = rows_b.len();
    if r == 0 {
        return vec![0.0; m * h];
    }
    let b = DMatrix::<f64>::from_fn(r, m, |i, c| f64::from(values[rows_b[i] * m + c]));
    let z = DMatrix::<f64>::from_fn(r, h, |i, d| rows_z[i][d]);
    let gram = b.transpose() * &b + DMatrix::identity(m, m) * RIDGE;
    let a = gram
        .cholesky()
        .expect("a ridge keeps it positive definite")
        .solve(&(b.transpose() * z)); // [M × H]
    (0..m)
        .flat_map(|i| (0..h).map(move |d| (i, d)))
        .map(|(i, d)| a[(i, d)] as f32)
        .collect()
}

/// Every row of `x` (`[N × F]` row-major counts) drawn again at `depth`
/// reads: shares from the row's Dirichlet posterior (counts plus ½), then a
/// multinomial; a pseudobulk sample at a cell's depth.
pub fn resample(x: &[f32], f: usize, depth: u64, rng: &mut StdRng) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    let mut p = vec![0f32; f];
    for (row, o) in x.chunks(f).zip(out.chunks_mut(f)) {
        for (pi, &c) in p.iter_mut().zip(row) {
            *pi = Gamma::new(f64::from(c) + 0.5, 1.0).map_or(0.0, |d| d.sample(rng)) as f32;
        }
        let s: f32 = p.iter().sum::<f32>().max(1e-30);
        p.iter_mut().for_each(|v| *v /= s);
        super::rbm::multinomial(depth, &p, o, rng);
    }
    out
}

/// Every row of `x` (`[N × F]` row-major) scaled to `depth` reads.
pub fn scale_rows(x: &mut [f32], f: usize, depth: f32) {
    for row in x.chunks_mut(f) {
        let n: f32 = row.iter().sum::<f32>().max(1.0);
        row.iter_mut().for_each(|v| *v *= depth / n);
    }
}

/// `k` groups of the rows of `x` (`[N × D]`) by k-means (k-means++ start,
/// seeded), every row's group.
pub fn kmeans(x: &DMatrix<f32>, k: usize, seed: u64) -> Vec<usize> {
    use rand::RngExt;
    use rand::SeedableRng;
    let n = x.nrows();
    if k <= 1 || n == 0 {
        return vec![0; n];
    }
    let k = k.min(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let dist =
        |a: usize, c: &[f32]| -> f32 { x.row(a).iter().zip(c).map(|(p, q)| (p - q).powi(2)).sum() };
    let mut centres: Vec<Vec<f32>> = vec![x.row(rng.random_range(0..n)).iter().copied().collect()];
    while centres.len() < k {
        let d: Vec<f32> = (0..n)
            .map(|i| {
                centres
                    .iter()
                    .map(|c| dist(i, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let total: f32 = d.iter().sum();
        let mut u = rng.random::<f32>() * total;
        let mut pick = n - 1;
        for (i, &di) in d.iter().enumerate() {
            if u < di {
                pick = i;
                break;
            }
            u -= di;
        }
        centres.push(x.row(pick).iter().copied().collect());
    }
    let mut label = vec![0usize; n];
    for _ in 0..100 {
        let mut moved = false;
        for (i, lab) in label.iter_mut().enumerate() {
            let best = (0..k)
                .min_by(|&a, &b| dist(i, &centres[a]).total_cmp(&dist(i, &centres[b])))
                .expect("a centre");
            moved |= best != *lab;
            *lab = best;
        }
        for (c, centre) in centres.iter_mut().enumerate() {
            let members: Vec<usize> = (0..n).filter(|&i| label[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            for (d, v) in centre.iter_mut().enumerate() {
                *v = members.iter().map(|&i| x[(i, d)]).sum::<f32>() / members.len() as f32;
            }
        }
        if !moved {
            break;
        }
    }
    label
}

/// `[N × dims]`: rows' expression states for an ordering, the leading
/// principal components of their centred `log1p` reads per 10⁴ (both tracks
/// of every gene summed; `x` `[N × F]` row-major).
pub fn row_states(x: &[f32], f: usize, dims: usize) -> DMatrix<f32> {
    let g = f / 2;
    let n = x.len() / f;
    let mut m = DMatrix::<f32>::zeros(n, g);
    for i in 0..n {
        let row = &x[i * f..(i + 1) * f];
        let total: f32 = row.iter().sum::<f32>().max(1.0);
        for j in 0..g {
            m[(i, j)] = ((row[j] + row[g + j]) / total * SCALE).ln_1p();
        }
    }
    for j in 0..g {
        let mean = m.column(j).mean();
        m.column_mut(j).add_scalar_mut(-mean);
    }
    let d = dims.min(n).min(g).max(1);
    let svd = m.svd(true, false);
    let u = svd.u.expect("asked for U");
    DMatrix::from_fn(n, d, |i, c| u[(i, c)] * svd.singular_values[c])
}

/// Spearman correlation of two equal-length series (ties broken by order).
pub fn rank_correlation(a: &[f32], b: &[f32]) -> f64 {
    let ranks = |x: &[f32]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
        let mut r = vec![0f64; x.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (a, b) = (ranks(a), ranks(b));
    let n = a.len().max(1) as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        c += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    c / (va * vb).sqrt().max(1e-12)
}
