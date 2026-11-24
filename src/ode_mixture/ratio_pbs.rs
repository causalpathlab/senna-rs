//! Pseudobulks for the ODE by splicing, not by state: cells grouped on how
//! far each gene's unspliced share in them sits from that gene's own, so a
//! pseudobulk holds cells at one phase of the dynamics rather than cells
//! that look alike.
//!
//! Every cell and gene with reads, given its `n = u + s` reads there, has
//! `u ~ Binomial(n, π)` with `logit π = η_g + o_c`: `η_g` the gene's
//! unspliced share over all cells (from every row's reads the collapse
//! already summed), `o_c` the cell's own shift of every gene
//! at once (how unspliced the cell is overall, a nuclear or technical share
//! that says nothing of the dynamics, fitted by Newton over the cell's genes
//! and taken out). What is left, the binomial residual
//! `z = (u − nπ) / √(nπ(1−π))`, is 0 where a gene has no reads and grows
//! with the evidence where it has, clipped to `±√n` (the most `n` reads
//! say at an even share, so one stray read at a share near 0 cannot steer
//! its cell). Cells are sketched by a seeded random
//! projection of `z`, next to one of their levels (every gene's two tracks
//! summed, log of its share of the cell's reads, centred over the cells):
//! on simulated cells the ratio alone places time and lineage worse than
//! the levels, and the two together best. The sketch's fuzzy kNN graph is
//! coarsened by heavy-edge matching into pseudobulks, and their reads
//! summed. A cell's overall unspliced share moves neither half: the offset
//! takes it out of the ratio, and the levels sum the tracks. A cell held
//! out of the grouping (QC's near-empty ones) joins the pseudobulk most of
//! the grouped cells of its collapse pseudobulk joined, so every cell has
//! one and is summed into it, as the collapse sums every cell.
//!
//! Nothing is cells × genes: two passes over the backend's sparse columns,
//! each cell's nonzeros at a time, with reads divided by their batch's
//! fold first (batch adjustment before anything else); the sketch is
//! `2·`[`SKETCH_DIM`] × the grouped cells.

use super::blocks::logit;
use super::counts::{track_rows, Role};
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::layout::{coarsened, CoarsenArgs};
use legume_numeric::matrix::traits::{MatOps, SampleOps};
use log::info;
use nalgebra::DMatrix;
use std::sync::{Arc, Mutex};

/// Each half's sketch dimension (ratio, levels)…
pub const SKETCH_DIM: usize = 200;
/// …the singular directions kept of each…
pub const RANK: usize = 30;
/// …and every cell's neighbours in the kNN graph.
pub const KNN: usize = 15;

/// How many pseudobulks [`ratio_pbs`] makes, and how it reads the data.
#[derive(Clone, Copy, Debug)]
pub struct RatioArgs {
    /// Pseudobulks to coarsen into.
    pub n_pbs: usize,
    pub block_size: usize,
    pub seed: u64,
}

/// The pseudobulks [`ratio_pbs`] makes.
pub struct RatioPbs {
    /// Every cell's pseudobulk.
    pub member: Vec<usize>,
    /// `[D × P]` every row's reads in every pseudobulk, batch folds divided
    /// out.
    pub counts: DMatrix<f32>,
    /// Cells per pseudobulk.
    pub n_cells: Vec<f32>,
}

/// A logit shift of a cell's every gene is clamped to this.
const MAX_OFFSET: f32 = 10.0;
/// Every gene's unspliced share is kept this far from 0 and 1.
const SHARE_FLOOR: f64 = 1e-3;

/// What every pass reads with.
struct Pass<'a> {
    /// Every row's (gene, unspliced?) among the genes with both tracks.
    role: &'a [Role],
    /// `[D × batches]`.
    delta: Option<&'a DMatrix<f32>>,
    /// Every cell of the data: is it grouped?
    included: &'a [bool],
}

impl Pass<'_> {
    /// Every included cell of columns `lb..ub` as its genes' `(gene, u, s)`,
    /// reads over the batch's fold, sorted by gene; `f(cell, entries)`.
    fn for_cells(
        &self,
        data: &SparseIoVec,
        (lb, ub): (usize, usize),
        mut f: impl FnMut(usize, &[(u32, f32, f32)]),
    ) -> anyhow::Result<()> {
        let x = data.read_columns_csc(lb..ub)?;
        let batch = self.delta.map(|_| data.get_batch_membership(lb..ub));
        let mut entries: Vec<(u32, f32, f32)> = Vec::new();
        for (j, col) in x.col_iter().enumerate() {
            let cell = lb + j;
            if !self.included[cell] {
                continue;
            }
            entries.clear();
            for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                let Some((g, unspliced)) = self.role[r] else {
                    continue;
                };
                let g = g as u32;
                let v = v / self.fold(r, batch.as_ref().map(|b| b[j]));
                entries.push(if unspliced { (g, v, 0.0) } else { (g, 0.0, v) });
            }
            entries.sort_unstable_by_key(|e| e.0);
            entries.dedup_by(|b, a| {
                let same = a.0 == b.0;
                if same {
                    a.1 += b.1;
                    a.2 += b.2;
                }
                same
            });
            f(cell, &entries);
        }
        Ok(())
    }

    /// Row `r`'s fold in batch `batch` (1 with no batches).
    fn fold(&self, r: usize, batch: Option<usize>) -> f32 {
        match (self.delta, batch) {
            (Some(d), Some(b)) => d[(r, b)].max(1e-8),
            _ => 1.0,
        }
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A cell's logit shift of every gene: Newton on the binomial likelihood of
/// its `(gene, u, s)` at the genes' logit shares `eta`.
pub(crate) fn cell_offset(entries: &[(u32, f32, f32)], eta: &[f32]) -> f32 {
    let mut o = 0f32;
    for _ in 0..25 {
        let (mut grad, mut hess) = (0f32, 0f32);
        for &(g, u, s) in entries {
            let n = u + s;
            let p = sigmoid(eta[g as usize] + o);
            grad += u - n * p;
            hess += n * p * (1.0 - p);
        }
        if hess <= 1e-6 {
            break;
        }
        let step = grad / hess;
        o = (o + step).clamp(-MAX_OFFSET, MAX_OFFSET);
        if step.abs() < 1e-4 {
            break;
        }
    }
    o
}

/// A cell's binomial residuals of its genes, `(gene, z)`, at the genes'
/// logit shares `eta` shifted by the cell's own offset.
pub(crate) fn residuals(entries: &[(u32, f32, f32)], eta: &[f32]) -> Vec<(u32, f32)> {
    let o = cell_offset(entries, eta);
    entries
        .iter()
        .filter(|e| e.1 + e.2 > 0.0)
        .map(|&(g, u, s)| {
            let n = u + s;
            let p = sigmoid(eta[g as usize] + o);
            let bound = n.sqrt();
            let z = (u - n * p) / (n * p * (1.0 - p)).sqrt().max(1e-6);
            (g, z.clamp(-bound, bound))
        })
        .collect()
}

/// `[K × N]` sketch columns as their top `rank` singular directions,
/// `[rank × N]` scores `(V·D)ᵀ` at unit length; `centre`d over the columns
/// first when asked.
fn top_directions(mut x: DMatrix<f32>, centre: bool, rank: usize) -> anyhow::Result<DMatrix<f32>> {
    use legume_numeric::matrix::traits::RandomizedAlgs;
    x.normalize_columns_inplace();
    if centre {
        let mean = x.column_mean();
        x.column_iter_mut().for_each(|mut c| c -= &mean);
        x.normalize_columns_inplace();
    }
    let (_, d, v) = x.rsvd(rank.clamp(1, x.nrows().min(x.ncols())))?;
    let mut scores = (v * DMatrix::from_diagonal(&d)).transpose();
    scores.normalize_columns_inplace();
    Ok(scores)
}

/// Pseudobulks of `data`'s cells by their splicing residuals (see the
/// module). `delta` (`[D × batches]`) the batch folds, `included` the cells
/// to group, `collapsed` every cell's collapse pseudobulk (where the rest
/// go), `row_totals` every row's reads over the cells (batch-adjusted).
pub fn ratio_pbs(
    data: &SparseIoVec,
    delta: Option<&DMatrix<f32>>,
    included: &[bool],
    collapsed: &[usize],
    row_totals: &[f64],
    args: RatioArgs,
) -> anyhow::Result<RatioPbs> {
    let rows = data.row_names()?;
    let (genes, role) = track_rows(&rows);
    let n_genes = genes.len();
    anyhow::ensure!(
        n_genes > 0,
        "no gene with both tracks to group the cells by"
    );
    let n = data.num_columns();
    anyhow::ensure!(
        included.len() == n && collapsed.len() == n,
        "a flag and a collapse pseudobulk per cell"
    );
    anyhow::ensure!(row_totals.len() == rows.len(), "a total per row");
    let cells: Vec<usize> = (0..n).filter(|&c| included[c]).collect();
    anyhow::ensure!(!cells.is_empty(), "no cell to group");
    // Every grouped cell's column of the sketch.
    let mut column = vec![usize::MAX; n];
    for (j, &c) in cells.iter().enumerate() {
        column[c] = j;
    }
    let pass = Pass {
        role: &role,
        delta,
        included,
    };
    let block = Some(args.block_size.max(1));

    // Every gene's reads per track, so its logit unspliced share.
    let mut totals = vec![[0f64; 2]; n_genes];
    for (track, &x) in role.iter().zip(row_totals) {
        if let Some((g, unspliced)) = *track {
            totals[g][usize::from(unspliced)] += x;
        }
    }
    let eta: Vec<f32> = totals
        .iter()
        .map(|&[s, u]| {
            let p = ((u + 1.0) / (u + s + 2.0)).clamp(SHARE_FLOOR, 1.0 - SHARE_FLOOR);
            logit(p) as f32
        })
        .collect();

    // 1: every cell's sketch: its residuals, then its levels.
    let basis_kg = DMatrix::<f32>::rnorm_seeded(SKETCH_DIM, n_genes, args.seed);
    let level_kg = DMatrix::<f32>::rnorm_seeded(SKETCH_DIM, n_genes, args.seed.wrapping_add(1));
    // The two halves apart, over the grouped cells only.
    let mut halves = (
        DMatrix::<f32>::zeros(SKETCH_DIM, cells.len()),
        DMatrix::<f32>::zeros(SKETCH_DIM, cells.len()),
    );
    struct Sketching<'a> {
        pass: &'a Pass<'a>,
        eta: &'a [f32],
        basis_kg: &'a DMatrix<f32>,
        level_kg: &'a DMatrix<f32>,
        column: &'a [usize],
    }
    type Halves = (DMatrix<f32>, DMatrix<f32>);
    data.visit_columns_by_block(
        &|(lb, ub), data: &SparseIoVec, s: &Sketching, out: Arc<Mutex<&mut Halves>>| {
            let k = s.basis_kg.nrows();
            let mut ratio = DMatrix::<f32>::zeros(k, ub - lb);
            let mut levels = DMatrix::<f32>::zeros(k, ub - lb);
            s.pass.for_cells(data, (lb, ub), |cell, entries| {
                let mut col = ratio.column_mut(cell - lb);
                for (g, z) in residuals(entries, s.eta) {
                    col.axpy(z, &s.basis_kg.column(g as usize), 1.0);
                }
                let mut col = levels.column_mut(cell - lb);
                let total: f32 = entries.iter().map(|e| e.1 + e.2).sum();
                if total > 0.0 {
                    for &(g, u, sp) in entries {
                        let x = ((u + sp) / total * 1e4).ln_1p();
                        col.axpy(x, &s.level_kg.column(g as usize), 1.0);
                    }
                }
            })?;
            let mut out = out.lock().expect("sketch lock");
            for cell in (lb..ub).filter(|&c| s.pass.included[c]) {
                out.0.set_column(s.column[cell], &ratio.column(cell - lb));
                out.1.set_column(s.column[cell], &levels.column(cell - lb));
            }
            Ok(())
        },
        &Sketching {
            pass: &pass,
            eta: &eta,
            basis_kg: &basis_kg,
            level_kg: &level_kg,
            column: &column,
        },
        &mut halves,
        block,
    )?;
    // Each half denoised: every cell unit length (the levels centred over
    // the cells first, the common profile out), the half's top `rank`
    // singular directions, scores `U·D`, unit length again. kNN on the raw
    // residuals would find neighbours in the binomial noise.
    let ratio = top_directions(halves.0, false, RANK)?;
    let levels = top_directions(halves.1, true, RANK)?;
    let mut sketch = DMatrix::<f32>::zeros(ratio.nrows() + levels.nrows(), cells.len());
    sketch.rows_mut(0, ratio.nrows()).copy_from(&ratio);
    sketch
        .rows_mut(ratio.nrows(), levels.nrows())
        .copy_from(&levels);
    sketch.normalize_columns_inplace();

    // The included cells' kNN graph coarsened into pseudobulks.
    let groups = coarsened(
        &sketch,
        &CoarsenArgs {
            knn: KNN,
            n_landmarks: args.n_pbs.clamp(1, cells.len()),
            block_size: args.block_size.max(1),
            seed: args.seed,
        },
    )?;
    drop(sketch);
    let n_pb = groups.n_pb();
    let member = with_the_rest(&cells, &groups.membership, collapsed, n_pb);
    let mut n_cells = vec![0f32; n_pb];
    member.iter().for_each(|&p| n_cells[p] += 1.0);
    info!(
        "ode: {} cells in {n_pb} pseudobulks by splicing and levels ({} genes, 2 × {SKETCH_DIM} dims to 2 × {RANK})",
        cells.len(),
        n_genes,
    );

    // 2: every pseudobulk's reads, every cell's summed in.
    let mut counts = DMatrix::<f32>::zeros(rows.len(), n_pb);
    struct Summing<'a> {
        pass: &'a Pass<'a>,
        member: &'a [usize],
    }
    data.visit_columns_by_block(
        &|(lb, ub), data: &SparseIoVec, s: &Summing, out: Arc<Mutex<&mut DMatrix<f32>>>| {
            let x = data.read_columns_csc(lb..ub)?;
            let batch = s.pass.delta.map(|_| data.get_batch_membership(lb..ub));
            let mut local: Vec<(usize, usize, f32)> = Vec::with_capacity(x.nnz());
            for (j, col) in x.col_iter().enumerate() {
                let p = s.member[lb + j];
                let b = batch.as_ref().map(|b| b[j]);
                for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                    local.push((p, r, v / s.pass.fold(r, b)));
                }
            }
            // The block's cells share few pseudobulks: merged before the
            // lock is taken.
            local.sort_unstable_by_key(|e| (e.0, e.1));
            local.dedup_by(|b, a| {
                let same = (a.0, a.1) == (b.0, b.1);
                if same {
                    a.2 += b.2;
                }
                same
            });
            let mut out = out.lock().expect("counts lock");
            for (p, r, v) in local {
                out[(r, p)] += v;
            }
            Ok(())
        },
        &Summing {
            pass: &pass,
            member: &member,
        },
        &mut counts,
        block,
    )?;
    Ok(RatioPbs {
        member,
        counts,
        n_cells,
    })
}

/// Every cell's pseudobulk: the grouped `cells`' from `grouped`, and every
/// other cell the one most grouped cells of its `collapsed` pseudobulk
/// joined (the most joined over all where its collapse pseudobulk has none
/// grouped).
pub(crate) fn with_the_rest(
    cells: &[usize],
    grouped: &[usize],
    collapsed: &[usize],
    n_pb: usize,
) -> Vec<usize> {
    use std::collections::HashMap;
    let mut tally: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    let mut overall = vec![0usize; n_pb];
    let mut member = vec![usize::MAX; collapsed.len()];
    for (&c, &p) in cells.iter().zip(grouped) {
        member[c] = p;
        *tally.entry(collapsed[c]).or_default().entry(p).or_default() += 1;
        overall[p] += 1;
    }
    let most = |counts: &mut dyn Iterator<Item = (usize, usize)>| {
        counts
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|(p, _)| p)
    };
    let fallback = most(&mut overall.iter().copied().enumerate()).expect("a pseudobulk");
    let majority: HashMap<usize, usize> = tally
        .iter()
        .map(|(&q, counts)| {
            let p = most(&mut counts.iter().map(|(&p, &n)| (p, n))).expect("a cell");
            (q, p)
        })
        .collect();
    for (c, m) in member.iter_mut().enumerate() {
        if *m == usize::MAX {
            *m = majority.get(&collapsed[c]).copied().unwrap_or(fallback);
        }
    }
    member
}
