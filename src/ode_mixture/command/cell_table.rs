//! Every cell's readout: all cells read out at the climb's mode a block at a
//! time and placed on the map, and the per-cell tables written from them.

use super::{f32_columns, keep_mask, lineage_index, write_rows};
use crate::ode_mixture::coarsen::coarse_reads;
use crate::ode_mixture::counts::for_cell_blocks;
use crate::ode_mixture::likelihood::Reads;
use crate::ode_mixture::model::lineage_label;
use crate::ode_mixture::output::MapBasis;
use crate::ode_mixture::readout::{read_cells, Mode};
use data_beans::alg::feature_coarsening::FeatureCoarsening;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use nalgebra::DMatrix;

/// Every cell's readout, as the tables have it.
pub(super) struct CellTable {
    cells: Vec<Box<str>>,
    lineage: Vec<Box<str>>,
    /// `[L][C]` the lineages' posteriors.
    p_lineage: Vec<Vec<f32>>,
    tau: Vec<f32>,
    tau_sd: Vec<f32>,
    tau_lo: Vec<f32>,
    tau_hi: Vec<f32>,
    /// Every cell's raw unspliced share (U+1)/(U+S+2) over all two-track genes.
    ufrac: Vec<f32>,
    /// `[C × k]` row-major, every cell's place on the map and its velocity.
    place: Vec<f32>,
    velocity: Vec<f32>,
}

/// What [`read_out_cells`] reads: the data, the fit's genes and their
/// coarsening and fold, the cells QC kept (`None`: all).
pub(super) struct CellSource<'a> {
    pub(super) data: &'a data_beans::sparse_io_vector::SparseIoVec,
    pub(super) names: &'a [Box<str>],
    pub(super) fc: Option<&'a FeatureCoarsening>,
    pub(super) delta: Option<&'a DMatrix<f32>>,
    pub(super) keep: Option<&'a [usize]>,
}

/// Every cell of `data` read out at `mode` a block at a time (QC's
/// near-empty cells, those not in `keep`, left out), placed on `map` with
/// its velocity there.
pub(super) fn read_out_cells(
    CellSource {
        data,
        names,
        fc,
        delta,
        keep,
    }: CellSource<'_>,
    mode: &Mode,
    map: &MapBasis,
    block_size: usize,
) -> anyhow::Result<CellTable> {
    let lineages = mode.params.path.len();
    let all_cells = data.column_names()?;
    let kept_cell = keep.map(|k| keep_mask(Some(k), all_cells.len()));
    let mut t = CellTable {
        cells: Vec::new(),
        lineage: Vec::new(),
        p_lineage: vec![Vec::new(); lineages],
        tau: Vec::new(),
        tau_sd: Vec::new(),
        tau_lo: Vec::new(),
        tau_hi: Vec::new(),
        ufrac: Vec::new(),
        place: Vec::new(),
        velocity: Vec::new(),
    };
    let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    for_cell_blocks(data, names, block_size, delta, |b| {
        let (start, u, s) = (b.start, &b.unspliced, &b.spliced);
        let rows: Vec<usize> = (0..u.nrows())
            .filter(|i| kept_cell.as_ref().is_none_or(|k| k[start + i]))
            .collect();
        // A block QC emptied: nothing to read out.
        if rows.is_empty() {
            return Ok(());
        }
        let share = b.unspliced_share();
        let block = Reads::from_matrices(u.select_rows(&rows).cast(), s.select_rows(&rows).cast());
        let block = match fc {
            Some(fc) => coarse_reads(fc, &block),
            None => block,
        };
        let out = read_cells(&block, mode);
        t.cells
            .extend(rows.iter().map(|&i| all_cells[start + i].clone()));
        for p in &out.lineage {
            let best = lineage_index(p);
            t.lineage
                .push(best.map_or_else(|| "NA".into(), lineage_label));
        }
        for (l, col) in t.p_lineage.iter_mut().enumerate() {
            col.extend(
                out.lineage
                    .iter()
                    .map(|p| p.get(l).map_or(f32::NAN, |&x| x as f32)),
            );
        }
        t.tau.extend(f32s(&out.time_mean));
        t.tau_sd.extend(f32s(&out.time_sd));
        t.tau_lo.extend(f32s(&out.time_lo));
        t.tau_hi.extend(f32s(&out.time_hi));
        t.ufrac.extend(rows.iter().map(|&i| share[i]));
        let total = &block.u + &block.s;
        for c in 0..rows.len() {
            let x: Vec<f64> = total.row(c).iter().copied().collect();
            let dlog: Vec<f64> = out.velocity_log.row(c).iter().copied().collect();
            t.place.extend(f32s(&map.place(&x)));
            t.velocity.extend(f32s(&map.velocity(&x, &dlog)));
        }
        Ok(())
    })?;
    Ok(t)
}

impl CellTable {
    /// `{out}.cell_time.parquet`, and the map's `cell_embedding` and
    /// `cell_velocity` (`k` components).
    pub(super) fn write(&self, out: &str, k: usize) -> anyhow::Result<()> {
        let p_cols: Vec<(Box<str>, Vec<f32>)> = self
            .p_lineage
            .iter()
            .enumerate()
            .map(|(l, v)| (format!("p_L{}", l + 1).into(), v.clone()))
            .collect();
        let mut cols: Vec<(Box<str>, Column)> = vec![
            ("lineage".into(), Column::Str(&self.lineage)),
            ("tau".into(), Column::F32(&self.tau)),
            ("tau_sd".into(), Column::F32(&self.tau_sd)),
            ("tau_lo".into(), Column::F32(&self.tau_lo)),
            ("tau_hi".into(), Column::F32(&self.tau_hi)),
            ("ufrac".into(), Column::F32(&self.ufrac)),
        ];
        cols.extend(f32_columns(&p_cols));
        write_named_table(
            &format!("{out}.cell_time.parquet"),
            "cell",
            &self.cells,
            &cols,
        )?;
        write_rows(
            &format!("{out}.cell_embedding.parquet"),
            &self.cells,
            "pc",
            &self.place,
            k,
        )?;
        write_rows(
            &format!("{out}.cell_velocity.parquet"),
            &self.cells,
            "pc",
            &self.velocity,
            k,
        )
    }
}
