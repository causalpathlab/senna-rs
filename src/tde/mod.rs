//! `senna tde` (Temporal Divergence Embedding): `senna bge` whose phase-1 units
//! also observe which finest pseudobulks share their time (see [`context`]).
//! One likelihood: the context is one more categorical per unit, weighted
//! like its counts, with no weight of its own. Design: `docs/tde-plan.md`.

pub(crate) mod args;
pub(crate) mod context;

pub use args::TdeArgs;

use graph_embedding_util as ge;
use legume_numeric::matrix::common_io::read_lines_of_words_delim;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// A pseudobulk unit's name (`l{level}:pb{index}`), level and value.
type PbRow<T> = (Box<str>, i32, T);
/// A time bin's name, as in the time table.
type BinName = Box<str>;

/// The time context over a fit's units, from each cell's time bin. Keeps the
/// pseudobulk units' time profiles for `{out}.unit_time.parquet`.
struct TimeContext {
    cell_bin: Vec<Option<u32>>,
    bins: Vec<Box<str>>,
    /// Per pseudobulk unit: its name (`l{level}:pb{index}`, as in
    /// `{out}.pb_embedding.parquet`), level and time profile.
    pb_profiles: Mutex<Vec<PbRow<Vec<f32>>>>,
}

impl ge::UnitContextBuilder for TimeContext {
    fn build(
        &self,
        units: &ge::UnitTable,
        cell_to_pb_per_level: &[Vec<usize>],
    ) -> anyhow::Result<ge::UnitContext> {
        let total: Vec<f32> = (0..units.n_units()).map(|u| units.total_of(u, 0)).collect();
        let profile = context::profiles(
            &units.level,
            &units.source_index,
            cell_to_pb_per_level,
            &self.cell_bin,
            self.bins.len(),
        )?;
        *self.pb_profiles.lock().expect("unpoisoned") = (0..units.n_pb_units)
            .map(|u| {
                let l = units.level[u];
                (
                    format!("l{l}:pb{}", units.source_index[u]).into(),
                    i32::from(l),
                    profile[u].clone(),
                )
            })
            .collect();
        context::context_from_profiles(
            &profile,
            &units.level,
            &units.source_index,
            cell_to_pb_per_level.len(),
            &total,
        )
    }
}

/// The time context from each cell's continuous mean τ (see
/// [`context::continuous_context`]). Keeps the pseudobulk units' mean τ for
/// `{out}.unit_time.parquet`.
struct ContinuousTime {
    tau: Vec<Option<f32>>,
    pb_tau: Mutex<Vec<PbRow<f32>>>,
}

impl ge::UnitContextBuilder for ContinuousTime {
    fn build(
        &self,
        units: &ge::UnitTable,
        cell_to_pb_per_level: &[Vec<usize>],
    ) -> anyhow::Result<ge::UnitContext> {
        let total: Vec<f32> = (0..units.n_units()).map(|u| units.total_of(u, 0)).collect();
        let mean = context::unit_mean_tau(
            &units.level,
            &units.source_index,
            cell_to_pb_per_level,
            &self.tau,
        );
        *self.pb_tau.lock().expect("unpoisoned") = (0..units.n_pb_units)
            .filter_map(|u| {
                let l = units.level[u];
                mean[u].map(|t| {
                    (
                        format!("l{l}:pb{}", units.source_index[u]).into(),
                        i32::from(l),
                        t,
                    )
                })
            })
            .collect();
        let h = context::kernel_width(cell_to_pb_per_level.last().expect("a level"), &self.tau)?;
        info!("tde: time kernel width {h:.4} (pooled spread of τ within the finest pseudobulks)");
        context::continuous_context(
            &units.level,
            &units.source_index,
            cell_to_pb_per_level,
            &self.tau,
            &total,
        )
    }
}

impl ContinuousTime {
    /// `{out}.unit_time.parquet`: every timed pseudobulk unit's level and
    /// mean τ.
    fn write(&self, out: &str) -> anyhow::Result<()> {
        let rows = self.pb_tau.lock().expect("unpoisoned");
        let names: Vec<Box<str>> = rows.iter().map(|r| r.0.clone()).collect();
        let level: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let tau: Vec<f32> = rows.iter().map(|r| r.2).collect();
        let table: Vec<(Box<str>, Column)> = vec![
            ("level".into(), Column::I32(&level)),
            ("tau".into(), Column::F32(&tau)),
        ];
        write_named_table(&format!("{out}.unit_time.parquet"), "pb", &names, &table)?;
        info!("Wrote {out}.unit_time.parquet");
        Ok(())
    }
}

/// Each barcode's τ from `file`'s `column` (see [`time_bins`] for the
/// layout); `None` when missing or not a number.
fn time_values(
    file: &str,
    column: &str,
    barcodes: &[Box<str>],
) -> anyhow::Result<Vec<Option<f32>>> {
    let table = read_lines_of_words_delim(file, &['\t'][..], 0)?;
    let col = table
        .header
        .iter()
        .position(|h| h.as_ref() == column)
        .ok_or_else(|| anyhow::anyhow!("{file}: no column `{column}` in {:?}", table.header))?;
    let by_barcode: HashMap<&str, f32> = table
        .lines
        .iter()
        .filter_map(|l| Some((l.first()?.as_ref(), l.get(col)?.parse::<f32>().ok()?)))
        .filter(|(_, t)| t.is_finite())
        .collect();
    let tau: Vec<Option<f32>> = barcodes
        .iter()
        .map(|b| by_barcode.get(b.as_ref()).copied())
        .collect();
    let n = tau.iter().flatten().count();
    anyhow::ensure!(n > 0, "{file}: no barcode has a numeric `{column}`");
    info!(
        "tde: {n} of {} cells have a continuous time in `{column}`",
        tau.len()
    );
    Ok(tau)
}

impl TimeContext {
    /// `{out}.unit_time.parquet`: every pseudobulk unit's level and share of
    /// each time bin.
    fn write(&self, out: &str) -> anyhow::Result<()> {
        let rows = self.pb_profiles.lock().expect("unpoisoned");
        let names: Vec<Box<str>> = rows.iter().map(|r| r.0.clone()).collect();
        let level: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let shares: Vec<Vec<f32>> = (0..self.bins.len())
            .map(|t| rows.iter().map(|r| r.2[t]).collect())
            .collect();
        let mut table: Vec<(Box<str>, Column)> = vec![("level".into(), Column::I32(&level))];
        table.extend(
            self.bins
                .iter()
                .cloned()
                .zip(shares.iter().map(|v| Column::F32(v))),
        );
        write_named_table(&format!("{out}.unit_time.parquet"), "pb", &names, &table)?;
        info!("Wrote {out}.unit_time.parquet");
        Ok(())
    }
}

/// Each barcode's time bin from `file` (tab-separated, header line, barcodes
/// in the first column, the time in `column`): bins are the column's distinct
/// values, in sorted order.
fn time_bins(
    file: &str,
    column: &str,
    barcodes: &[Box<str>],
) -> anyhow::Result<(Vec<Option<u32>>, Vec<BinName>)> {
    let table = read_lines_of_words_delim(file, &['\t'][..], 0)?;
    let col = table
        .header
        .iter()
        .position(|h| h.as_ref() == column)
        .ok_or_else(|| anyhow::anyhow!("{file}: no column `{column}` in {:?}", table.header))?;
    let mut values: Vec<&str> = table
        .lines
        .iter()
        .filter_map(|l| l.get(col).map(AsRef::as_ref))
        .collect();
    values.sort_unstable();
    values.dedup();
    let bin_of: HashMap<&str, u32> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i as u32))
        .collect();
    let by_barcode: HashMap<&str, u32> = table
        .lines
        .iter()
        .filter_map(|l| Some((l.first()?.as_ref(), bin_of[l.get(col)?.as_ref()])))
        .collect();
    let cell_bin: Vec<Option<u32>> = barcodes
        .iter()
        .map(|b| by_barcode.get(b.as_ref()).copied())
        .collect();
    let n_labelled = cell_bin.iter().flatten().count();
    anyhow::ensure!(
        n_labelled > 0,
        "{file}: no barcode matches the data's cells"
    );
    info!(
        "tde: {n_labelled} of {} cells have a time in `{column}`, {} bins",
        cell_bin.len(),
        values.len()
    );
    Ok((cell_bin, values.iter().map(|&v| v.into()).collect()))
}

pub fn fit_tde(args: &TdeArgs) -> anyhow::Result<()> {
    if args.continuous {
        anyhow::ensure!(
            !args.collapse_by_time,
            "--collapse-by-time cuts by bins; continuous time does not enter the collapse yet"
        );
        let built: OnceLock<Arc<ContinuousTime>> = OnceLock::new();
        crate::bge::fit_bge_with(&args.bge, &|unified| {
            let ctx = Arc::new(ContinuousTime {
                tau: time_values(&args.time, &args.time_column, &unified.barcodes)?,
                pb_tau: Mutex::new(Vec::new()),
            });
            let _ = built.set(ctx.clone());
            Ok(crate::bge::FitHooks {
                unit_context: Some(ctx as crate::bge::UnitContextArc),
                strata: None,
            })
        })?;
        return match built.get() {
            Some(ctx) => ctx.write(&args.bge.out),
            None => Ok(()),
        };
    }
    let built: OnceLock<Arc<TimeContext>> = OnceLock::new();
    crate::bge::fit_bge_with(&args.bge, &|unified| {
        let (cell_bin, bins) = time_bins(&args.time, &args.time_column, &unified.barcodes)?;
        // Strata follow the count backend's column order; `0` mixes freely.
        let strata = if args.collapse_by_time {
            let names = unified.count_backend().column_names()?;
            let (backend_bin, _) = time_bins(&args.time, &args.time_column, &names)?;
            Some(
                backend_bin
                    .iter()
                    .map(|b| b.map_or(0, |t| t as usize + 1))
                    .collect(),
            )
        } else {
            None
        };
        let ctx = Arc::new(TimeContext {
            cell_bin,
            bins,
            pb_profiles: Mutex::new(Vec::new()),
        });
        let _ = built.set(ctx.clone());
        Ok(crate::bge::FitHooks {
            unit_context: Some(ctx as crate::bge::UnitContextArc),
            strata,
        })
    })?;
    match built.get() {
        Some(ctx) => ctx.write(&args.bge.out),
        None => Ok(()),
    }
}
