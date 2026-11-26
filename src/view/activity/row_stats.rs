//! A per-cell statistic over some of a count source's rows (those whose
//! names match a pattern), and as a rate per the run's own totals: m6A
//! sites detected per 1k of the cell's UMIs, say. The pass over the rows
//! is a background load, shared by every view of the run.

use super::*;
use crate::view::colour_scale::Transform;
use crate::view::CellStat;
use data_beans::utilities::name_matching::names_matching;
use std::sync::Mutex;
use std::time::Duration;

/// A total or a count of features detected, over the rows whose names
/// match `rows` (every row: `None`; a regex matched case-insensitively,
/// as data-beans' `--row-name-pattern` is), divided by the run's own total
/// count per cell and scaled by `per_run` when that is set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowStat {
    pub measure: CellStat,
    /// A regular expression over feature names, case-insensitive.
    pub rows: Option<Box<str>>,
    /// Per this many of the cell's counts in the run's own data.
    pub per_run: Option<u32>,
    /// Divided by the same statistic over the rows matching this, as
    /// methylated over methylated and unmethylated.
    pub per_rows: Option<Box<str>>,
}

impl RowStat {
    /// Checked: a pattern that is no regular expression, or a zero scale,
    /// is refused.
    pub fn new(
        measure: CellStat,
        rows: Option<&str>,
        per_run: Option<u32>,
    ) -> Result<Self, String> {
        if let Some(p) = rows {
            names_matching(&[], p).map_err(|e| format!("rows: {e}"))?;
        }
        if per_run == Some(0) {
            return Err("per run UMI is a positive count (1000: per 1k)".into());
        }
        Ok(Self {
            measure,
            rows: rows.map(Into::into),
            per_run,
            per_rows: None,
        })
    }

    /// This statistic divided by the same over the rows matching
    /// `pattern`; not with a rate per the run's totals.
    pub fn per_rows(mut self, pattern: Option<&str>) -> Result<Self, String> {
        let Some(p) = pattern else {
            return Ok(self);
        };
        names_matching(&[], p).map_err(|e| format!("per_rows: {e}"))?;
        if self.per_run.is_some() {
            return Err("per_rows or per_run_umi, not both".into());
        }
        self.per_rows = Some(p.into());
        Ok(self)
    }

    /// What the key says: `features detected matching /methylated$ per 1k
    /// run UMIs`.
    #[must_use]
    pub fn name(&self) -> String {
        let mut name = match self.measure {
            CellStat::Total => "total counts",
            CellStat::Detected => "features detected",
            CellStat::Mean => "mean count",
        }
        .to_string();
        if let Some(p) = &self.rows {
            name += &format!(" matching {p}");
        }
        match self.per_run {
            Some(k) if k % 1000 == 0 => name += &format!(" per {}k run UMIs", k / 1000),
            Some(k) => name += &format!(" per {k} run UMIs"),
            None => {}
        }
        if let Some(p) = &self.per_rows {
            name += &format!(" / matching {p}");
        }
        name
    }
}

/// Each column's total and features detected over some rows, and how many
/// rows, read once.
type RowSums = legume_tui::work::Loading<(Vec<f32>, Vec<f32>, usize)>;

/// One read: of the extra file it holds (the run's own counts: `None`),
/// over the rows matching a pattern.
struct RowRead {
    extra: Option<Arc<ExtraFiles>>,
    pattern: Box<str>,
    sums: Arc<RowSums>,
}

/// The [`RowSums`] reads asked for, shared by every view of the run.
#[derive(Clone, Default)]
pub(crate) struct RowSumLoads(Arc<Mutex<Vec<RowRead>>>);

impl RowSumLoads {
    /// The read of `source`'s rows matching `pattern`, made on first ask.
    /// Reads of an extra file no longer open are dropped.
    fn of(&self, loads: &RunLoads, source: Source, pattern: &str) -> Result<Arc<RowSums>, String> {
        let extra = match source {
            Source::Observed => None,
            Source::Extra => Some(loads.extra.clone().ok_or("no extra data file is open")?),
            Source::Expected => return Err("the model's expectation is not counts".into()),
        };
        let same = |a: &Option<Arc<ExtraFiles>>, b: &Option<Arc<ExtraFiles>>| match (a, b) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (a, b) => a.is_none() && b.is_none(),
        };
        let mut reads = self.0.lock().map_err(|e| e.to_string())?;
        if extra.is_some() {
            reads.retain(|r| r.extra.is_none() || same(&r.extra, &extra));
        }
        if let Some(r) = reads
            .iter()
            .find(|r| same(&r.extra, &extra) && *r.pattern == *pattern)
        {
            return Ok(r.sums.clone());
        }
        let (files, pattern_owned) = (loads.files.clone(), pattern.to_string());
        let held = extra.clone();
        let read = Arc::new(RowSums::new(move || {
            let o = match &held {
                Some(x) => &x.get().as_ref().map_err(Clone::clone)?.observed,
                None => &files.get().as_ref().map_err(Clone::clone)?.observed,
            };
            let rows =
                names_matching(&o.features.names, &pattern_owned).map_err(|e| e.to_string())?;
            if rows.is_empty() {
                return Err(format!("no feature name matches {pattern_owned}"));
            }
            let (total, detected) = o.column_stats(Some(&rows))?;
            Ok((total, detected, rows.len()))
        }));
        reads.push(RowRead {
            extra,
            pattern: pattern.into(),
            sums: read.clone(),
        });
        Ok(read)
    }

    /// The longest a read has been running, while any runs.
    pub(crate) fn busy(&self) -> Option<Duration> {
        let reads = self.0.lock().ok()?;
        reads.iter().filter_map(|r| r.sums.busy()).max()
    }

    /// How many of the reads have answered.
    pub(crate) fn answered(&self) -> usize {
        self.0.lock().map_or(0, |reads| {
            reads.iter().filter(|r| r.sums.try_get().is_some()).count()
        })
    }
}

impl Activity {
    /// `stat` on each of the points `names` (the view `key`), from count
    /// source `source`; a rate per the run's own totals is of both.
    pub(super) fn row_stat_levels(
        &mut self,
        stat: &RowStat,
        source: Source,
        key: usize,
        names: &[Box<str>],
    ) -> Result<Levels, String> {
        let rows = self.rows(source, key, names)?;
        // Every read asked for at once, so none waits for another.
        let run_totals = stat
            .per_run
            .map(|_| self.loads.totals.answer(self.patient).map(drop));
        // The denominator's read starts before the numerator is waited on.
        if let Some(p) = &stat.per_rows {
            self.loads.row_sums.of(&self.loads, source, p)?;
        }
        let raw = self.row_measure(stat.measure, stat.rows.as_deref(), source)?;
        run_totals.transpose()?;
        if let Some(p) = &stat.per_rows {
            let den = self.row_measure(stat.measure, Some(p), source)?;
            let values = Self::on_points(&raw, &rows)
                .iter()
                .zip(Self::on_points(&den, &rows))
                .map(|(&n, d)| if d > 0.0 { n / d } else { f32::NAN })
                .collect();
            // A fraction starts at zero.
            return Ok(Levels::from_values(values, false).starting_at_zero());
        }
        let on_points = Self::on_points(&raw, &rows);
        let Some(k) = stat.per_run else {
            // Counts on ln(1 + count) unless the colour scale says
            // otherwise; the key's ticks are counts.
            let natural = Transform::for_counts(stat.measure != CellStat::Detected);
            return Ok(Levels::from_values(on_points, false).on(natural));
        };
        let run_rows = self.rows(Source::Observed, key, names)?;
        let umis = Self::on_points(&self.observed()?.totals()?.0, &run_rows);
        let values = on_points
            .iter()
            .zip(&umis)
            .map(|(&v, &u)| if u > 0.0 { v / u * k as f32 } else { f32::NAN })
            .collect();
        // A rate starts at zero, as a share does.
        Ok(Levels::from_values(values, true))
    }

    /// `measure` per column of `source` over the rows matching `pattern`
    /// (every row: `None`).
    fn row_measure(
        &mut self,
        measure: CellStat,
        pattern: Option<&str>,
        source: Source,
    ) -> Result<Vec<f32>, String> {
        let pick = |total: &[f32], detected: &[f32], n: usize| match measure {
            CellStat::Total => total.to_vec(),
            CellStat::Detected => detected.to_vec(),
            CellStat::Mean => {
                let n = n.max(1) as f32;
                total.iter().map(|t| t / n).collect()
            }
        };
        match pattern {
            Some(p) => {
                let read = self.loads.row_sums.of(&self.loads, source, p)?;
                let (total, detected, n) = read.answer(self.patient)?;
                Ok(pick(total, detected, *n))
            }
            None => {
                if source == Source::Observed {
                    self.loads.totals.answer(self.patient)?;
                }
                let counts = self.counts(source)?;
                let n = counts.features.names.len();
                let (total, detected) = counts.totals()?;
                Ok(pick(total, detected, n))
            }
        }
    }
}
