//! Opening the run's files on the side: its data (observed counts and
//! batches), an extra data file, and the model tables, each as a
//! background load.

use super::*;
use senna::run_manifest::RunManifest;
use senna::senna_input::{
    match_cells, read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Open the run's data files once: the observed counts, and each cell's
/// batch when there is more than one. Slow on a large run (every data file
/// is opened), so the screen does it on the side.
pub(crate) fn read_run_data(m: &RunManifest, dir: &Path) -> anyhow::Result<RunRead> {
    anyhow::ensure!(!m.data.input.is_empty(), "the manifest lists no data files");
    let files = m.data_inputs(dir);
    if let Some(gone) = files.iter().find(|f| !Path::new(f.as_ref()).exists()) {
        anyhow::bail!("the data is not here ({gone})");
    }
    let batch_files = m.data_batches(dir);
    let reload = senna::multiome_layout::recorded_layout(m.data.multiome.as_ref(), files.len())?;
    info!("view: opening {} data file(s)", files.len());
    let SparseDataWithBatch { data, batch, .. } =
        read_data_on_shared_rows(reload.apply(ReadSharedRowsArgs {
            data_files: files,
            batch_files: (!batch_files.is_empty()).then_some(batch_files),
            keep_empty_barcodes: true,
            ..Default::default()
        })?)?;
    let cells = data.column_names()?;
    anyhow::ensure!(
        cells.len() == batch.len(),
        "{} cells but {} batch labels",
        cells.len(),
        batch.len()
    );
    // A colouring only when there is more than one batch.
    let batches = batch.iter().any(|b| *b != batch[0]).then(|| {
        crate::view::data::Labels::new(
            crate::view::data::LabelKind::Batch,
            cells.iter().cloned().zip(batch),
            &[],
        )
    });
    let cell_index = cells
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i as u32))
        .collect();
    let observed = Observed::new(data, cell_index)?;
    Ok(RunRead { observed, batches })
}

/// Read data files the run was not fitted on, their cells together (on the
/// features they share) matched to the run's `cells` ([`match_cells`]), and
/// each cell's totals with them. Refuses files with none of the run's cells.
pub(crate) fn read_extra(paths: &[PathBuf], cells: &[Box<str>]) -> anyhow::Result<ExtraRead> {
    anyhow::ensure!(!paths.is_empty(), "no extra data file given");
    let name = crate::view::files::names(paths);
    info!("view: opening {name} as extra counts");
    let SparseDataWithBatch { data, .. } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: paths.iter().map(|p| p.to_string_lossy().into()).collect(),
        keep_empty_barcodes: true,
        ..Default::default()
    })?;
    let cell_index = match_cells(cells, &data.column_names()?);
    anyhow::ensure!(
        !cell_index.is_empty(),
        "none of the run's {} cells are in {name}",
        cells.len()
    );
    let observed = Observed::new(data, cell_index)?;
    observed.totals().map_err(anyhow::Error::msg)?;
    Ok(ExtraRead {
        name: name.into(),
        observed,
        of: cells.len(),
    })
}

/// A data file the run was not fitted on, opened in the view: its counts
/// with the run's cell names, and how many of the run's cells it has.
pub(crate) struct ExtraRead {
    pub name: Box<str>,
    pub(super) observed: Observed,
    pub of: usize,
}

impl ExtraRead {
    /// How many of the run's cells the file has.
    pub fn matched(&self) -> usize {
        self.observed.cell_index.len()
    }
}

/// An extra data file, read once on a worker thread.
pub(crate) type ExtraFiles = legume_tui::work::Loading<ExtraRead>;

/// What reading a run's data files gives: its observed counts, and each
/// cell's batch when there is more than one.
pub(crate) struct RunRead {
    pub(super) observed: Observed,
    pub batches: Option<crate::view::data::Labels>,
}

/// A run's data files, read once on a worker thread and shared by every
/// view of the run.
pub(crate) type RunFiles = legume_tui::work::Loading<RunRead>;

/// A run's model of expression, read once on a worker thread.
pub(crate) type ModelTables = legume_tui::work::Loading<Expected>;

/// A run's feature embedding, read once on a worker thread.
pub(crate) type FeatureTables = legume_tui::work::Loading<crate::view::features::FeatureEmbedding>;

/// A run's stage 2 (`senna donor-deviation`), read once on a worker thread.
pub(crate) type Stage2Tables = legume_tui::work::Loading<crate::view::stage2::Stage2>;

/// The read of the stage 2 the run at `dir` records (manifest `m`).
fn stage2_of(m: &RunManifest, dir: &Path) -> Arc<Stage2Tables> {
    let (m, dir) = (m.clone(), dir.to_path_buf());
    Arc::new(Stage2Tables::new(move || {
        crate::view::stage2::Stage2::load(&m, &dir).map_err(|e| e.to_string())
    }))
}

/// A run's slow reads, each started once on a worker thread and shared by
/// every view of the run: its data files, its model of expression, and
/// its feature embedding.
#[derive(Clone)]
pub(crate) struct RunLoads {
    pub files: Arc<RunFiles>,
    pub model: Arc<ModelTables>,
    pub features: Arc<FeatureTables>,
    /// Stage 2 of the run, when it has one: read with the rest.
    pub stage2: Arc<Stage2Tables>,
    /// Whether the run records a stage 2 to read.
    stage2_recorded: bool,
    /// Every cell's totals in the run's own counts, read once the data
    /// files are, on first asked: a pass over every column.
    pub totals: Arc<CellTotals>,
    /// Per-cell totals over the rows matching a pattern, by source and
    /// pattern, each read on first asked.
    pub row_sums: RowSumLoads,
    /// A data file opened in the view, when one is.
    pub extra: Option<Arc<ExtraFiles>>,
}

/// The run's own counts' per-cell totals, kept in [`Observed`] once read.
pub(crate) type CellTotals = legume_tui::work::Loading<()>;

/// A totals read waiting on `files`, the run's data files.
pub(super) fn totals_of(files: &Arc<RunFiles>) -> Arc<CellTotals> {
    let files = files.clone();
    Arc::new(CellTotals::new(move || {
        let read = files.get().as_ref().map_err(Clone::clone)?;
        read.observed.totals().map(|_| ())
    }))
}

impl RunLoads {
    /// The reads of the run at `dir` (manifest `m`), none started.
    pub(crate) fn new(m: &RunManifest, dir: &Path) -> Self {
        let run = || (m.clone(), dir.to_path_buf());
        let (fm, fd) = run();
        let (mm, md) = run();
        let (em, ed) = run();
        let files = Arc::new(RunFiles::new(move || {
            read_run_data(&fm, &fd).map_err(|e| e.to_string())
        }));
        Self {
            totals: totals_of(&files),
            files,
            model: Arc::new(ModelTables::new(move || {
                load_expected(&mm, &md).map_err(|e| e.to_string())
            })),
            features: Arc::new(FeatureTables::new(move || {
                crate::view::features::FeatureEmbedding::load(Some(&(em, ed)))
            })),
            stage2_recorded: !m.stage2.is_empty(),
            stage2: stage2_of(m, dir),
            row_sums: RowSumLoads::default(),
            extra: None,
        }
    }

    /// Read the stage 2 the run at `dir` records now (manifest `m`), in
    /// place of the one read before: `senna donor-deviation` added or
    /// replaced it since. Started at once when there is one.
    pub(crate) fn renew_stage2(&mut self, m: &RunManifest, dir: &Path) {
        self.stage2 = stage2_of(m, dir);
        self.stage2_recorded = !m.stage2.is_empty();
        if self.stage2_recorded {
            self.stage2.start();
        }
    }

    /// No run to read from, answered already: no worker thread answers
    /// in the middle of a test.
    #[cfg(test)]
    pub(crate) fn none() -> Self {
        const NO: &str = "no run to read from";
        Self {
            totals: Arc::new(CellTotals::ready(Err(NO.into()))),
            files: Arc::new(RunFiles::ready(Err(NO.into()))),
            model: Arc::new(ModelTables::ready(Err(NO.into()))),
            features: Arc::new(FeatureTables::ready(Err(NO.into()))),
            stage2: Arc::new(Stage2Tables::ready(Err(NO.into()))),
            stage2_recorded: false,
            row_sums: RowSumLoads::default(),
            extra: None,
        }
    }

    /// Start every read (each once).
    pub(crate) fn start(&self) {
        self.files.start();
        self.model.start();
        self.features.start();
        if self.stage2_recorded {
            self.stage2.start();
        }
    }

    /// What is still being read, and for how long: the longest first.
    pub(crate) fn busy(&self) -> Vec<(&'static str, std::time::Duration)> {
        let mut out: Vec<_> = [
            ("model tables", self.model.busy()),
            ("data files", self.files.busy()),
            ("feature embedding", self.features.busy()),
            ("cell totals", self.totals.busy()),
            ("stage 2", self.stage2.busy()),
            ("row statistics", self.row_sums.busy()),
            (
                "extra data file",
                self.extra.as_ref().and_then(|x| x.busy()),
            ),
        ]
        .into_iter()
        .filter_map(|(what, t)| t.map(|t| (what, t)))
        .collect();
        out.sort_by_key(|(_, t)| std::cmp::Reverse(*t));
        out
    }

    /// How many of the reads have answered.
    pub(crate) fn answered(&self) -> usize {
        usize::from(self.files.try_get().is_some())
            + usize::from(self.model.try_get().is_some())
            + usize::from(self.features.try_get().is_some())
            + usize::from(self.totals.try_get().is_some())
            + usize::from(self.stage2.try_get().is_some())
            + self.row_sums.answered()
            + usize::from(self.extra.as_ref().is_some_and(|x| x.try_get().is_some()))
    }
}
