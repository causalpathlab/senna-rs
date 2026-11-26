//! The cell cycle as a second axis of batch: each cell's cycle state cut
//! into a few bins, and the cell's batch label made `batch|bin`, so the
//! collapse's per-gene batch δ is fitted per batch and bin and every model
//! that reads batch reads the cycle too. No exact phase is called: the bins
//! only separate cells.
//!
//! - `difference`: bins of S − G2/M. Resting cells, low in both, share the
//!   balanced bin with cells between S and G2/M, so the S-vs-G2/M axis is
//!   modelled and cycling-vs-resting is kept.
//! - `all`: bins of the standardised (S, G2/M), cycling-vs-resting too.
//!
//! The bins are k-means clusters (legume-numeric's seeded k-means) of those
//! values, k chosen from 1 to `--cycle-bins-max` by BIC under a spherical
//! Gaussian with one variance (X-means, Pelleg & Moore 2000), or fixed by
//! `--cycle-bins`. Nothing is binned when one bin is chosen, or when too few
//! cycle genes match the features.
//!
//! Groups of similar cells that sit almost wholly in one bin keep plain
//! batch labels: the groups are [`senna::coarse_groups`] of the fit's random
//! projection with the S and G2/M scores regressed out, so a group the
//! cycle alone made is not taken for one. A batch's bin with too few cells
//! merges into its nearest bin. Labels are joined by legume-genomic-types'
//! [`genomic_data::batch_label`] (`rep1_wt|cc1`).

use super::check::ONE_PHASE;
use super::score::{score, Scores};
use clap::{Args, ValueEnum};
use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::batch_label;
use genomic_data::cell_cycle::CycleGenes;
use legume_numeric::matrix::common_io::{read_lines, write_lines};
use legume_numeric::matrix::kmeans::KmeansRowsFit;
use log::info;
use nalgebra::DMatrix;
use senna::coarse_groups::seeded_kmeans;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// What of the cell cycle a fit models as batch.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CycleMode {
    /// Nothing.
    #[default]
    None,
    /// S against G2/M; cycling against resting kept.
    Difference,
    /// The whole cycle, cycling against resting too.
    All,
}

/// The flags a fit takes for the cell-cycle covariate.
#[derive(Args, Debug, Clone, Default, Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CycleCovariateArgs {
    #[arg(
        long = "cell-cycle",
        value_enum,
        default_value_t = CycleMode::None,
        help = "Model the cell cycle as a second axis of batch: difference (S against G2/M, resting kept) or all"
    )]
    pub mode: CycleMode,

    #[arg(
        long,
        value_name = "FILE",
        help = "With --cell-cycle: cell-cycle genes of your own (phase, symbol[, current]) [default: Tirosh et al. 2016]"
    )]
    pub cycle_genes: Option<Box<str>>,

    #[arg(
        long,
        help = "With --cell-cycle: this many cycle bins [default: chosen by BIC up to --cycle-bins-max]"
    )]
    pub cycle_bins: Option<usize>,

    #[arg(
        long,
        default_value_t = 10,
        help = "With --cell-cycle: most cycle bins BIC may choose"
    )]
    pub cycle_bins_max: usize,
}

impl CycleCovariateArgs {
    /// Refuse `--cell-cycle` where it does not yet work (`what`).
    ///
    /// # Errors
    /// `--cell-cycle` was given.
    pub fn ensure_off(&self, what: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.mode == CycleMode::None,
            "--cell-cycle does not yet work with {what}"
        );
        Ok(())
    }

    /// Refuse `--cell-cycle` on loads it cannot score, before anything is
    /// read: multiome (cells glued across files) or a carried pseudobulk
    /// reference (columns that are not cells).
    ///
    /// # Errors
    /// `--cell-cycle` with either.
    pub fn ensure_loadable(&self, multiome: bool, pb_reference: bool) -> anyhow::Result<()> {
        if multiome {
            self.ensure_off("multiome loads, where cells are glued across files")?;
        }
        if pb_reference {
            self.ensure_off("a carried pseudobulk reference (--pb-reference)")?;
        }
        Ok(())
    }
}

/// A bin with fewer cells in a batch merges into its nearest bin.
pub const MIN_LEVEL_CELLS: usize = 10;
/// Groups of similar cells the one-phase rule looks at, and the projection
/// components they are clustered on.
const GROUPS: usize = 16;
const GROUP_DIMS: usize = 10;
const SEED: u64 = 42;

/// How the cycle was cut into bins, recorded beside the run.
#[derive(Debug, Serialize)]
pub struct CycleModel {
    pub mode: CycleMode,
    pub genes: Box<str>,
    pub matched: [usize; 2],
    /// Bins chosen, and the BIC of each k tried (k = 1 first).
    pub k: usize,
    pub bic: Vec<f64>,
    /// The bins' centres in the binned values (`difference`: S − G2/M;
    /// `all`: S and G2/M, standardised by `standardised`).
    pub centroids: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub standardised: Option<Standardised>,
    /// Cells left with plain batch labels: in a group that sits at least
    /// [`ONE_PHASE`] in one bin, or in a batch whose bins all merged.
    pub plain_cells: usize,
    pub one_bin_groups: usize,
    /// Cells per composite label.
    pub levels: BTreeMap<Box<str>, usize>,
}

/// How `all` standardised (S, G2/M) before binning.
#[derive(Debug, Serialize)]
pub struct Standardised {
    pub centre: [f32; 2],
    pub scale: [f32; 2],
}

/// The composite labels and the model they came from.
pub struct CycleBatches {
    pub labels: Vec<Box<str>>,
    pub model: CycleModel,
    pub scores: Scores,
}

/// Each cell's `batch|bin` label, or `None` (with the reason logged) when
/// the cycle is not modelled: mode `none`, too few genes, or one bin.
/// `proj_kn` is the fit's random projection (dims × cells) under plain
/// batch labels.
///
/// # Errors
/// A gene table or column block that cannot be read.
pub fn cycle_batches(
    data: &SparseIoVec,
    batch: &[Box<str>],
    proj_kn: &DMatrix<f32>,
    args: &CycleCovariateArgs,
    block: usize,
) -> anyhow::Result<Option<CycleBatches>> {
    if args.mode == CycleMode::None {
        return Ok(None);
    }
    // A run that inherited `batch|ccK` labels bins its cells afresh.
    let mut plain_of: HashMap<&str, Box<str>> = HashMap::new();
    let batch: Vec<Box<str>> = batch
        .iter()
        .map(|b| {
            if let Some(p) = plain_of.get(b.as_ref()) {
                return Ok(p.clone());
            }
            let p = plain_batch(b)?;
            plain_of.insert(b, p.clone());
            Ok(p)
        })
        .collect::<anyhow::Result<_>>()?;
    let genes = match args.cycle_genes.as_deref() {
        Some(p) => CycleGenes::read(std::path::Path::new(p)).map_err(anyhow::Error::msg)?,
        None => CycleGenes::tirosh_2016(),
    };
    let scores = match score(data, &genes, block, false, SEED)? {
        Ok(s) => s,
        Err(why) => {
            info!("--cell-cycle: not modelled, {why}");
            return Ok(None);
        }
    };
    let (values, standardised) = binned_values(&scores, args.mode);
    let (bic, fit) = choose_bins(&values, args)?;
    let k = fit.centroids.nrows();
    if k < 2 {
        if args.cycle_bins.is_some() {
            info!("--cell-cycle: not modelled, --cycle-bins asked for one bin");
        } else {
            info!("--cell-cycle: not modelled, BIC chose one bin (BIC by k: {bic:?})");
        }
        return Ok(None);
    }
    let centroids: Vec<Vec<f32>> = (0..k)
        .map(|c| fit.centroids.row(c).iter().copied().collect())
        .collect();
    let mut bin: Vec<Option<usize>> = fit.labels.iter().map(|&b| Some(b)).collect();

    // Groups of similar cells sitting wholly in one bin keep plain labels.
    let groups = senna::coarse_groups::coarse_groups(
        proj_kn,
        &[&scores.s, &scores.g2m],
        GROUPS,
        GROUP_DIMS,
        SEED,
    )?;
    let one_bin_groups = clear_one_bin_groups(&mut bin, &groups, k);
    merge_sparse_levels(&mut bin, &batch, &centroids);

    // Each (batch, bin) label composed once.
    let mut composed: HashMap<(&str, Option<usize>), Box<str>> = HashMap::new();
    let mut levels: BTreeMap<Box<str>, usize> = BTreeMap::new();
    let mut labels: Vec<Box<str>> = Vec::with_capacity(batch.len());
    for (b, &c) in batch.iter().zip(&bin) {
        let label = match composed.get(&(b.as_ref(), c)) {
            Some(l) => l.clone(),
            None => {
                let l: Box<str> = match c {
                    Some(c) => batch_label::compose(&[b, &format!("cc{c}")])
                        .map_err(anyhow::Error::msg)?
                        .into(),
                    None => b.clone(),
                };
                composed.insert((b, c), l.clone());
                levels.insert(l.clone(), 0);
                l
            }
        };
        *levels.get_mut(&label).expect("counted") += 1;
        labels.push(label);
    }
    let plain_cells = bin.iter().filter(|c| c.is_none()).count();
    info!(
        "--cell-cycle {:?}: {k} bins (BIC by k: {bic:?}); {plain_cells} cells kept plain batch labels \
         ({one_bin_groups} group(s) at least {:.0}% one bin)",
        args.mode,
        100.0 * ONE_PHASE
    );
    Ok(Some(CycleBatches {
        labels,
        model: CycleModel {
            mode: args.mode,
            genes: genes.source.clone(),
            matched: [scores.s_rows.len(), scores.g2m_rows.len()],
            k,
            bic,
            centroids,
            standardised,
            plain_cells,
            one_bin_groups,
            levels,
        },
        scores,
    }))
}

/// One line of a batch file: a loaded cell (its index among the scored
/// cells) or a column QC set aside, with its plain batch label.
pub enum Line {
    Cell(usize),
    Plain(Box<str>),
}

/// The cycle as modelled for a fit: each loaded cell's `batch|bin` label
/// and, with an output prefix, the batch files written for the run to
/// record.
pub struct Modelled {
    pub labels: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
}

/// Where a fit writes its cycle model and batch files, and what the plain
/// labels of columns QC set aside come from (see [`column_layout`]).
pub struct CycleOutput<'a> {
    pub out: &'a str,
    pub batch_files: Option<&'a [Box<str>]>,
    pub ignore_batch: bool,
}

/// [`cycle_batches`] for a fit's loaded cells, and with `output` the
/// model, scores and batch files written ([`CycleBatches::write`] over
/// [`column_layout`]); `None` when the cycle is not modelled.
///
/// # Errors
/// As [`cycle_batches`], [`column_layout`] and [`CycleBatches::write`].
pub fn model_cycle(
    data: &SparseIoVec,
    batch: &[Box<str>],
    proj_kn: &DMatrix<f32>,
    args: &CycleCovariateArgs,
    block: usize,
    output: Option<CycleOutput<'_>>,
    reread: impl FnOnce() -> anyhow::Result<data_beans::aux::data_loading::SparseDataWithBatch>,
) -> anyhow::Result<Option<Modelled>> {
    let Some(cycle) = cycle_batches(data, batch, proj_kn, args, block)? else {
        return Ok(None);
    };
    let batch_files = output
        .map(|o| {
            let layout = column_layout(data, o.batch_files, o.ignore_batch, reread)?;
            cycle.write(o.out, &data.column_names()?, &layout)
        })
        .transpose()?;
    Ok(Some(Modelled {
        labels: cycle.labels,
        batch_files,
    }))
}

/// Every column of every data file as loaded before QC, file by file, so
/// batch files can be written whole: the loaded cells of `data` at their
/// places, and each column QC set aside with its plain label (`all` under
/// `ignore_batch`, else its line of the file's batch file, else, when the
/// labels were inferred from names or file names, the data read again by
/// `reread`, called only when some column was set aside).
///
/// # Errors
/// A batch file that cannot be read or does not match its data file.
pub fn column_layout(
    data: &SparseIoVec,
    batch_files: Option<&[Box<str>]>,
    ignore_batch: bool,
    reread: impl FnOnce() -> anyhow::Result<data_beans::aux::data_loading::SparseDataWithBatch>,
) -> anyhow::Result<Vec<Vec<Line>>> {
    let per_file = data.num_columns_by_data()?;
    let mut at: Vec<Vec<Option<usize>>> = per_file.iter().map(|&n| vec![None; n]).collect();
    for col in 0..data.num_columns() {
        for loc in data.column_locations(col) {
            at[loc.backend as usize][loc.local_col as usize] = Some(col);
        }
    }
    let set_aside = at.iter().flatten().any(Option::is_none);
    // With nothing set aside no plain label is written.
    let plain: Vec<Vec<Box<str>>> = if !set_aside || ignore_batch {
        per_file.iter().map(|&n| vec!["all".into(); n]).collect()
    } else if let Some(files) = batch_files {
        anyhow::ensure!(
            files.len() == per_file.len(),
            "{} batch files for {} data files",
            files.len(),
            per_file.len()
        );
        files
            .iter()
            .map(|f| read_lines(f))
            .collect::<anyhow::Result<_>>()?
    } else {
        let whole = reread()?;
        let mut batch = whole.batch.into_iter();
        per_file
            .iter()
            .map(|&n| batch.by_ref().take(n).collect())
            .collect()
    };
    let files = at
        .into_iter()
        .zip(plain)
        .enumerate()
        .map(|(f, (cols, plain))| {
            anyhow::ensure!(
                plain.len() == cols.len(),
                "--cell-cycle: data file {f} has {} columns but {} plain batch labels",
                cols.len(),
                plain.len()
            );
            cols.into_iter()
                .zip(&plain)
                .map(|(c, b)| match c {
                    Some(c) => Ok(Line::Cell(c)),
                    None => plain_batch(b).map(Line::Plain),
                })
                .collect()
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(files)
}

/// `label` without the cycle axis an earlier `--cell-cycle` fit gave it:
/// `rep1|cc0` is `rep1`, `rep1` stays.
///
/// # Errors
/// A label with any other second axis, which `--cell-cycle` cannot extend.
pub fn plain_batch(label: &str) -> anyhow::Result<Box<str>> {
    match batch_label::split(label).as_slice() {
        [b] => Ok((*b).into()),
        [b, c] if cycle_bin(c).is_some() => Ok((*b).into()),
        _ => anyhow::bail!(
            "--cell-cycle: batch label {label:?} already has more than one part (joined by '{}'); \
             give --cell-cycle plain batch labels",
            batch_label::AXIS_SEP
        ),
    }
}

/// The bin a label part names, `cc0` → 0; `None` for any other part.
fn cycle_bin(part: &str) -> Option<usize> {
    part.strip_prefix("cc")
        .filter(|k| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()))?
        .parse()
        .ok()
}

/// Refuse to score new cells against `model` when it was fitted with
/// `--cell-cycle`: its batch δ is keyed by `batch|ccK`, which new cells do
/// not carry until their cycle is binned by the model's own bins.
///
/// # Errors
/// A model fitted with `--cell-cycle`.
pub fn ensure_no_cycle_axis(model: &str, what: &str) -> anyhow::Result<()> {
    #[derive(serde::Deserialize, Default)]
    #[serde(default)]
    struct Collapse {
        cell_cycle: CycleCovariateArgs,
    }
    #[derive(serde::Deserialize, Default)]
    #[serde(default)]
    struct Recorded {
        collapse: Collapse,
    }
    let mode = senna::run_manifest::load_for(model)
        .and_then(|(m, _)| m.train_args_as::<Recorded>(model))
        .map_or(CycleMode::None, |r| r.collapse.cell_cycle.mode);
    anyhow::ensure!(
        mode == CycleMode::None,
        "{what} on {model}: that run was fitted with --cell-cycle {}, and new cells cannot \
         yet be given its cycle bins; refit without --cell-cycle to {what}",
        format!("{mode:?}").to_lowercase()
    );
    Ok(())
}

impl CycleBatches {
    /// Write the cycle model (`{out}.cell_cycle_model.json`), each kept
    /// cell's scores and label (`{out}.cell_cycle.parquet`), and one batch
    /// file per data file (`{out}.cell_cycle.batch.{i}.txt`), every column
    /// of the file a line: its composite label, or its plain one for a
    /// column QC set aside. Returns the batch files, in data-file order.
    ///
    /// # Errors
    /// A file that cannot be written.
    pub fn write(
        &self,
        out: &str,
        cells: &[Box<str>],
        layout: &[Vec<Line>],
    ) -> anyhow::Result<Vec<Box<str>>> {
        senna::embed_common::mkdir_parent(out)?;
        let model = format!("{out}.cell_cycle_model.json");
        std::fs::write(&model, serde_json::to_string_pretty(&self.model)?)?;
        self.write_cells(&format!("{out}.cell_cycle.parquet"), cells)?;
        let mut files: Vec<Box<str>> = Vec::with_capacity(layout.len());
        for (i, lines) in layout.iter().enumerate() {
            let path = format!("{out}.cell_cycle.batch.{i}.txt");
            let text: Vec<Box<str>> = lines
                .iter()
                .map(|line| match line {
                    Line::Cell(c) => self.labels[*c].clone(),
                    Line::Plain(b) => b.clone(),
                })
                .collect();
            write_lines(&text, &path)?;
            files.push(path.into());
        }
        info!(
            "--cell-cycle: wrote {model}, {out}.cell_cycle.parquet and {} batch file(s) {out}.cell_cycle.batch.*.txt",
            files.len()
        );
        Ok(files)
    }

    fn write_cells(&self, path: &str, cells: &[Box<str>]) -> anyhow::Result<()> {
        use legume_numeric::matrix::parquet::{write_named_table, Column};
        let s = &self.scores;
        let phase: Vec<Box<str>> = s.phase.iter().map(|p| p.name().into()).collect();
        write_named_table(
            path,
            "cell",
            cells,
            &[
                ("s_score".into(), Column::F32(&s.s)),
                ("g2m_score".into(), Column::F32(&s.g2m)),
                ("phase".into(), Column::Str(&phase)),
                ("angle".into(), Column::F32(&s.angle)),
                ("batch_label".into(), Column::Str(&self.labels)),
            ],
        )
    }
}

/// Each gene's cell-cycle effect, from the collapse's per-gene δ over the
/// batch levels (`log_delta`, genes × levels, the levels named): per bin,
/// the mean over batches of log δ at `batch|ccK`, centred over the bins, so
/// a gene's row says how much each bin was absorbed. Levels without a bin
/// (plain labels) are left out. Written to `{out}.cycle_effect.parquet`.
///
/// # Errors
/// A table that cannot be written.
pub fn write_cycle_effect(
    out: &str,
    genes: &[Box<str>],
    levels: &[Box<str>],
    log_delta: &DMatrix<f32>,
) -> anyhow::Result<()> {
    // Each bin (in bin order) and the levels it holds.
    let mut of_bin: BTreeMap<usize, (Box<str>, Vec<usize>)> = BTreeMap::new();
    for (l, level) in levels.iter().enumerate() {
        if let [_, bin] = batch_label::split(level).as_slice() {
            if let Some(k) = cycle_bin(bin) {
                of_bin
                    .entry(k)
                    .or_insert_with(|| ((*bin).into(), Vec::new()))
                    .1
                    .push(l);
            }
        }
    }
    if of_bin.is_empty() {
        return Ok(());
    }
    let (bins, members): (Vec<Box<str>>, Vec<Vec<usize>>) = of_bin.into_values().unzip();
    let mut effect: Vec<Vec<f32>> = vec![vec![0.0; genes.len()]; bins.len()];
    for g in 0..genes.len() {
        let means: Vec<f64> = members
            .iter()
            .map(|ls| {
                ls.iter()
                    .map(|&l| f64::from(log_delta[(g, l)]))
                    .sum::<f64>()
                    / ls.len() as f64
            })
            .collect();
        let centre = means.iter().sum::<f64>() / means.len() as f64;
        for (k, m) in means.iter().enumerate() {
            effect[k][g] = (m - centre) as f32;
        }
    }
    let cols: Vec<(Box<str>, legume_numeric::matrix::parquet::Column)> = bins
        .iter()
        .zip(&effect)
        .map(|(b, e)| (b.clone(), legume_numeric::matrix::parquet::Column::F32(e)))
        .collect();
    let path = format!("{out}.cycle_effect.parquet");
    legume_numeric::matrix::parquet::write_named_table(&path, "gene", genes, &cols)?;
    info!("--cell-cycle: wrote {path} (log δ by bin, centred per gene)");
    Ok(())
}

/// The values the bins cut, cells × dims: S − G2/M, or the standardised
/// (S, G2/M) and how they were standardised.
fn binned_values(s: &Scores, mode: CycleMode) -> (DMatrix<f32>, Option<Standardised>) {
    let n = s.s.len();
    match mode {
        CycleMode::Difference => {
            let d: Vec<f32> = s.s.iter().zip(&s.g2m).map(|(a, b)| a - b).collect();
            (DMatrix::from_vec(n, 1, d), None)
        }
        _ => {
            let (ms, ss) = senna::embed_common::mean_sd(&s.s);
            let (mg, sg) = senna::embed_common::mean_sd(&s.g2m);
            let z = DMatrix::from_fn(n, 2, |i, c| {
                if c == 0 {
                    (s.s[i] - ms) / ss
                } else {
                    (s.g2m[i] - mg) / sg
                }
            });
            (
                z,
                Some(Standardised {
                    centre: [ms, mg],
                    scale: [ss, sg],
                }),
            )
        }
    }
}

/// The bins: k-means for each k up to the maximum and the fit with the
/// least BIC (or the k asked for), with every k's BIC.
fn choose_bins(
    x: &DMatrix<f32>,
    args: &CycleCovariateArgs,
) -> anyhow::Result<(Vec<f64>, KmeansRowsFit)> {
    if let Some(k) = args.cycle_bins {
        anyhow::ensure!(k >= 1, "--cycle-bins is at least 1");
        return Ok((Vec::new(), seeded_kmeans(x, k, SEED)));
    }
    anyhow::ensure!(args.cycle_bins_max >= 1, "--cycle-bins-max is at least 1");
    let mut bics = Vec::with_capacity(args.cycle_bins_max);
    let mut best: Option<(f64, KmeansRowsFit)> = None;
    for k in 1..=args.cycle_bins_max {
        let fit = seeded_kmeans(x, k, SEED);
        let b = bic(x, &fit);
        bics.push(b);
        if best.as_ref().is_none_or(|(least, _)| b < *least) {
            best = Some((b, fit));
        }
    }
    let (_, fit) = best.expect("k = 1 at least");
    Ok((bics, fit))
}

/// BIC of a k-means fit as a mixture of spherical Gaussians with one
/// variance (X-means): `−2 log L + p log n`, `p = (k − 1) + k·d + 1`.
fn bic(x: &DMatrix<f32>, fit: &legume_numeric::matrix::kmeans::KmeansRowsFit) -> f64 {
    let (n, d) = (x.nrows(), x.ncols());
    let k = fit.centroids.nrows();
    let mut size = vec![0usize; k];
    let mut rss = 0f64;
    for (i, &c) in fit.labels.iter().enumerate() {
        size[c] += 1;
        rss += (0..d)
            .map(|j| f64::from(x[(i, j)] - fit.centroids[(c, j)]).powi(2))
            .sum::<f64>();
    }
    let (nf, df) = (n as f64, d as f64);
    let var = (rss / (df * (nf - k as f64).max(1.0))).max(1e-12);
    let log_l = size
        .iter()
        .filter(|&&m| m > 0)
        .map(|&m| m as f64 * (m as f64 / nf).ln())
        .sum::<f64>()
        - 0.5 * nf * df * (2.0 * std::f64::consts::PI * var).ln()
        - 0.5 * rss / var;
    let p = (k - 1) as f64 + k as f64 * df + 1.0;
    -2.0 * log_l + p * nf.ln()
}

/// Clear the bin of every cell in a group at least [`ONE_PHASE`] in one
/// bin; returns how many groups.
fn clear_one_bin_groups(bin: &mut [Option<usize>], groups: &[usize], k: usize) -> usize {
    let n_groups = groups.iter().copied().max().map_or(0, |g| g + 1);
    let label: Vec<Option<u32>> = bin.iter().map(|c| c.map(|c| c as u32)).collect();
    let one: Vec<bool> = senna::cell_labels::label_composition(groups, &label, n_groups, k)
        .iter()
        .map(|c| {
            c.as_ref()
                .is_some_and(|c| c.iter().copied().fold(0.0, f32::max) >= ONE_PHASE)
        })
        .collect();
    for (c, &g) in bin.iter_mut().zip(groups) {
        if one[g] {
            *c = None;
        }
    }
    one.iter().filter(|&&o| o).count()
}

/// Within each batch, merge a bin with fewer than [`MIN_LEVEL_CELLS`]
/// cells into the bin whose centre is nearest, smallest first; a batch left
/// with one bin keeps its plain label.
fn merge_sparse_levels(bin: &mut [Option<usize>], batch: &[Box<str>], centroids: &[Vec<f32>]) {
    let dist = |a: usize, b: usize| -> f32 {
        centroids[a]
            .iter()
            .zip(&centroids[b])
            .map(|(x, y)| (x - y).powi(2))
            .sum()
    };
    let mut cells_of: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, b) in batch.iter().enumerate() {
        cells_of.entry(b).or_default().push(i);
    }
    for cells in cells_of.values() {
        loop {
            let mut count: BTreeMap<usize, usize> = BTreeMap::new();
            for &i in cells {
                if let Some(c) = bin[i] {
                    *count.entry(c).or_default() += 1;
                }
            }
            if count.len() <= 1 {
                for &i in cells {
                    bin[i] = None;
                }
                break;
            }
            let Some((&small, &n)) = count.iter().min_by_key(|(_, &n)| n) else {
                break;
            };
            if n >= MIN_LEVEL_CELLS {
                break;
            }
            let into = count
                .keys()
                .copied()
                .filter(|&c| c != small)
                .min_by(|&a, &b| dist(small, a).total_cmp(&dist(small, b)))
                .expect("another bin");
            for &i in cells {
                if bin[i] == Some(small) {
                    bin[i] = Some(into);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bic_prefers_two_bins_for_two_clumps_and_one_for_one() {
        let two = DMatrix::from_fn(
            200,
            1,
            |i, _| if i < 100 { -3.0 } else { 3.0 } + (i % 7) as f32 * 0.05,
        );
        let one = DMatrix::from_fn(200, 1, |i, _| (i % 13) as f32 * 0.05);
        let args = CycleCovariateArgs {
            mode: CycleMode::Difference,
            cycle_bins_max: 3,
            ..Default::default()
        };
        assert_eq!(choose_bins(&two, &args).unwrap().1.centroids.nrows(), 2);
        assert_eq!(choose_bins(&one, &args).unwrap().1.centroids.nrows(), 1);
    }

    #[test]
    fn a_sparse_bin_merges_into_its_nearest_and_a_lone_bin_is_plain() {
        let centroids = vec![vec![0.0], vec![1.0], vec![10.0]];
        // Batch a: 60 in bin 0, 5 in bin 1 (under the floor: merges into 0,
        // the nearest), 60 in bin 2. Batch b: 30 in bin 2 only (one bin:
        // plain).
        let mut bin: Vec<Option<usize>> = std::iter::repeat_n(Some(0), 60)
            .chain(std::iter::repeat_n(Some(1), 5))
            .chain(std::iter::repeat_n(Some(2), 60))
            .chain(std::iter::repeat_n(Some(2), 30))
            .collect();
        let batch: Vec<Box<str>> = (0..155)
            .map(|i| if i < 125 { "a" } else { "b" }.into())
            .collect();
        merge_sparse_levels(&mut bin, &batch, &centroids);
        assert!(bin[60..65].iter().all(|&c| c == Some(0)));
        assert_eq!(bin[65], Some(2));
        assert!(bin[125..].iter().all(Option::is_none));
    }

    #[test]
    fn a_label_from_an_earlier_cycle_fit_loses_its_bin_and_others_are_refused() {
        assert_eq!(plain_batch("rep1").unwrap().as_ref(), "rep1");
        assert_eq!(plain_batch("rep1|cc3").unwrap().as_ref(), "rep1");
        assert!(plain_batch("rep1|donor").is_err());
        assert!(plain_batch("rep1|cc").is_err());
        assert!(plain_batch("a|b|cc0").is_err());
    }

    #[test]
    fn a_group_wholly_in_one_bin_keeps_plain_labels() {
        let mut bin: Vec<Option<usize>> = (0..40)
            .map(|i| Some(usize::from(i >= 20 && i % 2 == 0)))
            .collect();
        // Group 0: cells 0..20, all bin 0; group 1: mixed.
        let groups: Vec<usize> = (0..40).map(|i| usize::from(i >= 20)).collect();
        assert_eq!(clear_one_bin_groups(&mut bin, &groups, 2), 1);
        assert!(bin[..20].iter().all(Option::is_none));
        assert!(bin[20..].iter().all(Option::is_some));
    }

    #[test]
    fn the_cycle_effect_averages_batches_per_bin_and_centres_per_gene() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("r").to_string_lossy().into_owned();
        let genes: Vec<Box<str>> = vec!["G1".into(), "G2".into()];
        let levels: Vec<Box<str>> = ["a|cc0", "a|cc1", "b|cc0", "b|cc1", "c"]
            .iter()
            .map(|l| Box::from(*l))
            .collect();
        // G1: cc0 at 1 and 3 (mean 2), cc1 at 0 and 0 (mean 0), plain "c" ignored.
        // G2: no cycle effect.
        let log_delta =
            DMatrix::from_row_slice(2, 5, &[1.0, 0.0, 3.0, 0.0, 9.0, 0.5, 0.5, 0.5, 0.5, 9.0]);
        write_cycle_effect(&out, &genes, &levels, &log_delta).unwrap();
        use senna::embed_common::*;
        let m = Mat::from_parquet_with_row_names(&format!("{out}.cycle_effect.parquet"), Some(0))
            .unwrap();
        assert_eq!(m.cols, [Box::from("cc0"), Box::from("cc1")]);
        assert!((m.mat[(0, 0)] - 1.0).abs() < 1e-6 && (m.mat[(0, 1)] + 1.0).abs() < 1e-6);
        assert!(m.mat[(1, 0)].abs() < 1e-6 && m.mat[(1, 1)].abs() < 1e-6);
    }
}
