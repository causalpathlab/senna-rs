//! One feature's activity per cell, for colouring the cell map by a gene.
//!
//! Two sources:
//!
//! - **expected**: what the model predicts, from tables already on disk and so
//!   instant for any feature. Embedding runs: `ρ_g · z_n`, the feature's
//!   program in each cell without the cell's depth term. Topic runs:
//!   `log Σ_k θ_nk β_gk`. Only the relative level across cells means anything.
//! - **observed**: `ln(1 + count)` from the run's data files, one row read at a
//!   time. The backend is opened on first use.
//!
//! Values come back aligned to a set of points by cell name, ready to be
//! mapped onto a colour ramp.

use data_beans::utilities::name_matching::GeneIndex;
use rustc_hash::FxHashMap as HashMap;
use senna::embed_common::*;
use senna::run_manifest::{self, RunManifest};
use senna::senna_input::{read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Expected,
    Observed,
}

impl Source {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Source::Expected => "expected",
            Source::Observed => "observed",
        }
    }

    #[must_use]
    pub fn other(self) -> Self {
        match self {
            Source::Expected => Source::Observed,
            Source::Observed => Source::Expected,
        }
    }
}

/// A feature axis with the shared name matcher (exact, then symbol, then
/// flexible), so a marker named `GENE1` finds a row named `ENSG…_GENE1`.
struct Axis {
    names: Vec<Box<str>>,
    index: GeneIndex,
}

impl Axis {
    fn new(names: Vec<Box<str>>) -> Self {
        let index = GeneIndex::build(&names);
        Self { names, index }
    }
}

enum Model {
    /// `z` is N × H, `rho` is D × H.
    Embedding { z: Mat, rho: Mat },
    /// `log_theta` is N × K, `log_beta` is D × K.
    Topic { log_theta: Mat, log_beta: Mat },
}

struct Expected {
    features: Axis,
    cells: Vec<Box<str>>,
    model: Model,
}

struct Observed {
    data: SparseIoVec,
    features: Axis,
    cells: Vec<Box<str>>,
}

/// Values of one feature (or feature set) per point, `NaN` where the point
/// has no value, and the range the colour ramp spans.
pub struct Levels {
    pub values: Vec<f32>,
    pub lo: f32,
    pub hi: f32,
    /// Observed counts: a zero means "not detected" and is drawn muted rather
    /// than at the bottom of the ramp.
    pub zero_is_off: bool,
}

impl Levels {
    /// Position on the ramp in `[0, 1]`, or a negative number for "off".
    #[inline]
    #[must_use]
    pub fn t(&self, i: usize) -> f32 {
        let v = self.values[i];
        if !v.is_finite() || (self.zero_is_off && v <= 0.0) {
            return -1.0;
        }
        ((v - self.lo) / (self.hi - self.lo).max(1e-12)).clamp(0.0, 1.0)
    }

    fn from_values(values: Vec<f32>, zero_is_off: bool) -> Self {
        let mut sample: Vec<f32> = values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && (!zero_is_off || *v > 0.0))
            .collect();
        let (lo, hi) = if sample.is_empty() {
            (0.0, 1.0)
        } else {
            sample.sort_unstable_by(f32::total_cmp);
            let q = |p: f32| sample[((sample.len() - 1) as f32 * p) as usize];
            // Counts start the ramp at zero; model levels at a low quantile,
            // since their scale has no natural origin.
            let lo = if zero_is_off { 0.0 } else { q(0.02) };
            (lo, q(0.99).max(lo + 1e-6))
        };
        Self {
            values,
            lo,
            hi,
            zero_is_off,
        }
    }
}

/// Values per source cell, the cell names they belong to, and the feature's
/// name as its axis spells it.
type Raw<'a> = (Vec<f32>, &'a [Box<str>], Box<str>);

pub struct Activity {
    manifest: RunManifest,
    dir: PathBuf,
    expected: Option<Result<Expected, String>>,
    observed: Option<Result<Observed, String>>,
}

fn read_table(dir: &Path, rel: &str) -> anyhow::Result<MatWithNames<Mat>> {
    let path = run_manifest::resolve(dir, rel);
    Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))
}

impl Activity {
    #[must_use]
    pub fn new(manifest: RunManifest, dir: PathBuf) -> Self {
        Self {
            manifest,
            dir,
            expected: None,
            observed: None,
        }
    }

    fn load_expected(&self) -> anyhow::Result<Expected> {
        let m = &self.manifest;
        let o = &m.outputs;
        if let (Some(z_rel), Ok((rho_path, _))) = (
            o.cell_embedding.as_deref(),
            run_manifest::resolve_feature_embedding_for(m, &self.dir),
        ) {
            let z = read_table(&self.dir, z_rel)?;
            let rho = Mat::from_parquet_with_row_names(&rho_path, Some(0))?;
            anyhow::ensure!(
                z.mat.ncols() == rho.mat.ncols(),
                "cell embedding has {} dims, feature embedding {}",
                z.mat.ncols(),
                rho.mat.ncols()
            );
            return Ok(Expected {
                features: Axis::new(rho.rows),
                cells: z.rows,
                model: Model::Embedding {
                    z: z.mat,
                    rho: rho.mat,
                },
            });
        }
        if let (true, Some(theta_rel), Some(beta_rel)) = (
            m.kind.latent_is_log_simplex(),
            o.latent.as_deref(),
            o.gene_dictionary(),
        ) {
            let theta = read_table(&self.dir, theta_rel)?;
            let beta = read_table(&self.dir, beta_rel)?;
            anyhow::ensure!(
                theta.mat.ncols() == beta.mat.ncols(),
                "latent has {} topics, dictionary {}",
                theta.mat.ncols(),
                beta.mat.ncols()
            );
            return Ok(Expected {
                features: Axis::new(beta.rows),
                cells: theta.rows,
                model: Model::Topic {
                    log_theta: theta.mat,
                    log_beta: beta.mat,
                },
            });
        }
        anyhow::bail!("this run has no model tables to predict a feature from")
    }

    fn load_observed(&self) -> anyhow::Result<Observed> {
        let m = &self.manifest;
        anyhow::ensure!(!m.data.input.is_empty(), "the manifest lists no data files");
        let files: Vec<Box<str>> = m
            .data
            .input
            .iter()
            .map(|p| run_manifest::resolve(&self.dir, p).to_string_lossy().into())
            .collect();
        let reload =
            senna::multiome_layout::recorded_layout(m.data.multiome.as_ref(), files.len())?;
        info!(
            "view: opening {} data file(s) for observed counts",
            files.len()
        );
        let SparseDataWithBatch { data, .. } =
            read_data_on_shared_rows(reload.apply(ReadSharedRowsArgs {
                data_files: files,
                keep_empty_barcodes: true,
                ..Default::default()
            })?)?;
        Ok(Observed {
            features: Axis::new(data.row_names()?),
            cells: data.column_names()?,
            data,
        })
    }

    fn expected(&mut self) -> Result<&Expected, String> {
        if self.expected.is_none() {
            self.expected = Some(self.load_expected().map_err(|e| e.to_string()));
        }
        self.expected
            .as_ref()
            .expect("just set")
            .as_ref()
            .map_err(Clone::clone)
    }

    fn observed(&mut self) -> Result<&Observed, String> {
        if self.observed.is_none() {
            self.observed = Some(self.load_observed().map_err(|e| e.to_string()));
        }
        self.observed
            .as_ref()
            .expect("just set")
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Feature names on `source`'s axis, for search and completion.
    pub fn feature_names(&mut self, source: Source) -> Result<&[Box<str>], String> {
        Ok(match source {
            Source::Expected => &self.expected()?.features.names,
            Source::Observed => &self.observed()?.features.names,
        })
    }

    /// Raw values of `feature` per source cell, with the cell names they
    /// belong to and the name the axis spells the feature with.
    fn raw(&mut self, feature: &str, source: Source) -> Result<Raw<'_>, String> {
        match source {
            Source::Expected => {
                let e = self.expected()?;
                let g = e
                    .features
                    .index
                    .match_gene(feature)
                    .ok_or_else(|| format!("no feature {feature} in the model"))?;
                let values = match &e.model {
                    Model::Embedding { z, rho } => {
                        (z * rho.row(g).transpose()).iter().copied().collect()
                    }
                    Model::Topic {
                        log_theta,
                        log_beta,
                    } => {
                        let b = log_beta.row(g);
                        log_theta
                            .row_iter()
                            .map(|t| {
                                let s: Vec<f32> =
                                    t.iter().zip(b.iter()).map(|(a, b)| a + b).collect();
                                let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                m + s.iter().map(|v| (v - m).exp()).sum::<f32>().ln()
                            })
                            .collect()
                    }
                };
                Ok((values, &e.cells, e.features.names[g].clone()))
            }
            Source::Observed => {
                let o = self.observed()?;
                let g = o
                    .features
                    .index
                    .match_gene(feature)
                    .ok_or_else(|| format!("no feature {feature} in the data files"))?;
                let row = o
                    .data
                    .read_rows_csr(std::iter::once(g))
                    .map_err(|e| e.to_string())?;
                let mut values = vec![0f32; o.cells.len()];
                for (&c, &v) in row.col_indices().iter().zip(row.values()) {
                    values[c] = v.ln_1p();
                }
                Ok((values, &o.cells, o.features.names[g].clone()))
            }
        }
    }

    /// `feature` on each of `points` (by name).
    pub fn levels(
        &mut self,
        feature: &str,
        source: Source,
        point_index: &HashMap<Box<str>, usize>,
        n_points: usize,
    ) -> Result<(Levels, Box<str>), String> {
        let (raw, cells, spelled) = self.raw(feature, source)?;
        let mut values = vec![f32::NAN; n_points];
        for (c, name) in cells.iter().enumerate() {
            if let Some(&i) = point_index.get(name) {
                values[i] = raw[c];
            }
        }
        Ok((
            Levels::from_values(values, source == Source::Observed),
            spelled,
        ))
    }

    /// Mean ramp position over a feature set: each feature is first placed on
    /// its own ramp, so one highly expressed feature cannot carry the set.
    /// Returns the levels and how many of the features were found.
    pub fn set_levels(
        &mut self,
        features: &[Box<str>],
        source: Source,
        point_index: &HashMap<Box<str>, usize>,
        n_points: usize,
    ) -> Result<(Levels, usize), String> {
        let mut sum = vec![0f32; n_points];
        let mut seen = vec![false; n_points];
        let mut used = 0usize;
        for f in features {
            let Ok((levels, _)) = self.levels(f, source, point_index, n_points) else {
                continue;
            };
            used += 1;
            for i in 0..n_points {
                if levels.values[i].is_finite() {
                    seen[i] = true;
                    sum[i] += levels.t(i).max(0.0);
                }
            }
        }
        if used == 0 {
            return Err(match source {
                Source::Expected => "none of these features are in the model".into(),
                Source::Observed => "none of these features are in the data files".into(),
            });
        }
        let values = sum
            .iter()
            .zip(&seen)
            .map(|(&s, &ok)| if ok { s / used as f32 } else { f32::NAN })
            .collect();
        Ok((
            Levels::from_values(values, source == Source::Observed),
            used,
        ))
    }
}
