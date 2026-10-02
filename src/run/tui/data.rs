//! The data files a session embeds and the batch file of each.

use super::batches::Batch;
use crate::tui::browse::{is_data, size_of, Header, Wanted};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// What `senna run` browses for: count backends to embed, or batch label
/// files, several at once either way; or one file for a flag: an earlier
/// run's feature table, or any file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Data,
    Batch,
    Table,
    File,
}

impl Wanted for Pick {
    /// The file's size.
    type About = String;

    fn header(&self) -> Header {
        let (title, what) = match self {
            Pick::Data => ("Data to embed", "backends (.zarr, .zarr.zip, .h5)"),
            Pick::Batch => (
                "Batch labels, one file per data file",
                "label files (.txt, .tsv, .csv, gzipped too)",
            ),
            Pick::Table => (
                "An earlier run's feature table",
                "feature tables (.feature_embedding / .dictionary .parquet, .senna.json)",
            ),
            Pick::File => ("A file for this flag", "files"),
        };
        Header {
            title: title.into(),
            notes: Vec::new(),
            what,
            star: None,
            verb: "take",
        }
    }

    fn file(&self, path: &Path, name: &str) -> Option<String> {
        let wanted = match self {
            Pick::Data => is_data(name),
            Pick::Batch => is_batch(name),
            // A run's feature table, or its manifest.
            Pick::Table => {
                name.ends_with(".senna.json")
                    || senna::run_manifest::RHO_TABLE_SUFFIXES
                        .iter()
                        .any(|e| name.ends_with(e))
            }
            Pick::File => true,
        };
        wanted.then(|| size_of(path))
    }

    fn store(&self, _path: &Path, _name: &str) -> Option<String> {
        matches!(self, Pick::Data | Pick::File).then(String::new)
    }

    fn describe<'a>(&self, size: &'a String) -> Cow<'a, str> {
        Cow::Borrowed(size)
    }

    fn many(&self) -> bool {
        matches!(self, Pick::Data | Pick::Batch)
    }

    /// The newest feature table here, as the likely one: a run's ρ table or
    /// manifest before a `.dictionary.parquet`, which a topic run writes as
    /// a log-simplex β the flags refuse.
    fn best(&self, dir: &Path, files: &[(&str, &String)]) -> Option<String> {
        if *self != Pick::Table {
            return None;
        }
        let modified = |n: &str| {
            std::fs::metadata(dir.join(n))
                .and_then(|m| m.modified())
                .ok()
        };
        files
            .iter()
            .max_by_key(|(n, _)| (!n.ends_with(".dictionary.parquet"), modified(n)))
            .map(|(n, _)| n.to_string())
    }
}

/// Endings of batch label files: plain or gzipped text.
const BATCH_ENDINGS: &[&str] = &[".txt", ".tsv", ".csv", ".txt.gz", ".tsv.gz", ".csv.gz"];

fn is_batch(name: &str) -> bool {
    BATCH_ENDINGS.iter().any(|e| name.ends_with(e))
}

/// A data file and the batch of its cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    pub data: PathBuf,
    pub batch: Batch,
    /// Features × cells, or why the file does not open.
    pub info: String,
    /// Its cells, once read.
    pub cells: Option<usize>,
    /// The `@batch` tag of each cell, when its barcodes carry one: senna's
    /// own batches for the file then.
    pub tags: Option<Vec<String>>,
    /// Whether its rows read as gene counts for `gem`
    /// (`{gene}/count/{spliced|unspliced}`); `None` when unknown.
    pub gene_counts: Option<bool>,
}

impl Pair {
    /// A data file not yet described; [`describe`] fills that in.
    pub fn pending(data: PathBuf) -> Self {
        Pair {
            data,
            batch: Batch::Own,
            info: "reading…".into(),
            cells: None,
            tags: None,
            gene_counts: None,
        }
    }

    /// Take in what [`describe`] found.
    pub fn described(&mut self, d: Described) {
        self.info = d.info;
        self.cells = d.cells;
        self.tags = d.tags;
        self.gene_counts = d.gene_counts;
    }
}

/// What reading a data file found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Described {
    /// `2000 features × 5000 cells`, or why the file does not open.
    pub info: String,
    pub cells: Option<usize>,
    pub tags: Option<Vec<String>>,
    pub gene_counts: Option<bool>,
}

/// Its size, its barcodes' `@batch` tags, and whether the rows are gem's
/// gene counts; or why the file does not open.
pub fn describe(path: &Path) -> Described {
    use data_beans::sparse_io::open_sparse_matrix_by_path;
    let m = match open_sparse_matrix_by_path(&path.to_string_lossy()) {
        Ok(m) => m,
        Err(e) => {
            return Described {
                info: format!("does not open: {e}"),
                ..Described::default()
            }
        }
    };
    let cells = m.num_columns();
    let info = match (m.num_rows(), cells) {
        (Some(r), Some(c)) => format!("{r} features × {c} cells"),
        _ => "opens".to_string(),
    };
    // senna reads a batch from the barcodes when the first one carries
    // `@batch`, the tag after the last `@`.
    let tags = m
        .column_names()
        .ok()
        .filter(|c| c.first().is_some_and(|n| n.contains('@')));
    let tags = tags.map(|c| {
        c.iter()
            .map(|n| n.rsplit('@').next().unwrap_or(n).to_string())
            .collect()
    });
    // gem's own reading of its row grammar decides.
    let gene_counts = m
        .row_names()
        .ok()
        .map(|names| crate::gem::tracks::assign_tracks(&names).is_ok());
    Described {
        info,
        cells,
        tags,
        gene_counts,
    }
}

/// A file name without the endings data and label files carry.
#[must_use]
pub fn stem(path: &Path) -> String {
    let name = crate::tui::name(path);
    let mut name = data_beans::hdf5_io::strip_backend_suffix(&name).to_string();
    while let Some(end) = BATCH_ENDINGS
        .iter()
        .filter(|e| name.ends_with(*e))
        .max_by_key(|e| e.len())
    {
        name.truncate(name.len() - end.len());
    }
    name
}

/// Words that say a file holds labels, not which sample it is.
const LABEL_WORDS: &[&str] = &["batch", "batches", "label", "labels", "membership"];

/// A name reduced to what identifies its sample: lower case, label words
/// and separators dropped. `s1_batch` and `S1` both give `s1`.
fn key(name: &str) -> String {
    name.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty() && !LABEL_WORDS.contains(w))
        .collect()
}

/// Whether a batch file's name says it belongs to a data file's: the same
/// key. A longer one that extends it at a word boundary counts too
/// (`s1` and `s1_cells_batch`), never one whose number runs on (`s1`
/// and `s10`).
fn same_sample(data: &Path, batch: &Path) -> bool {
    let (d, b) = (key(&stem(data)), key(&stem(batch)));
    if d.is_empty() || b.is_empty() {
        return false;
    }
    if d == b {
        return true;
    }
    let (short, long) = if d.len() < b.len() {
        (&d, &b)
    } else {
        (&b, &d)
    };
    let runs_on = |a: Option<char>, b: Option<char>| {
        a.zip(b)
            .is_some_and(|(a, b)| a.is_ascii_digit() == b.is_ascii_digit())
    };
    long.starts_with(short.as_str())
        && !runs_on(short.chars().last(), long[short.len()..].chars().next())
}

/// How [`assign`] paired the files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Paired {
    /// Each data file found the one batch file named for it.
    ByName(usize),
    /// No name matched; as many batch files as data files went in order.
    InOrder,
    /// Neither: what could be matched by name was, the rest is left unset.
    Partly(usize),
}

/// For each data file, the one of `batches` named for its sample. When no
/// name matches at all and the counts agree, files go in the order listed,
/// and the caller says so. A data file that two batch files fit, or that
/// shares its one match with another, gets none.
#[must_use]
pub fn assign(data: &[PathBuf], batches: &[PathBuf]) -> (Vec<Option<PathBuf>>, Paired) {
    let fits: Vec<Vec<usize>> = data
        .iter()
        .map(|d| {
            (0..batches.len())
                .filter(|&j| same_sample(d, &batches[j]))
                .collect()
        })
        .collect();
    let shared = |j: usize| fits.iter().filter(|g| g.contains(&j)).count() > 1;
    let chosen: Vec<Option<PathBuf>> = fits
        .iter()
        .map(|f| match f[..] {
            [j] if !shared(j) => Some(batches[j].clone()),
            _ => None,
        })
        .collect();
    let matched = chosen.iter().flatten().count();
    if matched == data.len() {
        (chosen, Paired::ByName(matched))
    } else if fits.iter().all(Vec::is_empty) && batches.len() == data.len() {
        (batches.iter().cloned().map(Some).collect(), Paired::InOrder)
    } else {
        (chosen, Paired::Partly(matched))
    }
}

/// Label files in `dir` that say they hold labels: a label-file ending
/// and a label word in the name. Names are checked before the file
/// system is asked anything.
#[must_use]
pub fn label_files_in(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_lowercase();
            is_batch(&n)
                && LABEL_WORDS.iter().any(|w| n.contains(w))
                && e.file_type().is_ok_and(|t| !t.is_dir())
        })
        .map(|e| e.path())
        .collect();
    found.sort();
    found
}

/// The one of `labels` named for `data`'s sample; none unless exactly one
/// fits.
#[must_use]
pub fn beside(data: &Path, labels: &[PathBuf]) -> Option<PathBuf> {
    match labels
        .iter()
        .filter(|l| same_sample(data, l))
        .collect::<Vec<_>>()[..]
    {
        [one] => Some(one.clone()),
        _ => None,
    }
}

#[cfg(test)]
#[path = "tests/data.rs"]
mod tests;
