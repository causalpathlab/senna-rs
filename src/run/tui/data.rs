//! The data files a session embeds and the batch file of each.

use crate::tui::browse::{is_data, size_of, Header, Wanted};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// What `senna run` browses for: count backends to embed, or batch label
/// files; several at once either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Data,
    Batch,
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
        };
        wanted.then(|| size_of(path))
    }

    fn store(&self, _path: &Path, _name: &str) -> Option<String> {
        (*self == Pick::Data).then(String::new)
    }

    fn describe<'a>(&self, size: &'a String) -> Cow<'a, str> {
        Cow::Borrowed(size)
    }

    fn many(&self) -> bool {
        true
    }
}

/// Endings of batch label files: plain or gzipped text.
const BATCH_ENDINGS: &[&str] = &[".txt", ".tsv", ".csv", ".txt.gz", ".tsv.gz", ".csv.gz"];

fn is_batch(name: &str) -> bool {
    BATCH_ENDINGS.iter().any(|e| name.ends_with(e))
}

/// A data file and its batch labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    pub data: PathBuf,
    pub batch: Option<PathBuf>,
    /// Features × cells, or why the file does not open.
    pub info: String,
    /// Whether its rows read as gene counts for `gem`
    /// (`{gene}/count/{spliced|unspliced}`); `None` when unknown.
    pub gene_counts: Option<bool>,
}

impl Pair {
    /// A data file not yet described; [`describe`] fills that in.
    pub fn pending(data: PathBuf) -> Self {
        Pair {
            data,
            batch: None,
            info: "reading…".into(),
            gene_counts: None,
        }
    }
}

/// `2000 features × 5000 cells` and whether the rows are gem's gene
/// counts, or why the file does not open.
pub fn describe(path: &Path) -> (String, Option<bool>) {
    use data_beans::sparse_io::open_sparse_matrix_by_path;
    let opened = open_sparse_matrix_by_path(&path.to_string_lossy());
    match opened {
        Ok(m) => {
            let dims = match (m.num_rows(), m.num_columns()) {
                (Some(r), Some(c)) => format!("{r} features × {c} cells"),
                _ => "opens".to_string(),
            };
            // gem's own reading of its row grammar decides.
            let gene = m
                .row_names()
                .ok()
                .map(|names| crate::gem::tracks::assign_tracks(&names).is_ok());
            (dims, gene)
        }
        Err(e) => (format!("does not open: {e}"), None),
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

/// Give each pair the one batch file named for its sample. When no name
/// matches at all and the counts agree, files go in the order listed,
/// and the caller says so. A data file that two batch files fit, or that
/// shares its one match with another, is left without one.
pub fn assign(pairs: &mut [Pair], batches: &[PathBuf]) -> Paired {
    let fits: Vec<Vec<usize>> = pairs
        .iter()
        .map(|p| {
            (0..batches.len())
                .filter(|&j| same_sample(&p.data, &batches[j]))
                .collect()
        })
        .collect();
    let mut matched = 0;
    for (i, f) in fits.iter().enumerate() {
        let shared = |j: usize| fits.iter().filter(|g| g.contains(&j)).count() > 1;
        if let [j] = f[..] {
            if !shared(j) {
                pairs[i].batch = Some(batches[j].clone());
                matched += 1;
            }
        }
    }
    if matched == pairs.len() {
        Paired::ByName(matched)
    } else if fits.iter().all(Vec::is_empty) && batches.len() == pairs.len() {
        for (p, b) in pairs.iter_mut().zip(batches) {
            p.batch = Some(b.clone());
        }
        Paired::InOrder
    } else {
        Paired::Partly(matched)
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

/// Why the batch files as given cannot be passed: senna wants one per
/// data file, or none.
#[must_use]
pub fn batch_problem(pairs: &[Pair]) -> Option<String> {
    let with = pairs.iter().filter(|p| p.batch.is_some()).count();
    (with > 0 && with < pairs.len()).then(|| {
        format!(
            "{with} of {} data files have batch labels; give each one, or clear them all",
            pairs.len()
        )
    })
}

#[cfg(test)]
#[path = "tests/data.rs"]
mod tests;
