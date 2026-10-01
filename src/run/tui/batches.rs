//! What each data file's batch is, and the label files that say so.
//!
//! senna's own rule, with no `--batch-files`, puts every file's cells in a
//! batch named after the file, or after the `@batch` tag its barcodes
//! carry. A different name, two files made one batch, or labels renamed
//! all need a label file per data file, one label per cell: a run writes
//! the ones it needs under `{out}.batches/`, beside its script, so the
//! script stays a complete record.

use super::data::Pair;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// How a data file's cells get their batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Batch {
    /// senna's own rule: the file's name, or its barcodes' `@batch` tags.
    Own,
    /// Every cell in one batch of this name.
    Named(String),
    /// A label file, one label per cell.
    Labels {
        file: PathBuf,
        /// Each label and its cells, in the order first seen.
        counts: Vec<(String, usize)>,
        /// Labels given another name.
        renamed: BTreeMap<String, String>,
    },
}

impl Batch {
    /// `file` as this file's labels, counted.
    pub fn labels(file: PathBuf) -> anyhow::Result<Self> {
        let counts = count(&read(&file)?);
        Ok(Batch::Labels {
            file,
            counts,
            renamed: BTreeMap::new(),
        })
    }
}

/// Each distinct label in `labels` and how often it comes, in the order
/// first seen.
#[must_use]
pub fn count(labels: &[String]) -> Vec<(String, usize)> {
    let mut at: BTreeMap<&str, usize> = BTreeMap::new();
    let mut out: Vec<(String, usize)> = Vec::new();
    for l in labels {
        match at.get(l.as_str()) {
            Some(&i) => out[i].1 += 1,
            None => {
                at.insert(l, out.len());
                out.push((l.clone(), 1));
            }
        }
    }
    out
}

fn read(file: &Path) -> anyhow::Result<Vec<String>> {
    let lines = legume_numeric::matrix::common_io::read_lines(&file.to_string_lossy())?;
    Ok(lines.into_iter().map(String::from).collect())
}

/// The batch senna names a file's cells after: its base name.
#[must_use]
pub fn own_name(data: &Path) -> String {
    legume_numeric::matrix::common_io::basename(&data.to_string_lossy())
        .map_or_else(|_| super::data::stem(data), String::from)
}

/// The batches `pair` puts its cells in, each once, with their cells
/// where known.
#[must_use]
pub fn of(pair: &Pair) -> Vec<(String, Option<usize>)> {
    let named: Vec<(String, Option<usize>)> = match &pair.batch {
        Batch::Own => match &pair.tags {
            Some(tags) => count(tags).into_iter().map(|(l, n)| (l, Some(n))).collect(),
            None => vec![(own_name(&pair.data), pair.cells)],
        },
        Batch::Named(name) => vec![(name.clone(), pair.cells)],
        Batch::Labels {
            counts, renamed, ..
        } => counts
            .iter()
            .map(|(l, n)| (renamed.get(l).unwrap_or(l).clone(), Some(*n)))
            .collect(),
    };
    // Labels renamed alike are one batch.
    let mut out: Vec<(String, Option<usize>)> = Vec::new();
    for (name, cells) in named {
        match out.iter_mut().find(|(n, _)| *n == name) {
            Some(b) => b.1 = b.1.zip(cells).map(|(a, c)| a + c),
            None => out.push((name, cells)),
        }
    }
    out
}

/// Every batch across `pairs`: its name, the files with cells in it, and
/// its cells (`None` while a file is still being read).
#[must_use]
pub fn summary(pairs: &[Pair]) -> Vec<(String, usize, Option<usize>)> {
    let mut out: Vec<(String, usize, Option<usize>)> = Vec::new();
    for p in pairs {
        for (name, cells) in of(p) {
            match out.iter_mut().find(|(n, _, _)| *n == name) {
                Some(b) => {
                    b.1 += 1;
                    b.2 = b.2.zip(cells).map(|(a, c)| a + c);
                }
                None => out.push((name, 1, cells)),
            }
        }
    }
    out
}

/// A label file a run writes before it starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Written {
    pub path: PathBuf,
    pub content: Content,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Content {
    /// The same label for every cell.
    Repeat { label: String, cells: usize },
    /// One label per cell, as given.
    Each(Vec<String>),
    /// A label file's labels, some renamed.
    Renamed {
        from: PathBuf,
        renamed: BTreeMap<String, String>,
    },
}

/// The label files a run in `dir` with prefix `out` passes, one per data
/// file, and those it has to write first. None at all when every file
/// keeps senna's own rule.
pub fn files(
    pairs: &[Pair],
    dir: &Path,
    out: &str,
) -> Result<(Vec<PathBuf>, Vec<Written>), String> {
    if pairs.iter().all(|p| p.batch == Batch::Own) {
        return Ok((Vec::new(), Vec::new()));
    }
    let folder = dir.join(format!("{out}.batches"));
    let mut taken: Vec<String> = Vec::new();
    let mut passed = Vec::new();
    let mut written = Vec::new();
    for p in pairs {
        let cells = || {
            p.cells.ok_or_else(|| {
                format!(
                    "{}: its cells are not counted yet",
                    crate::tui::name(&p.data)
                )
            })
        };
        let content = match &p.batch {
            Batch::Labels { file, renamed, .. } if renamed.is_empty() => {
                passed.push(file.clone());
                continue;
            }
            Batch::Labels { file, renamed, .. } => Content::Renamed {
                from: file.clone(),
                renamed: renamed.clone(),
            },
            Batch::Named(name) => Content::Repeat {
                label: name.clone(),
                cells: cells()?,
            },
            Batch::Own => match &p.tags {
                Some(tags) => Content::Each(tags.clone()),
                None => Content::Repeat {
                    label: own_name(&p.data),
                    cells: cells()?,
                },
            },
        };
        // One file per data file, named for it, never two alike.
        let stem = super::data::stem(&p.data);
        let mut name = format!("{stem}.txt");
        let mut k = 1;
        while taken.contains(&name) {
            k += 1;
            name = format!("{stem}-{k}.txt");
        }
        taken.push(name.clone());
        let path = folder.join(name);
        passed.push(path.clone());
        written.push(Written { path, content });
    }
    Ok((passed, written))
}

/// Write `w`, never over an existing file.
pub fn write(w: &Written) -> anyhow::Result<()> {
    if let Some(dir) = w.path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&w.path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", w.path.display()))?;
    let mut f = std::io::BufWriter::new(f);
    match &w.content {
        Content::Repeat { label, cells } => {
            for _ in 0..*cells {
                writeln!(f, "{label}")?;
            }
        }
        Content::Each(labels) => {
            for l in labels {
                writeln!(f, "{l}")?;
            }
        }
        Content::Renamed { from, renamed } => {
            for l in read(from)? {
                writeln!(f, "{}", renamed.get(&l).unwrap_or(&l))?;
            }
        }
    }
    f.flush()?;
    Ok(())
}

#[cfg(test)]
#[path = "tests/batches.rs"]
mod tests;
