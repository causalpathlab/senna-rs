//! What `senna view` browses for: marker panels for `lupin annotate`, runs
//! to open, or a run's data file that is not where it was recorded. The
//! browser itself is [`crate::tui::browse`].
//!
//! A marker panel shows how many of its genes the run has; the panel that
//! covers the most is starred and chosen to begin with. A run shows its kind
//! and layouts; the newest is chosen to begin with. A data file of the
//! recorded name is chosen to begin with.

use super::*;
use crate::tui::browse::{is_data, Browser, Header, Outcome, Wanted};
use data_beans::utilities::name_matching::GeneIndex;
use senna::run_manifest::RunManifest;
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Files larger than this are not read as marker panels.
const PANEL_MAX_BYTES: u64 = 8 << 20;

/// Extensions that are never marker panels. With the size cap and the rules
/// in [`read_panel`], the same test `lupin annotate --tui` applies.
const NOT_PANELS: &[&str] = &[
    "parquet", "zarr", "zip", "h5", "h5ad", "json", "bam", "bai", "png", "pdf", "log",
];

/// Marker panels, counted against the run's genes when known, and whether
/// lupin also tests GO terms on the clusters (tab turns it on and off).
pub(super) struct Panels {
    pub genes: Option<GeneIndex>,
    pub go: bool,
}

/// A marker panel's size, and how much of it this run can use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Panel {
    pub types: usize,
    pub genes: usize,
    /// Genes this run has, when its feature names are known.
    pub found: Option<usize>,
}

impl Wanted for Panels {
    type About = Panel;

    fn header(&self) -> Header {
        Header {
            title: "Marker panel for lupin annotate".into(),
            notes: vec![if self.go {
                "with GO terms: lupin tests them on each cluster too (tab: without)".into()
            } else {
                "without GO terms (tab: test them on each cluster too)".into()
            }],
            what: "marker panels (gene, cell type per line)",
            star: Some(" ★ covers the most of this run's genes"),
            verb: "annotate",
        }
    }

    fn file(&self, path: &Path, _name: &str) -> Option<Panel> {
        read_panel(path, self.genes.as_ref())
    }

    /// Tab turns GO terms on and off.
    fn key(&mut self, k: &KeyEvent) -> bool {
        let tab = k.code == KeyCode::Tab;
        self.go ^= tab;
        tab
    }

    /// `12 types · 340 genes · 301 in this run (89%)`.
    fn describe<'a>(&self, p: &'a Panel) -> Cow<'a, str> {
        let mut s = format!(
            "{} type{} · {} genes",
            p.types,
            if p.types == 1 { "" } else { "s" },
            p.genes
        );
        if let Some(found) = p.found {
            let pct = 100 * found / p.genes.max(1);
            s.push_str(&format!(" · {found} in this run ({pct}%)"));
        }
        Cow::Owned(s)
    }

    /// Annotation needs types to choose between.
    fn refuse(&self, name: &str, p: &Panel) -> Option<String> {
        (p.types < 2).then(|| format!("{name} names one cell type; lupin needs several"))
    }

    /// The panel with the most genes in this run; with no feature names,
    /// the one with the most genes. Single-type files do not count:
    /// annotation needs types to choose between.
    fn best(&self, _dir: &Path, files: &[(&str, &Panel)]) -> Option<String> {
        files
            .iter()
            .filter(|(_, p)| p.types > 1)
            .map(|(n, p)| (p.found.unwrap_or(p.genes), *n))
            .max_by(|(a, x), (b, y)| a.cmp(b).then_with(|| y.len().cmp(&x.len())))
            .filter(|(score, _)| *score > 0)
            .map(|(_, n)| n.to_string())
    }
}

/// Run manifests (`*.senna.json`).
pub(super) struct Runs;

impl Wanted for Runs {
    /// `bge · umap, phate · annotated`.
    type About = String;

    fn header(&self) -> Header {
        Header {
            title: "Run to view".into(),
            notes: Vec::new(),
            what: "runs (*.senna.json)",
            star: None,
            verb: "open",
        }
    }

    fn file(&self, path: &Path, name: &str) -> Option<String> {
        name.ends_with(".senna.json").then(|| describe_run(path))
    }

    fn describe<'a>(&self, about: &'a String) -> Cow<'a, str> {
        Cow::Borrowed(about)
    }

    /// The most recently written run here.
    fn best(&self, dir: &Path, files: &[(&str, &String)]) -> Option<String> {
        files
            .iter()
            .max_by_key(|(n, _)| files::modified(&dir.join(n)))
            .map(|(n, _)| n.to_string())
    }
}

/// A data file recorded at this path, which is not there.
pub(super) struct Missing(pub String);

impl Wanted for Missing {
    type About = ();

    fn header(&self) -> Header {
        Header {
            title: format!("Where is {}?", files::name(Path::new(&self.0))),
            notes: vec![
                format!(
                    "the run's data is recorded at {}, which is not here;",
                    self.0
                ),
                "the file chosen is written into the run's manifest".into(),
                "looking from the folder senna view runs in (~ goes home)".into(),
            ],
            what: "data files (.zarr, .zarr.zip, .h5)",
            star: Some(" ★ the recorded name"),
            verb: "use it",
        }
    }

    fn file(&self, _path: &Path, name: &str) -> Option<()> {
        is_data(name).then_some(())
    }

    fn store(&self, _path: &Path, _name: &str) -> Option<()> {
        Some(())
    }

    fn describe<'a>(&self, (): &'a ()) -> Cow<'a, str> {
        Cow::Borrowed("")
    }

    fn best(&self, _dir: &Path, files: &[(&str, &())]) -> Option<String> {
        let name = files::name(Path::new(&self.0));
        files.iter().any(|(n, _)| *n == name).then_some(name)
    }
}

/// `bge · umap, phate · annotated`, or why the manifest does not read.
fn describe_run(path: &Path) -> String {
    match RunManifest::load(path) {
        Ok((m, _)) => {
            let mut parts = vec![m.kind.to_string()];
            let layouts: Vec<&str> = m.layout.methods.keys().map(String::as_str).collect();
            parts.push(if layouts.is_empty() {
                "no layout yet".into()
            } else {
                layouts.join(", ")
            });
            if m.annotate.argmax.is_some() {
                parts.push("annotated".into());
            }
            parts.join(" · ")
        }
        Err(_) => "does not read as a run".into(),
    }
}

/// `path` as a marker panel: `gene<TAB>type` or `gene,type` lines, read as
/// lupin reads them. `None` for a file that is not one.
fn read_panel(path: &Path, genes: Option<&GeneIndex>) -> Option<Panel> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if NOT_PANELS.contains(&ext.as_str()) {
        return None;
    }
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > PANEL_MAX_BYTES {
        return None;
    }
    let lines = legume_numeric::matrix::common_io::read_lines(&path.to_string_lossy()).ok()?;
    let mut types = std::collections::HashSet::new();
    let mut seen = std::collections::HashSet::new();
    let mut found = 0;
    let mut numeric = true;
    for line in &lines {
        let line = line.trim();
        let Some((gene, label)) = line.split_once('\t').or_else(|| line.split_once(',')) else {
            continue;
        };
        let (gene, label) = (
            gene.trim(),
            label.split('\t').next().unwrap_or(label).trim(),
        );
        if gene.is_empty()
            || label.is_empty()
            || gene.starts_with('#')
            || matches!(gene.to_lowercase().as_str(), "gene" | "symbol")
        {
            continue;
        }
        numeric &= label.parse::<f64>().is_ok();
        // One type however its words are separated, as lupin's `label_key`.
        types.insert(
            label
                .split(|c: char| c.is_whitespace() || c == ',' || c == '_')
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join("_"),
        );
        if seen.insert(gene.to_string()) && genes.is_some_and(|g| g.match_gene(gene).is_some()) {
            found += 1;
        }
    }
    // A table of numbers is not a panel, nor, when the run's genes are
    // known, a file that names none of them (a log read as `word,word`).
    let useless = genes.is_some() && found == 0;
    (!seen.is_empty() && !numeric && !useless).then_some(Panel {
        types: types.len(),
        genes: seen.len(),
        found: genes.map(|_| found),
    })
}

/// Browse for a run before the view opens: the browser alone, full screen.
/// `None` when cancelled.
pub fn pick_run() -> anyhow::Result<Option<PathBuf>> {
    let here = std::env::current_dir()?;
    let mut b = Browser::open(here, Runs, None);
    let level = log::max_level();
    log::set_max_level(log::LevelFilter::Off);
    let mut terminal = ratatui::init();
    let picked = (|| loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Block::default().style(page()), area);
            let lines = b.lines(usize::from(area.height).saturating_sub(2));
            popup(f, area, lines, 110, At::Middle, color::TEXT);
        })?;
        if let Event::Key(k) = event::read()? {
            if k.kind == KeyEventKind::Release {
                continue;
            }
            if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(None);
            }
            match b.key(k) {
                Outcome::Cancelled => return Ok(None),
                Outcome::Chosen(c) => return Ok(Some(c.file())),
                Outcome::Moved | Outcome::Ignored => {}
            }
        }
    })();
    ratatui::restore();
    log::set_max_level(level);
    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::browse::{list_dir, Entry};

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn panels_are_counted_against_the_run_and_the_widest_is_best() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        write(
            d,
            "small.tsv",
            "gene\tcelltype\nGENE1\tCT1 a\nGENE2\tCT1_a\nGENE3\tCT2\nNOPE\tCT2\n",
        );
        write(
            d,
            "wide.csv",
            "GENE1,CT1 a\nGENE2,CT1 a\nGENE3,CT2 b\nGENE4,CT3\n",
        );
        write(d, "one.tsv", "GENE1\tCT1\nGENE2\tCT1\nGENE4\tCT1\n");
        write(d, "table.tsv", "a\t1\nb\t2\nc\t3\n");
        write(d, "run.log", "GENE1\tCT1\nGENE4\tCT2\n");
        write(d, "other.tsv", "Gene1_x\tCT1\nGene4_x\tCT2\n");
        std::fs::create_dir(d.join("data.zarr")).unwrap();
        std::fs::create_dir(d.join("sub")).unwrap();
        std::fs::create_dir(d.join(".hidden")).unwrap();

        let names: Vec<Box<str>> = ["GENE1", "GENE2", "GENE3", "GENE4"].map(Box::from).to_vec();
        let want = Panels {
            genes: Some(GeneIndex::build(&names)),
            go: false,
        };
        let entries = list_dir(d, &want);
        let listed: Vec<&str> = entries.iter().map(Entry::name).collect();
        assert_eq!(
            listed,
            ["..", ".hidden", "sub", "one.tsv", "small.tsv", "wide.csv"]
        );
        assert_eq!(
            entries[4],
            Entry::File(
                "small.tsv".into(),
                Panel {
                    types: 2,
                    genes: 4,
                    found: Some(3)
                }
            )
        );
        // `one.tsv` finds as many genes but names a single type.
        let b = Browser::open(d.to_path_buf(), want, None);
        assert_eq!(b.current().map(Entry::name), Some("wide.csv"));
    }

    #[test]
    fn typing_narrows_and_going_up_returns_to_the_folder() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::create_dir(d.join("panels")).unwrap();
        write(&d.join("panels"), "set.tsv", "GENE1\tCT1\nGENE3\tCT2\n");
        write(d, "other.tsv", "GENE1\tCT1\nGENE3\tCT2\n");

        let mut b = Browser::open(
            d.to_path_buf(),
            Panels {
                genes: None,
                go: false,
            },
            None,
        );
        b.filter.push_str("pan");
        let names: Vec<&str> = b.shown().iter().map(|e| e.name()).collect();
        assert_eq!(names, ["..", "panels"]);

        b.dir.push("panels");
        b.read(None);
        assert_eq!(b.current().map(Entry::name), Some("set.tsv"));
        b.go_up();
        assert_eq!(b.current().map(Entry::name), Some("panels"));
    }

    #[test]
    fn runs_are_listed_with_their_kind_and_the_newest_is_chosen() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let old = RunManifest::new(senna::run_manifest::RunKind::Svd, "a");
        old.save(&d.join("a.senna.json")).unwrap();
        let new = RunManifest::new(senna::run_manifest::RunKind::Svd, "b");
        new.save(&d.join("b.senna.json")).unwrap();
        // Set the times outright: two writes can share one mtime.
        let at = |secs| std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs);
        for (name, secs) in [("a.senna.json", 1_000), ("b.senna.json", 2_000)] {
            let f = std::fs::File::options()
                .write(true)
                .open(d.join(name))
                .unwrap();
            f.set_modified(at(secs)).unwrap();
        }
        write(d, "c.pinto.json", "{}");
        write(d, "markers.tsv", "GENE1\tCT1\nGENE3\tCT2\n");

        let b = Browser::open(d.to_path_buf(), Runs, None);
        let names: Vec<&str> = b.shown().iter().map(|e| e.name()).collect();
        assert_eq!(names, ["..", "a.senna.json", "b.senna.json"]);
        assert_eq!(b.current().map(Entry::name), Some("b.senna.json"));
        let Entry::File(_, about) = b.shown()[1] else {
            panic!("a run")
        };
        assert!(about.starts_with("svd · no layout yet"), "{about}");
    }

    #[test]
    fn data_files_and_stores_are_offered_and_the_recorded_name_is_chosen() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::create_dir(d.join("rna.zarr")).unwrap();
        std::fs::create_dir(d.join("sub")).unwrap();
        for f in [
            "a.zarr.zip",
            "b.h5",
            "c.h5ad",
            "notes.txt",
            "run.senna.json",
        ] {
            std::fs::write(d.join(f), b"").unwrap();
        }
        let b = Browser::open(
            d.to_path_buf(),
            Missing("/elsewhere/a.zarr.zip".into()),
            None,
        );
        let names: Vec<&str> = b.shown().iter().map(|e| e.name()).collect();
        assert_eq!(names, ["..", "sub", "a.zarr.zip", "b.h5", "rna.zarr"]);
        assert_eq!(b.current().map(Entry::name), Some("a.zarr.zip"));
    }
}
