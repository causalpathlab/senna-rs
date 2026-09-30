//! A file browser, one directory at a time: folders, and the files that are
//! what is being looked for — marker panels for `lupin annotate`, or runs
//! for the view to open.
//!
//! A marker panel shows how many of its genes the run has; the panel that
//! covers the most is starred and chosen to begin with. A run shows its kind
//! and layouts; the newest is chosen to begin with.

use super::draw::{hint, selected};
use super::*;
use data_beans::utilities::name_matching::GeneIndex;
use senna::run_manifest::RunManifest;
use std::path::{Path, PathBuf};

/// Files larger than this are not read as marker panels.
const PANEL_MAX_BYTES: u64 = 8 << 20;

/// Extensions that are never marker panels. With the size cap and the rules
/// in [`read_panel`], the same test `lupin annotate --tui` applies.
const NOT_PANELS: &[&str] = &[
    "parquet", "zarr", "zip", "h5", "h5ad", "json", "bam", "bai", "png", "pdf", "log",
];

/// What the browser looks for.
pub(super) enum Want {
    /// Marker panels, counted against the run's genes when known.
    Panels(Option<GeneIndex>),
    /// Run manifests (`*.senna.json`).
    Runs,
}

/// A marker panel's size, and how much of it this run can use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Panel {
    pub types: usize,
    pub genes: usize,
    /// Genes this run has, when its feature names are known.
    pub found: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Entry {
    Up,
    Dir(String),
    Panel(String, Panel),
    /// A run: its name and a one-line description.
    Run(String, String),
}

impl Entry {
    fn name(&self) -> &str {
        match self {
            Entry::Up => "..",
            Entry::Dir(n) | Entry::Panel(n, _) | Entry::Run(n, _) => n,
        }
    }

    fn is_file(&self) -> bool {
        matches!(self, Entry::Panel(..) | Entry::Run(..))
    }
}

/// What a key did in the browser.
pub(super) enum Outcome {
    /// Not a browser key.
    Ignored,
    Moved,
    Cancelled,
    /// A file was chosen.
    Chosen(PathBuf, Entry),
}

pub(super) struct Browser {
    pub dir: PathBuf,
    want: Want,
    entries: Vec<Entry>,
    /// Typed letters narrow the list to names containing them.
    pub filter: String,
    /// Position among the shown entries.
    pub row: usize,
    /// The file chosen to begin with: the best panel, or the newest run.
    best: Option<String>,
}

impl Browser {
    pub(super) fn open(dir: PathBuf, want: Want, select: Option<&str>) -> Self {
        let mut b = Browser {
            dir,
            want,
            entries: Vec::new(),
            filter: String::new(),
            row: 0,
            best: None,
        };
        b.read(select);
        b
    }

    /// Read the current directory; the cursor goes to `select` if it is
    /// listed, else to the best file, else to the first entry after `..`.
    fn read(&mut self, select: Option<&str>) {
        self.entries = list_dir(&self.dir, &self.want);
        self.best = match self.want {
            Want::Panels(_) => best_panel(&self.entries),
            Want::Runs => newest(&self.dir, &self.entries),
        };
        self.filter.clear();
        let want = select.map(str::to_string).or_else(|| self.best.clone());
        self.row = want
            .and_then(|w| self.shown().iter().position(|e| e.name() == w))
            .unwrap_or(usize::from(self.shown().len() > 1));
    }

    /// The entries the filter lets through, `..` always first. Hidden ones
    /// only when the filter starts with `.`.
    pub fn shown(&self) -> Vec<&Entry> {
        let f = self.filter.to_lowercase();
        let hidden = f.starts_with('.');
        self.entries
            .iter()
            .filter(|e| {
                matches!(e, Entry::Up)
                    || ((hidden || !e.name().starts_with('.'))
                        && e.name().to_lowercase().contains(&f))
            })
            .collect()
    }

    fn go_up(&mut self) {
        let from = files::name(&self.dir);
        if let Some(parent) = self.dir.parent().map(Path::to_path_buf) {
            self.dir = parent;
            self.read(Some(&from));
        }
    }

    /// A key: move, open a folder, narrow, cancel, or choose a file.
    pub fn key(&mut self, k: KeyEvent) -> Outcome {
        let n = self.shown().len();
        let last = n.saturating_sub(1);
        match k.code {
            KeyCode::Esc => return Outcome::Cancelled,
            KeyCode::Up => self.row = self.row.saturating_sub(1),
            KeyCode::Down => self.row = (self.row + 1).min(last),
            KeyCode::PageUp => self.row = self.row.saturating_sub(10),
            KeyCode::PageDown => self.row = (self.row + 10).min(last),
            KeyCode::Home => self.row = 0,
            KeyCode::End => self.row = last,
            KeyCode::Left => self.go_up(),
            KeyCode::Backspace => {
                if self.filter.pop().is_none() {
                    self.go_up();
                } else {
                    self.row = usize::from(self.shown().len() > 1);
                }
            }
            KeyCode::Char('~') if self.filter.is_empty() => {
                if let Ok(home) = std::env::var("HOME") {
                    self.dir = PathBuf::from(home);
                    self.read(None);
                }
            }
            KeyCode::Enter | KeyCode::Right => {
                let Some(entry) = self.shown().get(self.row).map(|e| (*e).clone()) else {
                    return Outcome::Ignored;
                };
                match entry {
                    Entry::Up => self.go_up(),
                    Entry::Dir(name) => {
                        self.dir.push(name);
                        self.read(None);
                    }
                    e if k.code == KeyCode::Enter => {
                        return Outcome::Chosen(self.dir.join(e.name()), e);
                    }
                    _ => return Outcome::Ignored,
                }
            }
            KeyCode::Char(c) => {
                let on = self.shown().get(self.row).map(|e| e.name().to_string());
                self.filter.push(c);
                let shown = self.shown();
                // Stay where the cursor was if it still shows, else on the
                // best file, else on the first entry after `..`.
                let at = |name: &str| shown.iter().position(|e| e.name() == name);
                self.row = on
                    .as_deref()
                    .filter(|n| *n != "..")
                    .and_then(at)
                    .or_else(|| self.best.as_deref().and_then(at))
                    .unwrap_or(usize::from(shown.len() > 1));
            }
            _ => return Outcome::Ignored,
        }
        Outcome::Moved
    }

    /// The lines of the popup, `rows` entries tall at most.
    pub fn lines(&self, rows: usize) -> Vec<Line<'static>> {
        let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
        let (title, what, star, none) = match self.want {
            Want::Panels(_) => (
                "Marker panel for lupin annotate",
                "marker panels (gene, cell type per line)",
                Some(" ★ covers the most of this run's genes"),
                "annotate",
            ),
            Want::Runs => ("Run to view", "runs (*.senna.json)", None, "open"),
        };
        let mut out = vec![
            Line::from(Span::styled(format!(" {title}"), bold)),
            Line::from(Span::styled(format!(" {}", shown(&self.dir)), hint())),
        ];
        if !self.filter.is_empty() {
            out.push(Line::from(format!(" names with “{}”", self.filter)));
        }
        out.push(Line::from(""));
        let shown = self.shown();
        let first = self
            .row
            .saturating_sub(rows / 2)
            .min(shown.len().saturating_sub(rows));
        let width = shown
            .iter()
            .map(|e| e.name().chars().count() + 1)
            .max()
            .unwrap_or(0)
            .min(36);
        for (i, e) in shown.iter().enumerate().skip(first).take(rows) {
            let mark = if star.is_some() && self.best.as_deref() == Some(e.name()) {
                "★"
            } else {
                " "
            };
            let text = match e {
                Entry::Up => " ../".to_string(),
                Entry::Dir(n) => format!(" {n}/"),
                Entry::Panel(n, p) => format!("{mark}{n:<width$}  {}", describe(p)),
                Entry::Run(n, d) => format!(" {n:<width$}  {d}"),
            };
            let style = if i == self.row {
                selected()
            } else if e.is_file() {
                Style::default()
            } else {
                hint()
            };
            out.push(Line::from(Span::styled(text, style)));
        }
        if !shown.iter().any(|e| e.is_file()) {
            out.push(Line::from(Span::styled(
                format!("  no {what} here"),
                hint(),
            )));
        }
        out.push(Line::from(""));
        if let (Some(s), Some(_)) = (star, &self.best) {
            out.push(Line::from(Span::styled(s, hint())));
        }
        out.push(Line::from(Span::styled(
            format!(
                " ↑ ↓ choose   enter open / {none}   ← or backspace up   type to narrow   ~ home   esc cancel"
            ),
            hint(),
        )));
        out
    }
}

/// `12 types · 340 genes · 301 in this run (89%)`.
fn describe(p: &Panel) -> String {
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
    s
}

/// Folders (not hidden, not a `.zarr` store) and the wanted files, `..`
/// first, folders before files, each group by name.
fn list_dir(dir: &Path, want: &Want) -> Vec<Entry> {
    let mut dirs = Vec::new();
    let mut found = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        // Hidden entries are kept; `shown` lists them only on request.
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        if path.is_dir() {
            if !name.ends_with(".zarr") {
                dirs.push(Entry::Dir(name));
            }
            continue;
        }
        let entry = match want {
            Want::Panels(genes) => read_panel(&path, genes.as_ref()).map(|p| Entry::Panel(name, p)),
            Want::Runs => name
                .ends_with(".senna.json")
                .then(|| Entry::Run(name, describe_run(&path))),
        };
        found.extend(entry);
    }
    dirs.sort_by(|a, b| a.name().cmp(b.name()));
    found.sort_by(|a, b| a.name().cmp(b.name()));
    let mut out = vec![Entry::Up];
    out.extend(dirs);
    out.extend(found);
    out
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

/// The panel with the most genes in this run; with no feature names, the
/// one with the most genes. Single-type files do not count: annotation
/// needs types to choose between.
fn best_panel(entries: &[Entry]) -> Option<String> {
    entries
        .iter()
        .filter_map(|e| match e {
            Entry::Panel(n, p) if p.types > 1 => Some((p.found.unwrap_or(p.genes), n)),
            _ => None,
        })
        .max_by(|(a, x), (b, y)| a.cmp(b).then_with(|| y.len().cmp(&x.len())))
        .filter(|(score, _)| *score > 0)
        .map(|(_, n)| n.clone())
}

/// The most recently written run here.
fn newest(dir: &Path, entries: &[Entry]) -> Option<String> {
    entries
        .iter()
        .filter(|e| matches!(e, Entry::Run(..)))
        .max_by_key(|e| files::modified(&dir.join(e.name())))
        .map(|e| e.name().to_string())
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

/// A path as shown: relative to the working directory when it is under it.
pub fn shown(p: &Path) -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| p.strip_prefix(cwd).ok().map(Path::to_path_buf))
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Browse for a run before the view opens: the browser alone, full screen.
/// `None` when cancelled.
pub fn pick_run() -> anyhow::Result<Option<PathBuf>> {
    let here = std::env::current_dir()?;
    let mut b = Browser::open(here, Want::Runs, None);
    let level = log::max_level();
    log::set_max_level(log::LevelFilter::Off);
    let mut terminal = ratatui::init();
    let picked = (|| loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Block::default().style(draw::page()), area);
            let rows = usize::from(area.height).saturating_sub(10).max(3);
            draw::popup(f, area, b.lines(rows), 110, draw::At::Middle, color::TEXT);
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
                Outcome::Chosen(path, _) => return Ok(Some(path)),
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
            "gene\tcelltype\nCD3E\tT cell\nCD8A\tT_cell\nMS4A1\tB\nNOPE\tB\n",
        );
        write(
            d,
            "wide.csv",
            "CD3E,T cell\nCD8A,T cell\nMS4A1,B cell\nLYZ,Mono\n",
        );
        write(d, "one.tsv", "CD3E\tT\nCD8A\tT\nLYZ\tT\n");
        write(d, "table.tsv", "a\t1\nb\t2\nc\t3\n");
        write(d, "run.log", "CD3E\tT\nLYZ\tB\n");
        write(d, "mouse.tsv", "Cd3e_x\tT\nLyz2_x\tB\n");
        std::fs::create_dir(d.join("data.zarr")).unwrap();
        std::fs::create_dir(d.join("sub")).unwrap();
        std::fs::create_dir(d.join(".hidden")).unwrap();

        let names: Vec<Box<str>> = ["CD3E", "CD8A", "MS4A1", "LYZ"].map(Box::from).to_vec();
        let genes = GeneIndex::build(&names);
        let want = Want::Panels(Some(genes));
        let entries = list_dir(d, &want);
        let listed: Vec<&str> = entries.iter().map(Entry::name).collect();
        assert_eq!(
            listed,
            ["..", ".hidden", "sub", "one.tsv", "small.tsv", "wide.csv"]
        );
        assert_eq!(
            entries[4],
            Entry::Panel(
                "small.tsv".into(),
                Panel {
                    types: 2,
                    genes: 4,
                    found: Some(3)
                }
            )
        );
        // `one.tsv` finds as many genes but names a single type.
        assert_eq!(best_panel(&entries).as_deref(), Some("wide.csv"));

        let mut b = Browser::open(d.to_path_buf(), want, None);
        assert_eq!(b.shown()[b.row].name(), "wide.csv");
        // Hidden entries show only when asked for with a leading `.`.
        assert!(b.shown().iter().all(|e| e.name() != ".hidden"));
        b.filter.push('.');
        assert!(b.shown().iter().any(|e| e.name() == ".hidden"));
    }

    #[test]
    fn typing_narrows_and_going_up_returns_to_the_folder() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::create_dir(d.join("panels")).unwrap();
        write(&d.join("panels"), "immune.tsv", "CD3E\tT\nMS4A1\tB\n");
        write(d, "other.tsv", "CD3E\tT\nMS4A1\tB\n");

        let mut b = Browser::open(d.to_path_buf(), Want::Panels(None), None);
        b.filter.push_str("pan");
        let names: Vec<&str> = b.shown().iter().map(|e| e.name()).collect();
        assert_eq!(names, ["..", "panels"]);

        b.dir.push("panels");
        b.read(None);
        assert_eq!(b.shown()[b.row].name(), "immune.tsv");
        b.go_up();
        assert_eq!(b.shown()[b.row].name(), "panels");
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
        write(d, "markers.tsv", "CD3E\tT\nMS4A1\tB\n");

        let b = Browser::open(d.to_path_buf(), Want::Runs, None);
        let names: Vec<&str> = b.shown().iter().map(|e| e.name()).collect();
        assert_eq!(names, ["..", "a.senna.json", "b.senna.json"]);
        assert_eq!(b.shown()[b.row].name(), "b.senna.json");
        let Entry::Run(_, about) = b.shown()[1] else {
            panic!("a run")
        };
        assert!(about.starts_with("svd · no layout yet"), "{about}");
    }
}
