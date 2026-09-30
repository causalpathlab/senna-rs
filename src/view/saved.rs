//! What the viewer has saved: a log of each PDF and a small thumbnail of it,
//! kept in `.senna-view/` in the directory the viewer runs from, so the list
//! lasts from one session to the next. The runs' own directories are not
//! touched.

use super::files;
use image::RgbaImage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Width of a thumbnail on disk, in pixels.
pub(crate) const THUMB_WIDTH: u32 = 240;
/// Saves remembered; older ones go, with their thumbnails.
const KEEP: usize = 200;

/// One saved file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Entry {
    /// The file, as an absolute path.
    pub path: PathBuf,
    /// What was saved, as the save dialog put it.
    pub what: String,
    /// When, in seconds since the Unix epoch.
    pub when: u64,
    /// The thumbnail's file name in `thumbs/`.
    thumb: String,
}

impl Entry {
    pub fn name(&self) -> String {
        files::name(&self.path)
    }
}

/// Every save remembered in one directory, newest first.
pub(crate) struct Gallery {
    dir: PathBuf,
    entries: Vec<Entry>,
}

impl Gallery {
    /// The saves logged in `dir`, less those whose file or thumbnail is
    /// gone; thumbnails nothing refers to any more are deleted. An
    /// unreadable log starts an empty gallery.
    pub fn open(dir: &Path) -> Self {
        let thumbs = dir.join("thumbs");
        let logged: Vec<Entry> = files::read_json(&dir.join("saved.json")).unwrap_or_default();
        let entries: Vec<Entry> = logged
            .into_iter()
            .filter(|e| e.path.exists() && thumbs.join(&e.thumb).exists())
            .collect();
        if let Ok(listing) = std::fs::read_dir(&thumbs) {
            let kept: std::collections::HashSet<&str> =
                entries.iter().map(|e| e.thumb.as_str()).collect();
            for f in listing.filter_map(Result::ok) {
                if !kept.contains(f.file_name().to_string_lossy().as_ref()) {
                    let _ = std::fs::remove_file(f.path());
                }
            }
        }
        Self {
            dir: dir.into(),
            entries,
        }
    }

    /// The directory the viewer keeps its saves in: `.senna-view/` here.
    pub fn here() -> Self {
        Self::open(Path::new(".senna-view"))
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn thumb(&self, e: &Entry) -> PathBuf {
        self.dir.join("thumbs").join(&e.thumb)
    }

    /// Remember that `path` was just saved, showing `picture`: at the top,
    /// replacing an earlier save of the same file.
    pub fn add(&mut self, path: &Path, what: &str, picture: &RgbaImage) -> anyhow::Result<()> {
        let path = path.canonicalize()?;
        // Another viewer in this directory may have saved since: build on
        // the log as it is on disk, not on what this one read at start.
        self.entries = Self::open(&self.dir).entries;
        if let Some(i) = self.entries.iter().position(|e| e.path == path) {
            let old = self.entries.remove(i);
            let _ = std::fs::remove_file(self.thumb(&old));
        }
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let thumbs = self.dir.join("thumbs");
        std::fs::create_dir_all(&thumbs)?;
        let thumb = format!("{}.png", now.as_nanos());
        thumbnail(picture).save(thumbs.join(&thumb))?;
        self.entries.insert(
            0,
            Entry {
                path,
                what: what.into(),
                when: now.as_secs(),
                thumb,
            },
        );
        for old in self.entries.split_off(self.entries.len().min(KEEP)) {
            let _ = std::fs::remove_file(self.thumb(&old));
        }
        std::fs::write(
            self.dir.join("saved.json"),
            serde_json::to_string_pretty(&self.entries)?,
        )?;
        Ok(())
    }
}

/// `picture` shrunk to `THUMB_WIDTH` wide at its aspect.
fn thumbnail(picture: &RgbaImage) -> RgbaImage {
    let (w, h) = picture.dimensions();
    let th = (THUMB_WIDTH as f32 * h as f32 / w.max(1) as f32)
        .round()
        .max(1.0) as u32;
    image::imageops::resize(
        picture,
        THUMB_WIDTH,
        th,
        image::imageops::FilterType::Triangle,
    )
}

/// How long ago `when` was, at `now` (both seconds since the epoch).
pub(crate) fn ago(when: u64, now: u64) -> String {
    match now.saturating_sub(when) {
        s if s < 60 => "just now".into(),
        s if s < 3600 => format!("{} min ago", s / 60),
        s if s < 86_400 => format!("{} h ago", s / 3600),
        s => format!("{} d ago", s / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 90, 255])
        })
    }

    /// A saved file in `dir` with some bytes in it.
    fn file(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"%PDF").unwrap();
        p
    }

    #[test]
    fn saves_are_listed_newest_first_and_kept_for_next_time() {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join(".senna-view");
        let (a, b) = (file(work.path(), "a.pdf"), file(work.path(), "b.pdf"));
        let mut g = Gallery::open(&store);
        assert!(g.entries().is_empty());
        g.add(&a, "this view · 7 in · 300 dpi", &picture(800, 600))
            .unwrap();
        g.add(&b, "all 2 runs, one grid", &picture(1200, 400))
            .unwrap();
        let names = |g: &Gallery| -> Vec<String> { g.entries().iter().map(|e| e.name()).collect() };
        assert_eq!(names(&g), ["b.pdf", "a.pdf"]);
        // Each has a small thumbnail on disk, at the picture's aspect.
        let thumb = image::open(g.thumb(&g.entries()[1])).unwrap();
        assert_eq!(thumb.width(), THUMB_WIDTH);
        assert_eq!(thumb.height(), THUMB_WIDTH * 3 / 4);
        // A new viewer in the same directory sees the same list.
        let again = Gallery::open(&store);
        assert_eq!(names(&again), ["b.pdf", "a.pdf"]);
        assert_eq!(again.entries()[0].what, "all 2 runs, one grid");
    }

    #[test]
    fn saving_a_file_again_moves_it_to_the_top_without_a_copy() {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join(".senna-view");
        let (a, b) = (file(work.path(), "a.pdf"), file(work.path(), "b.pdf"));
        let mut g = Gallery::open(&store);
        g.add(&a, "first", &picture(100, 100)).unwrap();
        g.add(&b, "second", &picture(100, 100)).unwrap();
        g.add(&a, "first, redone", &picture(100, 100)).unwrap();
        let listed: Vec<(String, &str)> = g
            .entries()
            .iter()
            .map(|e| (e.name(), e.what.as_str()))
            .collect();
        assert_eq!(
            listed,
            [
                ("a.pdf".into(), "first, redone"),
                ("b.pdf".into(), "second")
            ]
        );
        // One thumbnail per entry: the replaced one is gone.
        let thumbs = std::fs::read_dir(store.join("thumbs")).unwrap().count();
        assert_eq!(thumbs, 2);
    }

    #[test]
    fn a_saved_file_deleted_since_is_dropped_on_open() {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join(".senna-view");
        let (a, b) = (file(work.path(), "a.pdf"), file(work.path(), "b.pdf"));
        let mut g = Gallery::open(&store);
        g.add(&a, "a", &picture(100, 100)).unwrap();
        g.add(&b, "b", &picture(100, 100)).unwrap();
        std::fs::remove_file(&a).unwrap();
        let again = Gallery::open(&store);
        let names: Vec<String> = again.entries().iter().map(|e| e.name()).collect();
        assert_eq!(names, ["b.pdf"]);
    }

    #[test]
    fn an_unreadable_log_starts_an_empty_gallery() {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join(".senna-view");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(store.join("saved.json"), b"not json").unwrap();
        assert!(Gallery::open(&store).entries().is_empty());
    }

    #[test]
    fn times_read_as_how_long_ago() {
        let now = 1_000_000;
        assert_eq!(ago(now - 20, now), "just now");
        assert_eq!(ago(now - 300, now), "5 min ago");
        assert_eq!(ago(now - 2 * 3600 - 100, now), "2 h ago");
        assert_eq!(ago(now - 3 * 86_400, now), "3 d ago");
        // A clock that went back is not "in the future".
        assert_eq!(ago(now + 50, now), "just now");
    }

    #[test]
    fn a_deleted_files_thumbnail_goes_with_it() {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join(".senna-view");
        let (a, b) = (file(work.path(), "a.pdf"), file(work.path(), "b.pdf"));
        let mut g = Gallery::open(&store);
        g.add(&a, "a", &picture(100, 100)).unwrap();
        g.add(&b, "b", &picture(100, 100)).unwrap();
        std::fs::remove_file(&a).unwrap();
        let again = Gallery::open(&store);
        let thumbs: Vec<_> = std::fs::read_dir(store.join("thumbs")).unwrap().collect();
        assert_eq!(thumbs.len(), 1);
        assert!(again.thumb(&again.entries()[0]).exists());
    }

    #[test]
    fn two_viewers_in_one_directory_keep_each_others_saves() {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join(".senna-view");
        let (a, b) = (file(work.path(), "a.pdf"), file(work.path(), "b.pdf"));
        let mut first = Gallery::open(&store);
        let mut second = Gallery::open(&store);
        first.add(&a, "from the first", &picture(100, 100)).unwrap();
        second
            .add(&b, "from the second", &picture(100, 100))
            .unwrap();
        let names: Vec<String> = Gallery::open(&store)
            .entries()
            .iter()
            .map(|e| e.name())
            .collect();
        assert_eq!(names, ["b.pdf", "a.pdf"]);
    }
}
