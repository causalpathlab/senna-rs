//! Labels moved by hand: kept beside the run in `{prefix}.view_labels.json`
//! (legume-tui's `OffsetBook`), by layout and round, and by label name, so
//! a re-clustering that renumbers groups keeps each move on its label.
//!
//! A move is in label line heights (em) from the label's anchor, so it
//! holds at any zoom, text size or page. A file that is there but cannot be
//! read is refused and never written over: moves made then last the
//! session only.

use super::*;
use legume_tui::map::labels::Placed;
use legume_tui::map::offsets::{OffsetBook, Offsets};
use std::collections::HashMap;
use std::path::PathBuf;

/// Every layout's moved labels: the file as read, and this session's moves
/// ahead of it.
#[derive(Clone, Debug)]
pub struct Places {
    path: PathBuf,
    book: Option<OffsetBook>,
    moved: HashMap<String, Offsets<String>>,
    /// Why the file is not written: it is there and could not be read.
    pub refused: Option<String>,
}

impl Places {
    #[must_use]
    pub fn path(prefix: &str) -> PathBuf {
        PathBuf::from(format!("{prefix}.view_labels.json"))
    }

    /// The moves kept beside the run at `prefix`; none when there is no
    /// file, and refused (kept for the session only) when it cannot be
    /// read.
    #[must_use]
    pub fn load(prefix: &str) -> Self {
        let path = Self::path(prefix);
        let (book, refused) = match OffsetBook::load(&path) {
            Ok(b) => (Some(b), None),
            Err(e) => (None, Some(unreadable(&path, &e))),
        };
        Self {
            path,
            book,
            moved: HashMap::new(),
            refused,
        }
    }

    /// Layout `id`'s moved labels, by name.
    #[must_use]
    pub fn layout(&self, id: &str) -> Offsets<String> {
        match (self.moved.get(id), &self.book) {
            (Some(o), _) => o.clone(),
            (None, Some(b)) => b.layout(id),
            (None, None) => Offsets::default(),
        }
    }

    /// Layout `id`'s moves are now `o`, for this session; [`Self::keep`]
    /// writes them.
    pub fn set(&mut self, id: &str, o: Offsets<String>) {
        self.moved.insert(id.to_string(), o);
    }

    /// Write layout `id`'s moves to the file. It is read again first, so
    /// the layouts another view of the run kept since stay.
    ///
    /// # Errors
    /// When the file was refused, or cannot be read or written now.
    pub fn keep(&mut self, id: &str) -> Result<(), String> {
        if let Some(why) = &self.refused {
            return Err(why.clone());
        }
        let o = self.layout(id);
        let mut book = OffsetBook::load(&self.path).map_err(|e| {
            let why = unreadable(&self.path, &e);
            self.refused = Some(why.clone());
            why
        })?;
        book.set_layout(id, &o);
        book.save()
            .map_err(|e| format!("could not write {}: {e}", self.path.display()))?;
        self.book = Some(book);
        Ok(())
    }
}

/// Why the file at `path` is left alone.
fn unreadable(path: &std::path::Path, e: &std::io::Error) -> String {
    format!(
        "{} could not be read ({e}), so it is left as it is",
        files::name(path)
    )
}

impl Scene {
    /// The map labels are moved on: its kind and method (a zoom's path
    /// included) in this round, which has its own prefix.
    #[must_use]
    pub fn layout_id(&self) -> String {
        let s = self.current();
        let round = files::name(std::path::Path::new(&self.data.prefix));
        format!("{}/{}@{round}", s.kind.title(), s.method)
    }

    /// The moved group labels of the map on screen, by group id: only a
    /// cell map's group labels move.
    #[must_use]
    pub fn group_offsets(&self) -> Offsets<u32> {
        if self.current().axis() != Axis::Cells || self.colour.is_none() {
            return Offsets::default();
        }
        let o = self.places.layout(&self.layout_id());
        if o.is_empty() {
            return Offsets::default();
        }
        self.levels()
            .iter()
            .enumerate()
            .filter_map(|(g, name)| Some((g as u32, o.get(&name.to_string())?)))
            .collect()
    }

    /// Group labels on screen are now at `o` (by group id), for this
    /// session; the moves of names not in this grouping stay.
    pub fn set_group_offsets(&mut self, o: &Offsets<u32>) {
        let id = self.layout_id();
        let mut by_name = self.places.layout(&id);
        for (g, name) in self.levels().iter().enumerate() {
            let name = name.to_string();
            by_name.reset(&name);
            if let Some(d) = o.get(&(g as u32)) {
                by_name.nudge(name, d);
            }
        }
        self.places.set(&id, by_name);
    }

    /// Move group `g`'s label by `d_em` line heights, for this session.
    pub fn move_label(&mut self, g: u32, d_em: (f32, f32)) {
        let Some(name) = self.levels().get(g as usize).map(ToString::to_string) else {
            return;
        };
        let id = self.layout_id();
        let mut by_name = self.places.layout(&id);
        by_name.nudge(name, d_em);
        self.places.set(&id, by_name);
    }

    /// Write the moves on this map beside the run; a refusal is noted.
    pub fn keep_label_places(&mut self) {
        if let Err(why) = self.places.keep(&self.layout_id()) {
            self.note = Some(format!("label moves last this session only: {why}"));
        }
    }

    /// A group label's line height in frame pixels, for text `cell_px`
    /// from the terminal: what one em of a move is.
    #[must_use]
    pub fn label_em(&self, cell_px: f32) -> f32 {
        self.label_font(self.text_px(cell_px)).line_height() as f32
    }

    /// Where the keyed labels of the last frame were placed.
    #[must_use]
    pub fn placed_labels(&self) -> Vec<Placed<u32>> {
        self.label_placed.borrow().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test scene, its run at `dir/r`.
    fn scene_in(dir: &std::path::Path) -> Scene {
        let mut s = crate::view::tests::scene();
        s.data.prefix = dir.join("r").to_string_lossy().into_owned();
        s.places = Places::load(&s.data.prefix);
        s
    }

    #[test]
    fn moves_are_kept_by_name_for_their_layout_and_round() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = scene_in(dir.path());
        s.move_label(1, (1.5, -0.5));
        assert_eq!(s.group_offsets().get(&1), Some((1.5, -0.5)));
        s.keep_label_places();
        assert!(s.note.is_none(), "{:?}", s.note);
        let back = Places::load(&s.data.prefix);
        let name = s.levels()[1].to_string();
        assert_eq!(back.layout(&s.layout_id()).get(&name), Some((1.5, -0.5)));
        // Another method's map, or another round, has its own.
        let umap = s.layout_id();
        s.set_space(2);
        assert_ne!(s.layout_id(), umap);
        assert!(s.group_offsets().is_empty());
        let other = scene_in(&dir.path().join("next"));
        assert!(other.group_offsets().is_empty());
        // A moved label is drawn where it was put.
        let mut s = scene_in(dir.path());
        let mut plain = scene_in(dir.path());
        plain.places = Places::load(&dir.path().join("none").to_string_lossy());
        let vp = render::fit(s.current().points.bounds, 200, 200);
        let at = |s: &Scene| {
            deck::frames(&[(s, vp, 16.0)], false);
            s.label_rect(1).unwrap()
        };
        let (moved, there) = (at(&s), at(&plain));
        let em = s.label_em(16.0);
        assert!(
            (moved[0] - there[0] - 1.5 * em).abs() <= 1.0,
            "{moved:?} {there:?}"
        );
        assert!(s.placed_labels().iter().any(|p| p.key == 1 && p.moved));
        s.move_label(1, (-1.5, 0.5));
        assert!(s.group_offsets().is_empty(), "back on its anchor");
    }

    #[test]
    fn a_file_that_cannot_be_read_is_never_written_over() {
        let dir = tempfile::tempdir().unwrap();
        let path = Places::path(&dir.path().join("r").to_string_lossy());
        std::fs::write(&path, "{ not a label file").unwrap();
        let mut s = scene_in(dir.path());
        assert!(s.places.refused.is_some());
        // Moves still show this session.
        s.move_label(1, (1.0, 0.0));
        assert_eq!(s.group_offsets().get(&1), Some((1.0, 0.0)));
        s.keep_label_places();
        let note = s.note.take().unwrap();
        assert!(note.contains("view_labels.json"), "{note}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{ not a label file"
        );
    }

    #[test]
    fn a_file_another_view_wrote_since_keeps_its_layouts() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = scene_in(dir.path());
        let mut b = scene_in(dir.path());
        b.set_space(2);
        b.move_label(1, (0.0, 2.0));
        b.keep_label_places();
        a.move_label(1, (1.0, 0.0));
        a.keep_label_places();
        let back = Places::load(&a.data.prefix);
        let name = a.levels()[1].to_string();
        assert_eq!(back.layout(&b.layout_id()).get(&name), Some((0.0, 2.0)));
        assert_eq!(back.layout(&a.layout_id()).get(&name), Some((1.0, 0.0)));
    }
}
