//! The scene's space, grouping, focus and round, and moving between them.

use super::*;

impl Scene {
    /// Read the cells' batches on a worker thread, unless the run has them;
    /// [`Self::poll_batch`] adds them when they come.
    pub fn start_batch(&mut self) {
        let has = self.data.labels.iter().any(|l| l.kind == LabelKind::Batch);
        let Some((m, dir)) = self.data.run.clone().filter(|_| !has) else {
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(data::batch_labels(&m, &dir));
        });
        self.batch = Some(rx);
    }

    /// Add the cells' batches once read. Returns whether they were added.
    pub fn poll_batch(&mut self) -> bool {
        let Some(rx) = &self.batch else {
            return false;
        };
        match rx.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.batch = None;
                false
            }
            Ok(read) => {
                self.batch = None;
                match read {
                    Ok(Some(labels)) => {
                        self.data.labels.push(labels);
                        true
                    }
                    Ok(None) => false,
                    Err(e) => {
                        log::warn!("view: skipping batch: {e}");
                        false
                    }
                }
            }
        }
    }

    pub(super) fn label_index(&self, kind: LabelKind) -> Option<usize> {
        self.data.labels.iter().position(|l| l.kind == kind)
    }

    /// Colour by grouping `li`, unfocused. A feature's activity would hide
    /// the groups, so it is cleared; returns whether there was one.
    pub(super) fn set_colour(&mut self, li: Option<usize>) -> bool {
        let cleared = self.clear_pick();
        self.colour = li;
        self.focus = None;
        self.refresh_groups();
        cleared
    }

    pub fn current(&self) -> &data::Space {
        &self.data.spaces[self.space]
    }

    /// Groupings that apply to the current axis.
    pub fn colour_choices(&self) -> Vec<usize> {
        let axis = self.current().axis();
        (0..self.data.labels.len())
            .filter(|&i| self.data.labels[i].axis() == axis)
            .collect()
    }

    pub(super) fn default_colour(&self, want: Option<&str>) -> Option<usize> {
        let choices = self.colour_choices();
        if let Some(w) = want {
            if let Some(&i) = choices
                .iter()
                .find(|&&i| self.data.labels[i].title().eq_ignore_ascii_case(w))
            {
                return Some(i);
            }
            log::warn!("view: no `{w}` grouping for {}", self.current().title());
        }
        choices.first().copied()
    }

    pub(super) fn refresh_groups(&mut self) {
        let Some(li) = self.colour else {
            self.groups = None;
            self.styles.clear();
            return;
        };
        let labels = &self.data.labels[li];
        self.styles = self.book.resolve(labels.title(), &labels.levels);
        if matches!(&self.groups, Some((s, l, ..)) if *s == self.space && *l == li) {
            return;
        }
        let g = labels.align(&self.current().points);
        let sizes = data::group_counts(g.iter().copied(), labels.levels.len());
        self.groups = Some((self.space, li, g, sizes));
    }

    /// Change group `g`'s style in the current grouping and save the book.
    pub fn restyle(&mut self, g: usize, f: impl FnOnce(&mut style::Style)) {
        let Some(li) = self.colour else { return };
        let labels = &self.data.labels[li];
        let (grouping, name) = (labels.title().to_string(), labels.levels[g].to_string());
        let mut st = self.book.get(&grouping, &name);
        f(&mut st);
        self.book.set(&grouping, &name, st);
        self.refresh_groups();
    }

    /// Write the style book beside the run (when the style menu closes).
    pub fn save_styles(&mut self) {
        if let Err(e) = self.book.save(&self.data.prefix) {
            self.note = Some(format!("could not save styles: {e}"));
        }
    }

    pub fn style_of(&self, g: usize) -> style::Style {
        self.colour.map_or_else(style::Style::plain, |li| {
            let l = &self.data.labels[li];
            self.book.get(l.title(), &l.levels[g])
        })
    }

    pub fn resolved(&self, g: usize) -> Option<&style::Resolved> {
        self.styles.get(g)
    }

    pub fn groups(&self) -> Option<&[u32]> {
        self.groups.as_ref().map(|(_, _, g, _)| g.as_slice())
    }

    /// Points per group of the grouping on screen.
    pub fn group_sizes(&self) -> Option<&[usize]> {
        self.groups.as_ref().map(|(.., n)| n.as_slice())
    }

    pub fn levels(&self) -> &[Box<str>] {
        self.colour
            .map_or(&[], |i| self.data.labels[i].levels.as_slice())
    }

    pub fn set_space(&mut self, space: usize) {
        let axis = self.current().axis();
        self.space = space;
        if self.current().axis() != axis {
            self.colour = self.default_colour(None);
            self.focus = None;
        }
        self.refresh_groups();
        self.refresh_activity();
    }

    /// The next view `step` places along (cells, features on cells, features)
    /// of the layout method on screen, zoomed layouts left out; when that
    /// method has one view only, of every method. `None` with nowhere to go.
    pub fn next_view(&self, step: isize) -> Option<usize> {
        // A zoomed layout steps from the map it was zoomed from.
        let root = self.root();
        let method = &self.data.spaces[root].method;
        let tops = |same: bool| -> Vec<usize> {
            let spaces = self.data.spaces.iter().enumerate();
            spaces
                .filter(|(_, s)| s.parent.is_none() && (!same || s.method == *method))
                .map(|(i, _)| i)
                .collect()
        };
        let views = Some(tops(true))
            .filter(|v| v.len() > 1)
            .unwrap_or_else(|| tops(false));
        let at = views.iter().position(|&i| i == root)? as isize;
        (views.len() > 1).then(|| views[(at + step).rem_euclid(views.len() as isize) as usize])
    }

    /// Swap in a reloaded or different round, keeping layout, grouping and
    /// focus by name; a zoomed layout falls back to its root.
    pub fn replace_data(&mut self, data: Dataset) {
        // A relabel draft uses the ids of the round it was made on; it must
        // never be applied to another round. Leave (saving the draft), and
        // come back below if this is the same round read again.
        let left = self
            .review
            .as_ref()
            .map(|r| (r.draft.round.clone(), r.overview.get(r.at).map(|o| o.id)));
        if left.is_some() {
            self.leave_review();
        }
        let (method, kind) = {
            let s = &self.data.spaces[self.root()];
            (s.method.clone(), s.kind)
        };
        let colour = self.colour.map(|i| self.data.labels[i].kind);
        let focus = self.focused_name();
        // A new annotation round of the same run points at the same model
        // and data files; keep what was read from them.
        let same_run = match (&self.data.run, &data.run) {
            (Some((a, ad)), Some((b, bd))) => {
                ad == bd
                    && serde_json::to_value(&a.outputs).ok()
                        == serde_json::to_value(&b.outputs).ok()
                    && serde_json::to_value(&a.data).ok() == serde_json::to_value(&b.data).ok()
            }
            _ => false,
        };
        // The same run read again keeps the batches it had; another run
        // reads its own on the side.
        let batch = self
            .data
            .labels
            .iter()
            .position(|l| l.kind == LabelKind::Batch)
            .map(|i| self.data.labels[i].clone());
        self.data = data;
        let has_batch = self.data.labels.iter().any(|l| l.kind == LabelKind::Batch);
        match batch {
            Some(b) if same_run && !has_batch => self.data.labels.push(b),
            _ if same_run && self.batch.is_some() => {}
            _ => self.start_batch(),
        }
        self.near = None;
        self.locked.clear();
        self.orders.borrow_mut().clear();
        self.feature_index.borrow_mut().take();
        self.name_index.borrow_mut().clear();
        self.medians.borrow_mut().take();
        self.space = self
            .data
            .spaces
            .iter()
            .position(|s| s.method == method && s.kind == kind)
            .unwrap_or(0);
        self.colour = colour
            .and_then(|t| {
                self.colour_choices()
                    .into_iter()
                    .find(|&i| self.data.labels[i].kind == t)
            })
            .or_else(|| self.default_colour(None));
        self.groups = None;
        self.refresh_groups();
        self.focus =
            focus.and_then(|f| self.levels().iter().position(|l| *l == f).map(|i| i as u32));
        self.shown = None;
        if same_run {
            if let Some(a) = self.activity.as_mut() {
                a.forget_views();
            }
        } else {
            self.activity = None;
            self.geometry = None;
            self.feature_embedding = None;
        }
        self.suggestions = None;
        self.refresh_activity();
        if let Some((round, id)) = left {
            self.resume_review(&round, id);
        }
    }

    /// After new data: relabel `round` again at cluster `id` if the data is
    /// that round read again (it changed on disk, or was reloaded); else say
    /// the draft stays with it.
    fn resume_review(&mut self, round: &std::path::Path, id: Option<i64>) {
        let same = self
            .data
            .round
            .as_ref()
            .is_some_and(|r| files::same_file(&r.path, round));
        if !same {
            let name = files::name(round);
            self.note = Some(format!(
                "a different round is open: relabel mode left, the draft stays with {name}"
            ));
            return;
        }
        match self.enter_review() {
            Ok(()) => {
                let at = self
                    .review
                    .as_ref()
                    .and_then(|r| r.overview.iter().position(|o| Some(o.id) == id));
                if let Some(i) = at {
                    self.visit(i);
                }
                self.note = Some("reloaded · still relabelling, at the same cluster".into());
            }
            Err(e) => self.note = Some(format!("reloaded · relabel mode left: {e}")),
        }
    }

    /// Colour by what the round changed against its source, and describe
    /// it: how many cells now carry each label.
    pub fn show_changes(&mut self) -> Vec<String> {
        let Some(li) = self.label_index(LabelKind::Changed) else {
            return vec!["no cell changed label".into()];
        };
        let labels = &self.data.labels[li];
        let count = data::group_counts(labels.by_name.values().copied(), labels.levels.len());
        let mut out = vec![format!("{} cells changed label:", labels.by_name.len())];
        let mut rows: Vec<(usize, &str)> = count
            .iter()
            .zip(&labels.levels)
            .map(|(&n, l)| (n, l.as_ref()))
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.0));
        out.extend(rows.iter().map(|(n, l)| format!("  {n:>6}  now {l}")));
        self.set_colour(Some(li));
        out
    }

    pub(super) fn cluster_labels(&self) -> Option<&data::Labels> {
        Some(&self.data.labels[self.label_index(LabelKind::Cluster)?])
    }

    /// What the round records about the cluster of cell `cell`, as panel lines.
    pub fn cluster_info(&self, cell: &str) -> Option<Vec<String>> {
        let id = self.cluster_id_of(cell)?;
        Some(self.data.round.as_ref()?.cluster_lines(&id.to_string()))
    }

    pub fn cluster_id_of(&self, cell: &str) -> Option<i64> {
        let c = self.cluster_labels()?;
        Some(c.ids[*c.by_name.get(cell)? as usize])
    }

    /// The label the current round gives cluster `id`.
    pub fn cluster_label(&self, id: i64) -> Option<String> {
        self.data.round.as_ref()?.label(&id.to_string())
    }

    /// Annotation and marker names, for label completion.
    pub fn known_labels(&self) -> Vec<Box<str>> {
        let mut out: Vec<Box<str>> = self
            .data
            .labels
            .iter()
            .filter(|l| matches!(l.kind, LabelKind::Annotation | LabelKind::Markers))
            .flat_map(|l| l.levels.iter().cloned())
            .collect();
        out.sort();
        out.dedup();
        out
    }

    pub fn focused_name(&self) -> Option<Box<str>> {
        self.focus.map(|f| self.levels()[f as usize].clone())
    }

    /// Every dot and all text bigger (`step` 1) or smaller (-1), by ×1.25
    /// within 0.4–4×; each group's own size and colour are kept.
    pub fn resize(&mut self, step: isize) {
        self.scale = (self.scale * 1.25_f32.powi(step as i32)).clamp(0.4, 4.0);
        self.note = Some(format!("dots and text ×{:.2} · < > resize", self.scale));
    }

    /// Labels on the map in turn: small, medium, large, largest, off.
    pub fn cycle_labels(&mut self) {
        self.step_labels(1);
    }

    /// The label size `step` along small, medium, large, largest, off,
    /// wrapping.
    pub fn step_labels(&mut self, step: isize) {
        let n = TEXT_SCALES.len() as isize + 1;
        let at = if self.show_labels {
            TEXT_SCALES
                .iter()
                .position(|&t| t >= self.text_scale)
                .unwrap_or(0) as isize
        } else {
            n - 1
        };
        let next = (at + step).rem_euclid(n) as usize;
        self.show_labels = next < TEXT_SCALES.len();
        if self.show_labels {
            self.text_scale = TEXT_SCALES[next];
        }
        self.note = Some(format!(
            "labels {} · t for the next size",
            self.labels_said()
        ));
    }

    /// The label size as said: small, medium, large, largest, or off.
    pub fn labels_said(&self) -> &'static str {
        const SIZES: [&str; 4] = ["small", "medium", "large", "largest"];
        if !self.show_labels {
            return "off";
        }
        let at = TEXT_SCALES
            .iter()
            .position(|&t| t >= self.text_scale)
            .unwrap_or(0);
        SIZES[at.min(SIZES.len() - 1)]
    }

    /// Colour by the grouping of `kind`, when the run has one.
    pub fn colour_by(&mut self, kind: LabelKind) {
        if let Some(li) = self.label_index(kind) {
            self.set_colour(Some(li));
        }
    }

    /// Next grouping. The feature shown is cleared, its suggestions stay.
    pub fn cycle_colour(&mut self) {
        self.step_colour(1);
    }

    /// The grouping `step` along those that apply here and then group
    /// colours, wrapping.
    pub fn step_colour(&mut self, step: isize) {
        let mut ring: Vec<Option<usize>> = self.colour_choices().into_iter().map(Some).collect();
        ring.push(None);
        let at = ring
            .iter()
            .position(|&c| c == self.colour)
            .unwrap_or(ring.len() - 1) as isize;
        let next = ring[(at + step).rem_euclid(ring.len() as isize) as usize];
        if self.set_colour(next) {
            self.note = Some("back to group colours · g brings the features back".into());
        }
    }

    /// What the map is coloured by, as said.
    pub fn colour_said(&self) -> &'static str {
        self.colour
            .map_or("groups", |c| self.data.labels[c].kind.title())
    }

    /// Focus the next or previous group. Activity and suggestions shown for
    /// the previous group no longer apply, so they are cleared.
    pub fn step_focus(&mut self, delta: i64) {
        let n = self.levels().len() as i64;
        if n == 0 {
            return;
        }
        let had_pick = self.clear_pick();
        if self.clear_suggestions() || had_pick {
            self.note = Some("n suggests features for this group".into());
        }
        self.focus = Some(match self.focus {
            None if delta > 0 => 0,
            None => (n - 1) as u32,
            Some(f) => ((f as i64 + delta).rem_euclid(n)) as u32,
        });
    }

    /// One-line description of the view.
    pub fn caption(&self) -> String {
        let s = self.current();
        let colour = self.colour.map_or("none".to_string(), |i| {
            self.data.labels[i].title().to_string()
        });
        let focus = self
            .focus
            .map(|f| format!(" · focus {}", self.levels()[f as usize]))
            .unwrap_or_default();
        let pick = match &self.pick {
            Some(Pick::One(f)) => format!(" · feature {f} ({}) · x clears", self.source.name()),
            Some(Pick::Markers(g)) => format!(" · {g} markers ({}) · x clears", self.source.name()),
            None => String::new(),
        };
        format!(
            "{} · {} · {} pts · colour {}{}{}",
            s.method,
            s.title(),
            s.points.order.len(),
            colour,
            focus,
            pick
        )
    }
}
