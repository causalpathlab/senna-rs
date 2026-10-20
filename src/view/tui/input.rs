//! Input: keys, mouse, feature search and the style menu.

use super::*;
use rayon::prelude::*;

impl App {
    /// A key press. Returns whether anything on screen may have changed
    /// (a cleared message counts).
    pub(super) fn key(&mut self, k: KeyEvent) -> bool {
        // ctrl+c quits from anywhere, a text field included.
        if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return true;
        }
        if self.modal.is_some() {
            return self.modal_key(k);
        }
        if self.menu.is_none() && !self.help && self.guard_recompute(k) {
            return true;
        }
        if self.label_edit.is_some() && self.menu.is_none() && !self.help {
            return self.label_key(k);
        }
        if self.scene.review.is_some() && !self.help && self.menu.is_none() {
            self.message = None;
            if self.review_key(k) {
                return true;
            }
        }
        if self.menu.is_some() {
            return self.menu_key(k);
        }
        self.message = None;
        if self.help {
            // The page scrolls; esc, q or ? closes it, other keys do nothing.
            match k.code {
                KeyCode::Esc | KeyCode::Char('q' | '?') => self.help = false,
                _ => match crate::tui::list_move(&k, self.help_scroll, self.help_max.get() + 1) {
                    Some(at) => self.help_scroll = at,
                    None => return false,
                },
            }
            return true;
        }
        if self.scene.chart.is_some() {
            if k.code == KeyCode::Esc {
                self.scene.chart = None;
                self.message = Some("back to the map".into());
                self.restart();
                return true;
            }
            if !chart_takes(k) {
                self.message =
                    Some("on a chart: H next chart or back to the map · esc the map".into());
                return true;
            }
        }
        // With a feature list in the sidebar, the arrows go through it.
        if crate::tui::list_delta(&k).is_some_and(|d| self.step_pane(d)) {
            return true;
        }
        let step = |app: &App| app.vp.map_or(0.0, |v| 0.1 * v.w.min(v.h) as f32);
        match k.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Esc | KeyCode::Char('x') => {
                let cleared = self.info.take().is_some()
                    || self.scene.clear_suggestions()
                    || self.scene.clear_pick()
                    || self.scene.focus.take().is_some();
                if !cleared {
                    self.message = Some("nothing to clear · q quits".into());
                }
                self.restart();
            }
            KeyCode::Char('e') if k.modifiers.contains(KeyModifiers::CONTROL) => self.ask_extra(),
            KeyCode::Char('e') => self.open_menu(),
            KeyCode::Char('g') => self.change(|s| s.step_feature(1)),
            KeyCode::Char('G') => self.change(|s| s.step_feature(-1)),
            KeyCode::Char('W') if self.agent.is_some() => self.follow_agent(),
            KeyCode::Char('a') => self.change(Scene::pick_marker_set),
            KeyCode::Char('o') => self.change_source(Scene::cycle_source),
            KeyCode::Char('/') => {
                if self.names.is_none() {
                    let names = self.scene.searchable();
                    let lower = names.iter().map(|n| n.to_lowercase()).collect();
                    self.names = Some(SearchNames { names, lower });
                }
                if self.names.as_ref().is_some_and(|n| n.names.is_empty()) {
                    self.names = None;
                    self.message = Some(match self.scene.loading_line() {
                        Some(loading) => format!("{loading} · / searches once it is read"),
                        None => "no feature names to search in this run".into(),
                    });
                } else {
                    self.modal = Some(Modal::Search(String::new(), Vec::new(), 0));
                }
            }
            KeyCode::Char('?') => {
                self.help = true;
                self.help_scroll = 0;
            }
            KeyCode::Char(EDIT_KEY) => self.start_label_edit(),
            KeyCode::Tab => self.step_view(1),
            KeyCode::BackTab => self.step_view(-1),
            KeyCode::Char('m') => self.step_method(1, false),
            KeyCode::Char('c') => self.change(Scene::cycle_colour),
            KeyCode::Char('y') => self.change(Scene::toggle_time),
            KeyCode::Char('u') => self.change_source(Scene::cycle_stat),
            // `{ }` step between views; the streamlines' density has its own.
            KeyCode::Char('(' | ')') => {
                self.scene
                    .change_stream_density(k.code == KeyCode::Char(')'));
                self.redecorate();
            }
            KeyCode::Char('v') => {
                self.scene.toggle_velocity();
                self.redecorate();
            }
            KeyCode::Char(']') => self.change(|s| s.step_focus(1)),
            KeyCode::Char('[') => self.change(|s| s.step_focus(-1)),
            KeyCode::Char('t') => {
                self.change_text(Scene::cycle_labels);
                self.redecorate();
            }
            // On a chart, `+` / `-` show more or fewer features per group.
            KeyCode::Char('+' | '=' | '-' | '_') if self.scene.chart.is_some() => {
                let more = matches!(k.code, KeyCode::Char('+' | '='));
                self.change_text(|s| s.change_top(more));
                self.restart();
            }
            KeyCode::Char('+' | '=') => {
                self.with_vp(|v| v.zoom_at(1.4, 0.5 * v.w as f32, 0.5 * v.h as f32, &FREE))
            }
            KeyCode::Char('-' | '_') => {
                self.with_vp(|v| v.zoom_at(1.0 / 1.4, 0.5 * v.w as f32, 0.5 * v.h as f32, &FREE))
            }
            KeyCode::Char('T') => self.resolve_topics(),
            KeyCode::Char('H') => {
                self.change_text(Scene::cycle_chart);
                self.restart();
            }
            KeyCode::Char('0') => {
                self.scene.zoom_to_root();
                self.vp = None;
            }
            KeyCode::Char('z') => self.start_zoom(),
            KeyCode::Char('n') => self.change(Scene::suggest),
            KeyCode::Char('L' | 'M' | 'K') => {
                self.message = Some("decisions are made in relabel mode: press R".into());
            }
            KeyCode::Char(' ' | 'b') if self.sidebar => self.hide_sidebar(),
            KeyCode::Char(' ' | 'b') => self.sidebar = true,
            KeyCode::Char(',') => self.step_round(true),
            KeyCode::Char('.') => self.step_round(false),
            KeyCode::Char('Z') | KeyCode::Backspace => {
                if self.scene.zoom_out() {
                    self.vp = None;
                } else {
                    self.message = Some("already at the top-level layout".into());
                }
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down => {
                let (sx, sy) = match k.code {
                    KeyCode::Left => (1.0, 0.0),
                    KeyCode::Right => (-1.0, 0.0),
                    KeyCode::Up => (0.0, 1.0),
                    _ => (0.0, -1.0),
                };
                let d = step(self);
                self.with_vp(|v| v.pan_px(sx * d, sy * d, &FREE));
            }
            KeyCode::Char('R') => self.toggle_review(),
            KeyCode::Char('r') => self.open_recompute(),
            KeyCode::Char('>') => self.change(|s| s.resize(1)),
            KeyCode::Char('<') => self.change(|s| s.resize(-1)),
            KeyCode::Char('A') => self.ask_markers(),
            KeyCode::Char('p') => {
                self.change_text(Scene::pin);
                self.redecorate();
            }
            _ => return false,
        }
        true
    }

    /// A key while searching. Returns whether anything changed.
    pub(super) fn search_key(&mut self, k: KeyEvent) -> bool {
        let Some(Modal::Search(query, hits, at)) = self.modal.as_mut() else {
            return false;
        };
        let last = hits.len().saturating_sub(1);
        match k.code {
            KeyCode::Esc => {
                self.modal = None;
                return true;
            }
            // The matches stay as the sidebar's list, the chosen one shown.
            KeyCode::Enter => {
                match self.modal.take() {
                    Some(Modal::Search(query, hits, at)) if at < hits.len() => {
                        let chosen = hits[at].clone();
                        self.change(|s| s.keep_matches(&query, hits, chosen));
                    }
                    _ => self.message = Some("no matching feature".into()),
                }
                return true;
            }
            // The cursor moves through the matches listed in the sidebar;
            // letters type into the query instead.
            _ if !matches!(k.code, KeyCode::Char(_)) && crate::tui::list_delta(&k).is_some() => {
                *at = crate::tui::list_move(&k, *at, last + 1).unwrap_or(*at);
            }
            // Type, backspace, ctrl+u clears; other control keys type nothing.
            _ if crate::tui::edit_line(query, &k) => self.research(),
            _ => return false,
        }
        true
    }

    /// The matches of the query typed so far, the cursor on the first.
    fn research(&mut self) {
        let Some(Modal::Search(query, hits, at)) = self.modal.as_mut() else {
            return;
        };
        *hits = self
            .names
            .as_ref()
            .map_or_else(Vec::new, |n| search(&n.names, &n.lower, query));
        *at = 0;
    }

    /// A key in the style menu. Returns whether anything changed.
    pub(super) fn menu_key(&mut self, k: KeyEvent) -> bool {
        let n = self.scene.levels().len();
        let Some(menu) = self.menu.as_mut() else {
            return false;
        };
        if n == 0 {
            self.menu = None;
            return true;
        }
        let row = menu.row;
        let step: i64 = match k.code {
            KeyCode::Left | KeyCode::Char('h') => -1,
            KeyCode::Right | KeyCode::Char('l') => 1,
            _ => 0,
        };
        match k.code {
            KeyCode::Enter | KeyCode::Esc | KeyCode::Char('e' | 'q') => {
                self.menu = None;
                self.scene.save_styles();
            }
            KeyCode::Up | KeyCode::Char('k') => menu.row = (row + n - 1) % n,
            KeyCode::Down | KeyCode::Char('j') => menu.row = (row + 1) % n,
            KeyCode::Tab => menu.field = (menu.field + 1) % FIELDS.len(),
            KeyCode::BackTab => menu.field = (menu.field + FIELDS.len() - 1) % FIELDS.len(),
            KeyCode::Char(' ') => self.scene.restyle(row, |st| st.hidden = !st.hidden),
            KeyCode::Backspace => self
                .scene
                .restyle(row, |st| *st = crate::view::style::Style::plain()),
            _ if step != 0 => {
                let field = menu.field;
                let current = self.scene.resolved(row).map(|r| r.colour);
                self.scene
                    .restyle(row, |st| adjust(st, field, step, current));
            }
            _ => return false,
        }
        if let Some(m) = &self.menu {
            self.scene.focus = Some(m.row as u32);
        }
        self.settle();
        true
    }

    pub(super) fn open_menu(&mut self) {
        let n = self.scene.levels().len();
        if self.scene.colour.is_none() || n == 0 {
            self.message = Some("choose a grouping with c first".into());
            return;
        }
        let row = self.scene.focus.map_or(0, |f| f as usize).min(n - 1);
        self.scene.focus = Some(row as u32);
        self.menu = Some(Menu { row, field: 0 });
        self.restart();
    }

    /// A mouse event. Returns whether anything changed.
    pub(super) fn mouse(&mut self, m: MouseEvent) -> bool {
        // Over the `?` page the wheel scrolls it.
        if self.help {
            let step = match m.kind {
                MouseEventKind::ScrollUp => KeyCode::Up,
                MouseEventKind::ScrollDown => KeyCode::Down,
                _ => return false,
            };
            return self.key(KeyEvent::from(step));
        }
        if let MouseEventKind::Down(MouseButton::Left) = m.kind {
            if self.panel_click(m.column, m.row) || self.pane_click(m.column, m.row) {
                return true;
            }
        }
        // A chart has no points to hover, click, pan or zoom.
        if self.scene.chart.is_some() {
            self.pointer.cancel();
            return false;
        }
        let area = MapArea {
            rect: self.map,
            cell: self.cell,
        };
        let (vp, scene) = (self.vp, &self.scene);
        let held = self.pointer.label();
        let action = self.pointer.handle(&area, &m, |px, py| {
            vp.and_then(|vp| scene.group_at(&vp, px, py))
        });
        // A label let go, wherever: where it was put is kept. One pressed
        // and let go where it was is a click on it.
        let let_go = held.filter(|_| !self.pointer.dragging());
        if let (Some(g), false) = (let_go, matches!(action, MapAction::Click { .. })) {
            self.scene.keep_label_places();
            let name = &self.scene.levels()[g as usize];
            self.message = Some(format!(
                "{name} moved, here and in saved figures · E moves labels by key, backspace there puts one back"
            ));
            self.change_text(|_| {});
            return true;
        }
        match action {
            MapAction::None => return false,
            MapAction::Hover { px, py } => return self.hover(Some((px, py))),
            MapAction::Away => return self.hover(None),
            MapAction::Pan { .. } | MapAction::Zoom { .. } => self.with_vp(|v| {
                action.apply(v, &FREE);
            }),
            MapAction::MoveLabel { key, dx, dy } => {
                let em = self.scene.label_em(self.cell.1);
                self.scene.move_label(key, (dx / em, dy / em));
                if let Some(e) = self.label_edit.as_mut() {
                    e.at = Some(key);
                    self.scene.editing = Some(key);
                }
                self.redecorate();
            }
            MapAction::Click { px, py, .. } => self.click(px, py),
        }
        true
    }

    /// Frame the cluster label under the pointer at `at` (none: it left
    /// the map) and say what a click on it does; clear that when the
    /// pointer leaves it. Returns whether it changed.
    fn hover(&mut self, at: Option<(f32, f32)>) -> bool {
        let under = match (at, self.vp) {
            (Some((px, py)), Some(vp)) => self.scene.group_at(&vp, px, py),
            _ => None,
        };
        if under == self.scene.hover {
            return false;
        }
        let was = std::mem::replace(&mut self.scene.hover, under);
        match under {
            Some(g) => {
                let name = &self.scene.levels()[g as usize];
                self.message = Some(format!(
                    "{name} · click for the features placed nearest this cluster, drag to move the label"
                ));
            }
            // Leaving a label takes its note with it.
            None if was.is_some() => self.message = None,
            None => {}
        }
        self.redecorate();
        true
    }

    /// `E`: move the labels on the map from the keyboard, the first as
    /// read chosen.
    fn start_label_edit(&mut self) {
        let placed = self.scene.placed_labels();
        if self.scene.current().axis() != crate::view::Axis::Cells || placed.is_empty() {
            self.message = Some(
                "no labels to move here: a cell map's group labels move (t shows labels)".into(),
            );
            return;
        }
        let edit = LabelEdit::enter(&placed);
        self.label_edit = Some(edit);
        self.scene.editing = edit.at;
        self.label_message();
        if let Some(why) = &self.scene.places.refused {
            let note = format!("moves last this session only: {why}");
            self.message = Some(match self.message.take() {
                Some(m) => format!("{m} · {note}"),
                None => note,
            });
        }
        self.redecorate();
    }

    /// A key while moving labels: tab chooses, arrows move, backspace puts
    /// back, esc (or `E`) leaves. Other keys wait, `?` and `q` aside.
    fn label_key(&mut self, k: KeyEvent) -> bool {
        let Some(mut edit) = self.label_edit else {
            return false;
        };
        let placed = self.scene.placed_labels();
        let mut o = self.scene.group_offsets();
        let did = if LabelEdit::<u32>::starts(&k) {
            EditKey::Leave
        } else {
            edit.key(&k, &placed, &mut o)
        };
        match did {
            EditKey::Changed => {
                self.scene.set_group_offsets(&o);
                self.scene.keep_label_places();
            }
            EditKey::Taken => {}
            EditKey::Leave => {
                self.label_edit = None;
                self.scene.editing = None;
                self.message = Some(format!(
                    "labels stay where they were put · kept in {}",
                    files::name(&crate::view::places::Places::path(&self.scene.data.prefix))
                ));
                self.change_text(|_| {});
                self.redecorate();
                return true;
            }
            EditKey::Pass => match k.code {
                KeyCode::Char('?') => {
                    self.help = true;
                    self.help_scroll = 0;
                    return true;
                }
                KeyCode::Char('q') => {
                    self.quit = true;
                    return true;
                }
                _ => {
                    self.message =
                        Some("moving labels: esc or E first, then the map's keys".into());
                    return true;
                }
            },
        }
        self.label_edit = Some(edit);
        self.scene.editing = edit.at;
        self.label_message();
        self.change_text(|_| {});
        self.redecorate();
        true
    }

    /// Which label the keys move, and how.
    fn label_message(&mut self) {
        let name = self
            .scene
            .editing
            .and_then(|g| self.scene.levels().get(g as usize))
            .map_or_else(|| "a label".to_string(), ToString::to_string);
        self.message = Some(format!(
            "moving {name} · tab next label · arrows move it · esc done"
        ));
    }

    /// Focus the group of the point nearest the click, and name the point.
    pub(super) fn click(&mut self, px: f32, py: f32) {
        let Some(vp) = self.vp else { return };
        // A cluster's label stands for the cluster: its features, not one
        // cell's.
        if let Some(g) = self.scene.group_at(&vp, px, py) {
            self.scene.focus = Some(g);
            self.scene.show_near_group(g);
            self.info = None;
            let name = &self.scene.levels()[g as usize];
            self.message = Some(format!("{name} · the features most up in this cluster"));
            self.settle();
            return;
        }
        let pts = &self.scene.current().points;
        let reach = 2.0 * self.cell.0.max(8.0);
        let best = pts
            .order
            .par_iter()
            .map(|&i| {
                let (x, y) = vp.to_px(pts.xy[i as usize]);
                (i as usize, (x - px).powi(2) + (y - py).powi(2))
            })
            .filter(|&(_, d)| d <= reach * reach)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((i, _)) = best else {
            self.message = None;
            return;
        };
        let name = pts.names[i].to_string();
        let group = self
            .scene
            .groups()
            .map(|g| g[i])
            .filter(|&g| g != crate::view::NONE);
        let on_features = self.scene.current().axis() == crate::view::Axis::Features;
        if !on_features {
            // The features nearest this cell; in relabel mode, also go to its
            // cluster.
            self.scene.show_near(&name);
            let clicked = self.scene.cluster_id_of(&name);
            if let (Some(id), Some(r)) = (clicked, self.scene.review.as_ref()) {
                if let Some(i) = r.overview.iter().position(|o| o.id == id) {
                    if i != r.at {
                        self.scene.visit(i);
                    }
                    self.settle();
                    return;
                }
            }
        }
        self.info = (!on_features)
            .then(|| self.scene.cluster_info(&name))
            .flatten();
        let mut msg = match group {
            Some(g) => {
                self.scene.focus = Some(g);
                format!("{name} · {}", self.scene.levels()[g as usize])
            }
            None => name.clone(),
        };
        // A feature clicked on a feature map shows the features nearest it,
        // and becomes the pick, so a cell map shows its activity on the way
        // back.
        if on_features {
            self.scene.show_near_feature(&name);
            self.scene.set_pick(Pick::One(name.into()));
            if self.scene.data.has_axis(crate::view::Axis::Cells) {
                msg.push_str(" · tab to a cell view for its activity");
            }
        }
        self.message = Some(msg);
        self.restart();
    }

    pub(super) fn with_vp(&mut self, f: impl FnOnce(&mut Viewport)) {
        if let Some(vp) = self.vp.as_mut() {
            f(vp);
            self.restart();
        }
    }

    /// A scene change that may show another source: its feature axis is
    /// the one `/` searches from now on.
    pub(super) fn change_source(&mut self, f: impl FnOnce(&mut Scene)) {
        self.names = None;
        self.change(f);
    }

    /// Run a scene change, surface its note, and redraw.
    pub(super) fn change(&mut self, f: impl FnOnce(&mut Scene)) {
        f(&mut self.scene);
        self.settle();
    }

    /// Run a scene change that leaves the map as it is (only text changes),
    /// and surface its note.
    pub(super) fn change_text(&mut self, f: impl FnOnce(&mut Scene)) {
        f(&mut self.scene);
        if let Some(note) = self.scene.note.take() {
            self.message = Some(note);
        }
    }

    /// After the scene changed: show its note, if any, and redraw.
    pub(super) fn settle(&mut self) {
        self.change_text(|_| {});
        self.restart();
    }

    /// One terminal event. Returns whether the screen needs drawing.
    pub(super) fn handle(&mut self, ev: Event) -> bool {
        // While lupin works on a request, nothing may change underneath it.
        if self.relabeling.is_some() {
            if let Event::Key(k) = &ev {
                let ctrl_c =
                    k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL);
                if k.code == KeyCode::Char('q') || ctrl_c {
                    self.quit = true;
                }
            }
            if let Event::Resize(..) = ev {
                self.vp = None;
                return true;
            }
            return false;
        }
        let before = self.message.clone();
        let by_hand = matches!(&ev, Event::Key(k) if k.kind != KeyEventKind::Release)
            || matches!(&ev, Event::Mouse(m) if !matches!(m.kind, MouseEventKind::Moved));
        // What is shown, to tell whether the human changed it.
        let shown = |a: &App| {
            let s = &a.scene;
            (
                s.space,
                s.colour,
                s.focus,
                s.pick.clone(),
                s.source,
                a.from.clone(),
                a.zooming.is_some(),
                a.vp,
            )
        };
        let followed =
            (by_hand && self.agent.as_ref().is_some_and(|a| a.following())).then(|| shown(self));
        let changed = match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => self.key(k),
            Event::Mouse(m) => self.mouse(m),
            Event::Resize(..) => {
                self.vp = None;
                true
            }
            _ => false,
        };
        // The human moved the camera or changed what is shown: the view is
        // theirs until W.
        if followed.is_some_and(|before| before != shown(self)) {
            self.pause_agent();
        }
        changed || self.message != before
    }
}

/// Step one style property of a group by `step` (±1).
fn adjust(
    st: &mut crate::view::style::Style,
    field: usize,
    step: i64,
    current: Option<color::Rgb>,
) {
    match field {
        0 => {
            let sw = swatches();
            let now = st.colour.or_else(|| current.map(color::to_srgb_senna));
            // Start from the swatch nearest the colour shown now.
            let at = now.map_or(0, |c| {
                (0..sw.len())
                    .min_by_key(|&i| {
                        (0..3)
                            .map(|k| (i32::from(sw[i][k]) - i32::from(c[k])).pow(2))
                            .sum::<i32>()
                    })
                    .unwrap_or(0)
            });
            let n = sw.len() as i64;
            st.colour = Some(sw[(at as i64 + step).rem_euclid(n) as usize]);
        }
        1 => {
            let at = Shape::ALL.iter().position(|&s| s == st.shape).unwrap_or(0) as i64;
            st.shape = Shape::ALL[(at + step).rem_euclid(Shape::ALL.len() as i64) as usize];
        }
        2 => st.alpha = (st.alpha + 0.1 * step as f32).clamp(0.1, 1.0),
        3 => st.size = (st.size + 0.25 * step as f32).clamp(0.25, 4.0),
        _ => st.hidden = !st.hidden,
    }
}

/// The sidebar while searching: what was typed, then the matches, the one
/// under the cursor marked.
pub(super) fn search_rows(query: &str, hits: &[Box<str>], at: usize) -> Vec<PaneRow> {
    let mut out: Vec<PaneRow> = vec![
        (format!("search “{query}”"), None),
        ("↑ ↓ choose · enter or click shows · esc back".into(), None),
    ];
    if hits.is_empty() {
        out.push((
            if query.is_empty() {
                "  type part of a feature name".into()
            } else {
                "  no match".into()
            },
            None,
        ));
    }
    out.extend(hits.iter().enumerate().map(|(k, f)| {
        let mark = if k == at { "▸" } else { " " };
        (format!("{mark} {f}"), Some(f.clone()))
    }));
    out
}

/// Features matching `query`: exact name first, then symbol (the part after
/// an `ID_` prefix), then prefix, then substring; case-insensitive.
fn search(names: &[Box<str>], lower: &[String], query: &str) -> Vec<Box<str>> {
    const MAX: usize = 200;
    let q = query.to_lowercase();
    if q.is_empty() {
        return Vec::new();
    }
    let rank = |n: &str| {
        let symbol = n.rsplit_once('_').map_or(n, |(_, s)| s);
        if n == q || symbol == q {
            Some(0)
        } else if n.starts_with(&q) || symbol.starts_with(&q) {
            Some(1)
        } else if n.contains(&q) {
            Some(2)
        } else {
            None
        }
    };
    let mut hits: Vec<(u8, usize, &Box<str>)> = names
        .iter()
        .zip(lower)
        .filter_map(|(n, l)| rank(l).map(|r| (r, n.len(), n)))
        .collect();
    hits.sort();
    hits.into_iter()
        .take(MAX)
        .map(|(.., n)| n.clone())
        .collect()
}

/// Whether a chart takes `k`; the map's other keys would act unseen.
fn chart_takes(k: KeyEvent) -> bool {
    matches!(
        k.code,
        KeyCode::Char('H' | 'c' | '+' | '=' | '-' | '_' | 'T' | '?' | 'q')
    )
}

#[cfg(test)]
mod tests {
    use super::search;
    use super::*;

    fn app() -> App {
        App::new(
            crate::view::tests::scene(),
            Picker::halfblocks(),
            "r.senna.json".into(),
            "lupin".into(),
        )
    }

    fn press(app: &mut App, c: char) {
        app.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
    }

    #[test]
    fn a_chart_keeps_to_its_keys_and_esc_returns_to_the_map() {
        let mut a = app();
        a.scene.cycle_chart();
        assert!(a.scene.chart.is_some());
        let colour = a.scene.colour;
        press(&mut a, 'n');
        assert!(a.message.as_deref().unwrap().starts_with("on a chart"));
        assert!(a.scene.pick.is_none());
        press(&mut a, 'c');
        assert_ne!(a.scene.colour, colour);
        a.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(a.scene.chart.is_none());
    }

    #[test]
    fn merge_mode_keeps_to_its_keys() {
        let mut a = app();
        let draft = crate::view::review::Draft::default();
        let mut review = crate::view::relabel::Review::new(Vec::new(), draft);
        review.merge = Some(crate::view::relabel::MergeSel {
            cursor: 0,
            chosen: std::collections::BTreeSet::new(),
            levels: Vec::new(),
            best: None,
        });
        a.scene.review = Some(review);
        let space = a.scene.space;
        for c in ['m', 'z', 'L', 'a'] {
            press(&mut a, c);
            assert!(
                a.message.as_deref().unwrap().starts_with("in merge mode"),
                "{c}"
            );
        }
        assert_eq!(a.scene.space, space);
        assert!(a.zooming.is_none());
    }

    #[test]
    fn the_help_page_scrolls_and_closes_only_on_esc_q_or_question_mark() {
        let mut a = app();
        press(&mut a, '?');
        assert!(a.help);
        a.help_max.set(5);
        press(&mut a, 'j');
        a.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(a.help_scroll, 2);
        // A key of the map does nothing while the page is up.
        let colour = a.scene.colour;
        press(&mut a, 'c');
        assert!(a.help && a.scene.colour == colour);
        press(&mut a, 'q');
        assert!(!a.help && !a.quit);
    }

    #[test]
    fn ctrl_c_quits_and_control_keys_type_nothing_in_search() {
        let mut a = app();
        a.modal = Some(Modal::Search(String::new(), Vec::new(), 0));
        press(&mut a, 'g');
        a.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        match &a.modal {
            Some(Modal::Search(query, ..)) => assert_eq!(query, "g"),
            _ => panic!("search closed"),
        }
        a.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(a.quit);
    }

    #[test]
    fn ctrl_e_asks_for_an_extra_file_and_e_alone_opens_the_menu() {
        let mut a = app();
        let ctrl_e = KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL);
        a.key(ctrl_e);
        assert!(a.message.as_deref().unwrap().starts_with("no run"));
        a.scene.data.run = Some((
            senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Svd, "r"),
            std::env::temp_dir(),
        ));
        a.key(ctrl_e);
        assert!(a.menu.is_none());
        let Some(m) = &a.modal else {
            panic!("no browser")
        };
        assert!(matches!(m, Modal::ExtraFile(_)) && m.browser_lines(20).is_some());
        a.modal = None;
        press(&mut a, 'e');
        assert!(a.menu.is_some());
    }

    #[test]
    pub(super) fn search_ranks_exact_then_symbol_then_prefix_then_substring() {
        let names: Vec<Box<str>> = ["ENSG1_GENE10", "GENE1", "XGENE1Y", "ENSG2_GENE1", "OTHER"]
            .map(Into::into)
            .to_vec();
        let lower: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
        let hits = search(&names, &lower, "gene1");
        assert_eq!(&*hits[0], "GENE1");
        assert_eq!(&*hits[1], "ENSG2_GENE1");
        assert_eq!(&*hits[2], "ENSG1_GENE10");
        assert_eq!(&*hits[3], "XGENE1Y");
        assert_eq!(hits.len(), 4);
        assert!(search(&names, &lower, "").is_empty());
    }
}

#[cfg(test)]
mod label_tests {
    use super::*;
    use crate::view::places::Places;

    /// A view of the test scene, its run at `dir/r`, laid out on a kitty
    /// terminal and drawn.
    fn drawn_in(dir: &std::path::Path) -> App {
        let mut scene = crate::view::tests::scene();
        scene.data.prefix = dir.join("r").to_string_lossy().into_owned();
        scene.places = Places::load(&scene.data.prefix);
        let mut a = App::new(
            scene,
            super::super::attach_tests::kitty(),
            "r.senna.json".into(),
            "lupin".into(),
        );
        a.sidebar = false;
        a.layout(Rect::new(0, 0, 100, 40), STATUS_LINES);
        draw(&mut a);
        a
    }

    fn draw(a: &mut App) {
        if a.job.is_none() && a.base.is_none() {
            a.restart();
        }
        while a.job.is_some() {
            a.advance().unwrap();
        }
    }

    fn mouse(a: &mut App, kind: MouseEventKind, (column, row): (u16, u16)) {
        a.mouse(MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        });
    }

    /// The cell at the centre of group `g`'s label.
    fn label_cell(a: &App, g: u32) -> (u16, u16) {
        let [x0, y0, x1, y1] = a.scene.label_rect(g).expect("label drawn");
        (
            a.map.x + (0.5 * (x0 + x1) / a.cell.0) as u16,
            a.map.y + (0.5 * (y0 + y1) / a.cell.1) as u16,
        )
    }

    fn key(a: &mut App, code: KeyCode, mods: KeyModifiers) {
        a.key(KeyEvent::new(code, mods));
    }

    #[test]
    fn a_dragged_label_stays_where_it_is_put_after_a_reload() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = drawn_in(dir.path());
        let before = a.vp;
        let (col, row) = label_cell(&a, 1);
        mouse(&mut a, MouseEventKind::Down(MouseButton::Left), (col, row));
        mouse(
            &mut a,
            MouseEventKind::Drag(MouseButton::Left),
            (col + 3, row + 2),
        );
        // Let go off the map, on the status lines: the drag ends all the
        // same.
        mouse(&mut a, MouseEventKind::Up(MouseButton::Left), (0, 39));
        assert!(!a.pointer.dragging());
        assert_eq!(a.vp, before, "the label moved, not the map");
        let em = a.scene.label_em(a.cell.1);
        let want = (3.0 * a.cell.0 / em, 2.0 * a.cell.1 / em);
        let got = a.scene.group_offsets().get(&1).unwrap();
        assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4);
        // A drag of the map pans, and a later one starts afresh.
        mouse(&mut a, MouseEventKind::Down(MouseButton::Left), (2, 2));
        mouse(&mut a, MouseEventKind::Drag(MouseButton::Left), (4, 2));
        mouse(&mut a, MouseEventKind::Up(MouseButton::Left), (4, 2));
        assert_ne!(a.vp, before);
        // A label pressed and let go where it was is clicked, not moved.
        a.vp = before;
        a.restart();
        draw(&mut a);
        let at = label_cell(&a, 1);
        mouse(&mut a, MouseEventKind::Down(MouseButton::Left), at);
        mouse(&mut a, MouseEventKind::Up(MouseButton::Left), at);
        assert_eq!(a.scene.focus, Some(1));
        let got = a.scene.group_offsets().get(&1).unwrap();
        assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4);
        // Opened again, the run's label is where it was put.
        let b = drawn_in(dir.path());
        let got = b.scene.group_offsets().get(&1).unwrap();
        assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4);
        assert!(b
            .scene
            .placed_labels()
            .iter()
            .any(|p| p.key == 1 && p.moved));
    }

    #[test]
    fn shift_e_moves_labels_by_key_and_esc_leaves() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = drawn_in(dir.path());
        let space = a.scene.space;
        key(&mut a, KeyCode::Char('E'), KeyModifiers::SHIFT);
        let g = a.label_edit.and_then(|e| e.at).expect("a label chosen");
        assert_eq!(a.scene.editing, Some(g));
        assert!(a.context() == help::Context::Labels);
        key(&mut a, KeyCode::Right, KeyModifiers::NONE);
        assert_eq!(a.scene.group_offsets().get(&g), Some((0.5, 0.0)));
        key(&mut a, KeyCode::Up, KeyModifiers::SHIFT);
        assert_eq!(a.scene.group_offsets().get(&g), Some((0.5, -2.0)));
        // Kept at once.
        let back = Places::load(&a.scene.data.prefix);
        let name = a.scene.levels()[g as usize].to_string();
        assert_eq!(
            back.layout(&a.scene.layout_id()).get(&name),
            Some((0.5, -2.0))
        );
        // Tab chooses a label here, and does not step the view.
        key(&mut a, KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(a.scene.space, space);
        assert!(a.label_edit.is_some());
        a.label_edit = Some(LabelEdit { at: Some(g) });
        key(&mut a, KeyCode::Backspace, KeyModifiers::NONE);
        assert_eq!(a.scene.group_offsets().get(&g), None);
        // Other map keys wait for the mode to end.
        let colour = a.scene.colour;
        key(&mut a, KeyCode::Char('c'), KeyModifiers::NONE);
        assert_eq!(a.scene.colour, colour);
        key(&mut a, KeyCode::Esc, KeyModifiers::NONE);
        assert!(a.label_edit.is_none() && a.scene.editing.is_none());
        // Outside the mode, tab steps the view as ever.
        key(&mut a, KeyCode::Tab, KeyModifiers::NONE);
        assert_ne!(a.scene.space, space);
    }

    #[test]
    fn e_with_no_labels_to_move_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = drawn_in(dir.path());
        a.scene.show_labels = false;
        a.restart();
        draw(&mut a);
        key(&mut a, KeyCode::Char('E'), KeyModifiers::SHIFT);
        assert!(a.label_edit.is_none());
        assert!(a.message.as_deref().unwrap().contains("no labels"));
    }
}
