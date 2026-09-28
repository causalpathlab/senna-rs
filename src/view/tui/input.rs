//! Input: keys, mouse, feature search and the style menu.

use super::*;
use rayon::prelude::*;

impl App {
    pub(super) fn key(&mut self, k: KeyEvent) {
        if self.search.is_some() {
            self.search_key(k);
            return;
        }
        if self.prompt.is_some() {
            self.prompt_key(k);
            return;
        }
        if self.markers_input.is_some() {
            self.markers_key(k);
            return;
        }
        if self.scene.review.is_some() && !self.help && self.menu.is_none() {
            self.message = None;
            if self.review_key(k) {
                return;
            }
        }
        if self.menu.is_some() {
            self.menu_key(k);
            return;
        }
        self.message = None;
        if self.help {
            self.help = false;
            return;
        }
        let n_spaces = self.scene.data.spaces.len();
        let step = |app: &App| app.vp.map_or(0.0, |v| 0.1 * v.w.min(v.h) as f32);
        match k.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => self.quit = true,
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
            KeyCode::Char('e') => self.open_menu(),
            KeyCode::Char('g') => self.change(|s| s.step_feature(1)),
            KeyCode::Char('G') => self.change(|s| s.step_feature(-1)),
            KeyCode::Char('a') => self.change(Scene::pick_marker_set),
            KeyCode::Char('o') => {
                // The other source has its own feature axis to search.
                self.names = None;
                self.change(Scene::toggle_source);
            }
            KeyCode::Char('/') => {
                if self.names.is_none() {
                    let names = self.scene.searchable();
                    let lower = names.iter().map(|n| n.to_lowercase()).collect();
                    self.names = Some(SearchNames { names, lower });
                }
                if self.names.as_ref().is_some_and(|n| n.names.is_empty()) {
                    self.names = None;
                    self.message = Some("no feature names to search in this run".into());
                } else {
                    self.search = Some((String::new(), Vec::new()));
                }
            }
            KeyCode::Char('?') => self.help = true,
            KeyCode::Tab => self.switch_space((self.scene.space + 1) % n_spaces),
            KeyCode::BackTab => self.switch_space((self.scene.space + n_spaces - 1) % n_spaces),
            KeyCode::Char('m') => self.next_method(),
            KeyCode::Char('c') => self.change(Scene::cycle_colour),
            KeyCode::Char(']') => self.change(|s| s.step_focus(1)),
            KeyCode::Char('[') => self.change(|s| s.step_focus(-1)),
            KeyCode::Char('t') => self.change(|s| s.show_labels = !s.show_labels),
            KeyCode::Char('+' | '=') => {
                self.with_vp(|v| v.zoom_at(1.4, 0.5 * v.w as f32, 0.5 * v.h as f32))
            }
            KeyCode::Char('-' | '_') => {
                self.with_vp(|v| v.zoom_at(1.0 / 1.4, 0.5 * v.w as f32, 0.5 * v.h as f32))
            }
            KeyCode::Char('0' | 'r') => {
                self.scene.zoom_to_root();
                self.vp = None;
            }
            KeyCode::Char('z') => self.start_zoom(),
            KeyCode::Char('n') => self.change(Scene::suggest),
            KeyCode::Char('v' | 'L' | 'M' | 'K' | 'D') => {
                self.message = Some("decisions are made in relabel mode: press R".into());
            }
            KeyCode::Char('b') => {
                self.sidebar = !self.sidebar;
                if !self.sidebar {
                    self.message = Some("sidebar hidden · b to show".into());
                }
            }
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
                self.with_vp(|v| v.pan_px(sx * d, sy * d));
            }
            KeyCode::Char('s') => self.save(),
            KeyCode::Char('R') => self.toggle_review(),
            KeyCode::Char('A') => self.ask_markers(),
            KeyCode::Char('T') => self.change(Scene::cycle_text_size),
            KeyCode::Char('k' | 'l') => self.change(Scene::pin),
            _ => {}
        }
    }

    pub(super) fn search_key(&mut self, k: KeyEvent) {
        let Some((query, _)) = self.search.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                self.search = None;
                return;
            }
            KeyCode::Enter => {
                let pick = self
                    .search
                    .take()
                    .and_then(|(_, hits)| hits.into_iter().next());
                match pick {
                    Some(name) => self.change(|s| s.set_pick(Pick::One(name))),
                    None => self.message = Some("no matching feature".into()),
                }
                return;
            }
            KeyCode::Backspace => {
                query.pop();
            }
            KeyCode::Char(c) => query.push(c),
            _ => return,
        }
        let query = query.clone();
        let hits = self
            .names
            .as_ref()
            .map_or_else(Vec::new, |n| search(&n.names, &n.lower, &query));
        self.search = Some((query, hits));
    }

    pub(super) fn menu_key(&mut self, k: KeyEvent) {
        let n = self.scene.levels().len();
        let Some(menu) = self.menu.as_mut() else {
            return;
        };
        if n == 0 {
            self.menu = None;
            return;
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
            KeyCode::Char('r') => self
                .scene
                .restyle(row, |st| *st = crate::view::style::Style::plain()),
            _ if step != 0 => {
                let field = menu.field;
                let current = self.scene.resolved(row).map(|r| r.colour);
                self.scene
                    .restyle(row, |st| adjust(st, field, step, current));
            }
            _ => return,
        }
        if let Some(m) = &self.menu {
            self.scene.focus = Some(m.row as u32);
        }
        self.settle();
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

    pub(super) fn mouse(&mut self, m: MouseEvent) {
        let Some((px, py)) = self.cell_to_px(m.column, m.row) else {
            return;
        };
        match m.kind {
            MouseEventKind::ScrollUp => self.with_vp(|v| v.zoom_at(1.25, px, py)),
            MouseEventKind::ScrollDown => self.with_vp(|v| v.zoom_at(0.8, px, py)),
            MouseEventKind::Down(MouseButton::Left) => {
                self.drag = Some(Drag {
                    col: m.column,
                    row: m.row,
                    moved: false,
                });
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some(d) = self.drag.as_mut() {
                    let dx = (f32::from(m.column) - f32::from(d.col)) * self.cell.0;
                    let dy = (f32::from(m.row) - f32::from(d.row)) * self.cell.1;
                    *d = Drag {
                        col: m.column,
                        row: m.row,
                        moved: true,
                    };
                    self.with_vp(|v| v.pan_px(dx, dy));
                }
            }
            // Releasing ends the drag; a press that never moved is a click.
            MouseEventKind::Up(MouseButton::Left) if self.drag.take().is_some_and(|d| !d.moved) => {
                self.click(px, py);
            }
            _ => {}
        }
    }

    /// Focus the group of the point nearest the click, and name the point.
    pub(super) fn click(&mut self, px: f32, py: f32) {
        let Some(vp) = self.vp else { return };
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
        // A feature clicked on a feature map becomes the pick, so the cell map
        // shows its activity on the way back.
        if on_features {
            self.scene.set_pick(Pick::One(name.into()));
            msg.push_str(" · tab to a cell view for its activity");
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

    pub(super) fn cell_to_px(&self, col: u16, row: u16) -> Option<(f32, f32)> {
        let m = self.map;
        (col >= m.x && row >= m.y && col < m.x + m.width && row < m.y + m.height).then(|| {
            (
                (f32::from(col - m.x) + 0.5) * self.cell.0,
                (f32::from(row - m.y) + 0.5) * self.cell.1,
            )
        })
    }

    /// Run a scene change, surface its note, and redraw.
    pub(super) fn change(&mut self, f: impl FnOnce(&mut Scene)) {
        f(&mut self.scene);
        self.settle();
    }

    /// After the scene changed: show its note, if any, and redraw.
    pub(super) fn settle(&mut self) {
        if let Some(note) = self.scene.note.take() {
            self.message = Some(note);
        }
        self.restart();
    }

    pub(super) fn handle(&mut self, ev: Event) {
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
            }
            return;
        }
        match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => self.key(k),
            Event::Mouse(m) => self.mouse(m),
            Event::Resize(..) => self.vp = None,
            _ => {}
        }
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
            let enc = color::encoder();
            let now = st
                .colour
                .or_else(|| current.map(|c| c.map(|v| enc.encode(v))));
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

/// Features matching `query`: exact name first, then symbol (the part after
/// an `ID_` prefix), then prefix, then substring; case-insensitive.
fn search(names: &[Box<str>], lower: &[String], query: &str) -> Vec<Box<str>> {
    const MAX: usize = 8;
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

#[cfg(test)]
mod tests {
    use super::search;

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
