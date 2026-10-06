//! Several views open at once, and the one loop that drives them. One view
//! is on screen at a time; `w` shows them all in a grid, each tile the
//! view's current picture, where the mouse or the arrows choose one and a
//! click or enter opens it. Saving happens here too, since a PDF can take in
//! every view. Every view's background work (a zoom, lupin, a reloaded file)
//! goes on whichever is on screen.

use super::*;
use crate::view::deck;
use crate::view::pdf;
use crate::view::saved::{ago, Gallery};
use rayon::prelude::*;

/// One tile of the grid: its frame on screen and its picture.
struct Tile {
    rect: Rect,
    job: Option<Job>,
    proto: Option<Protocol>,
}

/// Whether a toast's time is up.
pub(super) fn expired(toast: &Option<(String, std::time::Instant)>) -> bool {
    toast
        .as_ref()
        .is_some_and(|(_, until)| std::time::Instant::now() >= *until)
}

/// What fills the terminal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Grid,
    View(usize),
}

pub(super) struct Deck {
    apps: Vec<App>,
    /// The view on screen, or last on screen while the grid shows.
    at: usize,
    picker: Picker,
    cell: (f32, f32),
    grid: bool,
    /// The tile under the pointer or the arrow keys.
    hover: usize,
    tiles: Vec<Tile>,
    /// The area the tiles were laid out in, and its columns.
    area: Rect,
    cols: usize,
    save: Option<SaveDialog>,
    toast: Option<(String, std::time::Instant)>,
    /// Said on the grid's status line until the next key.
    note: Option<String>,
    /// What was last drawn; the terminal is cleared when that changes, so
    /// the images of one screen never linger under the next.
    drawn: Option<Screen>,
    /// Figures saved here, this session and before, shown on the left.
    gallery: Gallery,
    show_saved: bool,
    /// Thumbnails ready to draw, by thumbnail file, at `thumb_rows` rows;
    /// `None` for one that could not be read, so it is not read again.
    thumbs: std::collections::HashMap<std::path::PathBuf, Option<Protocol>>,
    thumb_rows: u16,
    /// The first save the strip shows, as it scrolls.
    saved_top: usize,
    /// The save chosen in the strip, while keys go to it.
    saved_at: Option<usize>,
    /// Moving a save: where to, as typed.
    moving: Option<MoveDialog>,
    quit: bool,
}

impl Deck {
    /// The views `apps`, beside the figures saved in `gallery`.
    pub(super) fn new(apps: Vec<App>, picker: Picker, gallery: Gallery) -> Self {
        let f = picker.font_size();
        Self {
            apps,
            at: 0,
            cell: (f32::from(f.width.max(1)), f32::from(f.height.max(1))),
            picker,
            grid: false,
            hover: 0,
            tiles: Vec::new(),
            area: Rect::default(),
            cols: 1,
            save: None,
            toast: None,
            note: None,
            drawn: None,
            gallery,
            show_saved: true,
            thumbs: Default::default(),
            thumb_rows: 0,
            saved_top: 0,
            saved_at: None,
            moving: None,
            quit: false,
        }
    }

    pub(super) fn run(mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        if self.apps.len() > 1 {
            self.apps[0].message = Some(format!(
                "{} runs open · w shows them all · {{ }} step between them",
                self.apps.len()
            ));
        }
        let mut dirty = true;
        while !self.quit {
            let size = terminal.size()?;
            let full = Rect::new(0, 0, size.width, size.height);
            let area = self.content(full);
            self.prepare_thumbs(full);
            if self.grid {
                self.layout_grid(area);
            } else {
                self.apps[self.at].layout(area);
            }
            dirty |= self.background();
            dirty |= if self.grid {
                self.advance_tiles()?
            } else {
                self.apps[self.at].advance()?
            };
            if expired(&self.toast) {
                self.toast = None;
                dirty = true;
            }
            if std::mem::take(&mut dirty) {
                self.draw(terminal)?;
            }
            if event::poll(self.wait())? {
                // Drain everything queued so a burst of scroll events costs
                // one re-render, not one per event.
                loop {
                    let ev = event::read()?;
                    // A slow answer (a run read again, a model's tables
                    // read on the first click) says it is loading.
                    let (changed, spun) = if self.save.is_some() {
                        (self.event(terminal, ev), false)
                    } else {
                        crate::tui::busy::during("loading", || self.event(terminal, ev))
                    };
                    if spun {
                        self.drawn = None;
                    }
                    dirty |= changed? || spun;
                    if self.quit || !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Every view's background work. A view on screen that changed needs
    /// drawing; a view in the grid that changed needs its tile again.
    /// Returns whether the screen needs drawing.
    fn background(&mut self) -> bool {
        let mut dirty = false;
        for (k, app) in self.apps.iter_mut().enumerate() {
            let on_screen = !self.grid && k == self.at;
            if !app.background(on_screen) {
                continue;
            }
            if on_screen {
                dirty = true;
            } else if let Some(t) = self.tiles.get_mut(k).filter(|_| self.grid) {
                t.job = None;
                t.proto = None;
            }
        }
        dirty
    }

    /// How long to wait for input before the next round of work.
    fn wait(&self) -> Duration {
        if self.save.is_some() {
            Duration::from_millis(250)
        } else if self.grid {
            if self.tiles.iter().any(|t| t.job.is_some()) {
                Duration::ZERO
            } else if self.toast.is_some() {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(250)
            }
        } else {
            self.apps[self.at].wait()
        }
    }

    fn screen(&self) -> Screen {
        if self.grid {
            Screen::Grid
        } else {
            Screen::View(self.at)
        }
    }

    fn draw(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let screen = self.screen();
        if self.drawn != Some(screen) {
            terminal.clear()?;
            self.drawn = Some(screen);
        }
        terminal.draw(|f| {
            match screen {
                Screen::Grid => self.draw_grid(f),
                Screen::View(k) => self.apps[k].draw(f),
            }
            if let Some(strip) = self.strip(f.area()) {
                self.draw_saved(f, strip);
            }
            if let Some(d) = &self.save {
                self.draw_save(f, d);
            }
            if let Some(m) = &self.moving {
                draw_move(f, m);
            }
        })?;
        Ok(())
    }

    /// One terminal event, to the save dialog, the grid, the deck's own
    /// keys or the view on screen. Returns whether the screen needs drawing.
    fn event(&mut self, terminal: &mut DefaultTerminal, ev: Event) -> anyhow::Result<bool> {
        if let Event::Resize(..) = ev {
            self.drawn = None;
            self.area = Rect::default();
        }
        if self.save.is_some() {
            if let Event::Key(k) = ev {
                if k.kind != KeyEventKind::Release {
                    self.save_event(terminal, k)?;
                }
            }
            return Ok(true);
        }
        if self.moving.is_some() {
            if let Event::Key(k) = ev {
                if k.kind != KeyEventKind::Release {
                    self.move_key(k);
                }
            }
            return Ok(true);
        }
        let size = terminal.size()?;
        let full = Rect::new(0, 0, size.width, size.height);
        if self.strip(full).is_none() {
            // Hidden, or too narrow a screen to show it.
            self.saved_at = None;
        }
        if let Event::Mouse(m) = ev {
            if let Some(changed) = self.strip_mouse(full, m) {
                return Ok(changed);
            }
            if matches!(m.kind, MouseEventKind::Down(_)) {
                // A click elsewhere hands the keys back.
                self.saved_at = None;
            }
        }
        if self.saved_at.is_some() {
            if let Event::Key(k) = ev {
                if k.kind != KeyEventKind::Release {
                    self.strip_key(full, k);
                }
                return Ok(true);
            }
        }
        if self.grid {
            return Ok(self.grid_event(ev));
        }
        if let Event::Key(k) = ev {
            if k.kind != KeyEventKind::Release && self.deck_key(k) {
                return Ok(true);
            }
        }
        let app = &mut self.apps[self.at];
        let changed = app.handle(ev);
        self.quit = app.quit;
        Ok(changed)
    }

    /// The deck's keys while a view is on screen, unless the view is
    /// typing into a box or a menu. Returns whether the key was one.
    fn deck_key(&mut self, k: KeyEvent) -> bool {
        let n = self.apps.len();
        let app = &mut self.apps[self.at];
        if app.modal.is_some() || app.menu.is_some() {
            return false;
        }
        let several = |app: &mut App| {
            if n == 1 {
                app.message = Some(
                    "one view open · d copies it beside this one · several -f runs open side by side"
                        .into(),
                );
            }
            n > 1
        };
        match k.code {
            KeyCode::Char('l') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                self.drawn = None;
                app.refresh();
            }
            KeyCode::Char('w') if several(app) => {
                self.grid = true;
                self.hover = self.at;
                self.note = None;
                // The view just left may have changed; the others kept theirs.
                if let Some(t) = self.tiles.get_mut(self.at) {
                    t.job = None;
                    t.proto = None;
                }
            }
            KeyCode::Char('{' | '}') if several(app) => {
                let step = if k.code == KeyCode::Char('{') { -1 } else { 1 };
                self.at = (self.at as isize + step).rem_euclid(n as isize) as usize;
                self.announce();
            }
            KeyCode::Char('w' | '{' | '}') => {}
            KeyCode::Char('d') => {
                self.duplicate(self.at);
                self.at += 1;
                self.grid = true;
                self.note = None;
            }
            KeyCode::Char('s') => self.save = Some(self.new_save_dialog(Scope::View)),
            KeyCode::Char('f') => self.toggle_saved(),
            KeyCode::Char('F') => self.focus_saved(),
            _ => return false,
        }
        self.apps[self.at].help = false;
        true
    }

    /// Say which session just came on screen.
    fn announce(&mut self) {
        let n = self.apps.len();
        let title = self.title(self.at);
        let app = &mut self.apps[self.at];
        app.message = Some(format!(
            "view {}/{n} · {title} · {}",
            self.at + 1,
            app.scene.caption()
        ));
    }

    /// A session's name: its run's file, numbered when the run is open more
    /// than once.
    fn title(&self, k: usize) -> String {
        let from = &self.apps[k].from;
        let name = files::name(from);
        let copies = self.apps.iter().filter(|a| &a.from == from).count();
        if copies > 1 {
            let nth = self.apps[..k].iter().filter(|a| &a.from == from).count() + 1;
            format!("{name} ({nth})")
        } else {
            name
        }
    }

    /// Put a copy of session `k` right after it: same view, same camera,
    /// its own state from here on.
    fn duplicate(&mut self, k: usize) {
        let app = &self.apps[k];
        let mut copy = App::new(
            app.scene.duplicate(),
            self.picker.clone(),
            app.from.clone(),
            app.lupin.clone(),
        );
        copy.vp = app.vp;
        copy.sidebar = app.sidebar;
        copy.message = None;
        self.apps.insert(k + 1, copy);
        self.hover = k + 1;
        // Every tile moves: lay them out again.
        self.tiles.clear();
        self.drawn = None;
    }

    /// Close session `k`, unless it is the last or lupin is working for it.
    /// Returns what to say when it stays.
    fn close(&mut self, k: usize) -> Option<String> {
        if self.apps.len() == 1 {
            return Some("the last view stays open · q quits".into());
        }
        if self.apps[k].relabeling.is_some() {
            return Some("lupin is working for this view · close it once it answers".into());
        }
        self.apps.remove(k);
        self.tiles.clear();
        self.drawn = None;
        if self.at > k || self.at == self.apps.len() {
            self.at -= 1;
        }
        self.hover = self.hover.min(self.apps.len() - 1);
        None
    }

    fn open(&mut self, k: usize) {
        if k < self.apps.len() {
            self.at = k;
            self.grid = false;
            self.announce();
        }
    }

    /// Lay the tiles out in `area` (less the status lines), starting a
    /// picture for any tile without one.
    fn layout_grid(&mut self, area: Rect) {
        let body = Rect {
            height: area.height.saturating_sub(STATUS_LINES),
            ..area
        };
        let n = self.apps.len();
        if body != self.area || self.tiles.len() != n {
            self.area = body;
            let (cols, rows) = deck::shape(
                n,
                f32::from(body.width) * self.cell.0,
                f32::from(body.height) * self.cell.1,
            );
            self.cols = cols;
            let cut =
                |at: u16, len: u16, k: usize, of: usize| at + (usize::from(len) * k / of) as u16;
            self.tiles = (0..n)
                .map(|k| {
                    let (c, r) = (k % cols, k / cols);
                    let x = cut(body.x, body.width, c, cols);
                    let y = cut(body.y, body.height, r, rows);
                    Tile {
                        rect: Rect::new(
                            x,
                            y,
                            cut(body.x, body.width, c + 1, cols) - x,
                            cut(body.y, body.height, r + 1, rows) - y,
                        ),
                        job: None,
                        proto: None,
                    }
                })
                .collect();
        }
        for (tile, app) in self.tiles.iter_mut().zip(&self.apps) {
            if tile.job.is_some() || tile.proto.is_some() {
                continue;
            }
            let inner = Block::bordered().inner(tile.rect);
            let (w, h) = (
                (f32::from(inner.width) * self.cell.0) as usize,
                (f32::from(inner.height) * self.cell.1) as usize,
            );
            if w > 0 && h > 0 {
                // A chart is drawn whole, at once.
                if let Some(frame) = app.scene.chart_frame(w, h, self.cell.1, false) {
                    tile.proto = protocol(&self.picker, frame.to_image(), inner).ok();
                    continue;
                }
                let vp = deck::camera(app.vp, app.scene.current().points.bounds, w, h);
                tile.job = Some(Job::new(vp, &app.scene.layers()));
            }
        }
    }

    /// Draw one chunk of every tile still drawing, the tiles side by side
    /// on rayon's threads. Returns whether any picture changed.
    fn advance_tiles(&mut self) -> anyhow::Result<bool> {
        if self.tiles.iter().all(|t| t.job.is_none()) {
            return Ok(false);
        }
        let layers: Vec<Option<Vec<crate::view::render::Paint<'_>>>> = self
            .tiles
            .iter()
            .zip(&self.apps)
            .map(|(t, a)| t.job.as_ref().map(|_| a.scene.layers()))
            .collect();
        let done: Vec<bool> = self
            .tiles
            .par_iter_mut()
            .zip(&layers)
            .map(|(t, l)| match (t.job.as_mut(), l) {
                (Some(job), Some(l)) => job.step(l),
                _ => false,
            })
            .collect();
        drop(layers);
        // Labels go on finished tiles here: a scene keeps caches for them.
        let mut images = Vec::new();
        for (k, (tile, app)) in self.tiles.iter_mut().zip(&self.apps).enumerate() {
            let Some(job) = tile.job.as_ref() else {
                continue;
            };
            let img = if done[k] {
                let job = tile.job.take().expect("checked above");
                let vp = job.vp;
                let mut frame = job.finish();
                app.scene.decorate(&mut frame, &vp, self.cell.1);
                frame.to_image()
            } else {
                job.image()
            };
            images.push((k, img));
        }
        let picker = &self.picker;
        let tiles = &self.tiles;
        let protos: Vec<(usize, Protocol)> = images
            .into_par_iter()
            .map(|(k, img)| {
                let inner = Block::bordered().inner(tiles[k].rect);
                Ok((k, protocol(picker, img, inner)?))
            })
            .collect::<anyhow::Result<_>>()?;
        for (k, p) in protos {
            self.tiles[k].proto = Some(p);
        }
        Ok(true)
    }

    fn tile_at(&self, col: u16, row: u16) -> Option<usize> {
        let at = ratatui::layout::Position::new(col, row);
        self.tiles.iter().position(|t| t.rect.contains(at))
    }

    /// One event while the grid shows. Returns whether the screen changed.
    fn grid_event(&mut self, ev: Event) -> bool {
        let n = self.apps.len();
        match ev {
            Event::Mouse(m) => match m.kind {
                MouseEventKind::Moved | MouseEventKind::Drag(_) => {
                    match self.tile_at(m.column, m.row) {
                        Some(k) if k != self.hover => {
                            self.hover = k;
                            true
                        }
                        _ => false,
                    }
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    if let Some(k) = self.tile_at(m.column, m.row) {
                        self.open(k);
                    }
                    true
                }
                _ => false,
            },
            Event::Key(k) if k.kind != KeyEventKind::Release => {
                let (cols, h) = (self.cols, self.hover);
                self.note = None;
                match k.code {
                    KeyCode::Char('q') => self.quit = true,
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.quit = true;
                    }
                    KeyCode::Esc | KeyCode::Char('w') => self.open(self.at),
                    KeyCode::Enter | KeyCode::Char(' ') => self.open(h),
                    KeyCode::Char(d @ '1'..='9') => self.open(d as usize - '1' as usize),
                    KeyCode::Left | KeyCode::Char('h') => self.hover = h.saturating_sub(1),
                    KeyCode::Right | KeyCode::Char('l') => self.hover = (h + 1).min(n - 1),
                    KeyCode::Up | KeyCode::Char('k') => self.hover = h.saturating_sub(cols),
                    KeyCode::Down | KeyCode::Char('j') => {
                        if h + cols < n {
                            self.hover = h + cols;
                        }
                    }
                    KeyCode::Tab => self.hover = (h + 1) % n,
                    KeyCode::BackTab => self.hover = (h + n - 1) % n,
                    KeyCode::Char('s') => self.save = Some(self.new_save_dialog(Scope::Grid)),
                    KeyCode::Char('f') => self.toggle_saved(),
                    KeyCode::Char('F') => self.focus_saved(),
                    KeyCode::Char('d') => self.duplicate(h),
                    KeyCode::Char('X') | KeyCode::Delete => self.note = self.close(h),
                    KeyCode::Char('x') => {
                        self.note =
                            Some("X closes this view (shift, as it cannot be undone)".into());
                    }
                    _ => return false,
                }
                true
            }
            Event::Resize(..) => {
                self.area = Rect::default();
                true
            }
            _ => false,
        }
    }

    fn draw_grid(&self, f: &mut ratatui::Frame) {
        let page = page();
        f.render_widget(Block::default().style(page), f.area());
        let rule = Style::default().fg(rgb(color::MUTED));
        let bold = ratatui::style::Modifier::BOLD;
        for (k, (tile, app)) in self.tiles.iter().zip(&self.apps).enumerate() {
            let hovered = k == self.hover;
            let mark = if k == self.at { " ●" } else { "" };
            let title = format!(" {} {}{mark} ", k + 1, self.title(k));
            let mut block = Block::bordered()
                .title(Span::styled(
                    title,
                    if hovered {
                        Style::default().add_modifier(bold)
                    } else {
                        Style::default()
                    },
                ))
                .title_bottom(Line::from(Span::styled(
                    format!(" {} ", app.scene.current().method),
                    hint(),
                )))
                .style(page);
            block = if hovered {
                block
                    .border_type(ratatui::widgets::BorderType::Thick)
                    .border_style(Style::default().fg(rgb(color::TEXT)))
            } else {
                block.border_style(rule)
            };
            let inner = block.inner(tile.rect);
            f.render_widget(block, tile.rect);
            if let Some(p) = &tile.proto {
                f.render_widget(Image::new(p), inner);
            }
        }

        let [_, status] = Layout::vertical([Constraint::Min(1), Constraint::Length(STATUS_LINES)])
            .areas(f.area());
        let n = self.apps.len();
        let first = match &self.note {
            Some(note) => note.clone(),
            None => self.apps.get(self.hover).map_or_else(String::new, |a| {
                format!(
                    "{}/{n} · {} · {}",
                    self.hover + 1,
                    self.title(self.hover),
                    a.scene.caption()
                )
            }),
        };
        let busy = if self.tiles.iter().any(|t| t.job.is_some()) {
            "   · drawing…"
        } else {
            ""
        };
        let back = self.title(self.at.min(n - 1));
        let lines = vec![
            Line::from(format!(" {first}{busy}")),
            Line::from(" point or ← → ↑ ↓ to choose a view   click or enter opens it   1–9 by number   d copy it   X close it"),
            Line::from(Span::styled(
                format!(" w esc back to {back}   s save as PDF (this grid, or one page per run)   q quit"),
                hint(),
            )),
        ];
        f.render_widget(Paragraph::new(lines).style(page), status);
        if let Some((text, _)) = &self.toast {
            toast(f, f.area(), text);
        }
    }
}

/// Columns of the saved-figures strip on the left.
const SAVED_WIDTH: u16 = 28;

impl Deck {
    /// The strip of saved figures, when it shows: on the left of the body.
    fn strip(&self, full: Rect) -> Option<Rect> {
        let shows =
            self.show_saved && !self.gallery.entries().is_empty() && full.width >= 3 * SAVED_WIDTH;
        shows.then(|| Rect::new(0, 0, SAVED_WIDTH, full.height.saturating_sub(STATUS_LINES)))
    }

    /// Where the views or the grid go: beside the strip, when it shows.
    fn content(&self, full: Rect) -> Rect {
        match self.strip(full) {
            Some(s) => Rect {
                x: full.x + s.width,
                width: full.width - s.width,
                ..full
            },
            None => full,
        }
    }

    fn toggle_saved(&mut self) {
        self.show_saved = !self.show_saved;
        if !self.show_saved {
            self.saved_at = None;
        }
        self.relayout();
    }

    /// Everything moves over, or the strip's pictures do: lay out and draw
    /// afresh, so no image lingers where it was.
    fn relayout(&mut self) {
        self.area = Rect::default();
        self.drawn = None;
    }

    /// Keys go to the strip, on the save at its top: it shows if hidden.
    fn focus_saved(&mut self) {
        if self.gallery.entries().is_empty() {
            self.say("no figures saved here yet · s saves one".into());
            return;
        }
        if !self.show_saved {
            self.show_saved = true;
            self.relayout();
        }
        self.saved_at = Some(self.saved_top.min(self.gallery.entries().len() - 1));
    }

    /// Say `msg` where the screen shows it: a toast over the grid, else the
    /// view's own.
    fn say(&mut self, msg: String) {
        if self.grid {
            self.toast = Some((msg, std::time::Instant::now() + TOAST_FOR));
        } else {
            self.apps[self.at].pop(msg);
        }
    }

    /// Rows a thumbnail takes in the strip, at the typical 4:3 of a map.
    fn rows_per_thumb(&self) -> u16 {
        let px_wide = f32::from(SAVED_WIDTH - 2) * self.cell.0;
        (px_wide * 0.75 / self.cell.1).round().clamp(3.0, 12.0) as u16
    }

    /// How many saves the strip shows at once on a `full` screen.
    fn saved_shown(&self, full: Rect) -> usize {
        self.strip(full).map_or(0, |strip| {
            thumbs_that_fit(
                self.saved_block().inner(strip).height,
                self.rows_per_thumb(),
            )
        })
    }

    /// The furthest the strip scrolls, showing `shown` saves at once.
    fn last_top(&self, shown: usize) -> usize {
        self.gallery.entries().len().saturating_sub(shown.max(1))
    }

    /// Scroll so the strip starts at `top`, kept in range. Returns whether
    /// it moved.
    fn scroll_saved(&mut self, top: usize, shown: usize) -> bool {
        let top = top.min(self.last_top(shown));
        if top == self.saved_top {
            return false;
        }
        self.saved_top = top;
        self.drawn = None;
        true
    }

    /// Choose save `k`, scrolling the strip to show it.
    fn choose_saved(&mut self, k: usize, shown: usize) {
        let n = self.gallery.entries().len();
        if n == 0 {
            self.saved_at = None;
            return;
        }
        let k = k.min(n - 1);
        self.saved_at = Some(k);
        let shown = shown.max(1);
        if k < self.saved_top {
            self.scroll_saved(k, shown);
        } else if k >= self.saved_top + shown {
            self.scroll_saved(k + 1 - shown, shown);
        }
    }

    /// The save drawn at screen row `row` of the strip, if any.
    fn saved_at_row(&self, full: Rect, row: u16) -> Option<usize> {
        let inner = self.saved_block().inner(self.strip(full)?);
        let each = self.rows_per_thumb() + 3;
        let k = usize::from(row.checked_sub(inner.y + 1)? / each);
        (k < self.saved_shown(full))
            .then_some(self.saved_top + k)
            .filter(|&k| k < self.gallery.entries().len())
    }

    /// The mouse over the strip: the wheel scrolls it, a click chooses a
    /// save. `None` when the pointer is not on the strip.
    fn strip_mouse(
        &mut self,
        full: Rect,
        m: ratatui::crossterm::event::MouseEvent,
    ) -> Option<bool> {
        let strip = self.strip(full)?;
        if !strip.contains(ratatui::layout::Position::new(m.column, m.row)) {
            return None;
        }
        let shown = self.saved_shown(full);
        Some(match m.kind {
            MouseEventKind::ScrollUp => self.scroll_saved(self.saved_top.saturating_sub(1), shown),
            MouseEventKind::ScrollDown => self.scroll_saved(self.saved_top + 1, shown),
            MouseEventKind::Down(MouseButton::Left) => match self.saved_at_row(full, m.row) {
                Some(k) => {
                    self.choose_saved(k, shown);
                    true
                }
                None => false,
            },
            _ => false,
        })
    }

    /// A key while the strip has them: choose, move or remove a save.
    fn strip_key(&mut self, full: Rect, k: KeyEvent) {
        let Some(at) = self.saved_at else { return };
        let shown = self.saved_shown(full);
        let page = shown.max(1);
        let Some(e) = self.gallery.entries().get(at).cloned() else {
            self.saved_at = None;
            return;
        };
        match k.code {
            KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                self.quit = true;
            }
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Esc | KeyCode::Char('F') | KeyCode::Tab => self.saved_at = None,
            KeyCode::Up | KeyCode::Char('k') => self.choose_saved(at.saturating_sub(1), shown),
            KeyCode::Down | KeyCode::Char('j') => self.choose_saved(at + 1, shown),
            KeyCode::PageUp => self.choose_saved(at.saturating_sub(page), shown),
            KeyCode::PageDown => self.choose_saved(at + page, shown),
            KeyCode::Home => self.choose_saved(0, shown),
            KeyCode::End => self.choose_saved(usize::MAX, shown),
            KeyCode::Char('f') => self.toggle_saved(),
            KeyCode::Char('m') | KeyCode::Enter => {
                self.moving = Some(MoveDialog {
                    from: e.path.clone(),
                    to: crate::tui::shown(&e.path),
                    status: None,
                });
            }
            KeyCode::Char('x' | 'X') | KeyCode::Delete => {
                let delete = k.code == KeyCode::Char('X');
                let msg = match self.gallery.remove(&e.path, delete) {
                    Ok(()) if delete => format!("deleted {}", e.path.display()),
                    Ok(()) => format!(
                        "{} taken off the list · the PDF stays (X deletes it too)",
                        e.name()
                    ),
                    Err(e) => format!("not removed: {e}"),
                };
                self.say(msg);
                self.after_change(at, shown);
            }
            _ => {}
        }
    }

    /// The list changed under the strip: keep the choice in range. The
    /// views are laid out afresh only when the strip goes with the last
    /// save; else the strip alone is drawn again.
    fn after_change(&mut self, at: usize, shown: usize) {
        self.saved_top = self.saved_top.min(self.last_top(shown));
        self.choose_saved(at, shown);
        if self.gallery.entries().is_empty() {
            self.relayout();
        } else {
            self.drawn = None;
        }
    }

    /// A key in the move popup: type where to, enter moves.
    fn move_key(&mut self, k: KeyEvent) {
        let Some(m) = self.moving.as_mut() else {
            return;
        };
        m.status = None;
        if crate::tui::edit_line(&mut m.to, &k) {
            return;
        }
        match k.code {
            KeyCode::Esc => self.moving = None,
            KeyCode::Enter => {
                let to = crate::tui::home(m.to.trim());
                if to.is_empty() {
                    m.status = Some("type where to".into());
                    return;
                }
                let to = std::path::PathBuf::from(to);
                let from = m.from.clone();
                match self.gallery.relocate(&from, &to) {
                    Ok(now) => {
                        self.moving = None;
                        self.say(format!("moved to {}", crate::tui::shown(&now)));
                        self.drawn = None;
                    }
                    Err(e) => {
                        if let Some(m) = self.moving.as_mut() {
                            m.status = Some(format!("not moved: {e}"));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Thumbnail images for the saves the strip shows, built once each
    /// (again when their size in rows changes); none while it is hidden.
    fn prepare_thumbs(&mut self, full: Rect) {
        if self.strip(full).is_none() {
            return;
        }
        let rows = self.rows_per_thumb();
        if rows != self.thumb_rows {
            self.thumbs.clear();
            self.thumb_rows = rows;
        }
        let shown = self.saved_shown(full);
        // A smaller list, or a taller screen, may leave the top past the end.
        self.saved_top = self.saved_top.min(self.last_top(shown));
        let area = Rect::new(0, 0, SAVED_WIDTH - 2, rows);
        for e in self
            .gallery
            .entries()
            .iter()
            .skip(self.saved_top)
            .take(shown)
        {
            let path = self.gallery.thumb(e);
            if !self.thumbs.contains_key(&path) {
                let p = image::open(&path)
                    .ok()
                    .and_then(|img| protocol(&self.picker, img.to_rgba8(), area).ok());
                self.thumbs.insert(path, p);
            }
        }
    }

    /// The strip's frame: a rule on the map side, its title, and its keys
    /// along the bottom.
    fn saved_block(&self) -> Block<'static> {
        let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
        let (keys, style) = if self.saved_at.is_some() {
            (" ↑↓ m move x drop esc ", bold)
        } else {
            (" F choose · wheel scrolls ", hint())
        };
        Block::new()
            .borders(ratatui::widgets::Borders::RIGHT | ratatui::widgets::Borders::BOTTOM)
            .border_style(Style::default().fg(rgb(color::MUTED)))
            .title(Span::styled(
                format!(" saved ({}) · f hides ", self.gallery.entries().len()),
                bold,
            ))
            .title_bottom(Span::styled(keys, style))
            .style(page())
    }

    /// The saved figures, newest first: thumbnail, file name, how long ago
    /// and what was saved; as many as fit from the top of the scroll, and
    /// how many more lie above and below.
    fn draw_saved(&self, f: &mut ratatui::Frame, strip: Rect) {
        let block = self.saved_block();
        let inner = block.inner(strip);
        f.render_widget(block, strip);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let rows = self.thumb_rows;
        let shown = thumbs_that_fit(inner.height, rows);
        let n = self.gallery.entries().len();
        let mut y = inner.y + 1;
        for (k, e) in self
            .gallery
            .entries()
            .iter()
            .enumerate()
            .skip(self.saved_top)
            .take(shown)
        {
            let pic = Rect::new(inner.x + 1, y, inner.width.saturating_sub(1), rows);
            if let Some(Some(p)) = self.thumbs.get(&self.gallery.thumb(e)) {
                f.render_widget(Image::new(p), pic);
            }
            let chosen = self.saved_at == Some(k);
            let text = vec![
                Line::from(Span::styled(
                    format!("{}{}", if chosen { "▸" } else { " " }, e.name()),
                    if chosen {
                        selected()
                    } else {
                        Style::default().add_modifier(ratatui::style::Modifier::BOLD)
                    },
                )),
                Line::from(Span::styled(
                    format!(" {} · {}", ago(e.when, now), e.what),
                    hint(),
                )),
            ];
            f.render_widget(
                Paragraph::new(text).style(page()),
                Rect::new(inner.x, y + rows, inner.width, 2),
            );
            y += rows + 3;
        }
        // Above and below what fits, in the top row and under the last.
        let mut more = |text: String, y: u16| {
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(text, hint()))).style(page()),
                Rect::new(inner.x, y, inner.width, 1),
            );
        };
        if self.saved_top > 0 {
            more(format!(" ↑ {} more", self.saved_top), inner.y);
        }
        let below = n.saturating_sub(self.saved_top + shown);
        if below > 0 && y < inner.y + inner.height {
            more(format!(" ↓ {below} more"), y.saturating_sub(1));
        }
    }
}

/// Moving a saved PDF: from where, to where as typed, and what went wrong.
struct MoveDialog {
    from: std::path::PathBuf,
    to: String,
    status: Option<String>,
}

fn draw_move(f: &mut ratatui::Frame, m: &MoveDialog) {
    let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
    let lines = vec![
        Line::from(Span::styled(
            format!(" Move {}", files::name(&m.from)),
            bold,
        )),
        Line::from(Span::styled(
            format!(" now {}", crate::tui::shown(&m.from)),
            hint(),
        )),
        Line::from(""),
        Line::from(Span::styled(format!(" to  {}▏", m.to), selected())),
        Line::from(""),
        match &m.status {
            Some(s) => Line::from(Span::styled(format!(" {s}"), bold)),
            None => Line::from(Span::styled(
                " a folder (ending in /) keeps the name · folders are made as needed",
                hint(),
            )),
        },
        Line::from(" type the path (ctrl-u clears)   enter move   esc cancel"),
    ];
    popup(f, f.area(), lines, 76, At::Middle, color::TEXT);
}

/// How many saves the strip shows in `height` rows, each a thumbnail
/// `rows` tall, two lines of text and a gap, below one row of space.
fn thumbs_that_fit(height: u16, rows: u16) -> usize {
    let each = rows + 3;
    match height.checked_sub(each) {
        Some(rest) => usize::from(rest / each) + 1,
        None => 0,
    }
}

/// `stem`, or `stem-2`, `stem-3`, … : the first whose PDF does not exist yet.
fn free_stem(stem: &str) -> String {
    let taken = |s: &str| std::path::Path::new(&format!("{s}.pdf")).exists();
    if !taken(stem) {
        return stem.to_string();
    }
    (2..)
        .map(|k| format!("{stem}-{k}"))
        .find(|s| !taken(s))
        .expect("some number is free")
}

/// What a PDF takes in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// The session on screen.
    View,
    /// Every session, one grid on one page.
    Grid,
    /// Every session, one page each.
    Pages,
}

impl Scope {
    const ALL: [Scope; 3] = [Scope::View, Scope::Grid, Scope::Pages];

    fn name(self, n: usize) -> String {
        match self {
            Scope::View => "this view".into(),
            Scope::Grid => format!("all {n} runs, one grid"),
            Scope::Pages => format!("all {n} runs, one page each"),
        }
    }
}

/// Page widths offered, in inches: journal single and double column, then
/// larger.
const WIDTHS: [f32; 5] = [3.5, 5.0, 7.0, 10.0, 14.0];
/// Raster resolutions offered for the points.
const DPIS: [u32; 4] = [150, 300, 600, 1200];
const SAVE_FIELDS: [&str; 4] = ["what", "width", "dpi", "file"];

/// The save popup: what, how large, how fine, and where.
struct SaveDialog {
    scope: Scope,
    width: usize,
    dpi: usize,
    name: String,
    /// The name was typed, so a change of scope keeps it.
    typed: bool,
    field: usize,
    /// Screen size in pixels of what each scope pictures, for its aspect
    /// and to scale text: the view on screen, and the grid's area.
    view_px: (f32, f32),
    grid_px: (f32, f32),
    status: Option<String>,
    /// Enter was pressed once on a name that exists: the next one replaces it.
    replace: bool,
}

impl SaveDialog {
    /// Whether enter may write now. A file that exists is only replaced on
    /// a second enter, asked for here; a new name writes at once.
    fn ready(&mut self) -> bool {
        let path = self.path();
        if !path.exists() || self.replace {
            return true;
        }
        self.replace = true;
        self.status = Some(format!(
            "{} exists · enter again replaces it, or rename",
            path.display()
        ));
        false
    }

    /// Anything changed in the dialog: a replace has to be asked for again.
    fn changed(&mut self) {
        self.replace = false;
    }

    fn width_px(&self) -> usize {
        (WIDTHS[self.width] * DPIS[self.dpi] as f32).round() as usize
    }

    /// The picture's pixel size (the first page's, one page per run).
    fn size_px(&self) -> (usize, usize) {
        let (w, h) = match self.scope {
            Scope::Grid => self.grid_px,
            _ => self.view_px,
        };
        let wp = self.width_px();
        (wp, (wp as f32 * h / w.max(1.0)).round() as usize)
    }

    fn path(&self) -> std::path::PathBuf {
        files::with_pdf(&self.name)
    }
}

impl Deck {
    fn new_save_dialog(&self, scope: Scope) -> SaveDialog {
        let view_px = self.apps[self.at].view_px();
        let grid_px = if self.area.width > 0 {
            (
                f32::from(self.area.width) * self.cell.0,
                f32::from(self.area.height) * self.cell.1,
            )
        } else {
            view_px
        };
        let mut d = SaveDialog {
            scope: if self.apps.len() > 1 {
                scope
            } else {
                Scope::View
            },
            width: 2,
            dpi: 1,
            name: String::new(),
            typed: false,
            field: 0,
            view_px,
            grid_px,
            status: None,
            replace: false,
        };
        d.name = self.default_name(d.scope);
        d
    }

    /// The name proposed for `scope`, numbered past any file it would
    /// replace.
    fn default_name(&self, scope: Scope) -> String {
        let app = &self.apps[self.at];
        let first = here(&self.apps[0].scene.data.prefix);
        free_stem(&match scope {
            Scope::View => app.pdf_name(),
            Scope::Grid => format!("{first}.view.grid"),
            Scope::Pages => format!("{first}.view.pages"),
        })
    }

    /// A key in the save dialog: enter renders and writes the PDF, the rest
    /// edit the dialog.
    fn save_event(&mut self, terminal: &mut DefaultTerminal, k: KeyEvent) -> anyhow::Result<()> {
        if k.code == KeyCode::Enter {
            // An existing file is only replaced on a second enter.
            if !self.save.as_mut().is_some_and(SaveDialog::ready) {
                return Ok(());
            }
            if let Some(d) = self.save.as_mut() {
                d.status = Some("saving…".into());
            }
            self.draw(terminal)?;
            let d = self.save.take().expect("the dialog is open");
            match self.write_pdf(&d) {
                Ok(msg) => self.say(msg),
                Err(e) => {
                    self.save = Some(SaveDialog {
                        status: Some(format!("not saved: {e}")),
                        ..d
                    });
                }
            }
        } else {
            self.save_key(k);
        }
        if self.save.is_none() {
            // The dialog covered part of the picture, and a first save opens
            // the saved-figures strip: draw it all again.
            self.drawn = None;
        }
        Ok(())
    }

    fn save_key(&mut self, k: KeyEvent) {
        let n = self.apps.len();
        let Some(d) = self.save.as_mut() else { return };
        d.status = None;
        d.changed();
        let step: isize = match k.code {
            KeyCode::Left => -1,
            KeyCode::Right => 1,
            _ => 0,
        };
        let cycle = |at: usize, len: usize| (at as isize + step).rem_euclid(len as isize) as usize;
        // A new scope brings its own default name, unless one was typed.
        let mut rename = None;
        match k.code {
            KeyCode::Esc => self.save = None,
            KeyCode::Up | KeyCode::BackTab => d.field = (d.field + 3) % 4,
            KeyCode::Down | KeyCode::Tab => d.field = (d.field + 1) % 4,
            KeyCode::Left | KeyCode::Right => match d.field {
                0 if n > 1 => {
                    let at = Scope::ALL.iter().position(|&s| s == d.scope).unwrap_or(0);
                    d.scope = Scope::ALL[cycle(at, Scope::ALL.len())];
                    if !d.typed {
                        rename = Some(d.scope);
                    }
                }
                1 => d.width = cycle(d.width, WIDTHS.len()),
                2 => d.dpi = cycle(d.dpi, DPIS.len()),
                _ => {}
            },
            _ => {
                if crate::tui::edit_line(&mut d.name, &k) {
                    d.field = 3;
                    d.typed = true;
                }
            }
        }
        if let Some(scope) = rename {
            let name = self.default_name(scope);
            if let Some(d) = self.save.as_mut() {
                d.name = name;
            }
        }
    }

    fn draw_save(&self, f: &mut ratatui::Frame, d: &SaveDialog) {
        let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
        let n = self.apps.len();
        let values = [
            if n > 1 {
                format!("‹ {} ›", d.scope.name(n))
            } else {
                d.scope.name(n)
            },
            format!("‹ {} in ›", WIDTHS[d.width]),
            format!(
                "‹ {} ›  the points' resolution; text stays text",
                DPIS[d.dpi]
            ),
            format!(
                "{}▏{}",
                d.name,
                if files::has_pdf(&d.name) { "" } else { ".pdf" }
            ),
        ];
        let mut lines = vec![
            Line::from(Span::styled(" Save as PDF", bold)),
            Line::from(""),
        ];
        for (k, (field, value)) in SAVE_FIELDS.iter().zip(values).enumerate() {
            let here = k == d.field;
            let cursor = if here { " ▸ " } else { "   " };
            let style = if here { selected() } else { Style::default() };
            lines.push(Line::from(Span::styled(
                format!("{cursor}{field:<6} {value} "),
                style,
            )));
        }
        lines.push(Line::from(""));
        let (w, h) = d.size_px();
        let dpi = DPIS[d.dpi] as f32;
        let size = format!(
            " {w} × {h} px · {:.1} × {:.1} in{}",
            w as f32 / dpi,
            h as f32 / dpi,
            if d.scope == Scope::Pages {
                format!(" (first of {n} pages)")
            } else {
                String::new()
            }
        );
        lines.push(Line::from(size));
        let path = d.path();
        let note = match &d.status {
            Some(s) => Line::from(Span::styled(format!(" {s}"), bold)),
            None if d.name.trim().is_empty() => Line::from(Span::styled(" type a file name", bold)),
            None if w * h > deck::MAX_PIXELS => Line::from(Span::styled(
                " too large: choose a smaller width or dpi",
                bold,
            )),
            None if path.exists() => Line::from(Span::styled(
                format!(
                    " {} exists · enter twice replaces it, or rename",
                    path.display()
                ),
                bold,
            )),
            None => Line::from(""),
        };
        lines.push(note);
        lines.push(Line::from(
            " ↑ ↓ field   ← → change   type to rename (ctrl-u clears)   enter save   esc cancel",
        ));
        popup(f, f.area(), lines, 76, At::Middle, color::TEXT);
    }

    /// Render and write the PDF the dialog describes. Returns what to say.
    fn write_pdf(&mut self, d: &SaveDialog) -> Result<String, String> {
        let (w, h) = d.size_px();
        if d.name.trim().is_empty() {
            return Err("type a file name".into());
        }
        if w * h > deck::MAX_PIXELS {
            return Err("too large: choose a smaller width or dpi".into());
        }
        let dpi = DPIS[d.dpi];
        let view = |app| view_at(app, d.width_px(), self.cell.1);
        let pages = match d.scope {
            Scope::View => deck::view_pages(&[view(&self.apps[self.at])]),
            Scope::Pages => deck::view_pages(&self.apps.iter().map(view).collect::<Vec<_>>()),
            Scope::Grid => {
                let scenes: Vec<&Scene> = self.apps.iter().map(|a| &a.scene).collect();
                let cams: Vec<Option<Viewport>> = self.apps.iter().map(|a| a.vp).collect();
                let titles: Vec<String> = (0..self.apps.len()).map(|k| self.title(k)).collect();
                let cell = self.cell.1 * w as f32 / d.grid_px.0.max(1.0);
                vec![deck::render_grid(&scenes, &cams, &titles, (w, h), cell)]
            }
        };
        let path = d.path();
        pdf::write(&pages, dpi as f32, &path).map_err(|e| e.to_string())?;
        // Listed on the left with its first page as the thumbnail; failing
        // to list it never fails the save.
        let what = format!(
            "{} · {} in · {dpi} dpi",
            d.scope.name(self.apps.len()),
            WIDTHS[d.width]
        );
        let listed = match self.gallery.add(&path, &what, &pages[0].img) {
            Ok(()) => String::new(),
            Err(e) => format!(" · not listed: {e}"),
        };
        let pages = if pages.len() > 1 {
            format!("{} pages · ", pages.len())
        } else {
            String::new()
        };
        Ok(format!(
            "saved {} · {pages}{w} × {h} px at {dpi} dpi · text kept as text{listed}",
            path.display()
        ))
    }
}

/// A session's view for a page `wp` pixels wide: its own camera scaled to
/// that width at its aspect, and text scaled as much as the map is.
fn view_at(app: &App, wp: usize, cell_px: f32) -> (&Scene, Viewport, f32) {
    let (vw, vh) = app.view_px();
    let vw = vw.max(1.0);
    let hp = ((wp as f32 * vh / vw).round() as usize).max(1);
    let vp = deck::camera(app.vp, app.scene.current().points.bounds, wp, hp);
    (&app.scene, vp, cell_px * wp as f32 / vw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialog(name: &str) -> SaveDialog {
        SaveDialog {
            scope: Scope::View,
            width: 2,
            dpi: 1,
            name: name.into(),
            typed: false,
            field: 0,
            view_px: (800.0, 600.0),
            grid_px: (800.0, 600.0),
            status: None,
            replace: false,
        }
    }

    #[test]
    fn a_proposed_name_never_collides_with_a_saved_file() {
        let dir = tempfile::tempdir().unwrap();
        let stem = dir.path().join("run.view.umap.cells");
        let stem = stem.to_string_lossy();
        assert_eq!(free_stem(&stem), stem);
        std::fs::write(format!("{stem}.pdf"), b"%PDF").unwrap();
        assert_eq!(free_stem(&stem), format!("{stem}-2"));
        std::fs::write(format!("{stem}-2.pdf"), b"%PDF").unwrap();
        assert_eq!(free_stem(&stem), format!("{stem}-3"));
    }

    #[test]
    fn replacing_a_file_takes_a_second_enter() {
        let dir = tempfile::tempdir().unwrap();
        let taken = dir.path().join("fig.pdf");
        std::fs::write(&taken, b"%PDF").unwrap();

        // A new name saves at once.
        let mut d = dialog(&dir.path().join("new").to_string_lossy());
        assert!(d.ready());

        // An existing one asks first, then saves on the second enter.
        let mut d = dialog(&taken.to_string_lossy());
        assert!(!d.ready());
        assert!(d
            .status
            .as_deref()
            .unwrap()
            .contains("enter again replaces it"));
        assert!(d.ready());

        // Any change in between asks again.
        let mut d = dialog(&taken.to_string_lossy());
        assert!(!d.ready());
        d.changed();
        assert!(!d.ready());
    }

    #[test]
    fn the_strip_scrolls_to_the_chosen_save_and_drops_it() {
        let work = tempfile::tempdir().unwrap();
        let mut gallery = Gallery::open(&work.path().join(".senna-view"));
        let picture = image::RgbaImage::from_pixel(40, 30, image::Rgba([1, 2, 3, 255]));
        for k in 0..10 {
            let pdf = work.path().join(format!("{k}.pdf"));
            std::fs::write(&pdf, b"%PDF").unwrap();
            gallery.add(&pdf, "a view", &picture).unwrap();
        }
        let mut deck = Deck::new(Vec::new(), Picker::halfblocks(), gallery);
        // Over the grid, so what is said goes in a toast.
        deck.grid = true;
        let full = Rect::new(0, 0, 160, 50);
        let shown = deck.saved_shown(full);
        assert!((1..10).contains(&shown), "{shown} of 10 fit");
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);

        deck.focus_saved();
        assert_eq!(deck.saved_at, Some(0));
        // Down past the last one shown scrolls it into view.
        for _ in 0..shown {
            deck.strip_key(full, key(KeyCode::Down));
        }
        assert_eq!(deck.saved_at, Some(shown));
        assert_eq!(deck.saved_top, 1);
        deck.strip_key(full, key(KeyCode::End));
        assert_eq!(deck.saved_at, Some(9));
        assert_eq!(deck.saved_top, 10 - shown);
        // The wheel stops at either end.
        assert!(!deck.scroll_saved(99, shown));
        assert!(deck.scroll_saved(0, shown));

        // x takes the last off the list but keeps its PDF; the choice stays
        // in range.
        deck.strip_key(full, key(KeyCode::End));
        let last = deck.gallery.entries()[9].path.clone();
        deck.strip_key(full, key(KeyCode::Char('x')));
        assert_eq!(deck.gallery.entries().len(), 9);
        assert!(last.exists());
        assert_eq!(deck.saved_at, Some(8));
        // Esc hands the keys back.
        deck.strip_key(full, key(KeyCode::Esc));
        assert_eq!(deck.saved_at, None);
    }

    #[test]
    fn only_thumbnails_that_fit_the_strip_are_made() {
        // Each takes its rows, two lines of text and a gap; one row on top.
        assert_eq!(thumbs_that_fit(0, 6), 0);
        assert_eq!(thumbs_that_fit(8, 6), 0);
        assert_eq!(thumbs_that_fit(9, 6), 1);
        assert_eq!(thumbs_that_fit(17, 6), 1);
        assert_eq!(thumbs_that_fit(18, 6), 2);
        assert_eq!(thumbs_that_fit(40, 6), 4);
    }
}
