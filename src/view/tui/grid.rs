//! Several views open at once, and the one loop that drives them. One view
//! is on screen at a time; `w` shows them all in a grid, each tile the
//! view's current picture, where the mouse or the arrows choose one and a
//! click or enter opens it. Saving happens here too, since a PDF can take in
//! every view. Every view's background work (a zoom, lupin, a reloaded file)
//! goes on whichever is on screen.

use super::*;
use crate::view::deck;
use legume_tui::chrome::Chrome;
use legume_tui::gallery::strip::{self, Keymap, ProtocolPainter, Strip};
use legume_tui::gallery::{Gallery, Policy};
use legume_tui::graphics::Repick;
use legume_tui::look::Look;
use legume_tui::save::{Numbering, SaveDialog, SaveKey, SaveRequest, SaveTarget};
use legume_tui::Round;
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
    /// The save dialog, and what it offers to save.
    save: Option<(SaveDialog<Scope>, Target)>,
    toast: Option<(String, std::time::Instant)>,
    /// Said on the grid's status line until the next key.
    note: Option<String>,
    /// What was last drawn; the terminal is cleared when that changes, so
    /// the images of one screen never linger under the next.
    drawn: Option<Screen>,
    /// Figures saved here, this session and before, shown on the left in
    /// the strip.
    gallery: Gallery,
    strip: Strip<ProtocolPainter>,
    /// Moving a save: where to, as typed.
    moving: Option<MoveDialog>,
    quit: bool,
}

impl Deck {
    /// The views `apps`, beside the figures saved in `gallery`.
    pub(super) fn new(apps: Vec<App>, picker: Picker, gallery: Gallery) -> Self {
        Self {
            apps,
            at: 0,
            cell: cell(&picker),
            strip: Strip::new(
                ProtocolPainter {
                    picker: picker.clone(),
                },
                cell(&picker),
            ),
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
            moving: None,
            quit: false,
        }
    }

    /// Drive the views on `session`'s terminal until quit, or until `senna
    /// kill` asks them to stop. While detached nothing is drawn and the
    /// views' own work (lupin, senna, files changing) goes on at a slow
    /// poll; on attach the terminal is asked again (the protocol
    /// `graphics` names, where it names one) and the pictures made again
    /// as it needs
    /// ([`Self::repick`]).
    pub(super) fn run(
        mut self,
        terminal: &mut DefaultTerminal,
        session: &legume_tui::Session,
        graphics: Graphics,
    ) -> anyhow::Result<()> {
        if self.apps.len() > 1 {
            self.apps[0].message = Some(format!(
                "{} runs open · w shows them all · {{ }} step between them",
                self.apps.len()
            ));
        }
        let mut dirty = true;
        while !self.quit {
            let attached = session.attached();
            if attached {
                let size = terminal.size()?;
                let full = Rect::new(0, 0, size.width, size.height);
                let area = self.content(full);
                self.prepare_thumbs(full);
                if self.grid {
                    self.layout_grid(area);
                } else {
                    self.apps[self.at].layout(area, App::status_lines(full.width));
                }
            }
            dirty |= self.background();
            if attached {
                dirty |= if self.grid {
                    self.advance_tiles()?
                } else {
                    self.apps[self.at].advance()?
                };
            }
            if expired(&self.toast) {
                self.toast = None;
                dirty = true;
            }
            if attached && std::mem::take(&mut dirty) {
                self.draw(terminal)?;
            }
            session.status(&self.status());
            let mut repick = None;
            let picker = &self.picker;
            let round = session.poll_with(terminal, self.wait(), || {
                repick = Some(legume_tui::graphics::reprobe(
                    session,
                    picker,
                    graphics,
                    Fallback::Probe,
                ));
                Ok(())
            })?;
            if let Some(repick) = repick {
                self.repick(repick);
            }
            match round {
                Round::Event(mut ev) => {
                    // Drain everything queued so a burst of scroll events
                    // costs one re-render, not one per event.
                    loop {
                        dirty |= self.event(terminal, ev)?;
                        if self.quit || !event::poll(Duration::ZERO)? {
                            break;
                        }
                        ev = event::read()?;
                    }
                }
                Round::Repainted => dirty = true,
                // `senna kill` stops the view as q does; one heard by the
                // file browser before it is said again to this screen.
                Round::Stopped => self.quit = true,
                _ => {}
            }
        }
        Ok(())
    }

    /// Take what the terminal attached now needs of the pictures. One
    /// like the last keeps them; kitty's images, dropped by the client's
    /// reset, are encoded again as each view is next drawn. A picker of
    /// another kind or cell size has every view made again for it. Either
    /// way the tiles and thumbnails are made afresh at the next round.
    fn repick(&mut self, repick: Repick) {
        match repick {
            Repick::Resend => self.apps.iter_mut().for_each(App::resend),
            Repick::Changed(picker) => {
                self.cell = cell(&picker);
                for app in &mut self.apps {
                    app.repick(picker.clone());
                }
                self.strip.repaint(
                    ProtocolPainter {
                        picker: picker.clone(),
                    },
                    self.cell,
                );
                self.picker = picker;
            }
            // Same, or another word the crate may add: kept as they are.
            _ => return,
        }
        self.tiles.clear();
        self.strip.forget();
    }

    /// What `senna sessions` says the view is showing: the run on screen,
    /// its layout and view, or the grid.
    fn status(&self) -> String {
        if self.grid {
            return format!("grid of {} views", self.apps.len());
        }
        let s = self.apps[self.at].scene.current();
        format!("{} · {} {}", self.title(self.at), s.method, s.title())
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
            if let Some(strip) = self.strip_area(f.area()) {
                let keys = Keymap::default();
                let now = legume_tui::gallery::now();
                self.strip
                    .draw(f, strip, &self.gallery, &strip_look(), &keys, now);
            }
            if let Some((d, t)) = &self.save {
                d.render(f, f.area(), t, &Look::from(Chrome::themed(color::THEME)));
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
        let strip = self.strip_area(full);
        if strip.is_none() {
            // Hidden, or too narrow a screen to show it.
            self.strip.leave();
        }
        if let (Event::Mouse(m), Some(strip)) = (&ev, strip) {
            let top = self.strip.top();
            if let Some(changed) = self.strip.mouse(strip, m, &self.gallery) {
                self.scrolled_from(top);
                return Ok(changed);
            }
        }
        if let Event::Mouse(m) = &ev {
            if matches!(m.kind, MouseEventKind::Down(_)) {
                // A click elsewhere hands the keys back.
                self.strip.leave();
            }
        }
        if self.strip.chosen().is_some() {
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
        // Not over the help page, a box or menu being typed in, or a
        // relabel round; on a chart only saving and redrawing, as its
        // footer says.
        if app.help || app.modal.is_some() || app.menu.is_some() || app.scene.review.is_some() {
            return false;
        }
        let ctrl_l = k.code == KeyCode::Char('l') && k.modifiers.contains(KeyModifiers::CONTROL);
        if app.scene.chart.is_some() && !(ctrl_l || k.code == KeyCode::Char('s')) {
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
            KeyCode::Char('s') => self.open_save(Scope::View),
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
            let (cols, rows) = legume_tui::export::shape(
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
                    tile.proto = fit_protocol(&self.picker, frame.to_image(), inner).ok();
                    continue;
                }
                let vp = deck::camera(app.vp, app.scene.current().points.bounds, w, h);
                tile.job = Some(Job::new(vp, color::BACKGROUND, &app.scene.layers()));
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
                app.scene
                    .decorate(&mut frame, &vp, self.cell.1, app.stroke_floor);
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
                Ok((k, fit_protocol(picker, img, inner)?))
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
                    KeyCode::Char('s') => self.open_save(Scope::Grid),
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

/// Where the viewer keeps what it saved: `.senna-view/` in the directory
/// it runs from.
pub(super) const SAVED_DIR: &str = ".senna-view";

/// A name offered for a save goes `-2`, `-3`, … past files already there.
const NUMBERING: Numbering = Numbering::Suffix;

/// The saves logged in `dir`, kept as senna always has: a save whose file
/// is gone is dropped, and a log that cannot be read starts afresh.
pub(super) fn open_saved(dir: &std::path::Path) -> Gallery {
    Gallery::open(dir, Policy::plain())
}

/// The strip's look: the page's, with its rule as muted as the map's.
fn strip_look() -> Look {
    Look {
        border: Style::default().fg(rgb(color::MUTED)),
        ..Look::from(Chrome::themed(color::THEME))
    }
}

impl Deck {
    /// The strip of saved figures, when it shows: on the left, above the
    /// status area (the grid's, or the view's at full width).
    fn strip_area(&self, full: Rect) -> Option<Rect> {
        let status = if self.grid {
            STATUS_LINES
        } else {
            App::status_lines(full.width)
        };
        let body = Rect {
            height: full.height.saturating_sub(status),
            ..full
        };
        self.strip.area(body, &self.gallery)
    }

    /// Where the views or the grid go: beside the strip, when it shows.
    fn content(&self, full: Rect) -> Rect {
        match self.strip_area(full) {
            Some(s) => Rect {
                x: full.x + s.width,
                width: full.width - s.width,
                ..full
            },
            None => full,
        }
    }

    fn toggle_saved(&mut self) {
        self.strip.toggle();
        self.relayout();
    }

    /// Everything moves over, or the strip's pictures do: lay out and draw
    /// afresh, so no image lingers where it was.
    fn relayout(&mut self) {
        self.area = Rect::default();
        self.drawn = None;
    }

    /// The strip scrolled from `top`: its pictures moved, so the screen is
    /// drawn afresh.
    fn scrolled_from(&mut self, top: usize) {
        if self.strip.top() != top {
            self.drawn = None;
        }
    }

    /// Keys go to the strip, on the save at its top: it shows if hidden.
    fn focus_saved(&mut self) {
        let shown = self.strip.is_shown();
        if !self.strip.focus(&self.gallery) {
            self.say("no figures saved here yet · s saves one".into());
            return;
        }
        if !shown {
            self.relayout();
        }
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

    /// A key while the strip has them: choose, move or remove a save.
    fn strip_key(&mut self, full: Rect, k: KeyEvent) {
        let Some(area) = self.strip_area(full) else {
            self.strip.leave();
            return;
        };
        let top = self.strip.top();
        match self.strip.key(area, k, &self.gallery, &Keymap::default()) {
            strip::Action::Quit => self.quit = true,
            strip::Action::Relayout => self.relayout(),
            strip::Action::Move(at) => {
                let from = self.gallery.entries()[at].path.clone();
                self.moving = Some(MoveDialog {
                    to: crate::tui::shown(&from),
                    from,
                    status: None,
                });
            }
            strip::Action::Remove { at, delete } => {
                let e = self.gallery.entries()[at].clone();
                let msg = match self.gallery.remove(&e.path, delete) {
                    Ok(()) if delete => format!("deleted {}", e.path.display()),
                    Ok(()) => format!(
                        "{} taken off the list · the PDF stays (X deletes it too)",
                        e.name()
                    ),
                    Err(e) => format!("not removed: {e}"),
                };
                self.say(msg);
                // The views are laid out afresh only when the strip goes
                // with the last save; else the strip alone is drawn again.
                match self.strip.after_change(area, &self.gallery) {
                    strip::Action::Relayout => self.relayout(),
                    _ => self.drawn = None,
                }
            }
            _ => {}
        }
        self.scrolled_from(top);
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

    /// Thumbnails for the saves the strip shows, made once each; none
    /// while it is hidden.
    fn prepare_thumbs(&mut self, full: Rect) {
        if let Some(area) = self.strip_area(full) {
            self.strip.prepare(area, &self.gallery);
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
        Line::from(" type the path (ctrl+u clears)   enter move   esc cancel"),
    ];
    popup(f, f.area(), lines, 76, At::Middle, color::TEXT);
}

/// What a PDF takes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// What the save dialog offers, as the deck was when it opened.
struct Target {
    /// Sessions open.
    n: usize,
    /// The name the view on screen offers, and the first run's.
    view: String,
    first: String,
    /// Screen size in pixels of what each scope pictures, for its aspect
    /// and to scale text: the view on screen, and the grid's area.
    view_px: (f32, f32),
    grid_px: (f32, f32),
}

impl SaveTarget for Target {
    type Scope = Scope;

    fn scopes(&self) -> Vec<Scope> {
        if self.n > 1 {
            Scope::ALL.to_vec()
        } else {
            vec![Scope::View]
        }
    }

    fn scope_name(&self, s: Scope) -> String {
        s.name(self.n)
    }

    fn default_stem(&self, s: Scope) -> String {
        match s {
            Scope::View => self.view.clone(),
            Scope::Grid => format!("{}.view.grid", self.first),
            Scope::Pages => format!("{}.view.pages", self.first),
        }
    }

    fn exts(&self, _: Scope) -> Vec<String> {
        vec!["pdf".into()]
    }

    /// One page per run is as wide as the view on screen, its first page
    /// named in the size.
    fn aspect(&self, s: Scope) -> f32 {
        let (w, h) = match s {
            Scope::Grid => self.grid_px,
            _ => self.view_px,
        };
        w / h.max(1.0)
    }

    fn size_note(&self, s: Scope) -> String {
        match s {
            Scope::Pages => format!("(first of {} pages)", self.n),
            _ => String::new(),
        }
    }
}

impl Deck {
    /// What the save dialog offers now.
    fn save_target(&self) -> Target {
        let app = &self.apps[self.at];
        let view_px = app.view_px();
        let grid_px = if self.area.width > 0 {
            (
                f32::from(self.area.width) * self.cell.0,
                f32::from(self.area.height) * self.cell.1,
            )
        } else {
            view_px
        };
        Target {
            n: self.apps.len(),
            view: app.pdf_name(),
            first: here(&self.apps[0].scene.data.prefix),
            view_px,
            grid_px,
        }
    }

    /// Open the save dialog on `scope`, or on this view when one is open.
    fn open_save(&mut self, scope: Scope) {
        let t = self.save_target();
        let scope = if t.scopes().contains(&scope) {
            scope
        } else {
            Scope::View
        };
        self.save = Some((SaveDialog::new(&t, scope, NUMBERING), t));
    }

    /// A key in the save dialog: enter on a name that may be written
    /// renders and writes the PDF, the rest edit the dialog.
    fn save_event(&mut self, terminal: &mut DefaultTerminal, k: KeyEvent) -> anyhow::Result<()> {
        let Some((d, t)) = self.save.as_mut() else {
            return Ok(());
        };
        match d.key(k, t) {
            SaveKey::Open => {}
            SaveKey::Cancelled => self.save = None,
            SaveKey::Save(req) => {
                d.say("saving…");
                self.draw(terminal)?;
                let (mut d, t) = self.save.take().expect("the dialog is open");
                match self.write_pdf(&req, &t) {
                    Ok(msg) => self.say(msg),
                    Err(e) => {
                        d.say(format!("not saved: {e}"));
                        self.save = Some((d, t));
                    }
                }
            }
        }
        if self.save.is_none() {
            // The dialog covered part of the picture, and a first save opens
            // the saved-figures strip: draw it all again.
            self.drawn = None;
        }
        Ok(())
    }

    /// Render and write the PDF `req` asks for. Returns what to say.
    fn write_pdf(&mut self, req: &SaveRequest<Scope>, t: &Target) -> Result<String, String> {
        let (w, h) = req.px;
        let dpi = req.dpi;
        let view = |app| view_at(app, w, self.cell.1);
        let pages = match req.scope {
            Scope::View => deck::view_pages(&[view(&self.apps[self.at])]),
            Scope::Pages => deck::view_pages(&self.apps.iter().map(view).collect::<Vec<_>>()),
            Scope::Grid => {
                let scenes: Vec<&Scene> = self.apps.iter().map(|a| &a.scene).collect();
                let cams: Vec<Option<Viewport>> = self.apps.iter().map(|a| a.vp).collect();
                let titles: Vec<String> = (0..self.apps.len()).map(|k| self.title(k)).collect();
                let cell = self.cell.1 * w as f32 / t.grid_px.0.max(1.0);
                vec![deck::render_grid(&scenes, &cams, &titles, (w, h), cell)]
            }
        };
        legume_tui::export::write_pdf(&pages, dpi as f32, &req.path).map_err(|e| e.to_string())?;
        // Listed on the left with its first page as the thumbnail; failing
        // to list it never fails the save.
        let what = format!(
            "{} · {} in · {dpi} dpi",
            t.scope_name(req.scope),
            req.width_in
        );
        let listed = match self
            .gallery
            .add(&[&req.path], &what, Some(&pages[0].img), ())
        {
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
            req.path.display()
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

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// What the save dialog offers over `n` views, this one named `view`.
    fn target(n: usize, view: &str) -> Target {
        Target {
            n,
            view: view.into(),
            first: "run".into(),
            view_px: (800.0, 600.0),
            grid_px: (1200.0, 400.0),
        }
    }

    /// A gallery in `work` with a save of each of `names`, the last on top.
    fn gallery(work: &std::path::Path, names: &[&str]) -> Gallery {
        let mut g = open_saved(&work.join(SAVED_DIR));
        let picture = image::RgbaImage::from_pixel(40, 30, image::Rgba([1, 2, 3, 255]));
        for name in names {
            let pdf = work.join(name);
            std::fs::write(&pdf, b"%PDF").unwrap();
            g.add(&[&pdf], "a view", Some(&picture), ()).unwrap();
        }
        g
    }

    #[test]
    fn a_proposed_name_never_collides_with_a_saved_file() {
        let dir = tempfile::tempdir().unwrap();
        let stem = dir.path().join("run.view.umap.cells");
        let stem = stem.to_string_lossy();
        let t = target(1, &stem);
        let offered = |t: &Target| {
            SaveDialog::new(t, Scope::View, NUMBERING)
                .name()
                .to_string()
        };
        assert_eq!(offered(&t), stem);
        std::fs::write(format!("{stem}.pdf"), b"%PDF").unwrap();
        assert_eq!(offered(&t), format!("{stem}-2"));
        std::fs::write(format!("{stem}-2.pdf"), b"%PDF").unwrap();
        assert_eq!(offered(&t), format!("{stem}-3"));
    }

    #[test]
    fn replacing_a_file_takes_a_second_enter() {
        let dir = tempfile::tempdir().unwrap();
        let taken = dir.path().join("fig");
        std::fs::write(dir.path().join("fig.pdf"), b"%PDF").unwrap();
        let t = target(1, &taken.to_string_lossy());

        // A new name saves at once.
        let fresh = target(1, &dir.path().join("new").to_string_lossy());
        let mut d = SaveDialog::new(&fresh, Scope::View, NUMBERING);
        assert!(matches!(
            d.key(key(KeyCode::Enter), &fresh),
            SaveKey::Save(_)
        ));

        // An existing one asks first, then saves on the second enter.
        let mut d = SaveDialog::new(&t, Scope::View, NUMBERING);
        d.key(key(KeyCode::End), &t);
        for _ in 0.."-2".len() {
            d.key(key(KeyCode::Backspace), &t);
        }
        assert_eq!(d.name(), taken.to_string_lossy());
        assert_eq!(d.key(key(KeyCode::Enter), &t), SaveKey::Open);
        assert!(d.status().unwrap().contains("enter again replaces it"));
        let SaveKey::Save(req) = d.key(key(KeyCode::Enter), &t) else {
            panic!("the second enter saves");
        };
        assert_eq!(req.path, dir.path().join("fig.pdf"));
        assert_eq!((req.width_in, req.dpi, req.px), (7.0, 300, (2100, 1575)));

        // Another name in between asks again.
        let mut d = SaveDialog::new(&t, Scope::View, NUMBERING);
        d.key(key(KeyCode::End), &t);
        for _ in 0.."-2".len() {
            d.key(key(KeyCode::Backspace), &t);
        }
        assert_eq!(d.key(key(KeyCode::Enter), &t), SaveKey::Open);
        d.key(key(KeyCode::Char('x')), &t);
        d.key(key(KeyCode::Backspace), &t);
        assert_eq!(d.key(key(KeyCode::Enter), &t), SaveKey::Open);
    }

    #[test]
    fn several_views_offer_a_grid_and_a_page_each() {
        let one = target(1, "run.view.umap.cells");
        assert_eq!(one.scopes(), [Scope::View]);
        let two = target(2, "run.view.umap.cells");
        assert_eq!(two.scopes(), Scope::ALL);
        assert_eq!(two.scope_name(Scope::Grid), "all 2 runs, one grid");
        assert_eq!(two.default_stem(Scope::Grid), "run.view.grid");
        assert_eq!(two.default_stem(Scope::Pages), "run.view.pages");
        assert_eq!(two.size_note(Scope::Pages), "(first of 2 pages)");
        assert_eq!(two.exts(Scope::View), ["pdf"]);
        // A grid is as wide as the grid on screen, a view as the view.
        assert_eq!(two.aspect(Scope::Grid), 3.0);
        assert_eq!(two.aspect(Scope::Pages), 800.0 / 600.0);
        // ← → step through the scopes, each with its own name.
        let mut d = SaveDialog::new(&two, Scope::View, NUMBERING);
        d.key(key(KeyCode::Right), &two);
        assert_eq!(d.scope(), Scope::Grid);
        assert_eq!(d.name(), "run.view.grid");
        assert_eq!(d.px(&two), (2100, 700));
    }

    #[test]
    fn a_log_written_before_still_loads() {
        let work = tempfile::tempdir().unwrap();
        let store = work.path().join(SAVED_DIR);
        std::fs::create_dir_all(store.join("thumbs")).unwrap();
        let pdf = work.path().join("a.pdf");
        std::fs::write(&pdf, b"%PDF").unwrap();
        let thumb = image::RgbaImage::from_pixel(24, 18, image::Rgba([1, 2, 3, 255]));
        thumb.save(store.join("thumbs").join("17.png")).unwrap();
        let log = serde_json::json!([{
            "path": pdf.canonicalize().unwrap(),
            "what": "this view · 7 in · 300 dpi",
            "when": 1_000_000,
            "thumb": "17.png",
        }]);
        std::fs::write(store.join("saved.json"), log.to_string()).unwrap();

        let mut g = open_saved(&store);
        assert_eq!(g.entries().len(), 1);
        let e = &g.entries()[0];
        assert_eq!(
            (e.name().as_str(), e.what.as_str()),
            ("a.pdf", "this view · 7 in · 300 dpi")
        );
        assert!(g.thumb(e).unwrap().exists());

        // A new save goes on top and the old one stays, as it was written.
        let b = work.path().join("b.pdf");
        std::fs::write(&b, b"%PDF").unwrap();
        g.add(&[&b], "a view", Some(&thumb), ()).unwrap();
        let names: Vec<String> = open_saved(&store)
            .entries()
            .iter()
            .map(|e| e.name())
            .collect();
        assert_eq!(names, ["b.pdf", "a.pdf"]);
        let written: serde_json::Value =
            serde_json::from_slice(&std::fs::read(store.join("saved.json")).unwrap()).unwrap();
        assert_eq!(written[1]["thumb"], "17.png");
        assert_eq!(written[1]["when"], 1_000_000);
    }

    #[test]
    fn the_strip_scrolls_to_the_chosen_save_and_drops_it() {
        let work = tempfile::tempdir().unwrap();
        let names: Vec<String> = (0..10).map(|k| format!("{k}.pdf")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut deck = Deck::new(
            Vec::new(),
            Picker::halfblocks(),
            gallery(work.path(), &names),
        );
        // Over the grid, so what is said goes in a toast.
        deck.grid = true;
        let full = Rect::new(0, 0, 160, 50);
        let shown = deck.strip.shown(deck.strip_area(full).unwrap());
        assert!((1..10).contains(&shown), "{shown} of 10 fit");

        deck.focus_saved();
        assert_eq!(deck.strip.chosen(), Some(0));
        // Down past the last one shown scrolls it into view.
        for _ in 0..shown {
            deck.strip_key(full, key(KeyCode::Down));
        }
        assert_eq!(deck.strip.chosen(), Some(shown));
        assert_eq!(deck.strip.top(), 1);
        deck.strip_key(full, key(KeyCode::End));
        assert_eq!(deck.strip.chosen(), Some(9));
        assert_eq!(deck.strip.top(), 10 - shown);
        // The wheel stops at either end.
        assert!(!deck.strip.scroll(99, shown, &deck.gallery));
        assert!(deck.strip.scroll(0, shown, &deck.gallery));

        // x takes the last off the list but keeps its PDF; the choice stays
        // in range.
        deck.strip_key(full, key(KeyCode::End));
        let last = deck.gallery.entries()[9].path.clone();
        deck.strip_key(full, key(KeyCode::Char('x')));
        assert_eq!(deck.gallery.entries().len(), 9);
        assert!(last.exists());
        assert_eq!(deck.strip.chosen(), Some(8));
        // m asks where to move it.
        deck.strip_key(full, key(KeyCode::Char('m')));
        assert!(deck.moving.is_some());
        deck.moving = None;
        // Esc hands the keys back.
        deck.strip_key(full, key(KeyCode::Esc));
        assert_eq!(deck.strip.chosen(), None);
    }

    #[test]
    fn the_strip_sits_above_the_status_lines_beside_the_views() {
        let work = tempfile::tempdir().unwrap();
        let mut deck = Deck::new(
            Vec::new(),
            Picker::halfblocks(),
            gallery(work.path(), &["a.pdf"]),
        );
        deck.grid = true;
        let full = Rect::new(0, 0, 160, 50);
        assert_eq!(
            deck.strip_area(full),
            Some(Rect::new(0, 0, 28, 50 - STATUS_LINES))
        );
        assert_eq!(deck.content(full), Rect::new(28, 0, 132, 50));
        // Hidden, or on a narrow screen, the views take it all.
        assert_eq!(deck.strip_area(Rect::new(0, 0, 80, 50)), None);
        deck.toggle_saved();
        assert_eq!(deck.content(full), full);
    }

    use super::attach_tests::{kitty, view};

    #[test]
    fn after_an_attach_every_tile_and_thumbnail_is_made_again() {
        let work = tempfile::tempdir().unwrap();
        let mut deck = drawn_deck(Picker::halfblocks(), work.path());
        let full = Rect::new(0, 0, 160, 50);

        deck.repick(Repick::Changed(kitty()));
        assert_eq!(deck.cell, (8.0, 16.0));
        assert!(deck.apps.iter().all(|a| a.cell == (8.0, 16.0)));
        assert!(deck.tiles.is_empty() && deck.strip.made() == 0);
        // The next round lays the tiles out again and draws them anew.
        deck.prepare_thumbs(full);
        deck.layout_grid(deck.content(full));
        assert_eq!(deck.tiles.len(), 2);
        assert!(deck.tiles.iter().all(|t| t.job.is_some()));
        assert_eq!(deck.strip.made(), 1);
        assert_eq!(
            deck.strip.painter().picker.protocol_type(),
            ratatui_image::picker::ProtocolType::Kitty
        );
    }

    /// A deck of two views made with `picker`: the first drawn whole, the
    /// grid's tiles and a saved thumbnail too.
    fn drawn_deck(picker: Picker, work: &std::path::Path) -> Deck {
        let apps = vec![view(), view()]
            .into_iter()
            .map(|mut a| {
                a.repick(picker.clone());
                a
            })
            .collect();
        let mut deck = Deck::new(apps, picker, gallery(work, &["a.pdf"]));
        let full = Rect::new(0, 0, 160, 50);
        let area = deck.content(full);
        deck.apps[0].layout(area, STATUS_LINES);
        while deck.apps[0].advance().unwrap() {}
        deck.grid = true;
        deck.prepare_thumbs(full);
        deck.layout_grid(deck.content(full));
        while deck.advance_tiles().unwrap() {}
        assert!(deck.apps[0].proto.is_some() && deck.apps[0].base.is_some());
        assert!(deck.tiles.iter().all(|t| t.proto.is_some()));
        assert!(deck.strip.made() > 0);
        deck
    }

    #[test]
    fn a_terminal_like_the_last_keeps_the_pictures() {
        let work = tempfile::tempdir().unwrap();
        let mut deck = drawn_deck(Picker::halfblocks(), work.path());
        let vp = deck.apps[0].vp;
        deck.repick(Repick::Same);
        // Nothing made again: block characters are redrawn from the cells.
        assert!(deck.apps[0].job.is_none() && deck.apps[0].proto.is_some());
        assert_eq!(deck.apps[0].vp, vp);
        assert!(deck
            .tiles
            .iter()
            .all(|t| t.job.is_none() && t.proto.is_some()));
        assert!(deck.strip.made() > 0);
    }

    #[test]
    fn a_kitty_terminal_like_the_last_gets_its_pictures_sent_again() {
        let work = tempfile::tempdir().unwrap();
        let mut deck = drawn_deck(kitty(), work.path());
        let vp = deck.apps[0].vp;
        deck.repick(Repick::Resend);
        // Nothing encoded yet: each view waits until it is next drawn.
        assert!(deck.apps.iter().all(|a| a.proto.is_none()));
        // Then encoded again from the finished frame, no point drawn again.
        assert!(deck.apps[0].advance().unwrap());
        let app = &deck.apps[0];
        assert!(app.job.is_none(), "not drawn again");
        assert!(matches!(app.proto, Some(Protocol::Kitty(_))));
        assert_eq!(app.vp, vp);
        // The tiles and thumbnails are made again at the next round.
        assert!(deck.tiles.is_empty() && deck.strip.made() == 0);
        assert_eq!(deck.cell, (8.0, 16.0));
    }

    #[test]
    fn the_session_status_names_what_is_on_screen() {
        let work = tempfile::tempdir().unwrap();
        let gallery = open_saved(&work.path().join(SAVED_DIR));
        let mut deck = Deck::new(vec![view(), view()], Picker::halfblocks(), gallery);
        assert_eq!(deck.status(), "r.senna.json (1) · umap cells");
        deck.grid = true;
        assert_eq!(deck.status(), "grid of 2 views");
    }
}
