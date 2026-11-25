//! Several views open at once, and the one loop that drives them. One view
//! is on screen at a time; `w` shows them all in a grid, each tile the
//! view's current picture, where the mouse or the arrows choose one and a
//! click or enter opens it. Saving happens here too, since a PDF can take in
//! every view. Every view's background work (a zoom, lupin, a reloaded file)
//! goes on whichever is on screen.

use super::*;
use crate::view::deck;
use legume_tui::gallery::strip::{Keymap, ProtocolPainter, Strip};
use legume_tui::gallery::Gallery;
use legume_tui::graphics::Repick;
use legume_tui::save::SaveDialog;
use legume_tui::Look;
use legume_tui::Round;
use rayon::prelude::*;

mod save_dialog;
mod saved_strip;

use save_dialog::*;
use saved_strip::*;
pub(super) use saved_strip::{open_saved, SAVED_DIR};

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
                d.render(f, f.area(), t, &Look::from(CHROME));
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
            KeyCode::Char('D') => match self.donor_tiles(self.at) {
                Ok(()) => {
                    self.grid = true;
                    self.note = None;
                }
                Err(e) => self.apps[self.at].message = Some(e),
            },
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

    /// Beside view `k`, showing a gain map, one copy per donor drawing that
    /// donor's deviation at every cell's state: where each donor departs,
    /// side by side.
    fn donor_tiles(&mut self, k: usize) -> Result<(), String> {
        let scene = &mut self.apps[k].scene;
        let Some(crate::view::Pick::Gain { gene, .. }) = scene.pick.clone() else {
            return Err("D tiles a gain map by donor · show one first (i)".into());
        };
        let donors = scene.stage2_donors()?;
        for (j, donor) in donors.into_iter().enumerate() {
            self.duplicate(k + j);
            self.apps[k + j + 1]
                .scene
                .set_pick(crate::view::Pick::Gain {
                    gene: gene.clone(),
                    donor: Some(donor),
                });
        }
        Ok(())
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
            CHROME.toast(f, f.area(), text);
        }
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
#[path = "../../tests/tui_grid.rs"]
mod tests;
