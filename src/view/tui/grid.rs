//! Several views open at once, and the one loop that drives them. One view
//! is on screen at a time; `w` shows them all in a grid, each tile the
//! view's current picture, where the mouse or the arrows choose one and a
//! click or enter opens it. Saving happens here too, since a PDF can take in
//! every view. Every view's background work (a zoom, lupin, a reloaded file)
//! goes on whichever is on screen.

use super::draw::{page, popup, toast, At};
use super::*;
use crate::view::deck;
use crate::view::pdf;
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
    quit: bool,
}

impl Deck {
    pub(super) fn new(apps: Vec<App>, picker: Picker) -> Self {
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
            let area = Rect::new(0, 0, size.width, size.height);
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
                    dirty |= self.event(terminal, event::read()?)?;
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
            if let Some(d) = &self.save {
                self.draw_save(f, d);
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
        let dim = Style::default().fg(rgb(color::MUTED));
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
                    dim,
                )))
                .style(page);
            block = if hovered {
                block
                    .border_type(ratatui::widgets::BorderType::Thick)
                    .border_style(Style::default().fg(rgb(color::INK)))
            } else {
                block.border_style(dim)
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
                dim,
            )),
        ];
        f.render_widget(Paragraph::new(lines).style(page), status);
        if let Some((text, _)) = &self.toast {
            toast(f, f.area(), text);
        }
    }
}

fn has_pdf(name: &str) -> bool {
    name.trim().to_ascii_lowercase().ends_with(".pdf")
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
}

impl SaveDialog {
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
        let name = self.name.trim();
        if has_pdf(name) {
            name.into()
        } else {
            format!("{name}.pdf").into()
        }
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
        };
        d.name = self.default_name(d.scope);
        d
    }

    fn default_name(&self, scope: Scope) -> String {
        let app = &self.apps[self.at];
        let first = &self.apps[0].scene.data.prefix;
        match scope {
            Scope::View => app.pdf_name(),
            Scope::Grid => format!("{first}.view.grid"),
            Scope::Pages => format!("{first}.view.pages"),
        }
    }

    /// A key in the save dialog: enter renders and writes the PDF, the rest
    /// edit the dialog.
    fn save_event(&mut self, terminal: &mut DefaultTerminal, k: KeyEvent) -> anyhow::Result<()> {
        if k.code == KeyCode::Enter {
            if let Some(d) = self.save.as_mut() {
                d.status = Some("saving…".into());
            }
            self.draw(terminal)?;
            let d = self.save.take().expect("the dialog is open");
            match self.write_pdf(&d) {
                Ok(msg) if self.grid => {
                    self.toast = Some((msg, std::time::Instant::now() + TOAST_FOR));
                }
                Ok(msg) => self.apps[self.at].pop(msg),
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
            // The dialog covered part of the picture: draw it all again.
            self.drawn = None;
        }
        Ok(())
    }

    fn save_key(&mut self, k: KeyEvent) {
        let n = self.apps.len();
        let Some(d) = self.save.as_mut() else { return };
        d.status = None;
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
            KeyCode::Char('u') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                d.field = 3;
                d.typed = true;
                d.name.clear();
            }
            KeyCode::Backspace => {
                d.field = 3;
                d.typed = true;
                d.name.pop();
            }
            KeyCode::Char(c) if !k.modifiers.contains(KeyModifiers::CONTROL) => {
                d.field = 3;
                d.typed = true;
                d.name.push(c);
            }
            _ => {}
        }
        if let Some(scope) = rename {
            let name = self.default_name(scope);
            if let Some(d) = self.save.as_mut() {
                d.name = name;
            }
        }
    }

    fn draw_save(&self, f: &mut ratatui::Frame, d: &SaveDialog) {
        let dim = Style::default().fg(rgb(color::MUTED));
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
            format!("{}▏{}", d.name, if has_pdf(&d.name) { "" } else { ".pdf" }),
        ];
        let mut lines = vec![
            Line::from(Span::styled(" Save as PDF", bold)),
            Line::from(""),
        ];
        for (k, (field, value)) in SAVE_FIELDS.iter().zip(values).enumerate() {
            let here = k == d.field;
            let cursor = if here { " ▸ " } else { "   " };
            lines.push(Line::from(vec![
                Span::raw(format!("{cursor}{field:<6} ")),
                Span::styled(value, if here { bold } else { Style::default() }),
            ]));
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
                format!(" {} exists · enter replaces it", path.display()),
                bold,
            )),
            None => Line::from(""),
        };
        lines.push(note);
        lines.push(Line::from(Span::styled(
            " ↑ ↓ field   ← → change   type to rename (ctrl-u clears)   enter save   esc cancel",
            dim,
        )));
        popup(f, f.area(), lines, 76, At::Middle, color::INK);
    }

    /// Render and write the PDF the dialog describes. Returns what to say.
    fn write_pdf(&self, d: &SaveDialog) -> Result<String, String> {
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
        let pages = if pages.len() > 1 {
            format!("{} pages · ", pages.len())
        } else {
            String::new()
        };
        Ok(format!(
            "saved {} · {pages}{w} × {h} px at {dpi} dpi · text kept as text",
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
