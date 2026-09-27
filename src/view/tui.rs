//! Terminal front end: event loop, keys and mouse, progressive redraw.
//!
//! Each change of view starts a new render job. While a job runs the loop
//! polls without blocking, draws one chunk, and pushes the composite, so the
//! picture appears at once and fills in; any key or mouse event in between
//! simply replaces the job.

use super::color;
use super::decide::{Action, Decision, Watcher};
use super::files::{modified, same_file};
use super::render::{Job, Viewport};
use super::style::{swatches, Shape};
use super::{Graphics, Pick, Scene};
use image::DynamicImage;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::layout::{Constraint, Layout, Rect, Size};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};
use ratatui::DefaultTerminal;
use ratatui_image::picker::cap_parser::QueryStdioOptions;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::Protocol;
use ratatui_image::{Image, Resize};
use std::time::Duration;

const KEYS: &str =
    "tab space · c colour · [ ] focus · n suggest · g / feature · z zoom · e style · ? help";

const HELP: &[(&str, &str)] = &[
    (
        "tab / shift-tab",
        "next / previous space (cells, features on cells, features)",
    ),
    ("m", "next layout method"),
    (
        "c",
        "next grouping (annotation, cluster, topic, markers, none)",
    ),
    ("[  ]", "focus previous / next group; others turn gray"),
    ("click", "focus the group of the nearest point"),
    (
        "n",
        "suggest features (model-based; check with o): what sets the focused group apart, or what varies here",
    ),
    (
        "g  G",
        "next / previous feature (suggestions, else the focused group's markers)",
    ),
    ("/", "search a feature by name; enter shows it"),
    ("a", "activity of the focused group's whole marker set"),
    ("o", "expected (model) or observed (counts) activity"),
    ("x / esc", "clear feature, then focus"),
    ("z", "lay out the focused group's cells again, on their own"),
    (
        ",  .",
        "previous (source) / next annotation round; the file reloads on change",
    ),
    (
        "Z / backspace",
        "back to the layout this one was zoomed from",
    ),
    (
        "e",
        "style menu: colour, shape, opacity, size, visibility per group",
    ),
    ("b", "sidebar on / off (suggestions, cluster summary, style menu)"),
    ("L", "decide: label the focused / clicked cluster (asks label, then why)"),
    ("v  M", "mark clusters for a merge; merge the marked ones"),
    ("K", "decide: keep the cluster's current call"),
    ("A  D", "decide: add / drop the feature on screen in a cell type's markers"),
    ("t", "text labels on / off"),
    ("+  -  scroll", "zoom"),
    ("hjkl  arrows  drag", "pan"),
    ("0", "reset: top-level layout, whole map in view"),
    ("s", "save this view as PNG"),
    ("q", "quit"),
];

fn rgb(c: [u8; 3]) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

pub fn run(
    scene: Scene,
    graphics: Graphics,
    from: std::path::PathBuf,
    decisions: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = (|| {
        // A terminal that never answers must not stall startup; block
        // characters need no query at all.
        let mut picker = if graphics == Graphics::Blocks {
            Picker::halfblocks()
        } else {
            let options = QueryStdioOptions {
                timeout: Duration::from_millis(500),
                ..Default::default()
            };
            Picker::from_query_stdio_with_options(options).unwrap_or_else(|_| Picker::halfblocks())
        };
        match graphics {
            Graphics::Auto => {}
            Graphics::Kitty => picker.set_protocol_type(ProtocolType::Kitty),
            Graphics::Sixel => picker.set_protocol_type(ProtocolType::Sixel),
            Graphics::Iterm2 => picker.set_protocol_type(ProtocolType::Iterm2),
            Graphics::Blocks => picker.set_protocol_type(ProtocolType::Halfblocks),
        }
        execute!(std::io::stdout(), EnableMouseCapture)?;
        App::new(scene, picker, from, decisions).run(&mut terminal)
    })();
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

type ZoomResult = Result<super::sublayout::Laid, String>;

/// A zoom into one group, running on a worker thread.
struct Zooming {
    parent: usize,
    label: String,
    done: std::sync::mpsc::Receiver<ZoomResult>,
}

/// A decision being typed in the status line: first the label (unless the
/// action keeps the current one), then the rationale.
struct Prompt {
    decision: Decision,
    /// Typing the rationale (after the label).
    why: bool,
    input: String,
    /// Label completions offered for the current input.
    known: Vec<Box<str>>,
}

impl Prompt {
    fn subject(&self) -> String {
        let d = &self.decision;
        let ids: Vec<String> = d.clusters.iter().map(|c| format!("C{c}")).collect();
        match d.action {
            Action::Label => format!("label {}", ids.join(" ")),
            Action::Merge => format!("merge {}", ids.join(" ")),
            Action::Keep => format!("keep {} as {}", ids.join(" "), d.label),
            Action::MarkersAdd => format!("add {} to markers of", d.features.join(" ")),
            Action::MarkersDrop => format!("drop {} from markers of", d.features.join(" ")),
        }
    }

    fn line(&self) -> String {
        if self.why {
            format!(
                "{} · why? {}▏  (enter sends · esc cancels)",
                self.subject(),
                self.input
            )
        } else {
            let hint: Vec<&str> = self
                .known
                .iter()
                .filter(|k| k.to_lowercase().starts_with(&self.input.to_lowercase()))
                .take(4)
                .map(AsRef::as_ref)
                .collect();
            format!(
                "{} as: {}▏  tab: {}",
                self.subject(),
                self.input,
                hint.join(" · ")
            )
        }
    }
}

/// The style menu's cursor: which group, which property.
struct Menu {
    row: usize,
    field: usize,
}

const FIELDS: [&str; 5] = ["colour", "shape", "opacity", "size", "visible"];
const MENU_HINT: &str = "↑↓ group  ←→ change  tab property  space show/hide  r reset  enter done";

struct App {
    scene: Scene,
    picker: Picker,
    /// Pixels per terminal cell.
    cell: (f32, f32),
    map: Rect,
    vp: Option<Viewport>,
    job: Option<Job>,
    proto: Option<Protocol>,
    message: Option<String>,
    help: bool,
    drag: Option<(u16, u16, bool)>,
    /// Feature search in progress: the query and its current matches.
    search: Option<(String, Vec<Box<str>>)>,
    /// Searchable feature names, loaded on the first `/`.
    names: Option<Vec<Box<str>>>,
    /// Style menu: selected group and field.
    menu: Option<Menu>,
    /// A group being laid out on a worker thread.
    zooming: Option<Zooming>,
    /// The manifest on screen, its last-seen modification time, and when
    /// that was last checked.
    from: std::path::PathBuf,
    stamp: Option<std::time::SystemTime>,
    checked: std::time::Instant,
    /// Panel with the clicked cluster's summary and history.
    info: Option<Vec<String>>,
    /// Sidebar area, beside the map; empty when there is nothing to show.
    side: Rect,
    /// Whether the sidebar may open (`b` toggles).
    sidebar: bool,
    /// The `lupin relabel --watch` this view writes decisions for, if any.
    watcher: Option<Watcher>,
    decisions: Option<std::path::PathBuf>,
    /// Clusters marked for a merge (`v`), and the last one clicked.
    marked: Vec<i64>,
    clicked: Option<i64>,
    /// A decision being typed.
    prompt: Option<Prompt>,
    quit: bool,
}

impl App {
    fn new(
        scene: Scene,
        picker: Picker,
        from: std::path::PathBuf,
        decisions: Option<std::path::PathBuf>,
    ) -> Self {
        let f = picker.font_size();
        Self {
            watcher: Watcher::find(&from, decisions.as_deref()),
            decisions,
            marked: Vec::new(),
            clicked: None,
            prompt: None,
            stamp: modified(&from),
            from,
            checked: std::time::Instant::now(),
            info: None,
            side: Rect::default(),
            sidebar: true,
            scene,
            cell: (f32::from(f.width.max(1)), f32::from(f.height.max(1))),
            picker,
            map: Rect::default(),
            vp: None,
            job: None,
            proto: None,
            message: None,
            help: false,
            drag: None,
            search: None,
            names: None,
            menu: None,
            zooming: None,
            quit: false,
        }
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        while !self.quit {
            let area = terminal.size()?;
            self.layout(Rect::new(0, 0, area.width, area.height));
            if self.advance()? {
                terminal.draw(|f| self.draw(f))?;
            }
            if self.finish_zoom() {
                continue;
            }
            if self.job.is_none() && self.checked.elapsed() >= Duration::from_secs(1) {
                self.checked = std::time::Instant::now();
                if modified(&self.from) != self.stamp {
                    let from = self.from.clone();
                    self.open_round(&from, "reloaded");
                    continue;
                }
                if self.follow_watcher() {
                    continue;
                }
            }
            let wait = if self.job.is_some() {
                Duration::ZERO
            } else if self.zooming.is_some() {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(250)
            };
            if event::poll(wait)? {
                // Drain everything queued so a burst of scroll events costs
                // one re-render, not one per event.
                loop {
                    self.handle(event::read()?);
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
                terminal.draw(|f| self.draw(f))?;
            }
        }
        Ok(())
    }

    /// Map area and viewport for the current terminal size.
    /// Whether the sidebar has something to show.
    fn side_content(&self) -> bool {
        self.menu.is_some() || self.info.is_some() || self.scene.suggestion_lines().is_some()
    }

    fn layout(&mut self, area: Rect) {
        let body = Rect {
            height: area.height.saturating_sub(1),
            ..area
        };
        // The sidebar takes its columns from the map rather than covering it.
        let side_w = if self.sidebar && self.side_content() {
            (area.width / 3)
                .clamp(30, 48)
                .min(area.width.saturating_sub(20))
        } else {
            0
        };
        let map = Rect {
            width: body.width - side_w,
            ..body
        };
        self.side = Rect::new(body.x + map.width, body.y, side_w, body.height);
        if map != self.map {
            let had_map = self.map.width > 0 && self.map.height > 0;
            self.map = map;
            // Keep the camera where it was: same centre, same scale, a
            // different window onto it.
            match self.vp.as_mut() {
                Some(vp) if had_map => {
                    let (w, h) = (
                        (f32::from(map.width) * self.cell.0) as usize,
                        (f32::from(map.height) * self.cell.1) as usize,
                    );
                    vp.w = w;
                    vp.h = h;
                    self.restart();
                }
                _ => self.vp = None,
            }
        }
        if self.vp.is_none() {
            let (w, h) = self.map_px();
            self.vp = Some(Viewport::fit(self.scene.current().points.bounds, w, h));
            self.restart();
        }
    }

    fn map_px(&self) -> (usize, usize) {
        (
            (f32::from(self.map.width) * self.cell.0) as usize,
            (f32::from(self.map.height) * self.cell.1) as usize,
        )
    }

    fn restart(&mut self) {
        self.job = None;
        if let Some(vp) = self.vp {
            if vp.w > 0 && vp.h > 0 {
                self.job = Some(Job::new(vp, &self.scene.layers()));
            }
        }
    }

    /// Draw one chunk of the running job and turn the composite into an
    /// image. Returns whether a new image is ready.
    fn advance(&mut self) -> anyhow::Result<bool> {
        let Some(job) = self.job.as_mut() else {
            return Ok(false);
        };
        let layers = self.scene.layers();
        let done = job.step(&layers);
        let mut frame = job.composite();
        if done {
            self.scene.decorate(&mut frame, &job.vp, self.cell.1);
        }
        let img = frame.to_image(&color::Encoder::new());
        let size = Size::new(self.map.width, self.map.height);
        self.proto = Some(self.picker.new_protocol(
            DynamicImage::ImageRgba8(img),
            size,
            Resize::Fit(Some(image::imageops::FilterType::Triangle)),
        )?);
        if done {
            self.job = None;
        }
        Ok(true)
    }

    fn draw(&self, f: &mut ratatui::Frame) {
        let page = Style::default()
            .bg(rgb(color::BACKGROUND))
            .fg(rgb(color::INK));
        f.render_widget(Block::default().style(page), f.area());
        let [_, status] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(f.area());
        let map = self.map;
        if let Some(p) = &self.proto {
            f.render_widget(Image::new(p), map);
        }

        let left = match (&self.search, &self.prompt) {
            (Some((q, hits)), _) => {
                let shown: Vec<&str> = hits.iter().take(6).map(AsRef::as_ref).collect();
                format!("/{q}   {}", shown.join("  "))
            }
            (None, Some(p)) => p.line(),
            (None, None) => {
                let mut text = self.message.clone().unwrap_or_else(|| self.scene.caption());
                if !self.marked.is_empty() {
                    let ids: Vec<String> = self.marked.iter().map(|c| format!("C{c}")).collect();
                    text.push_str(&format!(" · marked {}", ids.join(" ")));
                }
                text
            }
        };
        let right = if self.job.is_some() {
            "drawing…"
        } else {
            KEYS
        };
        let pad = (status.width as usize)
            .saturating_sub(left.chars().count() + right.chars().count() + 2);
        let line = Line::from(vec![
            Span::raw(format!(" {left}")),
            Span::raw(" ".repeat(pad)),
            Span::styled(right, Style::default().fg(rgb(color::MUTED))),
        ]);
        f.render_widget(Paragraph::new(line).style(page), status);

        let side = self.side;
        if side.width > 0 {
            if let Some(menu) = &self.menu {
                self.draw_menu(f, side, menu, page);
            } else if let Some(lines) = self.info.clone().or_else(|| self.scene.suggestion_lines())
            {
                let text: Vec<Line> = lines.iter().map(|l| Line::from(format!(" {l}"))).collect();
                f.render_widget(
                    Paragraph::new(text)
                        .wrap(ratatui::widgets::Wrap { trim: false })
                        .block(side_block())
                        .style(page),
                    side,
                );
            }
        }

        if self.help {
            let w = 72.min(map.width);
            let h = (HELP.len() as u16 + 2).min(map.height);
            let r = Rect::new(
                map.x + (map.width - w) / 2,
                map.y + (map.height - h) / 2,
                w,
                h,
            );
            let lines: Vec<Line> = HELP
                .iter()
                .map(|(k, v)| Line::from(format!("  {k:<20} {v}")))
                .collect();
            f.render_widget(Clear, r);
            f.render_widget(
                Paragraph::new(lines)
                    .block(Block::bordered().border_style(Style::default().fg(rgb(color::MUTED))))
                    .style(page),
                r,
            );
        }
    }

    /// The style menu, docked on the right of the map.
    fn draw_menu(&self, f: &mut ratatui::Frame, map: Rect, menu: &Menu, page: Style) {
        let enc = color::Encoder::new();
        let to_color = |c: color::Rgb| {
            let [r, g, b] = c.map(|v| enc.encode(v));
            Color::Rgb(r, g, b)
        };
        let levels = self.scene.levels();
        let chrome = FIELDS.len() as u16 + 6;
        let r = map;
        let h = r.height;
        let list_rows = h.saturating_sub(chrome).max(1) as usize;
        let first = menu
            .row
            .saturating_sub(list_rows / 2)
            .min(levels.len().saturating_sub(list_rows));

        let dim = Style::default().fg(rgb(color::MUTED));
        let mut lines: Vec<Line> = Vec::new();
        for (g, level) in levels.iter().enumerate().skip(first).take(list_rows) {
            let st = self.scene.style_of(g);
            let res = self.scene.resolved(g);
            let mark = Span::styled(
                format!(" {} ", st.shape.glyph()),
                Style::default().fg(res.map_or(Color::Reset, |r| to_color(r.colour))),
            );
            let mut name = Style::default();
            if st.hidden {
                name = dim;
            }
            if g == menu.row {
                name = name.add_modifier(ratatui::style::Modifier::REVERSED);
            }
            lines.push(Line::from(vec![
                mark,
                Span::styled(level.to_string(), name),
            ]));
        }
        let st = self.scene.style_of(menu.row);
        let res = self.scene.resolved(menu.row);
        let values = [
            Line::from(vec![
                Span::raw("■■■■ "),
                Span::styled(
                    if st.colour.is_some() {
                        "custom"
                    } else {
                        "default"
                    },
                    dim,
                ),
            ])
            .style(Style::default().fg(res.map_or(Color::Reset, |r| to_color(r.colour)))),
            Line::from(format!("{} {}", st.shape.glyph(), st.shape.name())),
            Line::from(format!("{:.1}", st.alpha)),
            Line::from(format!("{:.2}×", st.size)),
            Line::from(if st.hidden { "hidden" } else { "shown" }),
        ];
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!(" {}", levels[menu.row]),
            Style::default().add_modifier(ratatui::style::Modifier::BOLD),
        )));
        for (k, (field, value)) in FIELDS.iter().zip(values).enumerate() {
            let cursor = if k == menu.field { " ▸ " } else { "   " };
            let mut spans = vec![Span::raw(format!("{cursor}{field:<8} "))];
            spans.extend(value.spans.into_iter().map(|s| s.patch_style(value.style)));
            lines.push(Line::from(spans));
        }
        lines.push(Line::from(Span::styled(format!(" {MENU_HINT}"), dim)));

        f.render_widget(
            Paragraph::new(lines)
                .wrap(ratatui::widgets::Wrap { trim: false })
                .block(side_block())
                .style(page),
            r,
        );
    }

    fn handle(&mut self, ev: Event) {
        match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => self.key(k),
            Event::Mouse(m) => self.mouse(m),
            Event::Resize(..) => self.vp = None,
            _ => {}
        }
    }

    fn with_vp(&mut self, f: impl FnOnce(&mut Viewport)) {
        if let Some(vp) = self.vp.as_mut() {
            f(vp);
            self.restart();
        }
    }

    /// Run a scene change, surface its note, and redraw.
    fn change(&mut self, f: impl FnOnce(&mut Scene)) {
        f(&mut self.scene);
        if let Some(note) = self.scene.note.take() {
            self.message = Some(note);
        }
        self.restart();
    }

    fn search_key(&mut self, k: KeyEvent) {
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
        let hits = search(self.names.as_deref().unwrap_or(&[]), &query);
        self.search = Some((query, hits));
    }

    fn open_menu(&mut self) {
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

    fn menu_key(&mut self, k: KeyEvent) {
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
            }
            KeyCode::Up | KeyCode::Char('k') => menu.row = (row + n - 1) % n,
            KeyCode::Down | KeyCode::Char('j') => menu.row = (row + 1) % n,
            KeyCode::Tab => menu.field = (menu.field + 1) % FIELDS.len(),
            KeyCode::BackTab => menu.field = (menu.field + FIELDS.len() - 1) % FIELDS.len(),
            KeyCode::Char(' ') => self.scene.restyle(row, |st| st.hidden = !st.hidden),
            KeyCode::Char('r') => self
                .scene
                .restyle(row, |st| *st = super::style::Style::plain()),
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
        if let Some(note) = self.scene.note.take() {
            self.message = Some(note);
        }
        self.restart();
    }

    fn key(&mut self, k: KeyEvent) {
        if self.search.is_some() {
            self.search_key(k);
            return;
        }
        if self.prompt.is_some() {
            self.prompt_key(k);
            return;
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
                    || !std::mem::take(&mut self.marked).is_empty()
                    || self.scene.clear_suggestions()
                    || self.scene.clear_pick()
                    || self.scene.focus.take().is_some();
                if !cleared && k.code == KeyCode::Esc {
                    self.quit = true;
                }
                self.restart();
            }
            KeyCode::Char('e') => self.open_menu(),
            KeyCode::Char('g') => self.change(|s| s.step_feature(1)),
            KeyCode::Char('G') => self.change(|s| s.step_feature(-1)),
            KeyCode::Char('a') => self.change(Scene::pick_marker_set),
            KeyCode::Char('o') => self.change(Scene::toggle_source),
            KeyCode::Char('/') => {
                if self.names.is_none() {
                    self.names = Some(self.scene.searchable());
                }
                if self.names.as_ref().is_some_and(Vec::is_empty) {
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
            KeyCode::Char('c') => {
                self.scene.cycle_colour();
                self.restart();
            }
            KeyCode::Char(']') => {
                self.scene.step_focus(1);
                self.restart();
            }
            KeyCode::Char('[') => {
                self.scene.step_focus(-1);
                self.restart();
            }
            KeyCode::Char('t') => {
                self.scene.show_labels = !self.scene.show_labels;
                self.restart();
            }
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
            KeyCode::Char('v') => self.toggle_mark(),
            KeyCode::Char('L') => self.begin(Action::Label),
            KeyCode::Char('M') => self.begin(Action::Merge),
            KeyCode::Char('K') => self.begin(Action::Keep),
            KeyCode::Char('A') => self.begin(Action::MarkersAdd),
            KeyCode::Char('D') => self.begin(Action::MarkersDrop),
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
            KeyCode::Char('h') | KeyCode::Left => {
                let d = step(self);
                self.with_vp(|v| v.pan_px(d, 0.0));
            }
            KeyCode::Char('l') | KeyCode::Right => {
                let d = step(self);
                self.with_vp(|v| v.pan_px(-d, 0.0));
            }
            KeyCode::Char('k') | KeyCode::Up => {
                let d = step(self);
                self.with_vp(|v| v.pan_px(0.0, d));
            }
            KeyCode::Char('j') | KeyCode::Down => {
                let d = step(self);
                self.with_vp(|v| v.pan_px(0.0, -d));
            }
            KeyCode::Char('s') => self.save(),
            _ => {}
        }
    }

    /// The cluster a decision is about: the focused cluster when colouring by
    /// cluster, else the one last clicked.
    fn target(&self) -> Option<i64> {
        self.scene.focused_cluster().or(self.clicked)
    }

    fn toggle_mark(&mut self) {
        let Some(c) = self.target() else {
            self.message = Some("focus or click a cluster to mark it".into());
            return;
        };
        if let Some(i) = self.marked.iter().position(|&m| m == c) {
            self.marked.remove(i);
        } else {
            self.marked.push(c);
        }
    }

    /// Start typing a decision of kind `action`.
    fn begin(&mut self, action: Action) {
        if self.watcher.is_none() {
            self.watcher = Watcher::find(&self.from, self.decisions.as_deref());
        }
        if self.watcher.is_none() {
            self.message = Some(
                "no `lupin relabel --watch` found beside this round; start one, or pass --decisions"
                    .into(),
            );
            return;
        }
        let mut decision = Decision {
            action,
            clusters: Vec::new(),
            features: Vec::new(),
            label: String::new(),
            rationale: String::new(),
            evidence: Vec::new(),
        };
        let mut input = String::new();
        match action {
            Action::Merge => {
                if self.marked.len() < 2 {
                    self.message = Some("mark two or more clusters with v first".into());
                    return;
                }
                decision.clusters = self.marked.clone();
            }
            Action::Label | Action::Keep => {
                let Some(c) = self.target() else {
                    self.message = Some("focus or click a cluster first".into());
                    return;
                };
                decision.clusters = vec![c];
                let (label, top) = self.scene.cluster_call(c);
                if let Some((call, support)) = top {
                    decision.evidence.push(serde_json::json!({
                        "kind": "marker", "term": call.clone(), "stat": "support", "value": support,
                    }));
                    input = call;
                }
                if action == Action::Keep {
                    let Some(l) = label else {
                        self.message = Some(format!("C{c} has no current call to keep"));
                        return;
                    };
                    decision.label = l;
                }
            }
            Action::MarkersAdd | Action::MarkersDrop => {
                let Some(Pick::One(f)) = self.scene.pick.clone() else {
                    self.message = Some("show a feature first (n, g or /)".into());
                    return;
                };
                if let Some(v) = self.scene.suggestion_score(&f) {
                    decision.evidence.push(serde_json::json!({
                        "kind": "marker", "term": f.as_ref(), "stat": "expected_lfc", "value": v,
                    }));
                }
                decision.features = vec![f];
                input = self
                    .scene
                    .focused_name()
                    .map(|n| n.to_string())
                    .unwrap_or_default();
            }
        }
        let why = !action.needs_label();
        self.prompt = Some(Prompt {
            decision,
            why,
            input: if why { String::new() } else { input },
            known: self.scene.known_labels(),
        });
    }

    fn prompt_key(&mut self, k: KeyEvent) {
        let Some(p) = self.prompt.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                self.prompt = None;
                self.message = Some("decision cancelled".into());
            }
            KeyCode::Backspace => {
                p.input.pop();
            }
            KeyCode::Tab if !p.why => {
                let low = p.input.to_lowercase();
                if let Some(k) = p.known.iter().find(|k| k.to_lowercase().starts_with(&low)) {
                    p.input = k.to_string();
                }
            }
            KeyCode::Enter => {
                let text = p.input.trim().to_string();
                if text.is_empty() {
                    return;
                }
                if p.why {
                    p.decision.rationale = text;
                    let p = self.prompt.take().expect("checked above");
                    self.send(&p.decision);
                } else {
                    p.decision.label = text;
                    p.why = true;
                    p.input.clear();
                }
            }
            KeyCode::Char(c) => p.input.push(c),
            _ => {}
        }
    }

    fn send(&mut self, d: &Decision) {
        let Some(w) = &self.watcher else { return };
        let line = d.to_json(&w.round_ref(&self.from));
        self.message = Some(match w.append(&line) {
            Ok(()) => {
                if d.action == Action::Merge {
                    self.marked.clear();
                }
                format!(
                    "sent: {} {} · lupin will write the next round",
                    d.action.name(),
                    d.label
                )
            }
            Err(e) => format!("could not write the decision: {e}"),
        });
    }

    /// Open the watcher's latest round when it moves on, and show a refused
    /// batch's reason. Returns whether a round was opened.
    fn follow_watcher(&mut self) -> bool {
        let Some(w) = self.watcher.as_mut() else {
            return false;
        };
        if !w.refresh() {
            return false;
        }
        if let Some(e) = &w.error {
            self.message = Some(format!("lupin refused the last decisions: {e}"));
        }
        let latest = w.latest.clone();
        match latest {
            Some(l) if !same_file(&l, &self.from) => {
                self.open_round(&l, "new round");
                true
            }
            _ => false,
        }
    }

    /// Start laying out the focused group's cells on a worker thread.
    fn start_zoom(&mut self) {
        if self.zooming.is_some() {
            return;
        }
        match self.scene.zoom_request() {
            Ok(super::ZoomRequest {
                label,
                names,
                geometry,
            }) => {
                let (tx, rx) = std::sync::mpsc::channel();
                let n = names.len();
                std::thread::spawn(move || {
                    let _ = tx.send(geometry.layout(&names).map_err(|e| e.to_string()));
                });
                self.message = Some(format!("laying out {n} cells of {label}…"));
                self.zooming = Some(Zooming {
                    parent: self.scene.space,
                    label,
                    done: rx,
                });
            }
            Err(e) => self.message = Some(e),
        }
    }

    /// Take a finished zoom, if any, and switch to it. Returns whether it did.
    fn finish_zoom(&mut self) -> bool {
        let Some(z) = &self.zooming else {
            return false;
        };
        let result = match z.done.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Err("the layout worker stopped".into())
            }
        };
        let z = self.zooming.take().expect("checked above");
        match result {
            Ok((names, xy)) => {
                let n = names.len();
                self.scene.add_zoomed(z.parent, &z.label, names, xy);
                self.vp = None;
                self.message = Some(format!(
                    "{} · {n} cells laid out on their own · Z to go back",
                    z.label
                ));
            }
            Err(e) => self.message = Some(e),
        }
        true
    }

    /// Open the manifest at `path` (a reload, or another round), keeping the
    /// camera, layout, grouping and focus where they still apply.
    fn open_round(&mut self, path: &std::path::Path, what: &str) {
        match super::Dataset::load(&path.to_string_lossy()) {
            Ok(data) => {
                let before = self.scene.current().points.bounds;
                self.scene.replace_data(data);
                self.stamp = modified(path);
                self.from = path.to_path_buf();
                self.info = None;
                if self.scene.current().points.bounds != before {
                    self.vp = None;
                }
                let name = path
                    .file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                self.message = Some(format!("{what} {name}"));
                self.restart();
            }
            Err(e) => {
                // Keep showing what we have; a half-written file will be
                // picked up on the next check.
                self.stamp = modified(path);
                self.message = Some(format!("could not open {}: {e}", path.display()));
            }
        }
    }

    /// Step to the source round (`back`) or to the round made from this one.
    fn step_round(&mut self, back: bool) {
        let round = self.scene.data.round.as_ref();
        let target = if back {
            round.and_then(|r| r.source.clone())
        } else {
            round.and_then(super::rounds::Round::newer)
        };
        match target {
            Some(p) => self.open_round(&p, if back { "source round" } else { "newer round" }),
            None => {
                self.message = Some(if back {
                    "this round records no source round".into()
                } else {
                    "no round made from this one next to it".into()
                });
            }
        }
    }

    fn switch_space(&mut self, i: usize) {
        self.scene.set_space(i);
        self.vp = None;
    }

    /// Same axis and title on the next method, if that method has it.
    fn next_method(&mut self) {
        let spaces = &self.scene.data.spaces;
        let cur = self.scene.current();
        let mut methods: Vec<&str> = Vec::new();
        for s in spaces {
            if !methods.contains(&s.method.as_str()) {
                methods.push(&s.method);
            }
        }
        if methods.len() < 2 {
            self.message = Some("only one layout method in this run".into());
            return;
        }
        let at = methods.iter().position(|m| *m == cur.method).unwrap_or(0);
        for k in 1..methods.len() {
            let m = methods[(at + k) % methods.len()];
            let same = spaces
                .iter()
                .position(|s| s.method == m && s.kind == cur.kind);
            let any = spaces.iter().position(|s| s.method == m);
            if let Some(i) = same.or(any) {
                self.switch_space(i);
                return;
            }
        }
    }

    fn save(&mut self) {
        let Some(vp) = self.vp else { return };
        let s = self.scene.current();
        let path = format!(
            "{}.view.{}.{}.png",
            self.scene.data.prefix,
            s.method,
            s.kind.slug()
        );
        let img = super::render_full(&self.scene, vp, self.cell.1);
        self.message = Some(match img.save(&path) {
            Ok(()) => format!("saved {path}"),
            Err(e) => format!("save failed: {e}"),
        });
    }

    fn cell_to_px(&self, col: u16, row: u16) -> Option<(f32, f32)> {
        let m = self.map;
        (col >= m.x && row >= m.y && col < m.x + m.width && row < m.y + m.height).then(|| {
            (
                (f32::from(col - m.x) + 0.5) * self.cell.0,
                (f32::from(row - m.y) + 0.5) * self.cell.1,
            )
        })
    }

    fn mouse(&mut self, m: MouseEvent) {
        let Some((px, py)) = self.cell_to_px(m.column, m.row) else {
            return;
        };
        match m.kind {
            MouseEventKind::ScrollUp => self.with_vp(|v| v.zoom_at(1.25, px, py)),
            MouseEventKind::ScrollDown => self.with_vp(|v| v.zoom_at(0.8, px, py)),
            MouseEventKind::Down(MouseButton::Left) => self.drag = Some((m.column, m.row, false)),
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((c, r, _)) = self.drag {
                    let dx = (f32::from(m.column) - f32::from(c)) * self.cell.0;
                    let dy = (f32::from(m.row) - f32::from(r)) * self.cell.1;
                    self.drag = Some((m.column, m.row, true));
                    self.with_vp(|v| v.pan_px(dx, dy));
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if let Some((_, _, false)) = self.drag.take() {
                    self.click(px, py);
                }
            }
            _ => {}
        }
    }

    /// Focus the group of the point nearest the click, and name the point.
    fn click(&mut self, px: f32, py: f32) {
        let Some(vp) = self.vp else { return };
        let pts = &self.scene.current().points;
        let reach = 2.0 * self.cell.0.max(8.0);
        let best = pts
            .order
            .iter()
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
            .filter(|&g| g != super::NONE);
        let on_features = self.scene.current().axis() == super::Axis::Features;
        if !on_features {
            self.clicked = self.scene.cluster_id_of(&name);
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
}

/// Features matching `query`: exact name first, then symbol (the part after
/// an `ID_` prefix), then prefix, then substring; case-insensitive.
fn search(names: &[Box<str>], query: &str) -> Vec<Box<str>> {
    const MAX: usize = 8;
    let q = query.to_lowercase();
    if q.is_empty() {
        return Vec::new();
    }
    let rank = |n: &str| {
        let n = n.to_lowercase();
        let symbol = n.rsplit_once('_').map_or(n.as_str(), |(_, s)| s);
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
        .filter_map(|n| rank(n).map(|r| (r, n.len(), n)))
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
    fn search_ranks_exact_then_symbol_then_prefix_then_substring() {
        let names: Vec<Box<str>> = ["ENSG1_GENE10", "GENE1", "XGENE1Y", "ENSG2_GENE1", "OTHER"]
            .map(Into::into)
            .to_vec();
        let hits = search(&names, "gene1");
        assert_eq!(&*hits[0], "GENE1");
        assert_eq!(&*hits[1], "ENSG2_GENE1");
        assert_eq!(&*hits[2], "ENSG1_GENE10");
        assert_eq!(&*hits[3], "XGENE1Y");
        assert_eq!(hits.len(), 4);
        assert!(search(&names, "").is_empty());
    }
}

/// Step one style property of a group by `step` (±1).
fn adjust(st: &mut super::style::Style, field: usize, step: i64, current: Option<color::Rgb>) {
    match field {
        0 => {
            let sw = swatches();
            let enc = color::Encoder::new();
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

/// The sidebar's frame: one thin rule on the map side, nothing else.
fn side_block() -> Block<'static> {
    Block::new()
        .borders(ratatui::widgets::Borders::LEFT)
        .border_style(Style::default().fg(rgb(color::MUTED)))
}
