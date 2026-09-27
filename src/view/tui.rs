//! Terminal front end: event loop, keys and mouse, progressive redraw.
//!
//! Each change of view starts a new render job. While a job runs the loop
//! polls without blocking, draws one chunk, and pushes the composite, so the
//! picture appears at once and fills in; any key or mouse event in between
//! simply replaces the job.

use super::color;
use super::render::{Job, Viewport};
use super::{Graphics, Scene};
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

const KEYS: &str = "tab space · m method · c colour · [ ] focus · t labels · s save · ? help";

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
    ("x / esc", "clear focus"),
    ("t", "text labels on / off"),
    ("+  -  scroll", "zoom"),
    ("hjkl  arrows  drag", "pan"),
    ("0", "reset view"),
    ("s", "save this view as PNG"),
    ("q", "quit"),
];

fn rgb(c: [u8; 3]) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

pub fn run(scene: Scene, graphics: Graphics) -> anyhow::Result<()> {
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
        App::new(scene, picker).run(&mut terminal)
    })();
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

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
    quit: bool,
}

impl App {
    fn new(scene: Scene, picker: Picker) -> Self {
        let f = picker.font_size();
        Self {
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
            let busy = self.job.is_some();
            let wait = if busy {
                Duration::ZERO
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
    fn layout(&mut self, area: Rect) {
        let map = Rect {
            height: area.height.saturating_sub(1),
            ..area
        };
        if map != self.map {
            self.map = map;
            self.vp = None;
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
        let [map, status] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(f.area());
        if let Some(p) = &self.proto {
            f.render_widget(Image::new(p), map);
        }

        let left = self.message.clone().unwrap_or_else(|| self.scene.caption());
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

    fn key(&mut self, k: KeyEvent) {
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
                if self.scene.focus.take().is_none() && k.code == KeyCode::Esc {
                    self.quit = true;
                }
                self.restart();
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
            KeyCode::Char('0' | 'r') => self.vp = None,
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
                .position(|s| s.method == m && s.title == cur.title);
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
            s.title.replace(' ', "_")
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
        self.message = Some(match group {
            Some(g) => {
                self.scene.focus = Some(g);
                format!("{name} · {}", self.scene.levels()[g as usize])
            }
            None => name,
        });
        self.restart();
    }
}
