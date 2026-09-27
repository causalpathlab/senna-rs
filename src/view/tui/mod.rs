//! Terminal front end: event loop, keys and mouse, progressive redraw.
//!
//! Each change of view starts a new render job. While a job runs the loop
//! polls without blocking, draws one chunk, and pushes the composite, so the
//! picture appears at once and fills in; any key or mouse event in between
//! simply replaces the job.

mod decisions;
mod draw;
mod input;

use decisions::Prompt;

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
    lupin: String,
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
        App::new(scene, picker, from, lupin).run(&mut terminal)
    })();
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

type ZoomResult = Result<super::sublayout::Laid, String>;

/// A decision lupin is applying on a worker thread.
struct Relabeling {
    merge: bool,
    done: std::sync::mpsc::Receiver<Result<crate::view::decide::Reply, String>>,
}

/// A zoom into one group, running on a worker thread.
struct Zooming {
    parent: usize,
    label: String,
    done: std::sync::mpsc::Receiver<ZoomResult>,
}

/// A left-button press: where the pointer last was, and whether it moved
/// (a drag pans; a press that did not move is a click).
struct Drag {
    col: u16,
    row: u16,
    moved: bool,
}

/// Feature names to search, with their lowercase forms computed once.
struct SearchNames {
    names: Vec<Box<str>>,
    lower: Vec<String>,
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
    drag: Option<Drag>,
    /// Feature search in progress: the query and its current matches.
    search: Option<(String, Vec<Box<str>>)>,
    /// Searchable feature names, loaded on the first `/`.
    names: Option<SearchNames>,
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
    /// A `lupin relabel --watch` on this chain, if one runs; its latest
    /// round is followed.
    watcher: Option<Watcher>,
    /// The lupin binary decisions are handed to.
    lupin: String,
    /// A decision lupin is applying.
    relabeling: Option<Relabeling>,
    /// Clusters marked for a merge (`v`), and the last one clicked.
    marked: Vec<i64>,
    clicked: Option<i64>,
    /// A decision being typed.
    prompt: Option<Prompt>,
    quit: bool,
}

impl App {
    fn new(scene: Scene, picker: Picker, from: std::path::PathBuf, lupin: String) -> Self {
        let f = picker.font_size();
        Self {
            watcher: Watcher::find(&from),
            lupin,
            relabeling: None,
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
            if self.finish_zoom() || self.finish_relabel() {
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
            } else if self.zooming.is_some() || self.relabeling.is_some() {
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

    /// Whether the sidebar has something to show.
    fn side_content(&self) -> bool {
        self.menu.is_some() || self.info.is_some() || self.scene.has_suggestions()
    }

    /// Map and sidebar areas and the viewport for the current terminal size.
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
            let (w, h) = self.map_px();
            match self.vp.as_mut() {
                Some(vp) if had_map => {
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
        drop(layers);
        let img = if done {
            let job = self.job.take().expect("checked above");
            let vp = job.vp;
            let mut frame = job.finish();
            self.scene.decorate(&mut frame, &vp, self.cell.1);
            frame.to_image()
        } else {
            job.image()
        };
        let size = Size::new(self.map.width, self.map.height);
        self.proto = Some(self.picker.new_protocol(
            DynamicImage::ImageRgba8(img),
            size,
            Resize::Fit(Some(image::imageops::FilterType::Triangle)),
        )?);
        Ok(true)
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
}
