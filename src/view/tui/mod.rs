//! Terminal front end: event loop, keys and mouse, progressive redraw.
//!
//! Each change of view starts a new render job. While a job runs the loop
//! polls without blocking, draws one chunk, and pushes the composite, so the
//! picture appears at once and fills in; any key or mouse event in between
//! simply replaces the job.

mod annotate;
mod browse;
mod decisions;
mod draw;
mod grid;
mod help;
mod input;
mod modal;
mod recompute;
mod relabel;

pub use browse::{pick_run, shown};
use decisions::Prompt;
use modal::Modal;

use super::color;
use super::decide::{Action, Decision, Mode, Rescore, Watcher};
use super::files::{self, modified, same_file};
use super::render::{Frame, Job, Viewport};
use super::style::{swatches, Shape};
use super::{Axis, Graphics, Pick, Scene};
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

/// A run prefix as a name in the directory the viewer runs from, where
/// figures are saved unless a path is typed.
fn here(prefix: &str) -> String {
    files::name(std::path::Path::new(prefix))
}

fn rgb(c: [u8; 3]) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

/// Open one session per run and hand the terminal to them.
pub fn run(
    sessions: Vec<(Scene, std::path::PathBuf)>,
    graphics: Graphics,
    lupin: String,
) -> anyhow::Result<()> {
    // Log lines written to the terminal would land over the screen: none
    // while it is ours. What matters reaches the status line instead.
    let level = log::max_level();
    log::set_max_level(log::LevelFilter::Off);
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
        let apps = sessions
            .into_iter()
            .map(|(scene, from)| App::new(scene, picker.clone(), from, lupin.clone()))
            .collect();
        grid::Deck::new(apps, picker).run(&mut terminal)
    })();
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    log::set_max_level(level);
    result
}

/// `img` ready to draw into `area`, scaled to fit it.
fn protocol(picker: &Picker, img: image::RgbaImage, area: Rect) -> anyhow::Result<Protocol> {
    Ok(picker.new_protocol(
        DynamicImage::ImageRgba8(img),
        Size::new(area.width, area.height),
        Resize::Fit(Some(image::imageops::FilterType::Triangle)),
    )?)
}

/// How long a toast stays up.
const TOAST_FOR: Duration = Duration::from_millis(2500);

/// Lines at the bottom: what is on screen, then two lines of keys.
const STATUS_LINES: u16 = 3;

/// The answer of work running on a worker thread.
struct Pending<T>(std::sync::mpsc::Receiver<Result<T, String>>);

impl<T: Send + 'static> Pending<T> {
    fn spawn(work: impl FnOnce() -> Result<T, String> + Send + 'static) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(work());
        });
        Self(rx)
    }

    /// The answer once it has come; `stopped` when the worker died first.
    fn poll(&self, stopped: &str) -> Option<Result<T, String>> {
        match self.0.try_recv() {
            Ok(r) => Some(r),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(Err(stopped.into())),
        }
    }
}

/// A decision lupin is applying on a worker thread.
struct Relabeling {
    job: RelabelJob,
    started: std::time::Instant,
    /// What was sent, one line per decision.
    sent: Vec<String>,
    /// The latest line lupin logged, for the status line.
    progress: std::sync::Arc<std::sync::Mutex<String>>,
    done: Pending<crate::view::decide::Reply>,
}

/// What a lupin call was for: the whole relabel draft, or a first
/// annotation of the run.
enum RelabelJob {
    Draft(Mode),
    Annotate,
}

/// A zoom into one group, running on a worker thread.
struct Zooming {
    parent: usize,
    label: String,
    done: Pending<super::sublayout::Laid>,
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
    /// The last finished frame before labels, and its viewport, so a
    /// change of labels only redraws them.
    base: Option<(Viewport, Frame)>,
    proto: Option<Protocol>,
    message: Option<String>,
    help: bool,
    drag: Option<Drag>,
    /// What is being typed in the status line, if anything.
    modal: Option<Modal>,
    /// Searchable feature names, loaded on the first `/`.
    names: Option<SearchNames>,
    menu: Option<Menu>,
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
    /// The cluster overview on the left of the map (relabel mode only).
    left: Rect,
    /// Whether the sidebar may open (`b` toggles).
    sidebar: bool,
    /// A `lupin relabel --watch` on this chain, whose latest round is followed.
    watcher: Option<Watcher>,
    lupin: String,
    relabeling: Option<Relabeling>,
    /// senna recomputing layouts or clusterings of this run (`r`).
    recomputing: Option<recompute::Recomputing>,
    /// lupin rescoring the round against the staged marker edits.
    rescoring: Option<Rescore>,
    /// A short popup over the map (lupin answered), and when it goes.
    toast: Option<(String, std::time::Instant)>,
    quit: bool,
}

impl App {
    fn new(scene: Scene, picker: Picker, from: std::path::PathBuf, lupin: String) -> Self {
        let f = picker.font_size();
        // Opened on an older round: say where the newest work is.
        let newer = scene
            .data
            .round
            .as_ref()
            .and_then(crate::view::rounds::Round::newer)
            .map(|n| {
                let name = files::name(&n);
                format!("a newer round exists ({name}) · . opens it")
            });
        Self {
            watcher: Watcher::find(&from),
            lupin,
            relabeling: None,
            recomputing: None,
            rescoring: None,
            toast: None,
            modal: None,
            stamp: modified(&from),
            from,
            checked: std::time::Instant::now(),
            info: None,
            side: Rect::default(),
            left: Rect::default(),
            sidebar: true,
            scene,
            cell: (f32::from(f.width.max(1)), f32::from(f.height.max(1))),
            picker,
            map: Rect::default(),
            vp: None,
            job: None,
            base: None,
            proto: None,
            message: newer,
            help: false,
            drag: None,
            names: None,
            menu: None,
            zooming: None,
            quit: false,
        }
    }

    /// Work that goes on whether or not this view is on screen: a zoom
    /// laid out, lupin answering, the file changing on disk, the watched
    /// chain moving on, a toast running out. Returns whether the view
    /// changed (on screen, lupin's elapsed time counts as a change).
    fn background(&mut self, on_screen: bool) -> bool {
        let mut changed = self.finish_zoom()
            | self.finish_relabel()
            | self.finish_recompute()
            | self.keep_scores_current();
        // Not while drawing on screen: a reload would restart the picture;
        // nor while senna rewrites the run, which reloads when it is done.
        let idle = (!on_screen || self.job.is_none()) && self.recomputing.is_none();
        if idle && self.checked.elapsed() >= Duration::from_secs(1) {
            self.checked = std::time::Instant::now();
            if modified(&self.from) != self.stamp {
                let from = self.from.clone();
                self.open_round(&from, "reloaded");
                changed = true;
            } else {
                changed |= self.follow_watcher();
            }
        }
        if grid::expired(&self.toast) {
            self.toast = None;
            changed = true;
        }
        // On screen, lupin's or senna's elapsed time keeps ticking.
        changed || (on_screen && (self.relabeling.is_some() || self.recomputing.is_some()))
    }

    /// How long to wait for input while this view is on screen.
    fn wait(&self) -> Duration {
        if self.job.is_some() {
            Duration::ZERO
        } else if self.zooming.is_some()
            || self.relabeling.is_some()
            || self.recomputing.is_some()
            || self.rescoring.is_some()
            || self.toast.is_some()
        {
            Duration::from_millis(50)
        } else {
            Duration::from_millis(250)
        }
    }

    /// The sidebar's text, when it shows text rather than the style menu.
    fn side_lines(&self) -> Option<Vec<String>> {
        self.scene
            .merge_lines()
            .or_else(|| self.scene.review_lines())
            .or_else(|| self.info.clone())
            .or_else(|| self.scene.suggestion_lines())
            .or_else(|| self.scene.near_lines())
    }

    /// Map and sidebar areas and the viewport for the current terminal size.
    fn layout(&mut self, area: Rect) {
        let body = Rect {
            height: area.height.saturating_sub(STATUS_LINES),
            ..area
        };
        // The sidebar takes its columns from the map rather than covering it.
        let side_w = if self.sidebar && (self.menu.is_some() || self.side_lines().is_some()) {
            (area.width / 3)
                .clamp(30, 48)
                .min(area.width.saturating_sub(20))
        } else {
            0
        };
        // Relabel mode adds the cluster overview on the other side.
        let left_w = if self.sidebar && self.scene.review.is_some() {
            (area.width / 5)
                .clamp(26, 40)
                .min(area.width.saturating_sub(side_w + 20))
        } else {
            0
        };
        let map = Rect {
            x: body.x + left_w,
            width: body.width - side_w - left_w,
            ..body
        };
        self.left = Rect::new(body.x, body.y, left_w, body.height);
        self.side = Rect::new(map.x + map.width, body.y, side_w, body.height);
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

    /// Pixel size of the view: its camera's, else the map area's.
    fn view_px(&self) -> (f32, f32) {
        match self.vp {
            Some(v) => (v.w as f32, v.h as f32),
            None => {
                let (w, h) = self.map_px();
                (w as f32, h as f32)
            }
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
        self.base = None;
        if self.scene.chart.is_some() {
            return self.draw_chart();
        }
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
            let base = job.finish();
            let mut frame = base.clone();
            self.scene.decorate(&mut frame, &vp, self.cell.1);
            self.base = Some((vp, base));
            frame.to_image()
        } else {
            job.image()
        };
        self.show(img)?;
        Ok(true)
    }

    fn show(&mut self, img: image::RgbaImage) -> anyhow::Result<()> {
        self.proto = Some(protocol(&self.picker, img, self.map)?);
        Ok(())
    }

    /// The chart on screen, made for the current grouping and drawn whole.
    fn draw_chart(&mut self) {
        let Some(vp) = self.vp.filter(|v| v.w > 0 && v.h > 0) else {
            return;
        };
        self.scene.refresh_chart();
        if let Some(frame) = self.scene.chart_frame(vp.w, vp.h, self.cell.1, false) {
            if let Err(e) = self.show(frame.to_image()) {
                self.message = Some(e.to_string());
            }
        }
        if let Some(note) = self.scene.note.take() {
            self.message = Some(note);
        }
    }

    /// Redraw labels and marks over the last finished frame, when only they
    /// changed. A running job decorates when it finishes.
    fn redecorate(&mut self) {
        if self.scene.chart.is_some() {
            return self.draw_chart();
        }
        if self.job.is_some() {
            return;
        }
        let img = match &self.base {
            Some((vp, base)) if Some(*vp) == self.vp => {
                let mut frame = base.clone();
                self.scene.decorate(&mut frame, vp, self.cell.1);
                frame.to_image()
            }
            _ => return self.restart(),
        };
        if self.show(img).is_err() {
            self.restart();
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
                let n = names.len();
                let done =
                    Pending::spawn(move || geometry.layout(&names).map_err(|e| e.to_string()));
                self.message = Some(format!("laying out {n} cells of {label}…"));
                self.zooming = Some(Zooming {
                    parent: self.scene.space,
                    label,
                    done,
                });
            }
            Err(e) => self.message = Some(e),
        }
    }

    /// Take a finished zoom, if any, and switch to it. Returns whether it did.
    fn finish_zoom(&mut self) -> bool {
        let Some(result) = self
            .zooming
            .as_ref()
            .and_then(|z| z.done.poll("the layout worker stopped"))
        else {
            return false;
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
                let name = files::name(path);
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

    /// Ctrl-R / Ctrl-L: read the run again and draw it afresh. In relabel
    /// mode only the drawing: a reload would end the mode.
    fn refresh(&mut self) {
        if self.scene.review.is_some() {
            self.message = Some("redrawn · the run reloads outside relabel mode".into());
            self.restart();
        } else {
            let from = self.from.clone();
            self.open_round(&from, "reloaded");
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

    /// Cells, features on cells, features: the next view of this layout.
    fn step_view(&mut self, step: isize) {
        let Some(i) = self.scene.next_view(step) else {
            self.message = Some(if self.scene.data.has_axis(Axis::Features) {
                "only one view in this run".into()
            } else {
                format!(
                    "no feature map; `senna layout {} --target features --from <run>` adds one",
                    self.scene.current().method
                )
            });
            return;
        };
        self.switch_space(i);
        let s = self.scene.current();
        self.message = Some(format!("{} · {}", s.method, s.title()));
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

    /// Where this view's PDF goes unless another name is typed (`.pdf`
    /// follows).
    fn pdf_name(&self) -> String {
        let s = self.scene.current();
        let what = match self.scene.chart.as_ref().map(|c| c.kind) {
            Some(crate::view::chart::Kind::Structure) => "structure".to_string(),
            Some(crate::view::chart::Kind::Heatmap) => "heatmap".to_string(),
            None => format!("{}.{}", s.method, s.kind.slug()),
        };
        format!("{}.view.{what}", here(&self.scene.data.prefix))
    }
}
