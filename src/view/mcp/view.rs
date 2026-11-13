//! The view tools: one run held open between calls, drawn as the
//! terminal viewer draws it.

use super::jobs::senna;
use super::{absolute, parse};
use crate::view::data::{Axis, Dataset, LabelKind, Labels, NONE};
use crate::view::recompute::Step;
use crate::view::render::{self, Viewport};
use crate::view::{
    color, deck, features, prepare_run, state, velocity, Pick, Scene, Source, VelocityStyle,
    ViewArgs,
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use image::codecs::png::CompressionType;
use legume_mcp::{Call, Content, ToolSpec};
use legume_tui::export::PageSize;
use senna::run_manifest;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Width of a map drawn without one: enough to read a layout by, small
/// enough to keep the image cheap for the agent.
const WIDTH: usize = 1024;

/// Largest side of a map drawn or exported.
const MAX_SIDE: usize = 4096;

/// Text size of a map drawn for the agent, as a terminal cell's height.
const CELL_PX: f32 = 18.0;

/// Width and resolution of an exported figure.
const EXPORT_WIDTH: usize = 2400;
const EXPORT_DPI: u32 = 300;

/// How far from a pixel of the last map `view_cell` looks for a point.
const REACH_PX: f32 = 12.0;

/// Features `view_markers` lists without a count.
const MARKERS: usize = 12;

/// A view tool's call on the run held open; `None` for another family's.
pub(super) fn call(
    open: &mut Option<Open>,
    name: &str,
    args: &Value,
    cx: &Call,
) -> Result<Option<Vec<Content>>> {
    let out = match name {
        "view_open" => {
            let open = open.insert(Open::new(&parse(args.clone())?, cx)?);
            cx.state(&open.view_state());
            vec![Content::Json(open.summary())]
        }
        "view_map" => {
            let open = held(open)?;
            let out = open.map(parse(args.clone())?)?;
            // What a person following the server sees next.
            cx.state(&open.view_state());
            out
        }
        "view_markers" => held(open)?.markers(parse(args.clone())?)?,
        "view_cell" => held(open)?.cell(parse(args.clone())?)?,
        "view_export" => held(open)?.export(parse(args.clone())?)?,
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// The run held open, for every view tool but `view_open`.
fn held(open: &mut Option<Open>) -> Result<&mut Open> {
    open.as_mut()
        .ok_or_else(|| anyhow!("no run open: call view_open first"))
}

/// An open run and the view on it.
pub(super) struct Open {
    scene: Scene,
    run: PathBuf,
    /// The world window asked for; `None` fits the whole map.
    window: Option<[f32; 4]>,
    /// The last map drawn, for `view_cell` pixels.
    drawn: Option<Viewport>,
}

/// `width`, or `default`, within what a picture may be.
fn width(width: Option<usize>, default: usize) -> Result<usize> {
    let width = width.unwrap_or(default);
    ensure!(
        (1..=MAX_SIDE).contains(&width),
        "width must be 1 to {MAX_SIDE}"
    );
    Ok(width)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    run: String,
    map: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(in crate::view) struct MapArgs {
    round: Option<String>,
    map: Option<String>,
    space: Option<String>,
    colour_by: Option<String>,
    focus: Option<Value>,
    feature: Option<String>,
    observed: Option<bool>,
    time: Option<bool>,
    velocity: Option<VelocityArg>,
    pub(in crate::view) bbox: Option<[f32; 4]>,
    fit: Option<bool>,
    center: Option<[f32; 2]>,
    zoom: Option<f32>,
    width: Option<usize>,
}

/// How the run's velocity field is drawn over the cells, or not.
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum VelocityArg {
    Stream,
    Arrows,
    Off,
}

impl VelocityArg {
    fn style(self) -> Option<VelocityStyle> {
        match self {
            Self::Stream => Some(VelocityStyle::Stream),
            Self::Arrows => Some(VelocityStyle::Arrows),
            Self::Off => None,
        }
    }
}

impl From<Option<VelocityStyle>> for VelocityArg {
    fn from(style: Option<VelocityStyle>) -> Self {
        match style {
            Some(VelocityStyle::Stream) => Self::Stream,
            Some(VelocityStyle::Arrows) => Self::Arrows,
            None => Self::Off,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MarkersArgs {
    group: String,
    n: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CellArgs {
    px: Option<[f32; 2]>,
    at: Option<[f32; 2]>,
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportArgs {
    png: Option<PathBuf>,
    pdf: Option<PathBuf>,
    width: Option<usize>,
    #[serde(default)]
    overwrite: bool,
}

/// The view's arguments as `senna view {run}` would take them.
fn view_args(run: &str, map: Option<&str>) -> Result<ViewArgs> {
    #[derive(clap::Parser)]
    struct Argv {
        #[command(flatten)]
        view: ViewArgs,
    }
    let mut argv = vec!["view", run];
    if let Some(m) = map {
        argv.extend(["--method", m]);
    }
    Ok(<Argv as clap::Parser>::try_parse_from(argv)?.view)
}

impl Open {
    fn new(a: &OpenArgs, cx: &Call) -> Result<Self> {
        let args = view_args(&a.run, a.map.as_deref())?;
        // Missing layouts and clusters, as `senna view` computes them, but
        // as children whose output stays off the protocol.
        prepare_run(&args, &a.run, &|why, argv, outputs| {
            cx.progress(why);
            let outputs: Vec<PathBuf> = outputs.iter().map(|o| absolute(Path::new(o))).collect();
            cx.run_making(senna(argv)?, &outputs, why).map(drop)
        })?;
        // A step that failed only warns, so a cancelled one is seen here.
        ensure!(!cx.is_cancelled(), "cancelled");
        cx.progress(&format!("reading {}", a.run));
        let mut data = Dataset::load(&a.run)?;
        data.with_batches();
        let scene = Scene::new(data, &args);
        Ok(Self::of(scene, absolute(Path::new(&a.run))))
    }

    fn of(scene: Scene, run: PathBuf) -> Self {
        Self {
            scene,
            run,
            window: None,
            drawn: None,
        }
    }

    fn summary(&self) -> Value {
        let s = &self.scene;
        let maps: Vec<Value> = s
            .data
            .spaces
            .iter()
            .filter(|sp| sp.parent.is_none())
            .map(|sp| json!({"map": sp.method, "space": sp.kind.slug(), "points": sp.points.names.len()}))
            .collect();
        let groupings: Vec<Value> = s
            .data
            .labels
            .iter()
            .map(|l| {
                json!({
                    "name": l.title(),
                    "of": if l.axis() == Axis::Cells { "cells" } else { "features" },
                    "groups": l.levels.len(),
                })
            })
            .collect();
        // What else a run can colour or draw the cells by.
        let velocity = s.velocity.is_some()
            || s.data
                .run
                .as_ref()
                .is_some_and(|(m, _)| velocity::Field::available(m));
        json!({
            "run": self.run,
            "maps": maps,
            "groupings": groupings,
            "time": s.data.times.is_some(),
            "velocity": velocity,
            "view": self.state(),
        })
    }

    /// The view as a watcher re-applies it: view_map's own arguments, set
    /// in full, with the run and the window last drawn.
    fn view_state(&self) -> Value {
        let s = &self.scene;
        let root = &s.data.spaces[s.root()];
        let feature = match &s.pick {
            Some(Pick::One(f)) => f.as_ref(),
            _ => "",
        };
        json!({
            "run": self.run,
            "round": s.data.round.as_ref().map(|r| absolute(&r.path)),
            "map": root.method,
            "space": root.kind.slug(),
            "colour_by": s.colour.map_or("none", |i| s.data.labels[i].title()),
            "focus": s.focused_name().unwrap_or_default(),
            "feature": feature,
            "observed": s.source != Source::Expected,
            "time": s.pick == Some(Pick::Time),
            "velocity": VelocityArg::from(s.velocity_style),
            "bbox": self.drawn.map(|vp| vp.window()),
        })
    }

    /// What the view shows: map, grouping, focus, feature and legend.
    fn state(&self) -> Value {
        let s = &self.scene;
        let sp = s.current();
        let legend: Vec<Value> = match s.group_sizes() {
            Some(sizes) => s
                .levels()
                .iter()
                .zip(sizes)
                .enumerate()
                .map(|(g, (name, &n))| {
                    let mut v = json!({"group": name, "points": n});
                    if let Some(r) = s.resolved(g) {
                        v["colour"] = hex(r.colour).into();
                    }
                    v
                })
                .collect(),
            None => Vec::new(),
        };
        // What is drawn, not what was asked: a feature the run cannot show
        // is left out, and the note says why.
        let feature = match (&s.pick, s.current_shown()) {
            (Some(Pick::One(f)), Some(_)) => Some(json!({"feature": f, "source": s.source.name()})),
            _ => None,
        };
        json!({
            "map": sp.method,
            "space": sp.kind.slug(),
            "bounds": sp.points.bounds,
            "colour_by": s.colour.map(|i| s.data.labels[i].title()),
            "focus": s.focused_name(),
            "feature": feature,
            "time": s.pick == Some(Pick::Time) && s.current_shown().is_some(),
            "velocity": VelocityArg::from(s.velocity_drawn()),
            "legend": legend,
        })
    }

    /// Every argument is checked against the run before anything changes,
    /// so a call that fails leaves the view as it was.
    fn map(&mut self, a: MapArgs) -> Result<Vec<Content>> {
        let width = width(a.width, WIDTH)?;
        if let Some(b) = a.bbox {
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            ensure!(
                b.iter().all(|v| v.is_finite()) && w > 0.0 && h > 0.0 && (w * h).is_finite(),
                "bbox is [x0, y0, x1, y1] with x0 < x1 and y0 < y1"
            );
        }
        ensure!(
            a.center.is_none_or(|c| c.iter().all(|v| v.is_finite())),
            "center is a finite [x, y]"
        );
        ensure!(
            a.zoom.is_none_or(|z| (1e-6..=1e6).contains(&z)),
            "zoom is between 1e-6 and 1e6"
        );
        if apply(&mut self.scene, &a)? {
            self.window = None;
        }
        self.move_window(&a, width);
        let vp = self.viewport(width);
        let png = self.draw(vp, CELL_PX, CompressionType::Best)?;
        self.drawn = Some(vp);
        let mut state = self.state();
        state["image"] = json!([vp.w, vp.h]);
        state["window"] = json!(vp.window());
        state["units_per_px"] = json!(vp.upp());
        if let Some(note) = self.scene.note.take() {
            state["note"] = note.into();
        }
        Ok(vec![Content::Png(png), Content::Json(state)])
    }

    /// Group `name` of the grouping shown.
    fn group(&self, name: &str) -> Result<u32> {
        let s = &self.scene;
        let li = s
            .colour
            .ok_or_else(|| anyhow!("no grouping shown: pass colour_by"))?;
        group_of(&s.data.labels[li], name)
    }

    /// The window `a` asks for. Zoom and center start from the window last
    /// drawn, or, fitted, from the one a fit at `width` would show.
    fn move_window(&mut self, a: &MapArgs, width: usize) {
        if a.fit == Some(true) {
            self.window = None;
        }
        if let Some(b) = a.bbox {
            self.window = Some(b);
        }
        if a.center.is_none() && a.zoom.is_none() {
            return;
        }
        let [x0, y0, x1, y1] = self.window.unwrap_or_else(|| self.viewport(width).window());
        let [cx, cy] = a.center.unwrap_or([0.5 * (x0 + x1), 0.5 * (y0 + y1)]);
        let k = a.zoom.filter(|z| *z > 0.0).unwrap_or(1.0);
        let (hw, hh) = (0.5 * (x1 - x0) / k, 0.5 * (y1 - y0) / k);
        self.window = Some([cx - hw, cy - hh, cx + hw, cy + hh]);
    }

    /// The camera for an image `width` wide at the window's shape.
    fn viewport(&self, width: usize) -> Viewport {
        let [x0, y0, x1, y1] = self.window.unwrap_or(self.scene.current().points.bounds);
        let aspect = (y1 - y0) / (x1 - x0).max(1e-6);
        let h = ((width as f32 * aspect).round() as usize).clamp(1, MAX_SIDE);
        match self.window {
            None => render::fit([x0, y0, x1, y1], width, h),
            Some(w) => render::frame(w, width, h),
        }
    }

    /// The view through `vp` as a PNG.
    fn draw(&self, vp: Viewport, cell_px: f32, level: CompressionType) -> Result<Vec<u8>> {
        let img = deck::frames(&[(&self.scene, vp, cell_px)], false)
            .remove(0)
            .to_image();
        png(img, level)
    }

    fn markers(&mut self, a: MarkersArgs) -> Result<Vec<Content>> {
        let g = self.group(&a.group)?;
        let n = a.n.unwrap_or(MARKERS);
        let s = &mut self.scene;
        ensure!(
            s.current().axis() == Axis::Cells,
            "markers are ranked on a map of cells"
        );
        let space = s.space;
        let mask: Vec<bool> = s
            .groups()
            .context("no grouping shown")?
            .iter()
            .map(|&x| x == g)
            .collect();
        let name = s.levels()[g as usize].clone();
        let table = s.marker_features(&name);
        let (activity, data) = s
            .activity_and_data()
            .context("no run manifest to read the model from")?;
        let universe = &data.spaces[space].points.names;
        let ranked = activity
            .suggest(space, universe, Some(&mask), n)
            .map_err(|e| anyhow!(e))?;
        let top: Vec<Value> = ranked
            .iter()
            .map(|(f, v)| json!({"feature": f, "score": v}))
            .collect();
        Ok(vec![Content::Json(json!({
            "group": name,
            "scored_by": "expected log fold change, the group over the rest of the map",
            "top": top,
            "marker_table": table,
        }))])
    }

    fn cell(&mut self, a: CellArgs) -> Result<Vec<Content>> {
        let points = &self.scene.current().points;
        let i = match (&a.name, a.px, a.at) {
            (Some(name), ..) => self
                .scene
                .point_of(self.scene.space, name)
                .ok_or_else(|| anyhow!("no point named {name} on this map"))?,
            // As a click picks: the nearest drawn point within reach.
            (None, Some([px, py]), _) => {
                let vp = self
                    .drawn
                    .context("no map drawn yet: call view_map, or pass at")?;
                let i = nearest(&points.xy, vp.to_data(px, py)).context("the map has no points")?;
                let (x, y) = vp.to_px(points.xy[i]);
                ensure!(
                    (x - px).hypot(y - py) <= REACH_PX,
                    "no point within {REACH_PX} px of [{px}, {py}]"
                );
                i
            }
            (None, None, Some(at)) => nearest(&points.xy, at).context("the map has no points")?,
            (None, None, None) => bail!("pass px, at or name"),
        };
        let name = &points.names[i];
        let axis = self.scene.current().axis();
        let groups: serde_json::Map<String, Value> = self
            .scene
            .data
            .labels
            .iter()
            .filter(|l| l.axis() == axis)
            .filter_map(|l| {
                let g = *l.by_name.get(name)?;
                (g != NONE).then(|| (l.title().to_string(), l.levels[g as usize].as_ref().into()))
            })
            .collect();
        let mut out = json!({"name": name, "at": points.xy[i], "groups": groups});
        let times = self
            .scene
            .data
            .times
            .as_ref()
            .filter(|_| axis == Axis::Cells);
        if let Some(t) = times.and_then(|t| t.get(name.as_ref())) {
            out["time"] = json!(t);
        }
        // A round's account of the cell's cluster, when it labels it.
        let s = &self.scene;
        if s.cluster_id_of(name)
            .is_some_and(|id| s.cluster_label(id).is_some())
        {
            out["round"] = json!(s.cluster_info(name));
        }
        Ok(vec![Content::Json(out)])
    }

    fn export(&mut self, a: ExportArgs) -> Result<Vec<Content>> {
        ensure!(a.png.is_some() || a.pdf.is_some(), "pass png, pdf or both");
        // Figures only: no other file is written, overwrite or not.
        for (path, ext) in a
            .png
            .iter()
            .map(|p| (p, "png"))
            .chain(a.pdf.iter().map(|p| (p, "pdf")))
        {
            ensure!(
                path.extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case(ext)),
                "{} is not a .{ext} file",
                path.display()
            );
        }
        for path in a.png.iter().chain(&a.pdf) {
            ensure!(
                a.overwrite || !path.exists(),
                "{} exists; pass overwrite to replace it",
                path.display()
            );
        }
        let vp = self.viewport(width(a.width, EXPORT_WIDTH)?);
        legume_tui::export::fits((vp.w, vp.h), EXPORT_DPI as f32).map_err(|e| anyhow!(e))?;
        // Text as large on the page as `senna view --pdf` sets it.
        let size = PageSize {
            width_in: vp.w as f32 / EXPORT_DPI as f32,
            dpi: EXPORT_DPI,
        };
        let cell_px = size.cell_px();
        let mut wrote = Vec::new();
        if let Some(path) = &a.png {
            std::fs::write(path, self.draw(vp, cell_px, CompressionType::Default)?)
                .with_context(|| format!("writing {}", path.display()))?;
            wrote.push(absolute(path));
        }
        if let Some(path) = &a.pdf {
            let pages = deck::view_pages(&[(&self.scene, vp, cell_px)]);
            legume_tui::export::write_pdf(&pages, EXPORT_DPI as f32, path)
                .with_context(|| format!("writing {}", path.display()))?;
            wrote.push(absolute(path));
        }
        Ok(vec![Content::Json(
            json!({"wrote": wrote, "image": [vp.w, vp.h]}),
        )])
    }
}

/// The view changes `a` asks for, beyond the window, applied to `scene`.
/// Every one is checked against the run before anything changes, so one
/// that fails leaves the scene as it was. Returns whether the map moved
/// (another layout, or a round with other bounds): a window kept from
/// before no longer applies. The terminal viewer applies an agent's view
/// with it as `view_map` does.
pub(in crate::view) fn apply(scene: &mut Scene, a: &MapArgs) -> Result<bool> {
    // A round the view does not show is read now and swapped in last.
    let round = match &a.round {
        Some(r) if !shows_round(scene, Path::new(r)) => Some(Dataset::load(r)?),
        _ => None,
    };
    let s = &*scene;
    let data = round.as_ref().unwrap_or(&s.data);
    let space = match (&a.map, &a.space) {
        (None, None) => None,
        (map, space) => {
            let cur = &s.data.spaces[s.root()];
            Some(find_space(
                data,
                map.as_deref().unwrap_or(&cur.method),
                space.as_deref().unwrap_or(cur.kind.slug()),
            )?)
        }
    };
    let axis = space.map_or(s.current().axis(), |at| data.spaces[at].axis());
    let choices: Vec<usize> = (0..data.labels.len())
        .filter(|&i| data.labels[i].axis() == axis)
        .collect();
    let colour = match &a.colour_by {
        Some(by) if by.eq_ignore_ascii_case("none") => Some(None),
        Some(by) => Some(Some(
            choices
                .iter()
                .copied()
                .find(|&i| data.labels[i].title().eq_ignore_ascii_case(by))
                .ok_or_else(|| {
                    let names: Vec<&str> =
                        choices.iter().map(|&i| data.labels[i].title()).collect();
                    anyhow!("no {by} grouping on this map; it has {}", names.join(", "))
                })?,
        )),
        None => None,
    };
    // The grouping shown once the rest applies: the one asked for, a
    // new axis's first, or the one shown now, by kind.
    let shown = colour.unwrap_or_else(|| match space {
        Some(at) if data.spaces[at].axis() != s.current().axis() => choices.first().copied(),
        // As replace_data keeps it: the same kind, else the first.
        _ => s.colour.and_then(|li| {
            let kind = s.data.labels[li].kind;
            choices
                .iter()
                .copied()
                .find(|&i| data.labels[i].kind == kind)
                .or_else(|| choices.first().copied())
        }),
    });
    let focus = match &a.focus {
        None => None,
        Some(f) => Some(match focus_name(f)? {
            None => None,
            Some(name) => {
                let li = shown.ok_or_else(|| anyhow!("no grouping shown: pass colour_by"))?;
                Some(group_of(&data.labels[li], &name)?)
            }
        }),
    };

    if a.time == Some(true) {
        ensure!(
            a.feature.as_deref().is_none_or(str::is_empty),
            "colour by time or by a feature, not both"
        );
        ensure!(data.times.is_some(), features::NO_TIMES);
    }
    // The field, read now when it is asked for and the view has none of
    // this run's: the round's run, when a round is swapped in.
    let has_field =
        s.velocity.is_some() && round.as_ref().is_none_or(|r| state::same_run(&s.data, r));
    let field = (a.velocity.and_then(VelocityArg::style).is_some() && !has_field)
        .then(|| velocity::Field::of_run(data.run.as_ref()))
        .transpose()?
        .map(std::sync::Arc::new);

    let mut moved = false;
    if let Some(data) = round {
        // Another round of the same layout keeps the window.
        let bounds = scene.current().points.bounds;
        scene.replace_data(data);
        moved |= scene.current().points.bounds != bounds;
    }
    if let Some(at) = space.filter(|&at| at != scene.space) {
        scene.set_space(at);
        moved = true;
    }
    // The grouping shown, asked for again, keeps its focus and feature.
    if let Some(li) = colour.filter(|&li| li != scene.colour) {
        scene.set_colour(li);
    }
    if let Some(focus) = focus {
        scene.focus = focus;
    }
    match a.observed {
        Some(true) => scene.source = scene.count_source(),
        Some(false) => scene.source = Source::Expected,
        None => {}
    }
    match a.feature.as_deref() {
        Some("") => {
            scene.clear_pick();
        }
        Some(f) => scene.set_pick(Pick::One(f.into())),
        None if a.observed.is_some() => scene.refresh_activity(),
        None => {}
    }
    match a.time {
        Some(true) => scene.set_pick(Pick::Time),
        Some(false) if scene.pick == Some(Pick::Time) => {
            scene.clear_pick();
        }
        _ => {}
    }
    if let Some(field) = field {
        scene.velocity = Some(field);
    }
    if let Some(v) = a.velocity {
        scene.velocity_style = v.style();
    }
    Ok(moved)
}

/// The view_map arguments a published view sets.
const VIEW_KEYS: [&str; 10] = [
    "round",
    "map",
    "space",
    "colour_by",
    "focus",
    "feature",
    "observed",
    "time",
    "velocity",
    "bbox",
];

/// An agent's published view ([`Open::view_state`]): the manifest it
/// shows, and the rest as view_map's arguments.
pub(in crate::view) fn agent_view(state: &Value) -> Result<(PathBuf, MapArgs)> {
    let shown = state["round"]
        .as_str()
        .or(state["run"].as_str())
        .context("a view state names its run")?
        .into();
    // What another build of senna publishes beyond these is left out.
    let fields: serde_json::Map<String, Value> = state
        .as_object()
        .context("a view state is an object")?
        .iter()
        .filter(|(k, _)| VIEW_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok((shown, parse(Value::Object(fields))?))
}

/// Whether the round at `path` is the one `scene` shows.
fn shows_round(scene: &Scene, path: &Path) -> bool {
    let shown = scene.data.round.as_ref().map(|r| r.path.as_path());
    shown.is_some_and(|r| run_manifest::same_file(r, path))
}

/// The root map `method` of `kind` in `data`, never a zoomed layout.
fn find_space(data: &Dataset, method: &str, kind: &str) -> Result<usize> {
    data.spaces
        .iter()
        .position(|sp| {
            sp.parent.is_none()
                && sp.method.eq_ignore_ascii_case(method)
                && sp.kind.slug().eq_ignore_ascii_case(kind)
        })
        .ok_or_else(|| anyhow!("no {method} map of {kind}; view_open lists the maps"))
}

/// Group `name` of `labels`: a level's name, or for clusters its id (`3`,
/// `C3` or lupin's `K3`, the ids of the round shown).
fn group_of(labels: &Labels, name: &str) -> Result<u32> {
    let by_name = labels
        .levels
        .iter()
        .position(|l| l.eq_ignore_ascii_case(name));
    let by_id = || {
        let id: i64 = name.trim_start_matches(['C', 'c', 'K', 'k']).parse().ok()?;
        (labels.kind == LabelKind::Cluster)
            .then(|| labels.ids.iter().position(|&i| i == id))
            .flatten()
    };
    by_name
        .or_else(by_id)
        .map(|g| g as u32)
        .ok_or_else(|| anyhow!("no group {name} in {}", labels.title()))
}

/// The group `focus` names, `None` to clear: a name or id, or as pinto
/// takes it, a list, of which senna focuses at most one.
fn focus_name(focus: &Value) -> Result<Option<String>> {
    let one = |v: &Value| match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        _ => bail!("focus is a group name or cluster id"),
    };
    match focus {
        Value::Array(list) => match list.as_slice() {
            [] => Ok(None),
            [g] => one(g).map(Some),
            _ => bail!("senna focuses one group at a time; pass a list of one"),
        },
        v => one(v).map(|s| (!s.is_empty()).then_some(s)),
    }
}

/// The point of `xy` nearest `at`.
fn nearest(xy: &[[f32; 2]], at: [f32; 2]) -> Option<usize> {
    use rayon::prelude::*;
    xy.par_iter()
        .enumerate()
        .map(|(i, p)| (i, (p[0] - at[0]).powi(2) + (p[1] - at[1]).powi(2)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

fn hex(c: color::Rgb) -> String {
    let [r, g, b] = color::to_srgb_senna(c);
    format!("#{r:02x}{g:02x}{b:02x}")
}

/// `img` as an opaque PNG: compressed hard for what goes to the agent
/// whole, less for a file.
fn png(img: image::RgbaImage, level: CompressionType) -> Result<Vec<u8>> {
    use image::codecs::png::{FilterType, PngEncoder};
    use image::ImageEncoder;
    let rgb = image::DynamicImage::ImageRgba8(img).into_rgb8();
    let mut out = Vec::new();
    PngEncoder::new_with_quality(&mut out, level, FilterType::Adaptive).write_image(
        &rgb,
        rgb.width(),
        rgb.height(),
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(out)
}

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let group = json!({"type": ["string", "integer"]});
    let xy = |what: &str| json!({"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": what});
    vec![
        spec(
            "view_open",
            "Open a run and summarize it: its maps (layout × points), its groupings, \
             whether it has cell times and a velocity field, and the view shown with its \
             groups' colours. A layout or clustering the run lacks \
             is computed first. Opening another run closes the one open. The user opens the \
             same run in a terminal with `senna view {run}.senna.json`.",
            json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "Run manifest ({out}.senna.json) or a lupin round's"},
                    "map": {"type": "string", "description": format!("Layout to start on ({}); computed when missing", Step::CellLayout.settings().join(", "))},
                },
                "required": ["run"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_map",
            "Draw the map as a PNG, with the state and legend as JSON. Every argument is \
             optional and stays set until changed; with none, the same view is drawn again.",
            json!({
                "type": "object",
                "properties": {
                    "round": {"type": "string", "description": "A lupin round of the run (its manifest), shown in its place"},
                    "map": {"type": "string", "description": "Layout method, as view_open lists the maps"},
                    "space": {"enum": ["cells", "features-on-cells", "features"], "description": "Which points"},
                    "colour_by": {"type": "string", "description": "A grouping view_open lists (annotation, cluster, topic, batch, …), or none"},
                    "focus": {"type": ["string", "integer", "array"], "items": group, "maxItems": 1, "description": "One group of the grouping shown, by name or cluster id (3, C3, K3), alone or as a list of one; the rest are dimmed. \"\" or [] clears"},
                    "feature": {"type": "string", "description": "Colour cells by one feature's level; \"\" shows the groups again"},
                    "observed": {"type": "boolean", "description": "The feature's observed counts instead of the model's expected level"},
                    "time": {"type": "boolean", "description": "Colour the cells by the run's time (a run view_open says has time); false shows the groups again"},
                    "velocity": {"enum": ["stream", "arrows", "off"], "description": "Draw the run's velocity field over the cells (a run view_open says has velocity), as streamlines or arrows, or not"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "World window [x0, y0, x1, y1]"},
                    "fit": {"type": "boolean", "description": "true: the whole map"},
                    "center": xy("Move the window's centre to this world point"),
                    "zoom": {"type": "number", "exclusiveMinimum": 0, "description": "Magnify the window about its centre: 2 zooms in, 0.5 out"},
                    "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": format!("Image width in pixels, this call only [default: {WIDTH}]")},
                },
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_markers",
            "The features that set one group of the grouping shown apart from the rest of \
             the map, by the model's expected log fold change; and the group's features in \
             the run's marker table, when it has one.",
            json!({
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "Group name, or cluster id (3, C3, K3)"},
                    "n": {"type": "integer", "minimum": 1, "description": format!("How many [default: {MARKERS}]")},
                },
                "required": ["group"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_cell",
            "The point nearest a pixel of the last map, a world point, or by name: its \
             position, its group in every grouping, its time when the run has times, and \
             the lupin round's account of its cluster when a round is shown.",
            json!({
                "type": "object",
                "properties": {
                    "px": xy("[column, row] pixel of the last view_map image"),
                    "at": xy("[x, y] world point"),
                    "name": {"type": "string", "description": "Cell or feature name"},
                },
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_export",
            "Write the current view as a PNG and/or a one-page PDF, text sized for print. \
             Refuses to replace a file unless overwrite.",
            json!({
                "type": "object",
                "properties": {
                    "png": {"type": "string", "description": "PNG path"},
                    "pdf": {"type": "string", "description": "PDF path"},
                    "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": format!("Image width in pixels [default: {EXPORT_WIDTH}]")},
                    "overwrite": {"type": "boolean", "description": "Replace files that exist"},
                },
                "additionalProperties": false,
            }),
        ),
    ]
}

#[cfg(test)]
#[path = "../tests/mcp.rs"]
mod tests;
