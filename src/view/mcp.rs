//! `senna mcp`: the viewer and senna's fits as Model Context Protocol tools,
//! served over stdio with legume-mcp.
//!
//! The view tools hold one run and one view between calls, as the terminal
//! viewer holds them: `view_map` changes only what it is given, then draws
//! the map. The run tools start `senna` as a child, as `senna run` does, so
//! stdout carries only the protocol and the child's log comes back as
//! progress. lupin rounds are read as they are on disk; making them is
//! lupin's own server.

use super::data::{Axis, Dataset, LabelKind, NONE};
use super::recompute::{Step, Target};
use super::render::{self, Viewport};
use super::{deck, Pick, Scene, Source, VelocityStyle, ViewArgs};
use crate::postprocess::{MIN_DIST, SPREAD};
use anyhow::{anyhow, bail, ensure, Context, Result};
use image::codecs::png::CompressionType;
use legume_mcp::{Call, Content, Handler, ToolSpec};
use legume_tui::export::PageSize;
use senna::run_manifest::{self, RunManifest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The fits `run_fit` runs: those `senna run` queues.
use crate::run::tui::METHODS;

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

/// What the agent is told when it connects, word for word the block the
/// README gives for clients that do not pass it on.
pub const INSTRUCTIONS: &str = "\
Use senna's MCP tools to embed single-cell data and look at the result; \
annotation is lupin's own server. Open the day with `notebook` (kind `plan`): \
the question, the data, and how the result will be checked. `run_fit` runs a \
fit to its end (`run_help` lists a method's flags) and returns its run \
manifest; an existing run is not fitted again. A long fit runs with \
`background`, as a job to follow with `job_wait`, in a session of `senna run` the user can watch with `senna attach`. \
`run_predict` places new cells in a trained model's latent, `run_update` \
continues a trained model over new data, and `run_layout` lays a run out \
again with chosen settings, umap's min_dist and spread among them. `view_open` opens a run (or a \
lupin round) and computes a missing layout or clustering first. `view_map` \
draws the map with its legend; its arguments stay set until changed, so pass \
only what changes. `view_markers` lists what sets a group apart, `view_cell` \
names a point and its groups, and `view_export` writes figures. World \
coordinates are the layout's, y up. Close with `notebook` (kind \
`conclusion`) and tell the user where the records are.";

/// The tools, and the run they have open.
#[derive(Default)]
pub struct Tools {
    open: Option<Open>,
}

/// An open run and the view on it.
struct Open {
    scene: Scene,
    run: PathBuf,
    /// The world window asked for; `None` fits the whole map.
    window: Option<[f32; 4]>,
    /// The last map drawn, for `view_cell` pixels.
    drawn: Option<Viewport>,
}

impl Handler for Tools {
    fn tools(&self) -> Vec<ToolSpec> {
        specs()
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> Result<Vec<Content>> {
        match name {
            "run_fit" => run_fit(parse(args)?, cx),
            "run_help" => run_help(parse(args)?),
            "run_predict" => run_predict(parse(args)?, cx),
            "run_update" => run_update(parse(args)?, cx),
            "run_layout" => run_layout(parse(args)?, cx),
            "view_open" => {
                let open = self.open.insert(Open::new(&parse(args)?, cx)?);
                cx.state(&open.view_state());
                Ok(vec![Content::Json(open.summary())])
            }
            "view_map" => {
                let open = self.open()?;
                let out = open.map(parse(args)?)?;
                // What a person following the server sees next.
                cx.state(&open.view_state());
                Ok(out)
            }
            "view_markers" => self.open()?.markers(parse(args)?),
            "view_cell" => self.open()?.cell(parse(args)?),
            "view_export" => self.open()?.export(parse(args)?),
            _ => bail!("unknown tool {name}"),
        }
    }
}

impl Tools {
    fn open(&mut self) -> Result<&mut Open> {
        self.open
            .as_mut()
            .ok_or_else(|| anyhow!("no run open: call view_open first"))
    }
}

fn parse<T: for<'de> Deserialize<'de>>(args: Value) -> Result<T> {
    serde_json::from_value(args).context("reading the arguments")
}

fn absolute(path: &Path) -> PathBuf {
    run_manifest::from_cwd(path)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitArgs {
    method: String,
    data: Vec<String>,
    out: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna {method} {data} -o {out} {args}` as a child, unless the run is
/// there already. Without an out, the run goes in today's results folder,
/// named by its first data file and method.
fn run_fit(a: FitArgs, cx: &Call) -> Result<Vec<Content>> {
    check_method(&a.method)?;
    ensure!(!a.data.is_empty(), "no data files");
    let out = match a.out {
        Some(out) => out,
        None => default_out(cx, &a.data[0], &a.method, run_manifest::default_path)?,
    };
    let mut argv = vec![a.method.clone()];
    argv.extend(a.data.iter().cloned());
    argv.extend(["-o".into(), out.clone()]);
    argv.extend(a.args);
    let task = Task {
        argv,
        made: vec![("run", run_manifest::default_path(&out).into())],
        summary: format!("senna {} fit of {}", a.method, a.data.join(", ")),
    };
    // legume-mcp refuses an out a background fit is still writing (naming
    // its job) until that fit has written the manifest, and fails a fit
    // that ends without it.
    let mut done = start(task, a.background, cx, |t, outputs| {
        background_fit(cx, &out, &t.argv, outputs, &t.summary)
    })?;
    // A background fit's run is not there yet: the agent opens it once
    // the job is done.
    if done.get("job").is_none() {
        done["kind"] = json!(kind(Path::new(done["run"].as_str().unwrap_or_default()))?);
    }
    Ok(vec![Content::Json(done)])
}

/// An out in today's results folder, named by the data file `data` and
/// `tag`: `{stem}_{tag}`, then `_2`, `_3`, …, the first whose made file
/// (`made` of the out) no running job is writing. Not legume-mcp's
/// `free_prefix`, which also passes over a name whose files exist: here
/// that name is taken, so a result already made is not made again.
fn default_out(cx: &Call, data: &str, tag: &str, made: fn(&str) -> String) -> Result<String> {
    let dir = cx
        .results_dir()
        .context("no results folder (records are off): pass out")?;
    let data = Path::new(data).file_name().unwrap_or_default();
    let stem = data.to_string_lossy();
    let stem = stem.split('.').next().unwrap_or("run");
    // As many as there could be jobs, and more.
    (1..=1000)
        .map(|k| match k {
            1 => format!("{stem}_{tag}"),
            k => format!("{stem}_{tag}_{k}"),
        })
        .map(|name| dir.join(name).to_string_lossy().into_owned())
        .find(|out| cx.ensure_free(&[absolute(Path::new(&made(out)))]).is_ok())
        .context("every default name is being written by a running job: pass out")
}

/// A senna command a run tool starts, and the files it makes, each under
/// the name the result gives it; the first shows it ran.
#[derive(Debug)]
struct Task {
    argv: Vec<String>,
    made: Vec<(&'static str, PathBuf)>,
    summary: String,
}

impl Task {
    fn outputs(&self) -> Vec<PathBuf> {
        self.made.iter().map(|(_, p)| absolute(p)).collect()
    }

    /// The made files by name, as the result gives them.
    fn named(&self) -> Value {
        let named = self
            .made
            .iter()
            .map(|(k, p)| (k.to_string(), json!(absolute(p))));
        Value::Object(named.collect())
    }

    /// The command line, as an error names it.
    fn line(&self) -> String {
        format!("senna {}", legume_mcp::command_line(&self.argv))
    }

    /// `senna {argv}` to its end, verbose, its log the call's progress;
    /// `guarded`, recorded with its outputs as its guard on replay, else
    /// recorded to run every time.
    fn run(&self, guarded: bool, cx: &Call) -> Result<()> {
        let outputs = if guarded { self.outputs() } else { vec![] };
        cx.progress(&self.summary);
        cx.run_making(senna(&self.argv)?, &outputs, &self.summary)
            .with_context(|| self.line())?;
        Ok(())
    }
}

/// `task` run, or with `background` started by `spawn` (given the task
/// and its outputs, giving the job as the agent is told it). A task whose
/// first file is there already is not run.
fn start(
    task: Task,
    background: bool,
    cx: &Call,
    spawn: impl FnOnce(&Task, &[PathBuf]) -> Result<Value>,
) -> Result<Value> {
    senna_parses(&task.argv)?;
    let mut done = task.named();
    let outputs = task.outputs();
    if outputs[0].exists() {
        done["ran"] = json!(false);
        done["note"] = json!("it is there already; another out runs anew");
    } else if background {
        let job = spawn(&task, &outputs).with_context(|| task.line())?;
        done.as_object_mut()
            .expect("named() is an object")
            .extend(job.as_object().cloned().unwrap_or_default());
    } else {
        task.run(true, cx)?;
        done["ran"] = json!(true);
    }
    Ok(done)
}

/// `task` as this server's job, which ends with the server.
fn server_job(task: &Task, outputs: &[PathBuf], cx: &Call) -> Result<Value> {
    Ok(json!({
        "job": cx.spawn_job(senna(&task.argv)?, outputs, &task.summary)?,
        "note": "it runs as this job until it ends or the server does; follow it with job_status or job_wait",
    }))
}

/// `run_predict`'s and `run_update`'s arguments.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelArgs {
    model: String,
    #[serde(default)]
    data: Vec<String>,
    out: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna {command} {data} --model {model} -o {out} {args}`.
fn model_argv(command: &str, a: &ModelArgs, out: &str) -> Vec<String> {
    let mut argv = vec![command.to_string()];
    argv.extend(a.data.iter().cloned());
    argv.extend(["--model".into(), a.model.clone(), "-o".into(), out.into()]);
    argv.extend(a.args.iter().cloned());
    argv
}

fn latent_path(out: &str) -> String {
    format!("{out}.latent.parquet")
}

/// `senna predict` into `out`: the latent and the per-cell predictive fit.
fn predict_task(a: &ModelArgs, out: &str) -> Task {
    Task {
        argv: model_argv("predict", a, out),
        made: vec![
            ("latent", latent_path(out).into()),
            ("predictive", format!("{out}.predictive.parquet").into()),
        ],
        summary: format!("senna predict of {} by {}", a.data.join(", "), a.model),
    }
}

/// The new cells in a trained model's latent, as `senna predict` puts them.
fn run_predict(a: ModelArgs, cx: &Call) -> Result<Vec<Content>> {
    let out = match &a.out {
        Some(out) => out.clone(),
        None => {
            let data = a
                .data
                .first()
                .context("no data files to name the out by: pass out")?;
            // By model too: the same cells in another model's latent are
            // another result.
            let model = Path::new(&a.model).file_name().unwrap_or_default();
            let tag = format!("{}_predict", model.to_string_lossy());
            default_out(cx, data, &tag, latent_path)?
        }
    };
    let done = start(predict_task(&a, &out), a.background, cx, |t, o| {
        server_job(t, o, cx)
    })?;
    Ok(vec![Content::Json(done)])
}

/// `senna update` into `out`: a new run.
fn update_task(a: &ModelArgs, out: &str) -> Task {
    let summary = match a.data.is_empty() {
        true => format!("senna update round of {}", a.model),
        false => format!("senna update of {} with {}", a.model, a.data.join(", ")),
    };
    Task {
        argv: model_argv("update", a, out),
        made: vec![("run", run_manifest::default_path(out).into())],
        summary,
    }
}

/// A trained model continued over new data, or a round on its own cells:
/// a new run.
fn run_update(a: ModelArgs, cx: &Call) -> Result<Vec<Content>> {
    let out = a.out.as_deref().context("pass out: the new run's prefix")?;
    let done = start(update_task(&a, out), a.background, cx, |t, o| {
        server_job(t, o, cx)
    })?;
    Ok(vec![Content::Json(done)])
}

#[derive(Deserialize, Clone, Copy, Default)]
#[serde(rename_all = "lowercase")]
enum LayoutTarget {
    #[default]
    Cells,
    Features,
}

impl LayoutTarget {
    fn step(self) -> Step {
        match self {
            Self::Cells => Step::CellLayout,
            Self::Features => Step::FeatureLayout,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayoutArgs {
    run: String,
    method: String,
    #[serde(default)]
    target: LayoutTarget,
    min_dist: Option<f32>,
    spread: Option<f32>,
    #[serde(default)]
    args: Vec<String>,
}

/// `senna layout {method} …` as `view_open` lays out the run `t`, with
/// the flags given.
fn layout_task(a: &LayoutArgs, t: &Target) -> Result<Task> {
    let step = a.target.step();
    one_of(&a.method, step.settings())
        .with_context(|| format!("senna's {} methods", step.label()))?;
    let mut argv = step.argv(t, &a.method);
    for (flag, v) in [("--umap-min-dist", a.min_dist), ("--umap-spread", a.spread)] {
        if let Some(v) = v {
            argv.extend([flag.into(), v.to_string()]);
        }
    }
    argv.extend(a.args.iter().cloned());
    // The kernel's range too, which clap leaves to the layout.
    if let crate::Commands::Layout {
        cmd: crate::LayoutCmd::Umap(umap),
    } = senna_parses(&argv)?.commands
    {
        umap.tumap_params()?;
    }
    let made = step.outputs(t, &a.method).remove(0);
    Ok(Task {
        argv,
        made: vec![("coords", made.into())],
        summary: format!("senna layout {} of {}", a.method, a.run),
    })
}

/// The run laid out again with the flags given, replacing that method's
/// layout. Always run, as the flags are the point, and recorded with no
/// outputs, so without a guard: a replay redoes it after any earlier
/// layout of the method. Its file is held and checked here instead, and
/// gets no row in results/INDEX.tsv (legume-mcp records no step that
/// replaces its outputs).
fn run_layout(a: LayoutArgs, cx: &Call) -> Result<Vec<Content>> {
    let task = layout_task(&a, &Target::load(&a.run)?)?;
    let outputs = task.outputs();
    cx.ensure_free(&outputs)?;
    task.run(false, cx)?;
    ensure!(
        outputs[0].exists(),
        "senna layout ended without {}",
        outputs[0].display()
    );
    let mut done = task.named();
    done["ran"] = json!(true);
    done["note"] = json!(format!("view_open the run with map {} to see it", a.method));
    Ok(vec![Content::Json(done)])
}

/// The fit `senna {argv}` in the background, from this server's folder:
/// on `senna run`'s Run screen in a session of its own, which a
/// person can attach to and which outlives this server; where there are
/// no sessions, a job of this server's. The job, as the agent is told it.
fn background_fit(
    cx: &Call,
    out: &str,
    argv: &[String],
    outputs: &[PathBuf],
    summary: &str,
) -> Result<Value> {
    #[cfg(unix)]
    {
        let mut watch = legume_mcp::own_command(["run", "--"])?;
        watch.args(argv);
        let label = Path::new(out)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
        let started = cx.start_session_job(legume_mcp::SessionJob {
            tool: crate::tui::TOOL,
            screen: "fit".into(),
            label,
            watch,
            record: senna(argv)?,
            outputs: outputs.to_vec(),
            summary: summary.into(),
        })?;
        let session = started.session;
        Ok(json!({
            "job": started.job,
            "session": session,
            "note": format!(
                "the fit runs as this job, on senna run's screen in session {session} (`senna attach {session}` shows it to the user), and goes on if this server ends; follow it with job_status or job_wait, then open the run with view_open once the job is done"
            ),
        }))
    }
    #[cfg(not(unix))]
    {
        let _ = out;
        Ok(json!({
            "job": cx.spawn_job(senna(argv)?, outputs, summary)?,
            "note": "the fit runs as this job until it ends or the server does; follow it with job_status or job_wait, then open the run with view_open once the job is done",
        }))
    }
}

/// `argv` as senna parses it: a call senna would refuse is refused here,
/// before senna is started, with clap's error on one line.
fn senna_parses(argv: &[String]) -> Result<crate::Cli> {
    let cli = <crate::Cli as clap::CommandFactory>::command();
    let matches = legume_mcp::cli::check(&cli, argv)?;
    Ok(<crate::Cli as clap::FromArgMatches>::from_arg_matches(
        &matches,
    )?)
}

/// `senna {argv}` as a child, verbose: its log is the call's progress.
fn senna(argv: &[String]) -> Result<Command> {
    // This server's own binary, found even when it was reinstalled since.
    let mut senna = legume_mcp::own_command(argv)?;
    if !argv.iter().any(|f| f == "-v" || f == "--verbose") {
        senna.arg("-v");
    }
    Ok(senna)
}

fn check_method(method: &str) -> Result<()> {
    one_of(method, &METHODS).context("senna's fits")
}

/// `name`, if it is one of `list`.
fn one_of(name: &str, list: &[impl AsRef<str>]) -> Result<()> {
    let list: Vec<&str> = list.iter().map(AsRef::as_ref).collect();
    ensure!(
        list.contains(&name),
        "{name} is not one of {}",
        list.join(", ")
    );
    Ok(())
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

/// The kind of fit the run `manifest` records.
fn kind(manifest: &Path) -> Result<Value> {
    let (m, _) = RunManifest::load(manifest)?;
    Ok(serde_json::to_value(m.kind)?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HelpArgs {
    method: String,
}

/// The commands `run_help` covers: the fits, and those the other run
/// tools start.
fn helped() -> Vec<String> {
    let others = ["predict", "update"].into_iter().map(String::from);
    let layouts = Step::CellLayout
        .settings()
        .iter()
        .map(|m| format!("layout {m}"));
    METHODS
        .iter()
        .map(|m| m.to_string())
        .chain(others)
        .chain(layouts)
        .collect()
}

/// The command's help, read here: nothing runs, so the session script
/// records no step.
fn run_help(a: HelpArgs) -> Result<Vec<Content>> {
    use clap::CommandFactory;
    one_of(&a.method, &helped())?;
    let mut cli = crate::Cli::command();
    cli.build();
    let mut sub = &mut cli;
    for word in a.method.split(' ') {
        sub = sub
            .find_subcommand_mut(word)
            .with_context(|| format!("no subcommand {}", a.method))?;
    }
    Ok(vec![Content::Text(sub.render_long_help().to_string())])
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    run: String,
    map: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(super) struct MapArgs {
    round: Option<String>,
    map: Option<String>,
    space: Option<String>,
    colour_by: Option<String>,
    focus: Option<Value>,
    feature: Option<String>,
    observed: Option<bool>,
    time: Option<bool>,
    velocity: Option<VelocityArg>,
    pub(super) bbox: Option<[f32; 4]>,
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

/// The view's arguments as `senna view -f {run}` would take them.
fn view_args(run: &str, map: Option<&str>) -> Result<ViewArgs> {
    #[derive(clap::Parser)]
    struct Argv {
        #[command(flatten)]
        view: ViewArgs,
    }
    let mut argv = vec!["view", "-f", run];
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
        super::prepare_run(&args, &a.run, &|why, argv, outputs| {
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
                .is_some_and(|(m, _)| super::velocity::Field::available(m));
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
pub(super) fn apply(scene: &mut Scene, a: &MapArgs) -> Result<bool> {
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
        ensure!(data.times.is_some(), super::features::NO_TIMES);
    }
    // The field, read now when it is asked for and the view has none of
    // this run's: the round's run, when a round is swapped in.
    let has_field = s.velocity.is_some()
        && round
            .as_ref()
            .is_none_or(|r| super::state::same_run(&s.data, r));
    let field = (a.velocity.and_then(VelocityArg::style).is_some() && !has_field)
        .then(|| super::velocity::Field::of_run(data.run.as_ref()))
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

/// The server's name, as `initialize` reports it and watchers find it.
pub(crate) const SERVER: &str = "senna";

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
pub(super) fn agent_view(state: &Value) -> Result<(PathBuf, MapArgs)> {
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
fn group_of(labels: &super::data::Labels, name: &str) -> Result<u32> {
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

fn hex(c: super::color::Rgb) -> String {
    let [r, g, b] = super::color::to_srgb_senna(c);
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

fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let group = json!({"type": ["string", "integer"]});
    let job = json!({"type": "boolean", "description": "Run as a job of this server and return its id at once; follow it with job_status or job_wait [default: false]"});
    let xy = |what: &str| json!({"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": what});
    vec![
        spec(
            "run_fit",
            "Fit a senna embedding as `senna {method} {data} -o {out} {args}` and return \
             the absolute path of its run manifest. An out whose manifest exists is not \
             fitted again; one a background fit is still writing is refused, naming its \
             job (job_wait for it). Fits can take long; progress carries the log.",
            json!({
                "type": "object",
                "properties": {
                    "method": {"enum": METHODS, "description": "The fit"},
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Data files (.zarr, .zarr.zip, .h5)"},
                    "out": {"type": "string", "description": "Output prefix; the run is {out}.senna.json [default: today's results folder, by data file and method]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help lists them, e.g. [\"-b\", \"batch.txt\"]"},
                    "background": {"type": "boolean", "description": "Run the fit as a job and return its id at once, for a fit longer than a call can wait; follow it with job_status or job_wait [default: false]"},
                },
                "required": ["method", "data"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_help",
            "Every flag of one fit, or of the command another run tool starts, with its \
             help, as `senna {method} --help` prints it.",
            json!({
                "type": "object",
                "properties": {"method": {"enum": helped()}},
                "required": ["method"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_predict",
            "Place new cells in a trained model's latent as `senna predict {data} --model \
             {model} -o {out} {args}`, and return the latent's path (and the per-cell \
             predictive fit's). The model is a topic, masked-topic, masked-vae, \
             masked-sbp, vae, svd, bge or simba run's prefix. An out whose latent exists is \
             not run again. Writes no run to view_open and no layout.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Trained run's prefix (its {out}, without .senna.json)"},
                    "data": {"type": "array", "items": {"type": "string"}, "description": "New data files (.zarr, .zarr.zip, .h5); none with --bulk in args"},
                    "out": {"type": "string", "description": "Output prefix: {out}.latent.parquet, {out}.predictive.parquet [default: today's results folder, by data file]"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (predict) lists them, e.g. [\"-b\", \"batch.txt\"]"},
                    "background": job,
                },
                "required": ["model"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_update",
            "Continue a trained model as `senna update {data} --model {model} -o {out} \
             {args}`: it absorbs the new data into a new run, or with no data runs a \
             round on its own cells. Returns the new run's manifest, for view_open. An \
             out whose manifest exists is not run again.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Parent run's prefix (its {out}, without .senna.json)"},
                    "data": {"type": "array", "items": {"type": "string"}, "description": "New data files to absorb; none makes a round"},
                    "out": {"type": "string", "description": "The new run's prefix, not the parent's"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (update) lists them; -b names the new data's batch files when the parent had batches"},
                    "background": job,
                },
                "required": ["model", "out"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "run_layout",
            "Lay out a run again with the settings given, as `senna layout {method} --from \
             {run} {args}`, replacing that method's layout in the run (e.g. umap with \
             min_dist 0.5 and [\"--knn\", \"30\"]; phate or tsne with \
             [\"--knn\", \"30\", \"--n-landmarks\", \"300\"]). Always runs. Then \
             view_open the run with that map to see it.",
            json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "Run manifest ({out}.senna.json)"},
                    "method": {"enum": Step::CellLayout.settings(), "description": "Layout method; the feature embedding takes umap or phate"},
                    "target": {"enum": ["cells", "features"], "description": "Lay out the cells, or the feature embedding on its own [default: cells]"},
                    "min_dist": {"type": "number", "minimum": 0, "description": format!("umap only: umap-learn's min_dist, in [0, spread] [default {MIN_DIST} when spread is given]; larger loosens groups into continua, smaller packs them. With neither this nor spread, t-UMAP")},
                    "spread": {"type": "number", "exclusiveMinimum": 0, "description": format!("umap only: umap-learn's spread, the scale of the map's distances [default {SPREAD} when min_dist is given]")},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (layout {method}) lists them"},
                },
                "required": ["run", "method"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_open",
            "Open a run and summarize it: its maps (layout × points), its groupings, \
             whether it has cell times and a velocity field, and the view shown with its \
             groups' colours. A layout or clustering the run lacks \
             is computed first. Opening another run closes the one open.",
            json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "Run manifest ({out}.senna.json) or a lupin round's"},
                    "map": {"type": "string", "description": "Layout to start on (umap, phate, tsne); computed when missing"},
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
#[path = "tests/mcp.rs"]
mod tests;
