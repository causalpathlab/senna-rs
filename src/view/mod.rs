//! `senna view`: a run's layouts, coloured by its groupings, in the terminal.

mod activity;
mod chart;
mod color;
mod data;
mod decide;
mod deck;
mod features;
mod files;
pub(crate) mod mcp;
mod paint;
mod places;
mod recompute;
mod relabel;
mod render;
mod review;
mod rounds;
mod stage2;
pub(crate) use rounds::typed_decisions;
mod state;
mod style;
mod sublayout;
mod tui;
mod velocity;
mod zoom;

use activity::{Activity, Levels, Source, Stat};
use color::Rgb;
use data::{Axis, Dataset, LabelKind, SpaceKind, NONE};
use data_beans::utilities::name_matching::GeneIndex;
use features::Centre;
use legume_tui::export::{Page, PageSize, PdfArgs};
use legume_tui::map::labels::{Hits, TEXT_SCALES};
use legume_tui::map::text::Font;
use render::{group_medians, Frame, Label, Paint, Tier, Viewport};
use senna::embed_common::*;

/// `--graphics`: the image protocol named, else the terminal asked.
pub use legume_tui::graphics::Graphics;

#[derive(Args, Debug)]
pub struct ViewArgs {
    #[arg(
        value_name = "MANIFEST",
        help = "Run manifest(s) (`{out}.senna.json`), as `senna view run.senna.json`; the same \
                as -f"
    )]
    pub manifests: Vec<Box<str>>,

    #[arg(
        long,
        short = 'f',
        num_args = 1..,
        help = "Run manifest(s) (`{out}.senna.json`); several open side by side (w shows the grid). \
                Omitted: browse to one",
        long_help = "Run manifest(s) (`{out}.senna.json`), also taken without the flag:\n\
                     `senna view run.senna.json`.\n\
                     Omitted, the view opens a file browser on the current directory\n\
                     to choose one (not with --pdf or --relabel).\n\
                     Every layout under `manifest.layout.methods` is available,\n\
                     together with clusters, annotation and topics the run carries.\n\
                     Several manifests (`-f a.senna.json b.senna.json`, or a glob)\n\
                     open as separate sessions: `w` shows them all in a grid, where\n\
                     hovering or the arrows choose one and a click or enter opens it.\n\
                     With --pdf, several manifests render as one grid page."
    )]
    pub from: Vec<Box<str>>,

    #[arg(
        long,
        help = "Start on this layout method (any `senna layout` method: tsne, umap, phate, lle, \
                stress, forceatlas2, diffusion, radial, sugiyama); computed first when the run \
                lacks it [default: the run's current layout, else tsne]"
    )]
    pub method: Option<Box<str>>,

    #[arg(
        long,
        help = "Start on this grouping (annotation, cluster, topic, batch, markers)"
    )]
    pub colour_by: Option<Box<str>>,

    #[arg(long, value_enum, default_value = "auto", help = "Image protocol")]
    pub graphics: Graphics,

    /// `--pdf`, `--width` and `--dpi`: the starting view saved to a PDF,
    /// with no terminal. Several manifests make one grid page.
    #[command(flatten)]
    pub export: PdfArgs,

    #[arg(
        long,
        help = "Start on this axis: cells, features-on-cells, or features"
    )]
    pub space: Option<Box<str>>,

    #[arg(long, help = "Start showing this feature's activity on the cells")]
    pub feature: Option<Box<str>>,

    #[arg(
        long,
        conflicts_with_all = ["exclusivity", "markers_of"],
        help = "Start on the stage-2 gain map (`senna donor-deviation`) of --feature, or of the gene ranked first (`i` in the view)"
    )]
    pub gain: bool,

    #[arg(
        long,
        requires = "gain",
        help = "With --gain, this donor's deviation drawn at every cell's state (`I` in the view)"
    )]
    pub donor: Option<Box<str>>,

    #[arg(
        long,
        conflicts_with_all = ["feature", "markers_of"],
        help = "Start coloured by stage-2 exclusivity: each cell's sample's distance to the nearest sample of another donor (`U` in the view)"
    )]
    pub exclusivity: bool,

    #[arg(
        long,
        help = "Start showing this group's marker-set activity",
        long_help = "Start showing the combined activity of one group's markers.\n\
                     The group is a name from the run's marker table."
    )]
    pub markers_of: Option<Box<str>>,

    #[arg(
        long,
        help = "Use observed counts for activity instead of the model's expectation"
    )]
    pub observed: bool,

    #[arg(
        long,
        value_enum,
        num_args = 0..=1,
        default_missing_value = "stream",
        help = "Draw the run's velocity field over the cells, as streamlines (the default) or arrows (a run with a cell_velocity table; v cycles it)"
    )]
    pub velocity: Option<VelocityStyle>,

    #[arg(
        long,
        help = "Streamlines' density: lanes across the map (default 40; ( and ) change it)"
    )]
    pub stream_density: Option<f32>,

    #[arg(
        long,
        help = "Start with this group focused (a name in the starting grouping)"
    )]
    pub focus: Option<Box<str>>,

    #[arg(
        long,
        help = "Start inside a fresh layout of this group's cells (a name in the starting grouping)"
    )]
    pub zoom_into: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        help = "Start on a chart instead of the map: structure (runs with topics, bge included) or heatmap (`H` in the view)"
    )]
    pub chart: Option<chart::Kind>,

    #[arg(
        long,
        help = "Print relabel mode's panel for the first cluster to visit, and exit"
    )]
    pub relabel: bool,

    #[arg(
        long,
        help = "lupin binary that applies annotation decisions (default: $SENNA_LUPIN, else `lupin`)"
    )]
    pub lupin: Option<Box<str>>,

    #[arg(
        long,
        help = "Open the run as it is: never compute a missing layout or clustering first",
        long_help = "Open the run as it is.\n\
                     By default a run missing a cell layout, a feature layout\n\
                     (runs with a feature embedding), or clusters of either gets\n\
                     them first, with `senna layout` and `senna clustering -m leiden`,\n\
                     recorded in the manifest so it happens once."
    )]
    pub no_compute: bool,

    #[arg(
        long,
        value_name = "FILE",
        num_args = 1..,
        help = "Also open these data files (.zarr, .zarr.zip, .h5), ones the run was not fitted on: their cells together, matched to the run's by name, shown as the extra source (ctrl+e opens them in the view)"
    )]
    pub extra: Vec<std::path::PathBuf>,

    #[arg(
        long,
        value_enum,
        help = "Start coloured by a per-cell statistic of the counts shown (the extra file's, with --extra; `u` in the view)"
    )]
    pub cell_stat: Option<CellStat>,

    #[arg(
        long,
        value_name = "SERVER",
        num_args = 0..=1,
        conflicts_with_all = ["from", "pdf", "relabel"],
        default_missing_value = "",
        help = "Follow what an agent does through `senna mcp`: the newest running server, or the one with this pid",
        long_help = "Follow what an agent does through `senna mcp`.\n\
                     The view shows the run, map, grouping, focus and window the agent\n\
                     shows, and a pane lists its latest calls, what the latest is doing\n\
                     and the last command it ran. Changing the view by hand stops\n\
                     following; W follows again.\n\
                     With no SERVER, the newest running one, waited for until one runs,\n\
                     and the next one once it ends; else the one with that pid."
    )]
    pub watch: Option<String>,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,
}

/// A per-cell statistic to start on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CellStat {
    /// `ln(1 + total count)`.
    Total,
    /// How many features have a count.
    Detected,
}

/// Features suggested for one view, best first.
struct Suggestions {
    space: usize,
    title: String,
    list: Vec<(Box<str>, f32)>,
}

/// A layer's draw order key: space, colour, focus, shown, merge, backdrop.
type OrderKey = (usize, Option<usize>, Option<u32>, u64, u64, bool);

/// What a zoom into one group needs: its name, its cells, and the geometry
/// table to lay them out from.
pub(crate) struct ZoomRequest {
    pub label: String,
    pub names: Vec<Box<str>>,
    pub geometry: std::sync::Arc<sublayout::Geometry>,
}

/// A feature, or one group's whole marker set, picked to show as activity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Pick {
    One(Box<str>),
    Markers(Box<str>),
    /// A run's time of each cell, from its `cell_time` table.
    Time,
    /// A summary of each cell over a count source's features.
    Stat(Stat),
    /// A gene's stage-2 log gain: each cell's own donor's deviation, or
    /// `donor`'s at every cell's state.
    Gain {
        gene: Box<str>,
        donor: Option<Box<str>>,
    },
    /// Each cell's stage-2 exclusivity: its sample's distance to the
    /// nearest matched sample of another donor.
    Exclusive,
}

impl Pick {
    /// The one feature the pick shows, on its activity or its gain map.
    pub(crate) fn feature(&self) -> Option<&str> {
        match self {
            Pick::One(f) | Pick::Gain { gene: f, .. } => Some(f),
            _ => None,
        }
    }

    /// The same kind of pick for feature `name`: its gain map with the same
    /// donor when this is a gain map, else its activity.
    pub(crate) fn for_feature(this: Option<&Self>, name: Box<str>) -> Self {
        match this {
            Some(Pick::Gain { donor, .. }) => Pick::Gain {
                gene: name,
                donor: donor.clone(),
            },
            _ => Pick::One(name),
        }
    }
}

/// Activity currently drawn: what it is of, where, and its levels.
#[derive(Clone)]
struct Shown {
    /// Distinguishes one computed activity from the next, for caches.
    id: u64,
    space: usize,
    pick: Pick,
    source: Source,
    title: String,
    levels: Levels,
}

/// How a run's velocity field is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum VelocityStyle {
    /// Streamlines through the field, as scVelo draws them.
    Stream,
    /// An arrow per grid bin.
    Arrows,
}

/// Every point's displacement on a map (its space index), as last drawn.
type Moved = (usize, std::sync::Arc<Vec<[f32; 2]>>);

/// Point of each name (its first, for a repeated name).
type NameIndex = std::collections::HashMap<Box<str>, usize>;

/// What is on screen.
pub(crate) struct Scene {
    pub data: Dataset,
    pub space: usize,
    /// Index into `data.labels`, when colouring by a grouping.
    pub colour: Option<usize>,
    pub focus: Option<u32>,
    pub show_labels: bool,
    /// Text size on the map, relative to the terminal's cell height.
    pub text_scale: f32,
    /// Every dot and all text on the map, together (`<` `>`), on top of
    /// each group's own size and `text_scale`.
    pub scale: f32,
    /// The group whose label is under the pointer, framed when drawn.
    pub hover: Option<u32>,
    /// Where group labels were drawn in the last frame, and its camera: what
    /// a click on the map can hit.
    label_hits: std::cell::RefCell<Hits<u32>>,
    /// Where the keyed labels of the last frame were placed, for moving
    /// them from the keyboard.
    label_placed: std::cell::RefCell<Vec<legume_tui::map::labels::Placed<u32>>>,
    /// Labels moved by hand, kept beside the run.
    pub places: places::Places,
    /// The label chosen to move from the keyboard (`E`), framed when drawn.
    pub editing: Option<u32>,
    /// Group id per point, and points per group, cached per (space, labels).
    groups: Option<(usize, usize, Vec<u32>, Vec<usize>)>,
    /// Resolved style per group of the current grouping.
    styles: Vec<style::Resolved>,
    /// Saved per-group styles for every grouping.
    pub book: style::Book,
    pub pick: Option<Pick>,
    pub source: Source,
    activity: Option<Activity>,
    shown: Option<Shown>,
    /// Each point's stage-2 row, by space ([`stage2::Stage2Rows`]).
    stage2_rows: stage2::Stage2Rows,
    ramp: Vec<Rgb>,
    suggestions: Option<Suggestions>,
    /// Draw orders computed for recent layer states; see `layers`.
    orders: std::cell::RefCell<Vec<(OrderKey, std::sync::Arc<Vec<u32>>)>>,
    shown_ids: u64,
    /// Name index of a feature space, for marking the picked feature.
    feature_index: std::cell::RefCell<Option<(usize, GeneIndex)>>,
    /// Point of each name in a space, for placing a clicked cell and the
    /// names near it; one per space, for the cell map and the feature map.
    name_index: std::cell::RefCell<std::collections::HashMap<usize, NameIndex>>,
    /// Group label anchors for a (space, grouping).
    medians: std::cell::RefCell<Option<((usize, usize), render::Medians)>>,
    /// The run's geometry table, read on the first zoom into a group.
    geometry: Option<std::sync::Arc<sublayout::Geometry>>,
    /// Whether to wait for a slow read not done yet (a PDF, the command
    /// line), or carry on and do again what needed it once it comes (the
    /// screen).
    pub patient: bool,
    pub review: Option<relabel::Review>,
    /// A chart drawn in place of the map (`H`), when one is on.
    pub chart: Option<chart::Chart>,
    /// Features near the last clicked cell, and sets locked on screen.
    pub near: Option<features::Near>,
    /// Whether the cells' batches, read with the data files, were added.
    batches_taken: bool,
    /// What to do again once a slow read answers.
    redo: Option<features::Ask>,
    /// How many of the run's slow reads had answered at the last look.
    answered: usize,
    /// How many neighbours a click lists.
    pub near_count: usize,
    pub locked: Vec<features::Near>,
    /// A message for the status line, taken by the front end.
    pub note: Option<String>,
    /// A data file the observed counts need and could not find (its place
    /// in `data.input` and recorded path), for the front end to ask where
    /// it is.
    pub missing_data: Option<(usize, String)>,
    /// Whether how the extra data file matched is still to be said.
    extra_pending: bool,
    /// A run's velocity field (read on the first `v`), whether it is drawn,
    /// and every point's displacement on the last map it was drawn on.
    velocity: Option<std::sync::Arc<velocity::Field>>,
    pub velocity_style: Option<VelocityStyle>,
    /// Streamlines' density: lanes across the visible map (`{` `}`).
    pub stream_lanes: f32,
    velocity_moved: std::cell::RefCell<Option<Moved>>,
}

impl Scene {
    fn new(data: Dataset, args: &ViewArgs) -> Self {
        let space = pick_space(&data, args.method.as_deref(), args.space.as_deref());
        let book = style::Book::load(&data.prefix);
        let places = places::Places::load(&data.prefix);
        let mut scene = Self {
            data,
            space,
            colour: None,
            focus: None,
            show_labels: true,
            text_scale: TEXT_SCALES[1],
            scale: 1.0,
            hover: None,
            label_hits: std::cell::RefCell::default(),
            label_placed: std::cell::RefCell::default(),
            places,
            editing: None,
            groups: None,
            styles: Vec::new(),
            book,
            pick: None,
            source: Source::Expected,
            activity: None,
            shown: None,
            ramp: color::activity_ramp(256),
            stage2_rows: Default::default(),
            geometry: None,
            patient: true,
            suggestions: None,
            review: None,
            chart: None,
            near: None,
            batches_taken: false,
            redo: None,
            answered: 0,
            near_count: features::NEAR_COUNTS[1],
            locked: Vec::new(),
            orders: std::cell::RefCell::new(Vec::new()),
            shown_ids: 0,
            feature_index: std::cell::RefCell::new(None),
            name_index: std::cell::RefCell::default(),
            medians: std::cell::RefCell::new(None),
            note: None,
            missing_data: None,
            extra_pending: false,
            velocity: None,
            velocity_style: None,
            stream_lanes: args.stream_density.unwrap_or(velocity::LANES),
            velocity_moved: std::cell::RefCell::new(None),
        };
        // `--colour-by tau` (or `time`) colours a run's cells by
        // their time; any other name is a grouping.
        let by_time = args
            .colour_by
            .as_deref()
            .is_some_and(|w| w.eq_ignore_ascii_case("tau") || w.eq_ignore_ascii_case("time"));
        let want = args.colour_by.as_deref().filter(|_| !by_time);
        scene.colour = scene.default_colour(want);
        scene.refresh_groups();
        if by_time {
            scene.toggle_time();
        }
        if let Some(style) = args.velocity {
            scene.show_velocity_as(style);
        }
        if args.observed {
            scene.source = Source::Observed;
        }
        if !args.extra.is_empty() {
            scene.open_extra(&args.extra);
        }
        if let Some(stat) = args.cell_stat {
            if scene.source == Source::Expected {
                scene.source = scene.count_source();
            }
            scene.set_pick(Pick::Stat(match stat {
                CellStat::Total => Stat::Total,
                CellStat::Detected => Stat::Detected,
            }));
        }
        if let Some(want) = args.focus.as_ref().or(args.zoom_into.as_ref()) {
            scene.focus = scene
                .levels()
                .iter()
                .position(|l| l == want)
                .map(|i| i as u32);
            if scene.focus.is_none() {
                log::warn!("view: no group `{want}` in the starting grouping");
            }
        }
        if args.zoom_into.is_some() && scene.focus.is_some() {
            let parent = scene.space;
            match scene.zoom_request().and_then(|z| {
                let (found, xy) = z.geometry.layout(&z.names).map_err(|e| e.to_string())?;
                Ok((z.label, found, xy))
            }) {
                Ok((label, names, xy)) => scene.add_zoomed(parent, &label, names, xy),
                Err(e) => log::warn!("view: {e}"),
            }
        }
        if let Some(g) = &args.markers_of {
            scene.set_pick(Pick::Markers(g.clone()));
        } else if let (true, Some(f)) = (args.gain, &args.feature) {
            scene.set_pick(Pick::Gain {
                gene: f.clone(),
                donor: args.donor.clone(),
            });
        } else if args.gain {
            scene.rank_gains(args.donor.clone());
        } else if args.exclusivity {
            scene.toggle_exclusive();
        } else if let Some(f) = &args.feature {
            scene.set_pick(Pick::One(f.clone()));
        }
        if let Some(note) = scene.note.take() {
            log::warn!("view: {note}");
        }
        scene
    }
}

impl Scene {
    /// A second scene on the same run showing what this one shows: the same
    /// map (zooms included), grouping, focus, feature and pinned names. Its
    /// caches start empty, and a relabel session stays with this one.
    pub(crate) fn duplicate(&self) -> Self {
        let mut s = Self {
            data: self.data.clone(),
            space: self.space,
            colour: self.colour,
            focus: self.focus,
            show_labels: self.show_labels,
            text_scale: self.text_scale,
            scale: self.scale,
            chart: self.chart.as_ref().map(chart::Chart::like),
            hover: None,
            label_hits: std::cell::RefCell::default(),
            label_placed: std::cell::RefCell::default(),
            places: self.places.clone(),
            editing: None,
            groups: None,
            styles: Vec::new(),
            book: self.book.clone(),
            // The activity on screen comes along, so nothing is read again.
            pick: self.pick.clone(),
            source: self.source,
            activity: None,
            shown: self.shown.clone(),
            ramp: self.ramp.clone(),
            // The same spaces, so the same rows.
            stage2_rows: self.stage2_rows.clone(),
            suggestions: None,
            orders: std::cell::RefCell::default(),
            shown_ids: self.shown_ids,
            feature_index: std::cell::RefCell::new(None),
            name_index: std::cell::RefCell::default(),
            medians: std::cell::RefCell::new(None),
            geometry: self.geometry.clone(),
            patient: self.patient,
            review: None,
            near: self.near.clone(),
            batches_taken: self.batches_taken,
            redo: None,
            answered: self.answered,
            near_count: self.near_count,
            locked: self.locked.clone(),
            note: None,
            missing_data: None,
            extra_pending: self.extra_pending,
            velocity: self.velocity.clone(),
            velocity_style: self.velocity_style,
            stream_lanes: self.stream_lanes,
            velocity_moved: std::cell::RefCell::new(self.velocity_moved.borrow().clone()),
        };
        s.refresh_groups();
        s
    }
}

fn pick_space(data: &Dataset, method: Option<&str>, space: Option<&str>) -> usize {
    data.spaces
        .iter()
        .position(|s| {
            method.is_none_or(|m| s.method.eq_ignore_ascii_case(m))
                && space.is_none_or(|t| s.kind.slug().eq_ignore_ascii_case(t))
        })
        .unwrap_or(0)
}

/// Run `senna` itself with `argv` on the way into the view, its progress
/// showing before the view opens. On the way to the screen it runs in
/// `session`, so a `senna kill` meanwhile stops it and then the view.
fn run_senna(
    session: Option<&legume_tui::Session>,
    why: &str,
    argv: &[String],
) -> anyhow::Result<()> {
    eprintln!("senna view: {why}; running `senna {}`", argv.join(" "));
    // This binary, found even when it was reinstalled while it ran.
    let mut senna = legume_mcp::own_command(argv)?;
    let status = match session {
        Some(session) => session.run_child(&mut senna)?,
        None => senna.status()?,
    };
    anyhow::ensure!(
        status.success(),
        "`senna {}` failed; see its messages above",
        argv.join(" ")
    );
    Ok(())
}

/// How [`prepare_run`] runs one step: `(why, argv, outputs)`.
type RunStep<'a> = dyn Fn(&str, &[String], &[String]) -> anyhow::Result<()> + 'a;

/// Compute what the view shows and the run lacks: a layout of the cells
/// (not for fne, which has none), of the feature embedding, and clusters of
/// each. Outputs go beside the manifest and are recorded in it, so this runs
/// once. Only a missing cell layout is an error; the rest warn and the view
/// opens without them. Each step runs through `run_step(why, argv,
/// outputs)`: in a session on the way to the screen, or as an MCP call.
fn prepare_run(args: &ViewArgs, from: &str, run_step: &RunStep) -> anyhow::Result<()> {
    use recompute::{Step, Target};
    use senna::run_manifest::RunManifest;
    let load = || RunManifest::load(std::path::Path::new(from));
    let (m, dir) = load()?;
    let target = Target::from_manifest(from, &m, &dir);
    let method = args
        .method
        .as_deref()
        .or(m.layout.current.as_deref())
        .unwrap_or("tsne")
        .to_string();
    // An output counts only when its file is here: one recorded but moved
    // or deleted is computed again.
    let here = |slot: &Option<String>| {
        slot.as_deref()
            .is_some_and(|p| senna::run_manifest::resolve(&dir, p).exists())
    };
    // The method asked for, else any: a feature map alone is no cell layout.
    let no_cells = |e: &senna::run_manifest::LayoutEntry| !here(&e.cell_coords);
    let wants_cell_layout = target.has_cells
        && match &args.method {
            Some(w) => m.layout.methods.get(w.as_ref()).is_none_or(no_cells),
            None => !here(&m.layout.cell_coords) && m.layout.methods.values().all(no_cells),
        };
    let run = |step: Step, method: &str, why: String| {
        run_step(
            &why,
            &step.argv(&target, method),
            &step.outputs(&target, method),
        )
    };
    let warn = |r: anyhow::Result<()>| {
        if let Err(e) = r {
            log::warn!("view: {e}; opening without it");
        }
    };

    if wants_cell_layout {
        let why = format!("{from} has no {method} layout of its cells yet");
        run(Step::for_method(&method), &method, why)?;
    }
    let (m, _) = load()?;
    if target.has_features && m.layout.methods.values().all(|e| !here(&e.feature_coords)) {
        // The cell layout's method when features can take it, else the default.
        let methods = Step::FeatureLayout.settings();
        let method = methods
            .iter()
            .find(|m| **m == method)
            .unwrap_or(&methods[0]);
        let why = format!("{from} has no layout of its feature embedding yet");
        warn(run(Step::FeatureLayout, method, why));
    }
    for (step, wanted) in [
        (
            // Only on a latent that is here: a missing one would fail anyway.
            Step::CellClusters,
            target
                .latent
                .as_deref()
                .is_some_and(|l| std::path::Path::new(l).exists())
                && !here(&m.cluster.clusters),
        ),
        (
            Step::FeatureClusters,
            target.has_features && !here(&m.cluster.feature_clusters),
        ),
    ] {
        if wanted {
            warn(run(step, "", format!("{from} has no {} yet", step.label())));
        }
    }
    Ok(())
}

impl ViewArgs {
    /// The manifests named, bare ones first and then those after `-f`.
    #[must_use]
    pub fn runs(&self) -> Vec<Box<str>> {
        self.manifests.iter().chain(&self.from).cloned().collect()
    }
}

pub fn run_view(args: &ViewArgs) -> anyhow::Result<()> {
    let named = args.runs();
    // A PDF or --relabel prints and exits; the rest is on screen.
    if args.export.pdf.is_some() || args.relabel {
        anyhow::ensure!(
            !named.is_empty(),
            "--pdf and --relabel need the run: pass <run>.senna.json"
        );
        return off_screen(args, &named);
    }
    // On screen, the session first. Hosted, this process only relays the
    // terminal from here on, and the program the host starts reads the
    // runs; with none named, live views are offered before a new one.
    let watching = args.watch.is_some();
    let session =
        crate::tui::TOOL.session("view", &args.session.options(watching || !named.is_empty()))?;
    let picked: Vec<Box<str>>;
    let mut agent = None;
    let from: &[Box<str>] = if let Some(which) = &args.watch {
        // Killed while waiting for the agent: nothing to open.
        let Some((followed, shown)) = agent_run(which, &session)? else {
            return Ok(());
        };
        agent = Some(followed);
        picked = vec![shown.to_string_lossy().into()];
        &picked
    } else if named.is_empty() {
        let Some(path) = tui::pick_run(&session)? else {
            return Ok(());
        };
        picked = vec![crate::tui::shown(&path).into()];
        &picked
    } else {
        &named
    };
    let mut scenes = read_scenes(args, from, Some(&session))?;
    start_charts(args, &mut scenes);
    // Killed while the runs were read, after the browser took the kill as
    // a stop: not opened only to close.
    if session.stop_requested() {
        return Ok(());
    }
    let views = scenes
        .into_iter()
        .zip(from)
        .map(|(s, f)| (s, std::path::PathBuf::from(f.as_ref())))
        .collect();
    tui::run(
        &session,
        views,
        agent
            .as_ref()
            .map(|(dir, next)| (dir.as_path(), next.as_deref())),
        args.graphics,
        args.lupin
            .as_deref()
            .map(String::from)
            .or_else(|| std::env::var("SENNA_LUPIN").ok())
            .unwrap_or_else(|| "lupin".into()),
    )
}

/// What `--watch` follows: the folder of a `senna mcp` server and, when
/// it follows the newest one, the servers' folders, where the next one is
/// found once it ends.
type Followed = (std::path::PathBuf, Option<std::path::PathBuf>);

/// The `senna mcp` server `which` names (its pid; else the newest running
/// one, waited for until one runs), and the manifest its agent shows,
/// waited for until it shows one; None if the session is stopped first.
fn agent_run(
    which: &str,
    session: &legume_tui::Session,
) -> anyhow::Result<Option<(Followed, std::path::PathBuf)>> {
    use anyhow::Context;
    use legume_mcp::watch;
    let pid = match which {
        "" => None,
        pid => Some(
            pid.parse::<u32>()
                .with_context(|| format!("{pid} is not a pid"))?,
        ),
    };
    anyhow::ensure!(
        watch::base().is_some(),
        "watching is off ({}=off): no server to follow",
        watch::ENV
    );
    let mut said = false;
    let server = loop {
        if let Some(server) = watch::find(mcp::SERVER, pid).into_iter().next() {
            break server;
        }
        if let Some(pid) = pid {
            anyhow::bail!("no running `senna mcp` with pid {pid}");
        }
        if session.stop_requested() {
            return Ok(None);
        }
        if !std::mem::replace(&mut said, true) {
            eprintln!("senna view: waiting for a `senna mcp` to start");
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    };
    let next = pid.is_none().then(watch::base).flatten();
    let mut said = false;
    loop {
        if let Some(state) = watch::state(&server.dir) {
            let shown = mcp::agent_view(&state)?.0;
            return Ok(Some(((server.dir, next), shown)));
        }
        if session.stop_requested() {
            return Ok(None);
        }
        anyhow::ensure!(
            watch::alive(server.meta.pid),
            "the agent's server ended before it opened a run"
        );
        if !std::mem::replace(&mut said, true) {
            eprintln!(
                "senna view: waiting for the agent to open a run (pid {})",
                server.meta.pid
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

/// The runs `from` read into scenes, what the view needs computed first:
/// in `session` on the way to the screen, plain for a PDF or --relabel.
fn read_scenes(
    args: &ViewArgs,
    from: &[Box<str>],
    session: Option<&legume_tui::Session>,
) -> anyhow::Result<Vec<Scene>> {
    use rayon::prelude::*;
    anyhow::ensure!(
        !(args.relabel && from.len() > 1),
        "--relabel reads one run; got {} manifests",
        from.len()
    );
    if !args.no_compute {
        for f in from {
            prepare_run(args, f, &|why, argv, _| run_senna(session, why, argv))?;
        }
    }
    // Reading runs is independent work; the scenes are built in order.
    let what = match from {
        [one] => format!("reading {}", crate::tui::name(std::path::Path::new(&**one))),
        many => format!("reading {} runs", many.len()),
    };
    // On screen, the cells' batches are read on the side (they mean opening
    // every data file); a PDF, --relabel or a batch colouring asked for
    // here waits for them.
    let wants_batch = args
        .colour_by
        .as_deref()
        .is_some_and(|c| c.eq_ignore_ascii_case("batch"));
    let aside = session.is_some() && !wants_batch;
    let mut data = legume_tui::work::during_inline(&what, || {
        from.par_iter()
            .map(|f| Dataset::load(f))
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    if !aside {
        data.iter_mut().for_each(Dataset::with_batches);
    }
    let mut scenes: Vec<Scene> = data.into_iter().map(|d| Scene::new(d, args)).collect();
    if aside {
        // The screen does not wait for a slow read: it starts them all now
        // and does again what needed one once it answers.
        for scene in &mut scenes {
            scene.patient = false;
            scene.start_loads();
        }
    }
    Ok(scenes)
}

/// The chart `--chart` asks for, started in every scene.
fn start_charts(args: &ViewArgs, scenes: &mut [Scene]) {
    if let Some(kind) = args.chart {
        for scene in scenes {
            scene.start_chart(kind);
        }
    }
}

/// `--relabel` or `--pdf` on the runs `from`, no screen.
fn off_screen(args: &ViewArgs, from: &[Box<str>]) -> anyhow::Result<()> {
    let mut scenes = read_scenes(args, from, None)?;
    if args.relabel {
        let scene = &mut scenes[0];
        match scene.enter_review() {
            Ok(()) => scene
                .review_lines()
                .unwrap_or_default()
                .iter()
                .for_each(|l| println!("{l}")),
            Err(e) => println!("{e}"),
        }
        scene.leave_review();
        return Ok(());
    }
    start_charts(args, &mut scenes);

    if let Some(path) = &args.export.pdf {
        let titles: Vec<String> = from
            .iter()
            .map(|f| files::name(std::path::Path::new(f.as_ref())))
            .collect();
        let size = args.export.size();
        let (page, (w, h)) = headless_page(&scenes, &titles, args.chart.is_some(), size)?;
        legume_tui::export::write_pdf(&[page], size.dpi as f32, path)?;
        info!(
            "Saved {} ({w} × {h} px at {} dpi, {} run(s))",
            path.display(),
            size.dpi,
            scenes.len()
        );
    }
    Ok(())
}

/// The page `--pdf` saves of `scenes` at `size`, and its size in pixels:
/// one map alone, else a grid of them named by `titles`.
fn headless_page(
    scenes: &[Scene],
    titles: &[String],
    chart: bool,
    size: PageSize,
) -> anyhow::Result<(Page, (usize, usize))> {
    // One map: a page of the map's own shape, as it would fill a screen
    // of that shape. A chart, or several views: a roughly 4:3 page of
    // 4:3 tiles.
    let (w, h) = if let ([scene], false) = (scenes, chart) {
        let [x0, y0, x1, y1] = scene.current().points.bounds;
        size.check((y1 - y0) / (x1 - x0).max(1e-6))
    } else {
        size.check_grid(scenes.len())
    }
    .map_err(|e| anyhow::anyhow!("{e} (--width, --dpi)"))?;
    let cell_px = size.cell_px();
    let page = if let [scene] = scenes {
        let vp = render::fit(scene.current().points.bounds, w, h);
        deck::view_pages(&[(scene, vp, cell_px)]).remove(0)
    } else {
        let refs: Vec<&Scene> = scenes.iter().collect();
        let cams = vec![None; scenes.len()];
        deck::render_grid(&refs, &cams, titles, (w, h), cell_px)
    };
    Ok((page, (w, h)))
}

#[cfg(test)]
#[path = "tests/snapshot.rs"]
mod snapshot;

#[cfg(test)]
#[path = "tests/scene.rs"]
mod tests;
