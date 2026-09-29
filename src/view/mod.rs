//! `senna view`: a run's layouts, coloured by its groupings, in the terminal.

mod activity;
mod color;
mod data;
mod decide;
mod deck;
mod features;
mod files;
mod paint;
mod pdf;
mod relabel;
mod render;
mod review;
mod rounds;
mod state;
mod style;
mod sublayout;
mod text;
mod tui;
mod zoom;

use activity::{Activity, Levels, Source};
use color::Rgb;
use data::{Axis, Dataset, LabelKind, SpaceKind, NONE};
use data_beans::utilities::name_matching::GeneIndex;
use features::Centre;
use render::{draw_labels, group_medians, Label, Paint, Tier, Viewport};
use senna::embed_common::*;
use text::Font;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum Graphics {
    /// Detect the terminal's image protocol; block characters if none.
    Auto,
    Kitty,
    Sixel,
    Iterm2,
    /// Coloured half-block characters: works everywhere, lowest resolution.
    Blocks,
}

#[derive(Args, Debug)]
pub struct ViewArgs {
    #[arg(
        long,
        short = 'f',
        num_args = 1..,
        required = true,
        help = "Run manifest(s) (`{out}.senna.json`); several open side by side (w shows the grid)",
        long_help = "Run manifest(s) (`{out}.senna.json`).\n\
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
        help = "Start on this layout method (umap, phate, tsne); computed first when the run lacks it"
    )]
    pub method: Option<Box<str>>,

    #[arg(
        long,
        help = "Start on this grouping (annotation, cluster, topic, markers)"
    )]
    pub colour_by: Option<Box<str>>,

    #[arg(long, value_enum, default_value = "auto", help = "Image protocol")]
    pub graphics: Graphics,

    #[arg(
        long,
        help = "Save the starting view to this PDF and exit (no terminal needed)",
        long_help = "Save the starting view to a PDF and exit: the points as an image\n\
                     at --dpi, the labels as text. Combine with --method / --space /\n\
                     --colour-by and --width; several manifests make one grid page."
    )]
    pub pdf: Option<Box<str>>,

    #[arg(
        long,
        help = "Start on this axis: cells, features-on-cells, or features"
    )]
    pub space: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 7.0,
        help = "Page width in inches (with --pdf); the height follows the view"
    )]
    pub width: f32,

    #[arg(
        long,
        default_value_t = 300,
        help = "Resolution of the points in dots per inch (with --pdf); text stays text"
    )]
    pub dpi: u32,

    #[arg(long, help = "Start showing this feature's activity on the cells")]
    pub feature: Option<Box<str>>,

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
        help = "Print suggested features for the starting view (with --pdf, show the top one)"
    )]
    pub suggest: bool,

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

/// Point of each name (its first, for a repeated name).
type NameIndex = std::collections::HashMap<Box<str>, usize>;

/// Text sizes on the map, as multiples of the terminal's cell height.
const TEXT_SCALES: [f32; 4] = [1.0, 1.4, 1.8, 2.4];

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
    /// The run's feature embedding, read on the first click on a feature.
    feature_embedding: Option<Result<features::FeatureEmbedding, String>>,
    pub review: Option<relabel::Review>,
    /// Features near the last clicked cell, and sets locked on screen.
    pub near: Option<features::Near>,
    pub locked: Vec<features::Near>,
    /// A message for the status line, taken by the front end.
    pub note: Option<String>,
}

impl Scene {
    fn new(data: Dataset, args: &ViewArgs) -> Self {
        let space = pick_space(&data, args.method.as_deref(), args.space.as_deref());
        let book = style::Book::load(&data.prefix);
        let mut scene = Self {
            data,
            space,
            colour: None,
            focus: None,
            show_labels: true,
            text_scale: TEXT_SCALES[1],
            groups: None,
            styles: Vec::new(),
            book,
            pick: None,
            source: Source::Expected,
            activity: None,
            shown: None,
            ramp: color::activity_ramp(256),
            geometry: None,
            feature_embedding: None,
            suggestions: None,
            review: None,
            near: None,
            locked: Vec::new(),
            orders: std::cell::RefCell::new(Vec::new()),
            shown_ids: 0,
            feature_index: std::cell::RefCell::new(None),
            name_index: std::cell::RefCell::default(),
            medians: std::cell::RefCell::new(None),
            note: None,
        };
        scene.colour = scene.default_colour(args.colour_by.as_deref());
        scene.refresh_groups();
        if args.observed {
            scene.source = Source::Observed;
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
            groups: None,
            styles: Vec::new(),
            book: self.book.clone(),
            // The activity on screen comes along, so nothing is read again.
            pick: self.pick.clone(),
            source: self.source,
            activity: None,
            shown: self.shown.clone(),
            ramp: self.ramp.clone(),
            suggestions: None,
            orders: std::cell::RefCell::default(),
            shown_ids: self.shown_ids,
            feature_index: std::cell::RefCell::new(None),
            name_index: std::cell::RefCell::default(),
            medians: std::cell::RefCell::new(None),
            geometry: self.geometry.clone(),
            feature_embedding: None,
            review: None,
            near: self.near.clone(),
            locked: self.locked.clone(),
            note: None,
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
/// showing before the view opens.
fn run_senna(why: &str, argv: &[&str]) -> anyhow::Result<()> {
    eprintln!("senna view: {why}; running `senna {}`", argv.join(" "));
    let status = std::process::Command::new(std::env::current_exe()?)
        .args(argv)
        .status()?;
    anyhow::ensure!(
        status.success(),
        "`senna {}` failed; see its messages above",
        argv.join(" ")
    );
    Ok(())
}

/// Compute what the view shows and the run lacks: a layout of the cells
/// (not for fne, which has none), of the feature embedding, and clusters of
/// each. Outputs go beside the manifest and are recorded in it, so this runs
/// once. Only a missing cell layout is an error; the rest warn and the view
/// opens without them.
fn prepare_run(args: &ViewArgs, from: &str) -> anyhow::Result<()> {
    use senna::run_manifest::RunManifest;
    let load = || RunManifest::load(std::path::Path::new(from));
    let (m, dir) = load()?;
    let out = senna::run_manifest::derive_out_prefix(from);
    let has_features = m.outputs.feature_embedding.is_some();
    let method = args
        .method
        .as_deref()
        .or(m.layout.current.as_deref())
        .unwrap_or("umap")
        .to_string();
    // The method asked for, else any: a feature map alone is no cell layout.
    let no_cells = |e: &senna::run_manifest::LayoutEntry| e.cell_coords.is_none();
    let wants_cell_layout = m.kind.has_cells()
        && match &args.method {
            Some(w) => m.layout.methods.get(w.as_ref()).is_none_or(no_cells),
            None => m.layout.cell_coords.is_none() && m.layout.methods.values().all(no_cells),
        };
    let warn = |r: anyhow::Result<()>| {
        if let Err(e) = r {
            log::warn!("view: {e}; opening without it");
        }
    };
    let cluster = |why: &str, extra: &[&str]| {
        let mut argv = vec!["clustering", "--from", from, "-m", "leiden", "-o", &out];
        argv.extend(extra);
        warn(run_senna(&format!("{from} has no {why} yet"), &argv));
    };

    if wants_cell_layout {
        run_senna(
            &format!("{from} has no {method} layout of its cells yet"),
            &["layout", &method, "--from", from, "--out", &out],
        )?;
    }
    let (m, _) = load()?;
    if has_features
        && m.layout
            .methods
            .values()
            .all(|e| e.feature_coords.is_none())
    {
        // Only umap and phate lay out features; t-SNE and tree refuse.
        let method = if method == "phate" { "phate" } else { "umap" };
        warn(run_senna(
            &format!("{from} has no layout of its feature embedding yet"),
            &[
                "layout", method, "--target", "features", "--from", from, "--out", &out,
            ],
        ));
    }
    let latent = m.outputs.geometry_latent().filter(|_| m.kind.has_cells());
    if let (Some(latent), None) = (latent, &m.cluster.clusters) {
        let latent = senna::run_manifest::resolve(&dir, latent);
        cluster("cell clusters", &["--latent", &latent.to_string_lossy()]);
    }
    if has_features && m.cluster.feature_clusters.is_none() {
        cluster("feature clusters", &["--target", "features"]);
    }
    Ok(())
}

pub fn run_view(args: &ViewArgs) -> anyhow::Result<()> {
    use rayon::prelude::*;
    anyhow::ensure!(
        !(args.relabel && args.from.len() > 1),
        "--relabel reads one run; got {} manifests",
        args.from.len()
    );
    if !args.no_compute {
        for from in &args.from {
            prepare_run(args, from)?;
        }
    }
    // Reading runs is independent work; the scenes are built in order.
    let data = args
        .from
        .par_iter()
        .map(|f| Dataset::load(f))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut scenes: Vec<Scene> = data.into_iter().map(|d| Scene::new(d, args)).collect();

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
    if args.suggest {
        let several = scenes.len() > 1;
        for (scene, from) in scenes.iter_mut().zip(&args.from) {
            if several {
                println!("== {from}");
            }
            scene.suggest();
            match scene.suggestion_lines() {
                Some(lines) => lines.iter().for_each(|l| println!("{l}")),
                None => println!("{}", scene.note.take().unwrap_or_default()),
            }
        }
    }

    if let Some(path) = &args.pdf {
        anyhow::ensure!(
            args.width > 0.0 && args.dpi > 0,
            "--width and --dpi must be positive"
        );
        let dpi = args.dpi as f32;
        let w = (args.width * dpi).round() as usize;
        // A roughly 4:3 page, its tiles 4:3 too.
        let (cols, rows) = deck::shape(scenes.len(), 4.0, 3.0);
        let h = (w as f32 * 0.75 * rows as f32 / cols as f32).round() as usize;
        anyhow::ensure!(
            w * h <= deck::MAX_PIXELS,
            "{w} × {h} px is too large; lower --width or --dpi"
        );
        // Labels about 8 pt on the page, whatever its size.
        let cell_px = dpi * 11.0 / 72.0;
        let page = if let [scene] = &scenes[..] {
            let vp = Viewport::fit(scene.current().points.bounds, w, h);
            deck::view_pages(&[(scene, vp, cell_px)]).remove(0)
        } else {
            let titles: Vec<String> = args
                .from
                .iter()
                .map(|f| files::name(std::path::Path::new(f.as_ref())))
                .collect();
            let refs: Vec<&Scene> = scenes.iter().collect();
            let cams = vec![None; scenes.len()];
            deck::render_grid(&refs, &cams, &titles, (w, h), cell_px)
        };
        pdf::write(&[page], dpi, std::path::Path::new(path.as_ref()))?;
        info!(
            "Saved {path} ({w} × {h} px at {} dpi, {} run(s))",
            args.dpi,
            scenes.len()
        );
        return Ok(());
    }

    let sessions = scenes
        .into_iter()
        .zip(&args.from)
        .map(|(s, f)| (s, std::path::PathBuf::from(f.as_ref())))
        .collect();
    tui::run(
        sessions,
        args.graphics,
        args.lupin
            .as_deref()
            .map(String::from)
            .or_else(|| std::env::var("SENNA_LUPIN").ok())
            .unwrap_or_else(|| "lupin".into()),
    )
}

#[cfg(test)]
mod tests {
    use super::data::{Labels, Points, Space};

    fn render_full(s: &Scene, vp: Viewport, cell_px: f32) -> image::RgbaImage {
        deck::frames(&[(s, vp, cell_px)], false)
            .remove(0)
            .to_image()
    }
    use super::*;

    fn pts(names: &[&str]) -> Points {
        let xy = (0..names.len())
            .map(|i| [i as f32, (i % 3) as f32])
            .collect();
        Points::new(names.iter().map(|&n| n.into()).collect(), xy)
    }

    fn space(method: &str, kind: SpaceKind, names: &[&str]) -> Space {
        Space {
            method: method.into(),
            kind,
            points: std::sync::Arc::new(pts(names)),
            backdrop: None,
            parent: None,
        }
    }

    fn scene() -> Scene {
        let cells = ["c1", "c2", "c3", "c4"];
        let genes = ["g1", "g2", "g3"];
        let data = Dataset {
            prefix: "r".into(),
            spaces: vec![
                space("umap", SpaceKind::Cells, &cells),
                space("umap", SpaceKind::Features, &genes),
                space("phate", SpaceKind::Cells, &cells),
            ],
            run: None,
            round: None,
            labels: vec![
                Labels::clusters(
                    LabelKind::Cluster,
                    [("c2", 1), ("c1", 0), ("c3", 1)].map(|(a, b)| (a.into(), b)),
                ),
                Labels::new(
                    LabelKind::Annotation,
                    [("c1", "CT1"), ("c4", "unassigned")].map(|(a, b)| (a.into(), b.into())),
                    &["unassigned"],
                ),
                Labels::new(
                    LabelKind::Markers,
                    [("g3", "CT1")].map(|(a, b)| (a.into(), b.into())),
                    &[],
                ),
            ],
        };
        let args = ViewArgs {
            from: vec!["r.senna.json".into()],
            method: None,
            colour_by: None,
            graphics: Graphics::Blocks,
            pdf: None,
            space: None,
            width: 7.0,
            dpi: 300,
            feature: None,
            markers_of: None,
            observed: false,
            focus: None,
            zoom_into: None,
            suggest: false,
            relabel: false,
            lupin: None,
            no_compute: true,
        };
        Scene::new(data, &args)
    }

    #[test]
    fn labels_join_points_by_name_and_skip_abstentions() {
        let s = scene();
        assert_eq!(s.data.labels[0].title(), "cluster");
        let g = s.groups().unwrap();
        assert_eq!(s.levels(), &["C0".into(), "C1".into()] as &[Box<str>]);
        assert_eq!(g, &[0, 1, 1, NONE]);
        assert_eq!(s.cluster_id_of("c2"), Some(1));
        assert_eq!(s.cluster_id_of("c4"), None);
        let ann = s.data.labels[1].align(&s.data.spaces[0].points);
        assert_eq!(ann, vec![0, NONE, NONE, NONE]);
    }

    #[test]
    fn colour_cycles_through_the_axis_groupings_then_none() {
        let mut s = scene();
        assert_eq!(s.colour, Some(0));
        s.cycle_colour();
        assert_eq!(s.colour, Some(1));
        s.cycle_colour();
        assert_eq!(s.colour, None);
        assert!(s.groups().is_none());
        s.cycle_colour();
        assert_eq!(s.colour, Some(0));
    }

    #[test]
    fn switching_axis_resets_colour_and_focus() {
        let mut s = scene();
        s.step_focus(1);
        assert_eq!(s.focus, Some(0));
        s.set_space(1);
        assert_eq!(s.colour, Some(2));
        assert_eq!(s.focus, None);
        assert_eq!(s.groups().unwrap(), &[NONE, NONE, 0]);
        // Same axis keeps the chosen grouping and focus.
        s.set_space(0);
        s.cycle_colour();
        s.step_focus(-1);
        let (colour, focus) = (s.colour, s.focus);
        s.set_space(2);
        assert_eq!((s.colour, s.focus), (colour, focus));
    }

    #[test]
    fn a_new_colouring_or_focus_clears_the_feature_on_screen() {
        let mut s = scene();
        s.pick = Some(Pick::One("g1".into()));
        s.cycle_colour();
        assert!(s.pick.is_none());
        assert!(s.note.take().is_some());
        s.pick = Some(Pick::One("g1".into()));
        s.step_focus(1);
        assert!(s.pick.is_none());
    }

    #[test]
    fn opening_another_round_leaves_relabel_mode_and_keeps_the_draft() {
        let dir = tempfile::tempdir().unwrap();
        let round = dir.path().join("r.senna.json");
        let mut s = scene();
        let mut draft = review::Draft {
            round: round.clone(),
            ..Default::default()
        };
        draft.cluster(1).verdict = Some(review::Verdict::Keep {
            label: "CT1".into(),
            rationale: "clear".into(),
        });
        s.review = Some(relabel::Review::new(Vec::new(), draft));
        let other = scene().data;
        s.replace_data(other);
        assert!(s.review.is_none());
        assert!(s
            .note
            .as_deref()
            .unwrap()
            .contains("draft stays with r.senna.json"));
        assert!(!review::Draft::load(&round).is_empty());
    }

    #[test]
    fn a_copy_shows_the_same_view_then_goes_its_own_way() {
        let mut s = scene();
        s.set_space(2);
        s.step_focus(1);
        s.text_scale = TEXT_SCALES[3];
        let mut copy = s.duplicate();
        assert_eq!(
            (copy.space, copy.colour, copy.focus, copy.text_scale),
            (s.space, s.colour, s.focus, s.text_scale)
        );
        assert_eq!(copy.groups(), s.groups());
        copy.cycle_colour();
        copy.set_space(0);
        assert_ne!(copy.colour, s.colour);
        assert_eq!(s.space, 2);
    }

    #[test]
    fn t_steps_label_sizes_then_off_then_small() {
        let mut s = scene();
        assert!(s.show_labels);
        assert_eq!(s.text_scale, TEXT_SCALES[1]);
        s.cycle_labels();
        s.cycle_labels();
        assert_eq!(s.text_scale, TEXT_SCALES[3]);
        s.cycle_labels();
        assert!(!s.show_labels);
        s.cycle_labels();
        assert!(s.show_labels);
        assert_eq!(s.text_scale, TEXT_SCALES[0]);
        assert!(s.note.take().unwrap().contains("small"));
    }

    #[test]
    fn focus_wraps_both_ways() {
        let mut s = scene();
        s.step_focus(-1);
        assert_eq!(s.focus, Some(1));
        s.step_focus(1);
        assert_eq!(s.focus, Some(0));
        s.step_focus(-1);
        assert_eq!(s.focus, Some(1));
    }

    #[test]
    fn zooming_nests_and_steps_back_one_level_or_to_the_top() {
        let mut s = scene();
        let pts = |n: &[&str]| n.iter().map(|&x| x.into()).collect::<Vec<Box<str>>>();
        let xy = vec![[0.0, 0.0], [1.0, 1.0]];
        s.add_zoomed(0, "C1", pts(&["c2", "c3"]), xy.clone());
        let first = s.space;
        assert_eq!(s.current().parent, Some(0));
        assert!(s.current().method.ends_with("C1"));
        s.add_zoomed(first, "C1", pts(&["c2"]), vec![[0.0, 0.0]]);
        assert_eq!(s.current().parent, Some(first));
        assert!(s.zoom_out());
        assert_eq!(s.space, first);
        s.add_zoomed(first, "C1", pts(&["c3"]), vec![[0.0, 0.0]]);
        assert!(s.zoom_to_root());
        assert_eq!(s.space, 0);
        assert!(!s.zoom_out());
        assert!(!s.zoom_to_root());
    }

    #[test]
    fn tab_steps_through_the_views_of_one_method_and_skips_zooms() {
        // umap cells (0), umap features (1), phate cells (2).
        let mut s = scene();
        assert_eq!(s.next_view(1), Some(1));
        assert_eq!(s.next_view(-1), Some(1));
        s.set_space(1);
        assert_eq!(s.next_view(1), Some(0));
        // phate has one view only: step through every method's views.
        s.set_space(2);
        assert_eq!(s.next_view(1), Some(0));
        assert_eq!(s.next_view(-1), Some(1));
        // A zoom steps from the map it came from, never into another zoom.
        s.set_space(0);
        let names: Vec<Box<str>> = vec!["c2".into(), "c3".into()];
        s.add_zoomed(0, "C1", names, vec![[0.0, 0.0], [1.0, 1.0]]);
        assert_eq!(s.next_view(1), Some(1));
        s.set_space(1);
        assert_eq!(s.next_view(1), Some(0));
    }

    #[test]
    fn start_options_pick_the_space() {
        let s = scene();
        assert_eq!(pick_space(&s.data, Some("phate"), None), 2);
        assert_eq!(pick_space(&s.data, Some("umap"), Some("features")), 1);
        assert_eq!(pick_space(&s.data, Some("tsne"), None), 0);
    }

    #[test]
    fn a_clicked_cell_is_drawn_with_edges_to_its_features() {
        let mut s = scene();
        let mut placed = space("umap", SpaceKind::FeaturesOnCells, &["g1", "g2", "g3"]);
        placed.points = std::sync::Arc::new(Points::new(
            ["g1", "g2", "g3"].map(Into::into).to_vec(),
            vec![[3.0, 0.0], [3.0, 2.0], [0.0, 2.0]],
        ));
        placed.backdrop = Some(0);
        s.data.spaces.push(placed);
        s.show_labels = false;
        assert_eq!(s.near_spaces(), (Some(0), Some(3)));
        let vp = Viewport::fit(s.current().points.bounds, 200, 200);
        let before = render_full(&s, vp, 16.0);
        s.near = Some(features::Near {
            name: "c1".into(),
            centre: features::Centre::Cell,
            features: vec![("g1".into(), 1.0), ("nowhere".into(), 0.5)],
            cells: Vec::new(),
        });
        let after = render_full(&s, vp, 16.0);
        // Halfway between c1 at (0, 0) and g1 at (3, 0): only the edge is there.
        let (x, y) = vp.to_px([1.5, 0.0]);
        let (x, y) = (x as u32, y as u32);
        assert_ne!(before.get_pixel(x, y), after.get_pixel(x, y));
        // Nothing is drawn towards the feature with no place.
        let (x, y) = vp.to_px([0.0, 1.5]);
        assert_eq!(
            before.get_pixel(x as u32, y as u32),
            after.get_pixel(x as u32, y as u32)
        );
    }
}
