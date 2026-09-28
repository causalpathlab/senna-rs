//! `senna view`: a run's layouts, coloured by its groupings, in the terminal.

mod activity;
mod color;
mod data;
mod decide;
mod features;
mod files;
mod paint;
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
use paint::render_full;
use render::{draw_labels, group_medians, Job, Label, Paint, Tier, Viewport};
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
        help = "Run manifest (`{out}.senna.json`) with at least one layout",
        long_help = "Run manifest (`{out}.senna.json`).\n\
                     Every layout under `manifest.layout.methods` is available,\n\
                     together with clusters, annotation and topics the run carries."
    )]
    pub from: Box<str>,

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
        help = "Render one frame to this PNG and exit (no terminal needed)",
        long_help = "Render the starting view to a PNG and exit.\n\
                     Combine with --method / --space / --colour-by and --size."
    )]
    pub png: Option<Box<str>>,

    #[arg(
        long,
        help = "Start on this axis: cells, features-on-cells, or features"
    )]
    pub space: Option<Box<str>>,

    #[arg(
        long,
        default_value = "1600x1200",
        help = "PNG size as WIDTHxHEIGHT (with --png)"
    )]
    pub size: Box<str>,

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
        help = "Print suggested features for the starting view (with --png, show the top one)"
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
    orders: std::cell::RefCell<Vec<(OrderKey, std::rc::Rc<Vec<u32>>)>>,
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
fn prepare_run(args: &ViewArgs) -> anyhow::Result<()> {
    use senna::run_manifest::RunManifest;
    let from: &str = &args.from;
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
    if !args.no_compute {
        prepare_run(args)?;
    }
    let data = Dataset::load(&args.from)?;
    let mut scene = Scene::new(data, args);

    if args.relabel {
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
        scene.suggest();
        match scene.suggestion_lines() {
            Some(lines) => lines.iter().for_each(|l| println!("{l}")),
            None => println!("{}", scene.note.take().unwrap_or_default()),
        }
    }

    if let Some(png) = &args.png {
        let (w, h) = args
            .size
            .split_once('x')
            .and_then(|(w, h)| Some((w.parse::<usize>().ok()?, h.parse::<usize>().ok()?)))
            .ok_or_else(|| anyhow::anyhow!("--size wants WIDTHxHEIGHT, got {}", args.size))?;
        let vp = Viewport::fit(scene.current().points.bounds, w, h);
        let img = render_full(&scene, vp, (h as f32 / 36.0).max(16.0));
        img.save(png.as_ref())?;
        info!("Saved {png} ({})", scene.caption());
        return Ok(());
    }

    tui::run(
        scene,
        args.graphics,
        std::path::PathBuf::from(args.from.as_ref()),
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
            points: pts(names),
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
            from: "r.senna.json".into(),
            method: None,
            colour_by: None,
            graphics: Graphics::Blocks,
            png: None,
            space: None,
            size: "10x10".into(),
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
        placed.points = Points::new(
            ["g1", "g2", "g3"].map(Into::into).to_vec(),
            vec![[3.0, 0.0], [3.0, 2.0], [0.0, 2.0]],
        );
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
