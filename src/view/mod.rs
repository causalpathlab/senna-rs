//! `senna view`: look at a run's layouts in the terminal.
//!
//! Every layout method the run has (umap, phate, tsne) on each axis it has
//! (cells, features placed on cells, features on their own), coloured by any
//! grouping the run carries (annotation, clusters, topics, markers). Drawn by
//! a progressive rasterizer on a light page with labels set on the map.

mod activity;
mod color;
mod data;
mod decide;
mod files;
mod relabel;
mod render;
mod review;
mod rounds;
mod style;
mod sublayout;
mod text;
mod tui;

use activity::{Activity, Levels, Source};
use color::Rgb;
use data::{Axis, Dataset, LabelKind, SpaceKind, NONE};
use data_beans::utilities::name_matching::GeneIndex;
use render::{draw_labels, group_medians, Job, Label, Paint, Viewport};
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

    #[arg(long, help = "Start on this layout method (umap, phate, tsne); computed first when the run lacks it")]
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
}

/// Features suggested for one view, best first.
struct Suggestions {
    space: usize,
    title: String,
    list: Vec<(Box<str>, f32)>,
}

/// What a layer's draw order depends on: its space, the grouping and focus
/// that rank its points, and which activity (by `Shown::id`) sorts them.
/// Space, colouring, focus, feature shown, merge selection, backdrop.
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

/// Text sizes on the map, as multiples of the terminal's cell height.
const TEXT_SCALES: [f32; 4] = [1.0, 1.4, 1.8, 2.4];

/// What is on screen: which space, which grouping, what is focused, and
/// which feature's activity (if any) is drawn over the cells.
pub(crate) struct Scene {
    pub data: Dataset,
    pub space: usize,
    /// Index into `data.labels`, when colouring by a grouping.
    pub colour: Option<usize>,
    pub focus: Option<u32>,
    pub show_labels: bool,
    /// Text size on the map, relative to the terminal's cell height.
    pub text_scale: f32,
    /// Group id per point, cached per (space, labels).
    groups: Option<(usize, usize, Vec<u32>)>,
    /// Resolved style per group of the current grouping.
    styles: Vec<style::Resolved>,
    /// Saved per-group styles for every grouping.
    pub book: style::Book,
    pub pick: Option<Pick>,
    pub source: Source,
    activity: Option<Activity>,
    shown: Option<Shown>,
    ramp: Vec<Rgb>,
    /// Suggested features for one view.
    suggestions: Option<Suggestions>,
    /// Draw orders computed for recent layer states; see `layers`.
    orders: std::cell::RefCell<Vec<(OrderKey, std::rc::Rc<Vec<u32>>)>>,
    /// Source of `Shown::id`.
    shown_ids: u64,
    /// Name index of a feature space, for marking the picked feature.
    feature_index: std::cell::RefCell<Option<(usize, GeneIndex)>>,
    /// Group label anchors for a (space, grouping).
    medians: std::cell::RefCell<Option<((usize, usize), render::Medians)>>,
    /// The run's geometry table, read on the first zoom into a group.
    geometry: Option<std::sync::Arc<sublayout::Geometry>>,
    /// Relabel mode, when on.
    pub review: Option<relabel::Review>,
    /// Features near the last clicked cell, and sets locked on screen.
    pub near: Option<relabel::Near>,
    pub locked: Vec<relabel::Near>,
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
            suggestions: None,
            review: None,
            near: None,
            locked: Vec::new(),
            orders: std::cell::RefCell::new(Vec::new()),
            shown_ids: 0,
            feature_index: std::cell::RefCell::new(None),
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

    pub fn current(&self) -> &data::Space {
        &self.data.spaces[self.space]
    }

    /// Groupings that apply to the current axis.
    pub fn colour_choices(&self) -> Vec<usize> {
        let axis = self.current().axis();
        (0..self.data.labels.len())
            .filter(|&i| self.data.labels[i].axis() == axis)
            .collect()
    }

    fn default_colour(&self, want: Option<&str>) -> Option<usize> {
        let choices = self.colour_choices();
        if let Some(w) = want {
            if let Some(&i) = choices
                .iter()
                .find(|&&i| self.data.labels[i].title().eq_ignore_ascii_case(w))
            {
                return Some(i);
            }
            log::warn!("view: no `{w}` grouping for {}", self.current().title());
        }
        choices.first().copied()
    }

    fn refresh_groups(&mut self) {
        let Some(li) = self.colour else {
            self.groups = None;
            self.styles.clear();
            return;
        };
        let labels = &self.data.labels[li];
        self.styles = self.book.resolve(labels.title(), &labels.levels);
        if matches!(&self.groups, Some((s, l, _)) if *s == self.space && *l == li) {
            return;
        }
        let g = labels.align(&self.current().points);
        self.groups = Some((self.space, li, g));
    }

    /// Change group `g`'s style in the current grouping and save the book.
    pub fn restyle(&mut self, g: usize, f: impl FnOnce(&mut style::Style)) {
        let Some(li) = self.colour else { return };
        let labels = &self.data.labels[li];
        let (grouping, name) = (labels.title().to_string(), labels.levels[g].to_string());
        let mut st = self.book.get(&grouping, &name);
        f(&mut st);
        self.book.set(&grouping, &name, st);
        self.refresh_groups();
    }

    /// Write the style book beside the run (when the style menu closes).
    pub fn save_styles(&mut self) {
        if let Err(e) = self.book.save(&self.data.prefix) {
            self.note = Some(format!("could not save styles: {e}"));
        }
    }

    /// Whether suggestions exist for the view on screen.
    pub fn has_suggestions(&self) -> bool {
        self.suggestions
            .as_ref()
            .is_some_and(|s| s.space == self.space)
    }

    /// Current style of group `g` in the current grouping.
    pub fn style_of(&self, g: usize) -> style::Style {
        self.colour.map_or_else(style::Style::plain, |li| {
            let l = &self.data.labels[li];
            self.book.get(l.title(), &l.levels[g])
        })
    }

    /// Resolved style of group `g`, for drawing swatches.
    pub fn resolved(&self, g: usize) -> Option<&style::Resolved> {
        self.styles.get(g)
    }

    pub fn groups(&self) -> Option<&[u32]> {
        self.groups.as_ref().map(|(_, _, g)| g.as_slice())
    }

    pub fn levels(&self) -> &[Box<str>] {
        self.colour
            .map_or(&[], |i| self.data.labels[i].levels.as_slice())
    }

    pub fn set_space(&mut self, space: usize) {
        let axis = self.current().axis();
        self.space = space;
        if self.current().axis() != axis {
            self.colour = self.default_colour(None);
            self.focus = None;
        }
        self.refresh_groups();
        self.refresh_activity();
    }

    /// What zooming into the focused group needs: its name, its cells, and
    /// the geometry table to lay them out from.
    pub fn zoom_request(&mut self) -> Result<ZoomRequest, String> {
        if self.current().axis() != Axis::Cells {
            return Err("zoom into a group from a cell view".into());
        }
        let (Some(f), Some(groups)) = (self.focus, self.groups()) else {
            return Err("focus a group first ([ ] or click), then press z".into());
        };
        let points = &self.current().points;
        let names: Vec<Box<str>> = groups
            .iter()
            .zip(&points.names)
            .filter(|&(&g, _)| g == f)
            .map(|(_, n)| n.clone())
            .collect();
        let label = self.levels()[f as usize].to_string();
        if self.geometry.is_none() {
            let (m, dir) = self
                .data
                .run
                .as_ref()
                .ok_or("no manifest to read the embedding from")?;
            let g = sublayout::Geometry::load(m, dir).map_err(|e| e.to_string())?;
            self.geometry = Some(std::sync::Arc::new(g));
        }
        let geometry = self.geometry.clone().expect("just loaded");
        Ok(ZoomRequest {
            label,
            names,
            geometry,
        })
    }

    /// Add a layout of one group, zoomed from `parent`, and switch to it.
    pub fn add_zoomed(
        &mut self,
        parent: usize,
        label: &str,
        names: Vec<Box<str>>,
        xy: Vec<[f32; 2]>,
    ) {
        let method = format!("{} › {label}", self.data.spaces[parent].method);
        self.data.spaces.push(data::Space {
            method,
            kind: SpaceKind::Cells,
            points: data::Points::new(names, xy),
            backdrop: None,
            parent: Some(parent),
        });
        self.focus = None;
        self.set_space(self.data.spaces.len() - 1);
    }

    /// Back to the layout this one was zoomed from. Returns whether there was one.
    pub fn zoom_out(&mut self) -> bool {
        match self.current().parent {
            Some(p) => {
                self.set_space(p);
                true
            }
            None => false,
        }
    }

    /// Swap in a reloaded or different round of the same run, keeping the
    /// layout, grouping and focused group the view had, matched by name. A
    /// zoomed layout falls back to the layout it was zoomed from.
    pub fn replace_data(&mut self, data: Dataset) {
        // A relabel draft uses the ids of the round it was made on; it must
        // never be applied to another round. Keep it with its own round.
        let left = self.review.as_ref().map(|r| r.draft.round.clone());
        if let Some(round) = left {
            self.leave_review();
            let name = round
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            self.note = Some(format!(
                "a different round is open: relabel mode left, the draft stays with {name}"
            ));
        }
        let (method, kind) = {
            let s = &self.data.spaces[self.root()];
            (s.method.clone(), s.kind)
        };
        let colour = self.colour.map(|i| self.data.labels[i].kind);
        let focus = self.focused_name();
        // A new annotation round of the same run points at the same model
        // and data files; keep what was read from them.
        let same_run = match (&self.data.run, &data.run) {
            (Some((a, ad)), Some((b, bd))) => {
                ad == bd
                    && serde_json::to_value(&a.outputs).ok()
                        == serde_json::to_value(&b.outputs).ok()
                    && serde_json::to_value(&a.data).ok() == serde_json::to_value(&b.data).ok()
            }
            _ => false,
        };
        self.data = data;
        self.near = None;
        self.locked.clear();
        self.orders.borrow_mut().clear();
        self.feature_index.borrow_mut().take();
        self.medians.borrow_mut().take();
        self.space = self
            .data
            .spaces
            .iter()
            .position(|s| s.method == method && s.kind == kind)
            .unwrap_or(0);
        self.colour = colour
            .and_then(|t| {
                self.colour_choices()
                    .into_iter()
                    .find(|&i| self.data.labels[i].kind == t)
            })
            .or_else(|| self.default_colour(None));
        self.groups = None;
        self.refresh_groups();
        self.focus =
            focus.and_then(|f| self.levels().iter().position(|l| *l == f).map(|i| i as u32));
        self.shown = None;
        if same_run {
            if let Some(a) = self.activity.as_mut() {
                a.forget_views();
            }
        } else {
            self.activity = None;
            self.geometry = None;
        }
        self.suggestions = None;
        self.refresh_activity();
    }

    /// Colour by what the round changed against its source, and describe
    /// it: how many cells now carry each label.
    pub fn show_changes(&mut self) -> Vec<String> {
        let Some(li) = self
            .data
            .labels
            .iter()
            .position(|l| l.kind == LabelKind::Changed)
        else {
            return vec!["no cell changed label".into()];
        };
        let labels = &self.data.labels[li];
        let mut count = vec![0usize; labels.levels.len()];
        for &g in labels.by_name.values() {
            count[g as usize] += 1;
        }
        let mut out = vec![format!("{} cells changed label:", labels.by_name.len())];
        let mut rows: Vec<(usize, &str)> = count
            .iter()
            .zip(&labels.levels)
            .map(|(&n, l)| (n, l.as_ref()))
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.0));
        out.extend(rows.iter().map(|(n, l)| format!("  {n:>6}  now {l}")));
        self.colour = Some(li);
        self.focus = None;
        self.clear_pick();
        self.refresh_groups();
        out
    }

    /// The run's clusters, when it has them.
    fn cluster_labels(&self) -> Option<&data::Labels> {
        self.data
            .labels
            .iter()
            .find(|l| l.kind == LabelKind::Cluster)
    }

    /// What the round records about the cluster of cell `cell`, as panel lines.
    pub fn cluster_info(&self, cell: &str) -> Option<Vec<String>> {
        let id = self.cluster_id_of(cell)?;
        Some(self.data.round.as_ref()?.cluster_lines(&id.to_string()))
    }

    /// The cluster id of cell `cell`, from the run's `cluster.clusters`.
    pub fn cluster_id_of(&self, cell: &str) -> Option<i64> {
        let c = self.cluster_labels()?;
        Some(c.ids[*c.by_name.get(cell)? as usize])
    }

    /// The label the current round gives cluster `id`, and its top call.
    pub fn cluster_call(&self, id: i64) -> (Option<String>, Option<(String, Option<f64>)>) {
        self.data
            .round
            .as_ref()
            .map_or((None, None), |r| r.call(&id.to_string()))
    }

    /// Cell-type names already in use (annotation and markers), for label
    /// completion.
    pub fn known_labels(&self) -> Vec<Box<str>> {
        let mut out: Vec<Box<str>> = self
            .data
            .labels
            .iter()
            .filter(|l| matches!(l.kind, LabelKind::Annotation | LabelKind::Markers))
            .flat_map(|l| l.levels.iter().cloned())
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Back to the top of the zoom chain. Returns whether it moved.
    /// The top of the current zoom chain.
    fn root(&self) -> usize {
        let mut at = self.space;
        while let Some(p) = self.data.spaces[at].parent {
            at = p;
        }
        at
    }

    pub fn zoom_to_root(&mut self) -> bool {
        let at = self.root();
        if at == self.space {
            return false;
        }
        self.set_space(at);
        true
    }

    fn markers(&self) -> Option<&data::Labels> {
        self.data
            .labels
            .iter()
            .find(|l| l.kind == LabelKind::Markers)
    }

    /// The focused group's name, whichever grouping it belongs to.
    pub fn focused_name(&self) -> Option<Box<str>> {
        self.focus.map(|f| self.levels()[f as usize].clone())
    }

    /// Marker features of `group`, matched loosely (case, and spaces, commas
    /// and underscores, which annotation tools rewrite).
    fn marker_features(&self, group: &str) -> Vec<Box<str>> {
        let norm = rounds::label_key;
        let Some(m) = self.markers() else {
            return Vec::new();
        };
        let Some(id) = m.levels.iter().position(|l| norm(l) == norm(group)) else {
            return Vec::new();
        };
        let mut out: Vec<Box<str>> = m
            .by_name
            .iter()
            .filter(|&(_, &g)| g as usize == id)
            .map(|(n, _)| n.clone())
            .collect();
        out.sort();
        out
    }

    /// Rank features for the view on screen: what distinguishes the focused
    /// group from the rest of the view, or, with nothing focused, what varies
    /// most here. Shows the first one straight away.
    pub fn suggest(&mut self) {
        const TOP: usize = 25;
        if self.current().axis() != Axis::Cells {
            self.note = Some("suggestions work on a cell view".into());
            return;
        }
        let space = self.space;
        let mask: Option<Vec<bool>> = match (self.focus, self.groups()) {
            (Some(f), Some(g)) => Some(g.iter().map(|&x| x == f).collect()),
            _ => None,
        };
        let title = match self.focused_name() {
            Some(name) => format!("features that set {name} apart in this view"),
            None => "features that vary most in this view".to_string(),
        };
        if self.activity().is_none() {
            self.note = Some("no manifest to read the model from".into());
            return;
        }
        let universe = &self.data.spaces[space].points.names;
        let activity = self.activity.as_mut().expect("made above");
        match activity.suggest(space, universe, mask.as_deref(), TOP) {
            Ok(list) if !list.is_empty() => {
                let first = list[0].0.clone();
                self.suggestions = Some(Suggestions { space, title, list });
                self.set_pick(Pick::One(first));
            }
            Ok(_) => self.note = Some("no feature stands out here".into()),
            Err(e) => self.note = Some(e),
        }
    }

    /// Drop the suggestions. Returns whether there were any.
    pub fn clear_suggestions(&mut self) -> bool {
        self.suggestions.take().is_some()
    }

    /// Panel text for the suggestions of the view on screen, marking the
    /// feature being shown.
    pub fn suggestion_lines(&self) -> Option<Vec<String>> {
        let Suggestions { space, title, list } = self.suggestions.as_ref()?;
        if *space != self.space {
            return None;
        }
        let shown = match &self.pick {
            Some(Pick::One(f)) => Some(f.as_ref()),
            _ => None,
        };
        let mut out = vec![title.clone(), "g / G step · o observed · x close".into()];
        out.extend(list.iter().enumerate().map(|(k, (f, v))| {
            let mark = if shown == Some(f.as_ref()) {
                "▸"
            } else {
                " "
            };
            format!("{mark} {:>2}  {f:<16} {v:>6.2}", k + 1)
        }));
        Some(out)
    }

    /// Features `g` steps through: the current suggestions, else the focused
    /// group's markers, else every marker grouped by type.
    fn feature_list(&self) -> Vec<Box<str>> {
        if let Some(sug) = self.suggestions.as_ref().filter(|s| s.space == self.space) {
            return sug.list.iter().map(|(f, _)| f.clone()).collect();
        }
        if let Some(name) = self.focused_name() {
            let own = self.marker_features(&name);
            if !own.is_empty() {
                return own;
            }
        }
        let Some(m) = self.markers() else {
            return Vec::new();
        };
        let mut all: Vec<(u32, Box<str>)> =
            m.by_name.iter().map(|(n, &g)| (g, n.clone())).collect();
        all.sort();
        all.into_iter().map(|(_, n)| n).collect()
    }

    pub fn step_feature(&mut self, delta: i64) {
        let list = self.feature_list();
        if list.is_empty() {
            self.note = Some("no marker table in this run; press / to search a feature".into());
            return;
        }
        let at = match &self.pick {
            Some(Pick::One(n)) => list.iter().position(|x| x == n),
            _ => None,
        };
        let n = list.len() as i64;
        let next = match at {
            Some(i) => (i as i64 + delta).rem_euclid(n),
            None if delta > 0 => 0,
            None => n - 1,
        };
        self.set_pick(Pick::One(list[next as usize].clone()));
    }

    pub fn set_pick(&mut self, pick: Pick) {
        self.pick = Some(pick);
        self.refresh_activity();
    }

    /// Show the focused group's marker set as one activity.
    pub fn pick_marker_set(&mut self) {
        let Some(name) = self.focused_name() else {
            self.note = Some("focus a group first ([ ] or click), then press a".into());
            return;
        };
        if self.marker_features(&name).is_empty() {
            self.note = Some(format!("no markers listed for {name}"));
            return;
        }
        self.set_pick(Pick::Markers(name));
    }

    pub fn toggle_source(&mut self) {
        self.source = self.source.other();
        self.refresh_activity();
    }

    /// Drop the activity view. Returns whether there was one.
    pub fn clear_pick(&mut self) -> bool {
        self.shown = None;
        self.pick.take().is_some()
    }

    /// Feature names searchable under the current source.
    pub fn searchable(&mut self) -> Vec<Box<str>> {
        let source = self.source;
        self.activity()
            .and_then(|a| a.feature_names(source).map(<[_]>::to_vec).ok())
            .unwrap_or_default()
    }

    fn activity(&mut self) -> Option<&mut Activity> {
        if self.activity.is_none() {
            let (m, dir) = self.data.run.clone()?;
            self.activity = Some(Activity::new(m, dir));
        }
        self.activity.as_mut()
    }

    /// Recompute the activity drawn for the current pick, space and source.
    /// Cell spaces only; on a feature space the pick is marked instead.
    fn refresh_activity(&mut self) {
        let Some(pick) = self.pick.clone() else {
            self.shown = None;
            return;
        };
        if self.current().axis() != Axis::Cells {
            self.shown = None;
            return;
        }
        let (space, source) = (self.space, self.source);
        if matches!(&self.shown, Some(s) if s.space == space && s.pick == pick && s.source == source)
        {
            return;
        }
        let set = match &pick {
            Pick::Markers(group) => self.marker_features(group),
            Pick::One(_) => Vec::new(),
        };
        if self.activity().is_none() {
            self.note = Some("no manifest to read activity from".into());
            return;
        }
        let names = &self.data.spaces[space].points.names;
        let activity = self.activity.as_mut().expect("made above");
        let result = match &pick {
            Pick::One(f) => activity
                .levels(f, source, space, names)
                .map(|(l, spelled)| (l, format!("{spelled} · {}", source.name()))),
            Pick::Markers(group) => activity
                .set_levels(&set, source, space, names)
                .map(|(l, used)| (l, format!("{group} markers ({used}) · {}", source.name()))),
        };
        match result {
            Ok((levels, title)) => {
                self.shown_ids += 1;
                self.shown = Some(Shown {
                    id: self.shown_ids,
                    space,
                    pick,
                    source,
                    title,
                    levels,
                });
            }
            Err(e) => {
                self.shown = None;
                self.note = Some(e);
            }
        }
    }

    /// Step to the next text size on the map.
    pub fn cycle_text_size(&mut self) {
        let at = TEXT_SCALES
            .iter()
            .position(|&t| t >= self.text_scale)
            .unwrap_or(0);
        let next = (at + 1) % TEXT_SCALES.len();
        self.text_scale = TEXT_SCALES[next];
        self.note = Some(format!(
            "text size {} of {} · T for the next",
            next + 1,
            TEXT_SCALES.len()
        ));
    }

    /// Colour by the grouping of `kind`, when the run has one.
    pub fn colour_by(&mut self, kind: LabelKind) {
        if let Some(li) = self.data.labels.iter().position(|l| l.kind == kind) {
            self.clear_pick();
            self.colour = Some(li);
            self.focus = None;
            self.refresh_groups();
        }
    }

    /// Next grouping. A feature's activity would hide the groups, so it is
    /// cleared (its suggestions stay; `g` brings them back).
    pub fn cycle_colour(&mut self) {
        if self.clear_pick() {
            self.note = Some("back to group colours · g brings the features back".into());
        }
        let choices = self.colour_choices();
        self.colour = match self
            .colour
            .and_then(|c| choices.iter().position(|&i| i == c))
        {
            Some(p) if p + 1 < choices.len() => Some(choices[p + 1]),
            Some(_) => None,
            None => choices.first().copied(),
        };
        self.focus = None;
        self.refresh_groups();
    }

    /// Focus the next or previous group. Activity and suggestions shown for
    /// the previous group no longer apply, so they are cleared.
    pub fn step_focus(&mut self, delta: i64) {
        let n = self.levels().len() as i64;
        if n == 0 {
            return;
        }
        let had_pick = self.clear_pick();
        if self.clear_suggestions() || had_pick {
            self.note = Some("n suggests features for this group".into());
        }
        self.focus = Some(match self.focus {
            None if delta > 0 => 0,
            None => (n - 1) as u32,
            Some(f) => ((f as i64 + delta).rem_euclid(n)) as u32,
        });
    }

    /// Layers for the renderer: the backdrop (if any) under the points.
    pub fn layers(&self) -> Vec<Paint<'_>> {
        let space = self.current();
        let mut layers = Vec::new();
        let mut keys = Vec::new();
        if let Some(b) = space.backdrop {
            layers.push(Paint {
                points: &self.data.spaces[b].points,
                groups: None,
                styles: &[],
                focus: None,
                selected: None,
                muted: true,
                size: 0.8,
                levels: None,
                order: None,
            });
            keys.push((b, None, None, 0, 0, true));
        }
        let shown = self.shown.as_ref().filter(|s| s.space == self.space);
        let merge = self.review.as_ref().and_then(|r| r.merge.as_ref());
        let selected = merge.map(|m| m.levels.as_slice());
        layers.push(Paint {
            points: &space.points,
            groups: self.groups(),
            styles: &self.styles,
            focus: self.focus,
            selected,
            muted: false,
            size: if space.backdrop.is_some() { 1.5 } else { 1.0 },
            levels: shown.map(|s| (&s.levels, self.ramp.as_slice())),
            order: None,
        });
        keys.push((
            self.space,
            self.colour,
            self.focus,
            shown.map_or(0, |s| s.id),
            merge.map_or(0, |m| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                m.chosen.hash(&mut h);
                h.finish() | 1
            }),
            false,
        ));
        // The order only changes with these keys, not with the camera: reuse
        // it across pans and zooms instead of re-ranking every point.
        let mut cache = self.orders.borrow_mut();
        for (paint, key) in layers.iter_mut().zip(keys) {
            let order = match cache.iter().find(|(k, _)| *k == key) {
                Some((_, o)) => o.clone(),
                None => {
                    let o = std::rc::Rc::new(paint.draw_order());
                    cache.insert(0, (key, o.clone()));
                    cache.truncate(4);
                    o
                }
            };
            paint.order = Some(order);
        }
        layers
    }

    /// Labels for a finished frame. Cells: each group's name at its median.
    /// Features: the features' own names (a group of features, such as one
    /// type's markers, is often scattered, so a median would point at
    /// nothing); labelled ones first, and every visible one when few remain.
    pub fn labels(&self, vp: &Viewport, cell_px: f32) -> Vec<Label> {
        let mut out = Vec::new();
        if !self.show_labels {
            return out;
        }
        let space = self.current();
        let groups = self.groups();
        let visible = |i: usize| {
            let (x, y) = vp.to_px(space.points.xy[i]);
            x >= 0.0 && y >= 0.0 && x < vp.w as f32 && y < vp.h as f32
        };

        if space.axis() == Axis::Cells {
            let Some(groups) = groups else { return out };
            let font = Font::for_cell_height(cell_px * self.text_scale, true);
            let n = self.levels().len();
            let key = (self.space, self.colour.unwrap_or(usize::MAX));
            let mut cached = self.medians.borrow_mut();
            if cached.as_ref().is_none_or(|(k, _)| *k != key) {
                *cached = Some((key, group_medians(&space.points, groups, n)));
            }
            let medians = cached.as_ref().map(|(_, m)| m.clone()).unwrap_or_default();
            drop(cached);
            for (g, m) in medians.into_iter().enumerate() {
                let Some((xy, size)) = m else { continue };
                if self.focus.is_some_and(|f| f as usize != g) || self.styles[g].hidden {
                    continue;
                }
                let (x, y) = vp.to_px(xy);
                out.push(Label {
                    text: self.levels()[g].to_string(),
                    x,
                    y,
                    ink: self.styles[g].ink,
                    priority: size as f32,
                    font,
                });
            }
            return out;
        }

        const MAX_NAMED: usize = 400;
        const NAME_ALL_BELOW: usize = 120;
        let font = Font::for_cell_height(cell_px * 0.8 * self.text_scale, false);
        let group_of = |i: usize| groups.map_or(NONE, |g| g[i]);
        let wanted = |g: u32| match self.focus {
            Some(f) => g == f,
            None => g != NONE,
        };
        let order = space.points.order.iter().map(|&i| i as usize);
        let mut named: Vec<usize> = order
            .clone()
            .filter(|&i| wanted(group_of(i)) && visible(i))
            .take(MAX_NAMED)
            .collect();
        let n_visible = order
            .clone()
            .filter(|&i| visible(i))
            .take(NAME_ALL_BELOW + 1)
            .count();
        if n_visible <= NAME_ALL_BELOW {
            named.extend(order.filter(|&i| !wanted(group_of(i)) && visible(i)));
        }
        let n = named.len();
        for (rank, i) in named.into_iter().enumerate() {
            let g = group_of(i);
            let (x, y) = vp.to_px(space.points.xy[i]);
            if g != NONE && self.styles[g as usize].hidden {
                continue;
            }
            let ink = if g == NONE {
                color::linear_rgb(color::INK)
            } else {
                self.styles[g as usize].ink
            };
            out.push(Label {
                text: space.points.names[i].to_string(),
                x,
                y: y - font.line_height() as f32 * 0.7,
                ink,
                priority: (n - rank) as f32,
                font,
            });
        }
        out
    }

    /// A colour key, for feature views only: their labels are feature names,
    /// so the group a colour stands for has to be spelled out once. Groups
    /// with the most points first.
    pub fn legend(&self) -> Vec<(String, Rgb, Rgb)> {
        const MAX_ENTRIES: usize = 24;
        let (Some(groups), Axis::Features) = (self.groups(), self.current().axis()) else {
            return Vec::new();
        };
        let mut count = vec![0usize; self.levels().len()];
        for &g in groups {
            if g != NONE {
                count[g as usize] += 1;
            }
        }
        let mut ids: Vec<usize> = (0..count.len())
            .filter(|&g| {
                count[g] > 0 && !self.styles[g].hidden && self.focus.is_none_or(|f| f as usize == g)
            })
            .collect();
        ids.sort_by_key(|&g| std::cmp::Reverse(count[g]));
        ids.truncate(MAX_ENTRIES);
        ids.into_iter()
            .map(|g| {
                (
                    self.levels()[g].to_string(),
                    self.styles[g].colour,
                    self.styles[g].ink,
                )
            })
            .collect()
    }

    /// Where the picked feature sits in a feature space, if it is one there.
    fn picked_point(&self) -> Option<usize> {
        let Some(Pick::One(name)) = &self.pick else {
            return None;
        };
        let space = self.current();
        if space.axis() != Axis::Features {
            return None;
        }
        let mut cached = self.feature_index.borrow_mut();
        if cached.as_ref().is_none_or(|(s, _)| *s != self.space) {
            *cached = Some((self.space, GeneIndex::build(&space.points.names)));
        }
        cached
            .as_ref()
            .and_then(|(_, index)| index.match_gene(name))
    }

    /// Labels, key and marks over a composited frame.
    pub fn decorate(&self, frame: &mut render::Frame, vp: &Viewport, cell_px: f32) {
        // The picked feature is marked even with labels off: it is the answer
        // to what was just asked for.
        let mut reserved = Vec::new();
        let mut labels = Vec::new();
        if let Some(i) = self.picked_point() {
            let (x, y) = vp.to_px(self.current().points.xy[i]);
            let ink = color::highlight_ink();
            render::draw_ring(frame, x, y, 0.45 * cell_px, ink);
            let font = Font::for_cell_height(cell_px * self.text_scale, true);
            labels.push(Label {
                text: self.current().points.names[i].to_string(),
                x,
                y: y - 0.45 * cell_px - font.line_height() as f32 * 0.6,
                ink,
                priority: f32::INFINITY,
                font,
            });
        }
        // Features near a clicked cell (and pinned names), at their places:
        // on a cell map, where the features sit among the cells; on a
        // feature map, the features' own points.
        let pos = match self.current().axis() {
            Axis::Cells => self.feature_positions(),
            Axis::Features => Some(&self.current().points),
        };
        if let Some(pos) = pos {
            let font = Font::for_cell_height(cell_px * 0.85 * self.text_scale, true);
            let ink = color::highlight_ink();
            for near in self.locked.iter().chain(&self.near) {
                for (rank, (f, _)) in near.features.iter().enumerate() {
                    let Some(i) = pos.names.iter().position(|n| n == f) else {
                        continue;
                    };
                    let (x, y) = vp.to_px(pos.xy[i]);
                    render::draw_ring(frame, x, y, 0.25 * cell_px, ink);
                    labels.push(Label {
                        text: f.to_string(),
                        x,
                        y: y - 0.25 * cell_px - font.line_height() as f32 * 0.6,
                        ink,
                        priority: 1e6 - rank as f32,
                        font,
                    });
                }
            }
        }
        let font = Font::for_cell_height(cell_px * 0.8 * self.text_scale, false);
        if let Some(s) = self.shown.as_ref().filter(|s| s.space == self.space) {
            reserved.push(render::draw_ramp_key(frame, &s.title, &self.ramp, font));
        } else if self.show_labels {
            reserved.extend(render::draw_legend(frame, &self.legend(), font));
        }
        if self.show_labels {
            labels.extend(self.labels(vp, cell_px));
        }
        draw_labels(frame, labels, &reserved);
    }

    /// One-line description of the view.
    pub fn caption(&self) -> String {
        let s = self.current();
        let colour = self.colour.map_or("none".to_string(), |i| {
            self.data.labels[i].title().to_string()
        });
        let focus = self
            .focus
            .map(|f| format!(" · focus {}", self.levels()[f as usize]))
            .unwrap_or_default();
        let pick = match &self.pick {
            Some(Pick::One(f)) => format!(" · feature {f} ({}) · x clears", self.source.name()),
            Some(Pick::Markers(g)) => format!(" · {g} markers ({}) · x clears", self.source.name()),
            None => String::new(),
        };
        format!(
            "{} · {} · {} pts · colour {}{}{}",
            s.method,
            s.title(),
            s.points.order.len(),
            colour,
            focus,
            pick
        )
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

/// Render the whole scene once, labels included.
pub(crate) fn render_full(scene: &Scene, vp: Viewport, cell_px: f32) -> image::RgbaImage {
    let layers = scene.layers();
    let mut job = Job::new(vp, &layers);
    while !job.step(&layers) {}
    let mut frame = job.finish();
    scene.decorate(&mut frame, &vp, cell_px);
    frame.to_image()
}

/// Run `senna layout` when the run has no layout yet, or not the method
/// asked for: the view shows layouts, and would otherwise only refuse.
/// Outputs go beside the manifest; progress shows before the view opens.
fn ensure_layout(args: &ViewArgs) -> anyhow::Result<()> {
    let (m, _) = senna::run_manifest::RunManifest::load(std::path::Path::new(args.from.as_ref()))?;
    let has_any = !m.layout.methods.is_empty() || m.layout.cell_coords.is_some();
    let want = match args.method.as_deref() {
        Some(w) if !m.layout.methods.contains_key(w) => w,
        Some(_) => return Ok(()),
        None if has_any => return Ok(()),
        None => "umap",
    };
    let out = senna::run_manifest::derive_out_prefix(&args.from);
    eprintln!(
        "senna view: {} has no {want} layout yet; running `senna layout {want}` first",
        args.from
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args(["layout", want, "--from", &args.from, "--out", &out])
        .status()?;
    anyhow::ensure!(
        status.success(),
        "`senna layout {want} --from {}` failed; see its messages above",
        args.from
    );
    Ok(())
}

pub fn run_view(args: &ViewArgs) -> anyhow::Result<()> {
    ensure_layout(args)?;
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
                Labels::clusters([("c2", 1), ("c1", 0), ("c3", 1)].map(|(a, b)| (a.into(), b))),
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
        s.review = Some(relabel::Review {
            order: vec![1],
            at: 0,
            draft,
            candidates: Vec::new(),
            fits: Vec::new(),
            target: None,
            rows: Vec::new(),
            row: 0,
            overview: Vec::new(),
            merge: None,
            preview: None,
        });
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
    fn start_options_pick_the_space() {
        let s = scene();
        assert_eq!(pick_space(&s.data, Some("phate"), None), 2);
        assert_eq!(pick_space(&s.data, Some("umap"), Some("features")), 1);
        assert_eq!(pick_space(&s.data, Some("tsne"), None), 0);
    }
}
