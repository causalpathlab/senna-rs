//! `senna view`: look at a run's layouts in the terminal.
//!
//! Every layout method the run has (umap, phate, tsne) on each axis it has
//! (cells, features placed on cells, features on their own), coloured by any
//! grouping the run carries (annotation, clusters, topics, markers). Drawn by
//! a progressive rasterizer on a light page with labels set on the map.

mod color;
mod data;
mod render;
mod text;
mod tui;

use color::{Encoder, Rgb};
use data::{Axis, Dataset, NONE};
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

    #[arg(long, help = "Start on this layout method (umap, phate, tsne)")]
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
}

/// What is on screen: which space, which grouping, what is focused.
pub(crate) struct Scene {
    pub data: Dataset,
    pub space: usize,
    /// Index into `data.labels`, when colouring by a grouping.
    pub colour: Option<usize>,
    pub focus: Option<u32>,
    pub show_labels: bool,
    /// Group id per point, cached per (space, labels).
    groups: Option<(usize, usize, Vec<u32>)>,
    palette: Vec<Rgb>,
    boost: f32,
}

impl Scene {
    fn new(data: Dataset, args: &ViewArgs) -> Self {
        let space = pick_space(&data, args.method.as_deref(), args.space.as_deref());
        let mut scene = Self {
            data,
            space,
            colour: None,
            focus: None,
            show_labels: true,
            groups: None,
            palette: Vec::new(),
            boost: 1.0,
        };
        scene.colour = scene.default_colour(args.colour_by.as_deref());
        scene.refresh_groups();
        scene
    }

    pub fn current(&self) -> &data::Space {
        &self.data.spaces[self.space]
    }

    /// Groupings that apply to the current axis.
    pub fn colour_choices(&self) -> Vec<usize> {
        let axis = self.current().axis;
        (0..self.data.labels.len())
            .filter(|&i| self.data.labels[i].axis == axis)
            .collect()
    }

    fn default_colour(&self, want: Option<&str>) -> Option<usize> {
        let choices = self.colour_choices();
        if let Some(w) = want {
            if let Some(&i) = choices
                .iter()
                .find(|&&i| self.data.labels[i].title.eq_ignore_ascii_case(w))
            {
                return Some(i);
            }
            log::warn!("view: no `{w}` grouping for {}", self.current().title);
        }
        choices.first().copied()
    }

    fn refresh_groups(&mut self) {
        let Some(li) = self.colour else {
            self.groups = None;
            self.palette.clear();
            return;
        };
        if matches!(&self.groups, Some((s, l, _)) if *s == self.space && *l == li) {
            return;
        }
        let labels = &self.data.labels[li];
        let g = labels.align(&self.current().points);
        let n = labels.levels.len();
        self.palette = (0..n).map(|i| color::category(i, n)).collect();
        let labelled = g.iter().filter(|&&v| v != NONE).count();
        self.boost = render::sparsity_boost(labelled as f32 / g.len().max(1) as f32);
        self.groups = Some((self.space, li, g));
    }

    pub fn groups(&self) -> Option<&[u32]> {
        self.groups.as_ref().map(|(_, _, g)| g.as_slice())
    }

    pub fn levels(&self) -> &[Box<str>] {
        self.colour
            .map_or(&[], |i| self.data.labels[i].levels.as_slice())
    }

    pub fn set_space(&mut self, space: usize) {
        let axis = self.current().axis;
        self.space = space;
        if self.current().axis != axis {
            self.colour = self.default_colour(None);
            self.focus = None;
        }
        self.refresh_groups();
    }

    pub fn cycle_colour(&mut self) {
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

    pub fn step_focus(&mut self, delta: i64) {
        let n = self.levels().len() as i64;
        if n == 0 {
            return;
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
        if let Some(b) = space.backdrop {
            layers.push(Paint {
                points: &self.data.spaces[b].points,
                groups: None,
                palette: &[],
                focus: None,
                muted: true,
                size: 0.8,
                boost: 1.0,
            });
        }
        layers.push(Paint {
            points: &space.points,
            groups: self.groups(),
            palette: &self.palette,
            focus: self.focus,
            muted: false,
            size: if space.backdrop.is_some() { 1.6 } else { 1.0 },
            boost: self.boost,
        });
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

        if space.axis == Axis::Cells {
            let Some(groups) = groups else { return out };
            let font = Font::for_cell_height(cell_px, true);
            let n = self.levels().len();
            for (g, m) in group_medians(&space.points, groups, n)
                .into_iter()
                .enumerate()
            {
                let Some((xy, size)) = m else { continue };
                if self.focus.is_some_and(|f| f as usize != g) {
                    continue;
                }
                let (x, y) = vp.to_px(xy);
                out.push(Label {
                    text: self.levels()[g].to_string(),
                    x,
                    y,
                    ink: color::category_ink(g),
                    priority: size as f32,
                    font,
                });
            }
            return out;
        }

        const MAX_NAMED: usize = 400;
        const NAME_ALL_BELOW: usize = 120;
        let font = Font::for_cell_height(cell_px * 0.8, false);
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
            let ink = if g == NONE {
                color::linear_rgb(color::INK)
            } else {
                color::category_ink(g as usize)
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
        let (Some(groups), Axis::Features) = (self.groups(), self.current().axis) else {
            return Vec::new();
        };
        let mut count = vec![0usize; self.levels().len()];
        for &g in groups {
            if g != NONE {
                count[g as usize] += 1;
            }
        }
        let mut ids: Vec<usize> = (0..count.len())
            .filter(|&g| count[g] > 0 && self.focus.is_none_or(|f| f as usize == g))
            .collect();
        ids.sort_by_key(|&g| std::cmp::Reverse(count[g]));
        ids.truncate(MAX_ENTRIES);
        ids.into_iter()
            .map(|g| {
                (
                    self.levels()[g].to_string(),
                    self.palette[g],
                    color::category_ink(g),
                )
            })
            .collect()
    }

    /// Labels and legend over a composited frame.
    pub fn decorate(&self, frame: &mut render::Frame, vp: &Viewport, cell_px: f32) {
        if !self.show_labels {
            return;
        }
        let font = Font::for_cell_height(cell_px * 0.8, false);
        let legend = render::draw_legend(frame, &self.legend(), font);
        draw_labels(frame, self.labels(vp, cell_px), legend.as_slice());
    }

    /// One-line description of the view.
    pub fn caption(&self) -> String {
        let s = self.current();
        let colour = self
            .colour
            .map_or("none".to_string(), |i| self.data.labels[i].title.clone());
        let focus = self
            .focus
            .map(|f| format!(" · focus {}", self.levels()[f as usize]))
            .unwrap_or_default();
        format!(
            "{} · {} · {} pts · colour {}{}",
            s.method,
            s.title,
            s.points.order.len(),
            colour,
            focus
        )
    }
}

fn pick_space(data: &Dataset, method: Option<&str>, space: Option<&str>) -> usize {
    let title = space.map(|s| s.replace('-', " "));
    data.spaces
        .iter()
        .position(|s| {
            method.is_none_or(|m| s.method.eq_ignore_ascii_case(m))
                && title
                    .as_deref()
                    .is_none_or(|t| s.title.eq_ignore_ascii_case(t))
        })
        .unwrap_or(0)
}

/// Render the whole scene once, labels included.
pub(crate) fn render_full(scene: &Scene, vp: Viewport, cell_px: f32) -> image::RgbaImage {
    let layers = scene.layers();
    let mut job = Job::new(vp, &layers);
    while !job.step(&layers) {}
    let mut frame = job.composite();
    scene.decorate(&mut frame, &vp, cell_px);
    frame.to_image(&Encoder::new())
}

pub fn run_view(args: &ViewArgs) -> anyhow::Result<()> {
    let data = Dataset::load(&args.from)?;
    let scene = Scene::new(data, args);

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

    tui::run(scene, args.graphics)
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

    fn space(method: &str, title: &'static str, axis: Axis, names: &[&str]) -> Space {
        Space {
            method: method.into(),
            title,
            axis,
            points: pts(names),
            backdrop: None,
        }
    }

    fn scene() -> Scene {
        let cells = ["c1", "c2", "c3", "c4"];
        let genes = ["g1", "g2", "g3"];
        let data = Dataset {
            prefix: "r".into(),
            spaces: vec![
                space("umap", "cells", Axis::Cells, &cells),
                space("umap", "features", Axis::Features, &genes),
                space("phate", "cells", Axis::Cells, &cells),
            ],
            labels: vec![
                Labels::from_pairs(
                    "cluster",
                    Axis::Cells,
                    [("c2", "C1"), ("c1", "C0"), ("c3", "C1")].map(|(a, b)| (a.into(), b.into())),
                    &[],
                ),
                Labels::from_pairs(
                    "annotation",
                    Axis::Cells,
                    [("c1", "CT1"), ("c4", "unassigned")].map(|(a, b)| (a.into(), b.into())),
                    &["unassigned"],
                ),
                Labels::from_pairs(
                    "markers",
                    Axis::Features,
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
        };
        Scene::new(data, &args)
    }

    #[test]
    fn labels_join_points_by_name_and_skip_abstentions() {
        let s = scene();
        assert_eq!(s.data.labels[0].title, "cluster");
        let g = s.groups().unwrap();
        assert_eq!(s.levels(), &["C1".into(), "C0".into()] as &[Box<str>]);
        assert_eq!(g, &[1, 0, 0, NONE]);
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
    fn start_options_pick_the_space() {
        let s = scene();
        assert_eq!(pick_space(&s.data, Some("phate"), None), 2);
        assert_eq!(pick_space(&s.data, Some("umap"), Some("features")), 1);
        assert_eq!(pick_space(&s.data, Some("tsne"), None), 0);
    }
}
