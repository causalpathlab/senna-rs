//! Charts drawn in place of the map, for the cells in view and the grouping
//! on screen (`c` changes it, as on the map), with legume-tui's charts:
//!
//! - **structure**: each cell's topic mixture as a thin stacked bar, cells
//!   side by side in one panel per group — the admixture plot of population
//!   genetics, as `lupin plot-topic` draws it. For runs with topics: topic
//!   runs, and `bge` runs (topics resolved from the cell embedding). Topics keep
//!   one colour each and one order everywhere (most prevalent at the bottom
//!   of a bar); inside a panel, cells go by their dominant topic, then by
//!   how dominant it is.
//! - **heatmap**: the top features of each group × the groups. A group's
//!   features are those whose level there most exceeds the next-highest
//!   group's, so each peaks in its own group and the rows fall into a
//!   diagonal; with one group in view, its highest levels. The model
//!   proposes candidates, the values shown choose. Each cell of the heatmap is the
//!   group's mean `ln(1 + count)` from the data files (the model's expected
//!   level when they cannot be read), z-scored per feature across the groups
//!   and clipped to `±CLIP`.

use super::activity::Source;
use super::color;
use super::render::Frame;
use super::Axis;
use super::Scene;
use legume_tui::chart::heatmap::{self, ByFn, Heat, HeatStyle, Rerank};
use legume_tui::chart::structure::{self, BarStyle, Figure, Grouping, Layout, Names, Weights};
use legume_tui::chart::{self, Step};
use legume_tui::map::text::Font;

pub use heatmap::CLIP;
/// Top features per group to begin with, and the range `+` / `-` allow.
pub const TOP_START: usize = 10;
const TOP_RANGE: (usize, usize) = (1, 100);

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Kind {
    Structure,
    Heatmap,
}

impl Kind {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Kind::Structure => "structure plot",
            Kind::Heatmap => "heatmap",
        }
    }
}

/// What a chart was made from: the space, the grouping, and the features
/// per group; made again when any changes.
type ChartKey = (usize, Option<usize>, usize);

pub struct Chart {
    pub kind: Kind,
    /// Top features per group, for the heatmap.
    pub top: usize,
    made: Option<(ChartKey, Result<Data, String>)>,
}

impl Chart {
    /// The same chart for another view of the run, made afresh there.
    #[must_use]
    pub fn like(&self) -> Self {
        Self {
            top: self.top,
            ..Self::new(self.kind)
        }
    }

    fn new(kind: Kind) -> Self {
        Self {
            kind,
            top: TOP_START,
            made: None,
        }
    }
}

enum Data {
    Structure(Structure),
    Heatmap(Heatmap),
}

/// Cells' topic mixtures, grouped into panels.
pub struct Structure {
    grouping: String,
    /// Each point's mixture, `k` values a point (`NaN`: none).
    theta: Vec<f32>,
    k: usize,
    /// The panels, and the topics from the bottom of a bar up.
    layout: Layout,
    /// Topic names by id.
    key: Vec<String>,
}

/// Top features × groups, z-scored per feature.
pub struct Heatmap {
    grouping: String,
    features: Vec<String>,
    groups: Vec<String>,
    /// Feature-major, `features.len() × groups.len()`; `NaN`: no value.
    z: Vec<f32>,
    source: Source,
    top: usize,
}

impl Scene {
    /// Map → structure plot (runs with topics) → heatmap → map.
    pub fn cycle_chart(&mut self) {
        if self.current().axis() != Axis::Cells {
            self.note = Some("charts are drawn from a cell view (tab)".into());
            return;
        }
        let topic = self.is_topic_run();
        let next = match self.chart.as_ref().map(|c| c.kind) {
            None if topic => Some(Kind::Structure),
            None | Some(Kind::Structure) => Some(Kind::Heatmap),
            Some(Kind::Heatmap) => None,
        };
        let top = self.chart.as_ref().map_or(TOP_START, |c| c.top);
        self.chart = next.map(|kind| Chart {
            top,
            ..Chart::new(kind)
        });
        self.note = Some(match next {
            Some(k) => format!("{} · H next · c groups by another grouping", k.name()),
            None => "back to the map".into(),
        });
    }

    /// Start on chart `kind` (`--chart`), made at once.
    pub fn start_chart(&mut self, kind: Kind) {
        self.chart = Some(Chart::new(kind));
        self.refresh_chart();
    }

    /// Whether the run records a latent (topics, or its own cell factors).
    #[must_use]
    pub fn run_has_latent(&self) -> bool {
        self.data
            .run
            .as_ref()
            .is_some_and(|(m, _)| m.outputs.latent.is_some())
    }

    /// Whether the run has topic mixtures: a topic run, or a `bge` run
    /// with its topics resolved.
    fn is_topic_run(&mut self) -> bool {
        self.activity()
            .is_some_and(super::activity::Activity::has_mixtures)
    }

    /// `+` / `-` on a heatmap: more or fewer features per group.
    pub fn change_top(&mut self, more: bool) {
        let Some(c) = self.chart.as_mut().filter(|c| c.kind == Kind::Heatmap) else {
            return;
        };
        c.top = chart::step_top(c.top, more, Step::Quarter, TOP_RANGE);
        self.note = Some(format!("top {} features per group", c.top));
    }

    /// Make the chart again if the space, grouping or feature count changed.
    pub fn refresh_chart(&mut self) {
        let Some(c) = self.chart.as_ref() else { return };
        let key = (self.space, self.colour, c.top);
        if c.made.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let (kind, top) = (c.kind, c.top);
        let data = match kind {
            _ if self.current().axis() != Axis::Cells => {
                Err("charts are drawn from a cell view (tab back to the cells)".into())
            }
            Kind::Structure => self.structure().map(Data::Structure),
            Kind::Heatmap => self.heatmap(top).map(Data::Heatmap),
        };
        // Not read yet: drawn again once it is, not kept as a failure.
        if crate::view::features::is_loading(&data) {
            return;
        }
        if let Some(c) = self.chart.as_mut() {
            c.made = Some((key, data));
        }
    }

    fn grouping_name(&self) -> Option<String> {
        self.colour
            .map(|i| self.data.labels[i].kind.title().to_string())
    }

    fn structure(&mut self) -> Result<Structure, String> {
        let grouping = self
            .grouping_name()
            .ok_or("colour the cells by a grouping first (c)")?;
        let groups = self.groups().ok_or("no grouping on screen")?.to_vec();
        let levels = self.levels().to_vec();
        let space = self.space;
        let (activity, data) = self
            .activity_and_data()
            .ok_or("no manifest to read the model from")?;
        let names = &data.spaces[space].points.names;
        let (theta, k) = activity.mixtures(space, names)?;
        Ok(build_structure(grouping, theta, k, &groups, &levels))
    }

    fn heatmap(&mut self, top: usize) -> Result<Heatmap, String> {
        let grouping = self
            .grouping_name()
            .ok_or("colour the cells by a grouping first (c)")?;
        let groups = self.groups().ok_or("no grouping on screen")?.to_vec();
        let levels = self.levels().to_vec();
        let n = levels.len();
        let space = self.space;
        let (activity, data) = self
            .activity_and_data()
            .ok_or("no manifest to read the model from")?;
        let names = &data.spaces[space].points.names;
        let sums = activity.group_sums(space, names, &groups, n)?;
        // Only groups with cells in view get a column.
        let mut size = vec![0usize; n];
        for &g in &groups {
            if let Some(s) = size.get_mut(g as usize) {
                *s += 1;
            }
        }
        let cols: Vec<usize> = (0..n).filter(|&g| size[g] > 0).collect();
        let (levels_fg, all) = activity.group_levels(&sums)?;
        let all = all.to_vec();
        let mut source = Source::Expected;
        let (picked, mut z) = pick(&levels_fg, n, &cols, top, |pool| {
            let pool: Vec<Box<str>> = pool.iter().map(|&f| all[f].clone()).collect();
            let (means, from) = activity.group_means(space, names, &groups, n, &pool)?;
            source = from;
            Ok(means)
        })?;
        heatmap::zscore_rows(&mut z, cols.len(), CLIP);
        Ok(Heatmap {
            grouping,
            features: picked.iter().map(|&f| all[f].to_string()).collect(),
            groups: cols.iter().map(|&g| levels[g].to_string()).collect(),
            z,
            source,
            top,
        })
    }

    /// The chart on screen, drawn `w × h` with text sized for `cell_px`;
    /// `None` on the map. With `keep_text`, text is kept for a PDF.
    #[must_use]
    pub fn chart_frame(&self, w: usize, h: usize, cell_px: f32, keep_text: bool) -> Option<Frame> {
        let c = self.chart.as_ref()?;
        let mut frame = Frame::blank(w, h, color::BACKGROUND);
        if keep_text {
            frame.keep_text();
        }
        let px = self.text_px(cell_px);
        let font = Font::for_cell_height(px, false);
        let bold = Font::for_cell_height(px, true);
        match c.made.as_ref().map(|(_, d)| d) {
            Some(Ok(d)) => draw(&mut frame, d, font, bold),
            Some(Err(e)) => chart::note(
                &mut frame,
                &format!("no {}: {e}", c.kind.name()),
                bold,
                &color::THEME,
            ),
            None => chart::note(
                &mut frame,
                &format!("drawing the {}…", c.kind.name()),
                bold,
                &color::THEME,
            ),
        }
        Some(frame)
    }
}

/// Panels by group in the grouping's order (cells without a group last, as
/// "unassigned"), cells in each by dominant topic then its weight; topics
/// by prevalence. Cells with no mixture (`NaN`) are left out.
fn build_structure(
    grouping: String,
    theta: Vec<f32>,
    k: usize,
    groups: &[u32],
    levels: &[Box<str>],
) -> Structure {
    let names: Vec<String> = levels.iter().map(ToString::to_string).collect();
    let layout = Layout::build(
        &Weights {
            k,
            w: &theta,
            valid: None,
        },
        // `NONE` (no group) is past every level: unassigned.
        &Grouping {
            of: groups,
            names: &names,
            order: None,
        },
    );
    Structure {
        grouping,
        theta,
        k,
        layout,
        key: (0..k).map(|t| format!("T{t}")).collect(),
    }
}

/// The top `top` features of each group shown (`cols`, of `n` groups),
/// from the model's levels `first` (feature-major, `n` a feature): they
/// propose candidates, `read` gives the values shown for those (`n` a
/// feature), and those choose, so a row peaks in its group in what is
/// drawn and a feature the data files lack is never picked. The chosen
/// features (by index into `first`) and their values in `cols`.
fn pick(
    first: &[f32],
    n: usize,
    cols: &[usize],
    top: usize,
    read: impl FnOnce(&[usize]) -> Result<Vec<f32>, String>,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let model = shown(first, n, cols);
    let pool = heatmap::candidates(&model, top);
    let means = read(&pool)?;
    let chosen = heatmap::top_by_margin(
        &model,
        top,
        Some(Rerank {
            rows: &pool,
            m: &shown(&means, n, cols),
        }),
    );
    let at: std::collections::HashMap<usize, usize> =
        pool.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let features: Vec<usize> = chosen.iter().map(|&(f, _)| f).collect();
    let picked = features
        .iter()
        .flat_map(|f| {
            let row = &means[at[f] * n..(at[f] + 1) * n];
            cols.iter().map(move |&g| row[g])
        })
        .collect();
    Ok((features, picked))
}

/// Feature-major levels `v`, `n` a feature, as a matrix of the groups
/// shown (`cols`).
fn shown<'a>(v: &'a [f32], n: usize, cols: &'a [usize]) -> ByFn<impl Fn(usize, usize) -> f32 + 'a> {
    let rows = v.len().checked_div(n).unwrap_or(0);
    ByFn {
        rows,
        cols: cols.len(),
        f: move |r, c| v[r * n + cols[c]],
    }
}

/// A made chart into `frame`, in the page's colours.
fn draw(frame: &mut Frame, d: &Data, font: Font, bold: Font) {
    match d {
        Data::Structure(s) => {
            let palette: Vec<color::Rgb> = (0..s.k).map(|t| color::category(t, s.k)).collect();
            let title = format!(
                "topic mixture per cell · one panel per {} · {} topics",
                s.grouping, s.k
            );
            structure::figure(
                frame,
                &s.layout,
                &Weights {
                    k: s.k,
                    w: &s.theta,
                    valid: None,
                },
                &BarStyle {
                    palette: &palette,
                    focus: None,
                    gap: structure::GAP,
                    hits: false,
                },
                &Figure {
                    title: Some(&title),
                    key: Some(&s.key),
                    names: Names::Auto,
                    font,
                    bold,
                    theme: &color::THEME,
                },
            );
        }
        Data::Heatmap(hm) => {
            let what = match hm.source {
                Source::Observed | Source::Extra => "mean ln(1 + count)",
                Source::Expected => "model's expected level (no count files)",
            };
            let title = format!(
                "top {} features per {} · {what}, z-scored per feature, clipped ±{CLIP}",
                hm.top, hm.grouping
            );
            // A heatmap that cannot be drawn says why in its place.
            let _ = heatmap::draw_heat(
                frame,
                &Heat {
                    title: &title,
                    rows: &hm.features,
                    cols: &hm.groups,
                    col_swatch: None,
                    z: &hm.z,
                    clip: CLIP,
                    bold_row: None,
                },
                font,
                bold,
                &color::THEME,
                &HeatStyle::for_theme(&color::THEME),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::data::NONE;

    fn levels(names: &[&str]) -> Vec<Box<str>> {
        names.iter().map(|&n| n.into()).collect()
    }

    fn rows(s: &Structure) -> Vec<(&str, Vec<u32>)> {
        s.layout
            .panels
            .iter()
            .map(|p| (p.name.as_str(), p.rows.clone()))
            .collect()
    }

    #[test]
    fn structure_panels_follow_groups_and_cells_their_dominant_topic() {
        // Three topics; topic 2 most prevalent, then 0, then 1.
        let theta = [
            0.1,
            0.1,
            0.8, // p0 group 0, dominant 2
            0.6,
            0.1,
            0.3, // p1 group 0, dominant 0
            0.2,
            0.0,
            0.8, // p2 group 1
            0.0,
            0.0,
            0.9, // p3 no group
            f32::NAN,
            f32::NAN,
            f32::NAN, // p4 no mixture
            0.3,
            0.1,
            0.6, // p5 group 0, dominant 2, weaker than p0
        ];
        let groups = [0, 0, 1, NONE, 0, 0];
        let s = build_structure(
            "cluster".into(),
            theta.to_vec(),
            3,
            &groups,
            &levels(&["A", "B"]),
        );
        // Most prevalent at the bottom of a bar.
        assert_eq!(s.layout.stack, [2, 0, 1]);
        // A: p0 (2 at 0.8), p5 (2 at 0.6), then p1 (dominant 0).
        assert_eq!(
            rows(&s),
            [
                ("A", vec![0, 5, 1]),
                ("B", vec![2]),
                ("unassigned", vec![3])
            ]
        );
        assert_eq!(s.key, ["T0", "T1", "T2"]);
    }

    fn structure_fixture() -> Structure {
        let k = 4;
        let theta: Vec<f32> = (0..60 * k)
            .map(|j| ((j * 37 % 11) as f32 + 1.0) / (12.0 * k as f32))
            .collect();
        let groups: Vec<u32> = (0..60)
            .map(|p| if p % 9 == 0 { NONE } else { p % 3 })
            .collect();
        build_structure(
            "cluster".into(),
            theta,
            k,
            &groups,
            &levels(&["CT1", "CT2", "CT3"]),
        )
    }

    fn heatmap_fixture() -> Heatmap {
        Heatmap {
            grouping: "cluster".into(),
            features: (1..=6).map(|i| format!("GENE{i}")).collect(),
            groups: ["CT1", "CT2", "CT3"].map(String::from).to_vec(),
            z: (0..18)
                .map(|j| {
                    if j == 7 {
                        f32::NAN
                    } else {
                        (j as f32 - 9.0) / 9.0
                    }
                })
                .collect(),
            source: Source::Expected,
            top: 2,
        }
    }

    /// A structure plot (bars stacked from the bottom), a heatmap and a
    /// note, drawn to the byte, pixels and kept text.
    #[test]
    fn charts_draw_as_they_did() {
        use crate::view::snapshot::{digest, runs_digest};
        let (font, bold) = (
            Font::for_cell_height(16.0, false),
            Font::for_cell_height(16.0, true),
        );
        let (s, hm) = (
            Data::Structure(structure_fixture()),
            Data::Heatmap(heatmap_fixture()),
        );
        let drawn = |f: &dyn Fn(&mut Frame), keep: bool| {
            let mut frame = Frame::blank(320, 240, color::BACKGROUND);
            if keep {
                frame.keep_text();
            }
            f(&mut frame);
            let runs = frame.take_text();
            (digest(&frame.to_image()), runs.len(), runs_digest(&runs))
        };
        let structure = |f: &mut Frame| draw(f, &s, font, bold);
        let heatmap = |f: &mut Frame| draw(f, &hm, font, bold);
        let said = |f: &mut Frame| chart::note(f, "drawing the heatmap…", bold, &color::THEME);
        assert_eq!(
            drawn(&structure, false),
            ((320, 240, 7610302172151140221), 0, 14695981039346656037)
        );
        assert_eq!(
            drawn(&structure, true),
            ((320, 240, 12213416596642415465), 9, 11116718457719180847)
        );
        assert_eq!(
            drawn(&heatmap, false),
            ((320, 240, 15222720487810110223), 0, 14695981039346656037)
        );
        assert_eq!(
            drawn(&heatmap, true),
            ((320, 240, 17814653105649234099), 14, 11586368578154238834)
        );
        assert_eq!(
            drawn(&said, false),
            ((320, 240, 12649403086188204864), 0, 14695981039346656037)
        );
    }

    /// The most prevalent topic sits at the bottom of every bar.
    #[test]
    fn bars_stack_the_most_prevalent_topic_at_the_bottom() {
        // Two cells of one group, topic 1 the more prevalent.
        let theta = vec![0.25, 0.75, 0.25, 0.75];
        let s = build_structure("cluster".into(), theta, 2, &[0, 0], &levels(&["A"]));
        assert_eq!(s.layout.stack, [1, 0]);
        let (font, bold) = (
            Font::for_cell_height(16.0, false),
            Font::for_cell_height(16.0, true),
        );
        let mut frame = Frame::blank(320, 240, color::BACKGROUND);
        draw(&mut frame, &Data::Structure(s), font, bold);
        let img = frame.to_image();
        let at = |y: u32| img.get_pixel(40, y).0;
        let srgb = |c: color::Rgb| {
            let [r, g, b] = color::to_srgb_senna(c);
            [r, g, b, 255]
        };
        // Bars run from 2 lines down to above the names; probe near each end.
        let lh = font.line_height() as u32;
        assert_eq!(at(2 * lh + 2), srgb(color::category(0, 2)));
        assert_eq!(at(240 - 2 * lh - 2), srgb(color::category(1, 2)));
    }

    /// Each group's top features by their margin over the next group; with
    /// one group shown, its top levels.
    #[test]
    fn each_group_takes_the_features_it_leads() {
        // Four features × two groups.
        let first = [3.0, 0.0, 2.0, 1.5, 1.0, 3.5, f32::NAN, 4.0];
        // The values shown for the candidates, here the model's own.
        let model = |pool: &[usize]| {
            Ok::<_, String>(
                pool.iter()
                    .flat_map(|&f| first[f * 2..f * 2 + 2].to_vec())
                    .collect(),
            )
        };
        let (f, _) = pick(&first, 2, &[0, 1], 2, model).unwrap();
        // 0 leads group 0 by 3, 1 by 0.5; 2 leads group 1 by 2.5; 3 has
        // one level and peaks nowhere.
        assert_eq!(f, [0, 1, 2]);
        // The counts choose among the model's candidates.
        let counts = |pool: &[usize]| {
            assert_eq!(pool, [0, 1, 2]);
            Ok(vec![1.0, 0.0, 0.0, 5.0, 0.0, 1.0])
        };
        let (f, means) = pick(&first, 2, &[0, 1], 1, counts).unwrap();
        assert_eq!(f, [0, 1]);
        assert_eq!(means, [1.0, 0.0, 0.0, 5.0]);
        // One group in view: its top levels, no runner-up needed.
        let (f, means) = pick(&first, 2, &[1], 2, model).unwrap();
        assert_eq!(f, [3, 2]);
        assert_eq!(means, [4.0, 3.5]);
        // A topic run's levels are logs of shares, all below zero: one
        // group still ranks them, by the model and by what is read.
        let logs = [-0.7, -0.9, -1.2, -0.7, -1.6, -1.6];
        let own = |pool: &[usize]| {
            Ok::<_, String>(
                pool.iter()
                    .flat_map(|&f| logs[f * 2..f * 2 + 2].to_vec())
                    .collect(),
            )
        };
        let (f, means) = pick(&logs, 2, &[1], 2, own).unwrap();
        assert_eq!(f, [1, 0]);
        assert_eq!(means, [-0.7, -0.9]);
        let counts = |pool: &[usize]| {
            assert_eq!(pool, [1, 0, 2]);
            Ok(vec![0.0, 0.5, 0.0, 0.25, 0.0, 2.0])
        };
        let (f, means) = pick(&logs, 2, &[1], 2, counts).unwrap();
        assert_eq!(f, [2, 1]);
        assert_eq!(means, [2.0, 0.5]);
        // What cannot be read is said.
        let fails = |_: &[usize]| Err::<Vec<f32>, String>("unreadable".into());
        assert_eq!(
            pick(&first, 2, &[0, 1], 2, fails).unwrap_err(),
            "unreadable"
        );
    }

    #[test]
    fn rows_are_z_scored_and_clipped() {
        let mut v = [1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 0.0, f32::NAN, 10.0];
        heatmap::zscore_rows(&mut v, 3, 1.0);
        let r = 1.0 / (2.0f32 / 3.0).sqrt();
        assert!((v[0] + r.min(1.0)).abs() < 1e-5 && v[1].abs() < 1e-6);
        assert_eq!(&v[3..6], [0.0, 0.0, 0.0]);
        assert!(v[7].is_nan() && v[6] == -1.0 && v[8] == 1.0);
    }

    /// A chart saved as a PDF page keeps its title, names and key as text.
    #[test]
    fn a_chart_pdf_keeps_its_text() {
        let mut scene = crate::view::tests::scene();
        for data in [
            Data::Structure(structure_fixture()),
            Data::Heatmap(heatmap_fixture()),
        ] {
            let kind = match data {
                Data::Structure(_) => Kind::Structure,
                Data::Heatmap(_) => Kind::Heatmap,
            };
            let mut c = Chart::new(kind);
            c.made = Some(((0, Some(0), TOP_START), Ok(data)));
            scene.chart = Some(c);
            let vp = super::super::render::fit([0.0, 0.0, 1.0, 1.0], 320, 240);
            let pages = super::super::deck::view_pages(&[(&scene, vp, 16.0)]);
            let pdf = legume_tui::export::pdf_bytes(&pages, 144.0).unwrap();
            // A run is written as a literal string, or in hex when it
            // has many bytes past ASCII.
            let has = |s: &str| {
                let hex: String = s.bytes().map(|b| format!("{b:02X}")).collect();
                [s.to_string(), hex.clone(), hex.to_lowercase()]
                    .iter()
                    .any(|s| pdf.windows(s.len()).any(|w| w == s.as_bytes()))
            };
            let words: &[&str] = match kind {
                Kind::Structure => &["topic mixture per cell", "CT2", "unassigned", "T3"],
                Kind::Heatmap => &["top 2 features per cluster", "GENE6", "CT3"],
            };
            for w in words {
                assert!(has(w), "{} lacks {w}", kind.name());
            }
        }
    }
}
