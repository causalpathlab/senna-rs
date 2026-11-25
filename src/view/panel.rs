//! A view as a figure panel: legume-tui's [`Panel`] (size in inches,
//! points and labels in points, a white page or none, group names on the
//! map, in a key or alone) over the scene's map. `senna view --pdf/--png`
//! with a panel flag and the `view_export` tool with a panel argument both
//! save through [`save`].

use super::color;
use super::data::NONE;
use super::render::{Frame, Job, Label, Paint, Viewport};
use super::Scene;
use anyhow::{anyhow, bail, ensure, Context, Result};
use legume_tui::export::{KeyEntry, Labels, Panel};
use legume_tui::map::labels::{place, Rect};
use legume_tui::map::offsets::Offsets;
use legume_tui::map::points::{auto_radius, visible_count, Mark, PointLayer};
use legume_tui::map::viewport::YAxis;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Where a panel is saved: any of a PNG, a PDF, and the key alone.
#[derive(Default)]
pub(crate) struct Files<'a> {
    pub png: Option<&'a Path>,
    pub pdf: Option<&'a Path>,
    pub legend: Option<&'a Path>,
}

/// What [`save`] wrote: the files, the panel's pixels, and a note when the
/// key did not hold every group.
pub(crate) struct Saved {
    pub wrote: Vec<PathBuf>,
    pub px: (usize, usize),
    pub note: Option<String>,
}

/// Save the scene's map as `panel` to `files`, `window` (data units) or
/// its whole map fitted to the panel.
///
/// # Errors
/// A chart on screen, a panel that cannot be drawn, a hidden group the
/// grouping does not have, or a file that cannot be written.
pub(crate) fn save(
    scene: &Scene,
    panel: &Panel,
    window: Option<[f32; 4]>,
    files: &Files<'_>,
) -> Result<Saved> {
    ensure!(
        scene.chart.is_none(),
        "a panel is a map: leave the chart first"
    );
    let px = panel.check().map_err(|e| anyhow!(e))?;
    let hidden = hidden(scene, panel)?;
    let entries = entries(scene, &hidden);
    // A PNG has its text in the pixels, a PDF as text; the legend is
    // either, as its extension says.
    let write = |path: &Path, frame: Frame, pdf: bool| -> Result<()> {
        if pdf {
            panel.write_pdf(&[panel.page(frame)], path)
        } else {
            panel
                .png(&frame)
                .and_then(|bytes| std::fs::write(path, bytes))
        }
        .with_context(|| format!("writing {}", path.display()))
    };
    let mut wrote = Vec::new();
    let mut note = None;
    for (path, pdf) in [(files.png, false), (files.pdf, true)] {
        let Some(path) = path else { continue };
        let (frame, n) = draw(scene, panel, window, &hidden, &entries, pdf);
        note = n;
        write(path, frame, pdf)?;
        wrote.push(path.to_path_buf());
    }
    if let Some(path) = files.legend {
        let pdf = is_pdf(path);
        let frame = panel
            .key_frame(&entries, color::BACKGROUND, pdf)
            .map_err(|e| anyhow!(e))?;
        write(path, frame, pdf)?;
        wrote.push(path.to_path_buf());
    }
    Ok(Saved { wrote, px, note })
}

/// Whether `path` names a PDF; anything else is written as a PNG.
fn is_pdf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
}

/// Which groups of the grouping shown the panel leaves out: those hidden
/// in the style menu and those it names, by name or cluster id (`10`,
/// `C10`, `K10`) as `view_map`'s focus takes them.
fn hidden(scene: &Scene, panel: &Panel) -> Result<Vec<bool>> {
    let mut hidden: Vec<bool> = scene.styles.iter().map(|s| s.hidden).collect();
    hidden.resize(scene.levels().len(), false);
    if panel.hidden.is_empty() {
        return Ok(hidden);
    }
    let Some(labels) = scene.colour.map(|i| &scene.data.labels[i]) else {
        bail!("no grouping shown to hide {} from", panel.hidden[0]);
    };
    for name in &panel.hidden {
        let Some(g) = labels.group(name) else {
            let some: Vec<&str> = labels.levels.iter().take(12).map(AsRef::as_ref).collect();
            bail!("no group {name} to hide; groups: {}", some.join(", "));
        };
        hidden[g] = true;
    }
    Ok(hidden)
}

/// The key's lines: every group with points, not hidden and in focus, in
/// the grouping's order, a dot in its colour and its name in the text's
/// near-black.
fn entries(scene: &Scene, hidden: &[bool]) -> Vec<KeyEntry> {
    if scene.current_shown().is_some() {
        return Vec::new();
    }
    let Some(sizes) = scene.group_sizes() else {
        return Vec::new();
    };
    let ink = color::linear_rgb(color::TEXT);
    scene
        .levels()
        .iter()
        .enumerate()
        .filter(|&(g, _)| {
            sizes.get(g).is_some_and(|&n| n > 0)
                && !hidden[g]
                && scene.focus.is_none_or(|f| f as usize == g)
        })
        .map(|(g, l)| (l.to_string(), scene.styles[g].colour, ink))
        .collect()
}

/// The map of the panel: its points, and its groups named on it or in a
/// key beside it. With `keep_text`, text is kept for a PDF. A note when
/// the key holds fewer than every group.
fn draw(
    scene: &Scene,
    panel: &Panel,
    window: Option<[f32; 4]>,
    hidden: &[bool],
    entries: &[KeyEntry],
    keep_text: bool,
) -> (Frame, Option<String>) {
    let px = panel.check().expect("checked by save");
    let areas = panel.areas(entries, px);
    let space = scene.current();
    let groups = scene.groups();
    let kept = |i: usize| groups.is_none_or(|g| g[i] == NONE || !hidden[g[i] as usize]);
    let bounds = window.unwrap_or_else(|| kept_bounds(&space.points.xy, kept, space.points.bounds));
    let [x0, y0, x1, y1] = areas.map;
    let n = (0..space.points.xy.len()).filter(|&i| kept(i)).count();
    let radius = panel.point_radius_px(n, (x1 - x0, y1 - y0));
    let vp = panel.viewport(bounds, areas.map, radius, px, YAxis::Up);

    let layers = scene.layers();
    // The map's own layer is the last; a backdrop keeps its size beside it.
    let main = layers.last().map_or(1.0, |l| l.size);
    let looks: Vec<Look<'_>> = layers
        .into_iter()
        .map(|paint| Look {
            radius: radius * paint.size / main,
            paint,
            hidden,
            alpha: panel.point_alpha,
            vp,
        })
        .collect();
    let mut job = Job::new(vp, panel.page_colour(color::BACKGROUND), &looks);
    if panel.is_transparent() {
        job = job.transparent();
    }
    while !job.step(&looks) {}
    let mut frame = job.finish();
    if keep_text {
        frame.keep_text();
    }

    let (regular, bold) = panel.fonts();
    let mut note = None;
    match (panel.labels, areas.key) {
        (Labels::Legend, Some(key)) => {
            panel.draw_key(&mut frame, &entries[..areas.shown], key);
            if areas.shown < entries.len() {
                note = Some(format!(
                    "the legend holds {} of {} groups: save it alone with legend",
                    areas.shown,
                    entries.len()
                ));
            }
        }
        (Labels::Map, _) => {
            let labels: Vec<_> = scene
                .labels(&vp, 1.0)
                .into_iter()
                .filter(|l| l.key.is_none_or(|g| !hidden[g as usize]))
                .map(|mut l| {
                    l.font = if l.font.is_bold() { bold } else { regular };
                    // Inside the panel rather than dropped at its edge.
                    let (hw, hh) = (
                        0.5 * l.font.width(&l.text) as f32 + 4.0,
                        0.5 * l.font.line_height() as f32 + 2.0,
                    );
                    l.x = l.x.clamp(hw, (px.0 as f32 - hw).max(hw));
                    l.y = l.y.clamp(hh, (px.1 as f32 - hh).max(hh));
                    l
                })
                .collect();
            let offsets = spread(&labels, px, scene.group_offsets());
            place(&mut frame, labels, &[], &offsets);
        }
        _ => {}
    }
    (frame, note)
}

/// Room on a small panel for labels that would collide where they are:
/// `moved` (the labels moved by hand) and, for each label in the order
/// [`place`] takes them, the nearest shift, in line heights, that keeps it
/// clear of those before it and inside the `(w, h)` panel. A moved label is
/// never dropped and gets a leader to its group when it is far; a label
/// with no room left is left to `place` to drop.
fn spread(labels: &[Label], (w, h): (usize, usize), moved: Offsets<u32>) -> Offsets<u32> {
    let mut order: Vec<&Label> = labels.iter().collect();
    order.sort_by(|a, b| b.tier.cmp(&a.tier).then(b.priority.total_cmp(&a.priority)));
    let (w, h) = (w as f32, h as f32);
    // As `place` boxes a label centred at (cx, cy).
    let rect = |l: &Label, (cx, cy): (f32, f32)| -> Rect {
        let (tw, th) = (l.font.width(&l.text) as f32, l.font.line_height() as f32);
        let (x0, y0) = (cx - 0.5 * tw, cy - 0.5 * th);
        [x0 - 3.0, y0 - 1.0, x0 + tw + 3.0, y0 + th + 1.0]
    };
    let clear = |r: Rect, taken: &[Rect]| {
        r[0] >= 0.0
            && r[1] >= 0.0
            && r[2] <= w
            && r[3] <= h
            && !taken
                .iter()
                .any(|p| r[0] < p[2] && p[0] < r[2] && r[1] < p[3] && p[1] < r[3])
    };
    // Shifts in line heights, nearest first: up to 3 lines up or down and
    // up to a label's width either side.
    let shifts = |l: &Label| {
        let th = l.font.line_height() as f32;
        let half = 0.5 * l.font.width(&l.text) as f32 / th;
        let mut s: Vec<(f32, f32)> = (-4..=4)
            .flat_map(|i| (-3..=3).map(move |j| (i as f32 * half, j as f32)))
            .collect();
        s.sort_by(|a, b| a.0.hypot(a.1).total_cmp(&b.0.hypot(b.1)));
        s
    };
    let mut out = moved;
    let mut taken: Vec<Rect> = Vec::new();
    // Labels moved by hand go first, where they were put.
    for l in order
        .iter()
        .filter(|l| l.key.is_some_and(|k| out.get(&k).is_some()))
    {
        let (dx, dy) = out.get(&l.key.expect("keyed")).expect("moved");
        let th = l.font.line_height() as f32;
        taken.push(rect(l, (l.x + dx * th, l.y + dy * th)));
    }
    for l in order {
        let Some(k) = l.key.filter(|k| out.get(k).is_none()) else {
            if l.key.is_none() {
                taken.push(rect(l, (l.x, l.y)));
            }
            continue;
        };
        let th = l.font.line_height() as f32;
        let Some((dx, dy)) = shifts(l)
            .into_iter()
            .find(|&(dx, dy)| clear(rect(l, (l.x + dx * th, l.y + dy * th)), &taken))
        else {
            continue;
        };
        taken.push(rect(l, (l.x + dx * th, l.y + dy * th)));
        if (dx, dy) != (0.0, 0.0) {
            out.nudge(k, (dx, dy));
        }
    }
    out
}

/// The bounds of the points `kept`, or `all` when none is.
fn kept_bounds(xy: &[[f32; 2]], kept: impl Fn(usize) -> bool, all: [f32; 4]) -> [f32; 4] {
    let empty = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let b = (0..xy.len()).filter(|&i| kept(i)).fold(empty, |b, i| {
        let [x, y] = xy[i];
        [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]
    });
    if b[0] <= b[2] {
        b
    } else {
        all
    }
}

/// A layer as the panel draws it: its hidden groups left out, every mark
/// at the panel's opacity, and the whole layer at the panel's radius.
struct Look<'a> {
    paint: Paint<'a>,
    hidden: &'a [bool],
    alpha: Option<f32>,
    /// The radius the layer is drawn at, in pixels.
    radius: f32,
    vp: Viewport,
}

impl PointLayer for Look<'_> {
    fn xy(&self, i: usize) -> [f32; 2] {
        self.paint.xy(i)
    }

    fn shuffled(&self) -> &[u32] {
        self.paint.shuffled()
    }

    fn order(&self) -> Arc<Vec<u32>> {
        self.paint.order()
    }

    fn mark(&self, i: usize) -> Option<Mark> {
        let g = self.paint.groups.map_or(NONE, |g| g[i]);
        if g != NONE && self.hidden.get(g as usize).copied().unwrap_or(false) {
            return None;
        }
        let m = self.paint.mark(i)?;
        Some(Mark {
            alpha: self.alpha.unwrap_or(m.alpha),
            ..m
        })
    }

    /// As the job sizes a layer, the multiplier that makes it `radius`.
    fn size(&self) -> f32 {
        self.radius / auto_radius(visible_count(&self.paint, &self.vp), &self.vp)
    }
}

/// A panel's height when none is asked for: `width_in` at the shape of
/// `bounds`, between 0.4 and 2.5 of the width as a page's is.
pub(crate) fn shaped_height(width_in: f32, [x0, y0, x1, y1]: [f32; 4]) -> f32 {
    width_in * ((y1 - y0) / (x1 - x0).max(1e-6)).clamp(0.4, 2.5)
}

/// Why panel flags without a file to save are refused, on screen or off.
pub(super) const NEEDS_A_FILE: &str =
    "panel flags (--height, --point-pt, --hide, ...) need a file: pass --pdf, --png or --legend";

/// `senna view --pdf/--png/--legend` with panel flags: the one run's
/// starting map saved as the panel they ask for, `--width` inches wide
/// and, without `--height`, as tall as the map's shape.
pub(super) fn from_flags(args: &super::ViewArgs, scenes: &[Scene]) -> Result<()> {
    let [scene] = scenes else {
        bail!("a panel is one run's map: pass one manifest");
    };
    let flags = &args.panel;
    ensure!(
        args.export.pdf.is_some() || flags.png.is_some() || flags.legend.is_some(),
        NEEDS_A_FILE
    );
    let width = args.export.width;
    let height = shaped_height(width, scene.current().points.bounds);
    let panel = flags.panel(width, height, args.export.dpi);
    let files = Files {
        png: flags.png.as_deref(),
        pdf: args.export.pdf.as_deref(),
        legend: flags.legend.as_deref(),
    };
    let saved = save(scene, &panel, None, &files)?;
    let (w, h) = saved.px;
    for path in &saved.wrote {
        log::info!(
            "Saved {} ({} × {} in, {w} × {h} px at {} dpi)",
            path.display(),
            panel.width_in,
            panel.height_in,
            panel.dpi
        );
    }
    if let Some(note) = saved.note {
        log::warn!("{note}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use legume_tui::map::text::Font;

    #[test]
    fn labels_at_one_place_are_spread_not_dropped() {
        let font = Font::for_cell_height(20.0, false);
        let label = |k: u32| Label {
            text: format!("C{k}"),
            x: 100.0,
            y: 100.0,
            ink: [0.0; 3],
            tier: 0,
            priority: k as f32,
            font,
            key: Some(k),
        };
        let labels: Vec<Label> = (0..4).map(label).collect();
        let o = spread(&labels, (200, 200), Offsets::default());
        // The most wanted stays put; each of the rest moves somewhere apart.
        assert!(o.get(&3).is_none());
        let moved: Vec<_> = (0..3).filter_map(|k| o.get(&k)).collect();
        assert_eq!(moved.len(), 3);
        assert!(moved.windows(2).all(|w| w[0] != w[1]));
        // A label moved by hand keeps its place.
        let by_hand: Offsets<u32> = [(3, (0.0, 2.0))].into_iter().collect();
        assert_eq!(
            spread(&labels, (200, 200), by_hand).get(&3),
            Some((0.0, 2.0))
        );
    }
}
