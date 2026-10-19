//! Several runs at once: their views drawn for a page, and one page of
//! them all. The grid's shape, the page's assembly and the PDF are
//! legume-tui's (`export`); what each tile shows is senna's.

use super::color;
use super::render::{self, Frame, Job, Paint, Viewport};
use super::Scene;
use legume_tui::export::{self, Page};
use rayon::prelude::*;

/// A camera for a `w × h` picture: `cam` (where the view was) scaled to
/// cover the same part of the map, or the whole layout fit when there is no
/// camera yet.
#[must_use]
pub(crate) fn camera(cam: Option<Viewport>, bounds: [f32; 4], w: usize, h: usize) -> Viewport {
    cam.map_or_else(|| render::fit(bounds, w, h), |v| v.resized(bounds, w, h))
}

/// Every view drawn to the end and labelled at its `cell_px`. The views go
/// side by side on rayon's threads: a job stamps its points one after
/// another, so views are the parallel unit. With `keep_text`, text comes
/// back as runs for a PDF instead of drawn into the pixels.
pub(crate) fn frames(views: &[(&Scene, Viewport, f32)], keep_text: bool) -> Vec<Frame> {
    // A view showing a chart is drawn whole; the rest are maps.
    let charts: Vec<Option<Frame>> = views
        .iter()
        .map(|(s, vp, cell_px)| s.chart_frame(vp.w, vp.h, *cell_px, keep_text))
        .collect();
    if charts.iter().any(Option::is_some) {
        let maps: Vec<(&Scene, Viewport, f32)> = views
            .iter()
            .zip(&charts)
            .filter(|(_, c)| c.is_none())
            .map(|(v, _)| *v)
            .collect();
        let mut maps = frames(&maps, keep_text).into_iter();
        return charts
            .into_iter()
            .map(|c| c.unwrap_or_else(|| maps.next().expect("one map frame per map view")))
            .collect();
    }
    let layers: Vec<Vec<Paint<'_>>> = views.iter().map(|(s, ..)| s.layers()).collect();
    let mut jobs: Vec<Job> = views
        .iter()
        .zip(&layers)
        .map(|((_, vp, _), l)| Job::new(*vp, color::BACKGROUND, l))
        .collect();
    jobs.par_iter_mut()
        .zip(&layers)
        .for_each(|(job, l)| while !job.step(l) {});
    drop(layers);
    jobs.into_iter()
        .zip(views)
        .map(|(job, &(scene, vp, cell_px))| {
            let mut frame = job.finish();
            if keep_text {
                frame.keep_text();
            }
            // A page shows any stroke: the screen's floor stays on screen.
            scene.decorate(&mut frame, &vp, cell_px, 1.0);
            frame
        })
        .collect()
}

/// Views as PDF pages, one per scene: the points at each viewport's size,
/// the text kept as text.
pub(crate) fn view_pages(views: &[(&Scene, Viewport, f32)]) -> Vec<Page> {
    frames(views, true)
        .into_iter()
        .map(|mut frame| Page {
            text: frame.take_text(),
            img: frame.to_image(),
        })
        .collect()
}

/// Every scene's current view in one `w × h` image, a grid of tiles named
/// at the bottom left. Each tile shows what its camera saw (`cams`, scaled
/// to the tile), or its whole layout without one. Text comes back as runs
/// for a PDF, not drawn into the pixels.
pub(crate) fn render_grid(
    scenes: &[&Scene],
    cams: &[Option<Viewport>],
    titles: &[String],
    (w, h): (usize, usize),
    cell_px: f32,
) -> Page {
    let (cols, rows) = export::shape(scenes.len(), w as f32, h as f32);
    let (tw, th) = (w / cols, h / rows);
    let views: Vec<(&Scene, Viewport, f32)> = scenes
        .iter()
        .zip(cams)
        .map(|(&s, &cam)| (s, camera(cam, s.current().points.bounds, tw, th), cell_px))
        .collect();
    let tiles = frames(&views, true).into_iter().zip(titles.iter().cloned());
    export::grid_page(tiles.collect(), (w, h), cell_px, &color::THEME)
}
