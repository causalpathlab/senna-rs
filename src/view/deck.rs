//! Several runs at once: the shape of their grid, and one image of them all.

use super::color;
use super::pdf::Page;
use super::render::{Frame, Job, Paint, Viewport};
use super::text::{self, Font};
use super::Scene;
use image::RgbaImage;
use rayon::prelude::*;

/// Pixels past which a picture is refused (its canvas is 12 bytes a pixel).
pub(crate) const MAX_PIXELS: usize = 80_000_000;

/// Columns and rows for `n` tiles in a `w × h` area: the shape whose tiles
/// have the longest shorter side (maps are roughly square), fewer columns on
/// a tie.
#[must_use]
pub(crate) fn shape(n: usize, w: f32, h: f32) -> (usize, usize) {
    let n = n.max(1);
    let side = |c: usize| (w / c as f32).min(h / n.div_ceil(c) as f32);
    let c = (1..=n).fold(1, |best, c| if side(c) > side(best) { c } else { best });
    (c, n.div_ceil(c))
}

/// A camera for a `w × h` picture: `cam` (where the view was) scaled to
/// cover the same part of the map, or the whole layout fit when there is no
/// camera yet.
#[must_use]
pub(crate) fn camera(cam: Option<Viewport>, bounds: [f32; 4], w: usize, h: usize) -> Viewport {
    match cam {
        Some(v) if v.w > 0 && v.h > 0 => {
            let k = (w as f32 / v.w as f32).min(h as f32 / v.h as f32);
            Viewport {
                scale: v.scale * k,
                w,
                h,
                ..v
            }
        }
        _ => Viewport::fit(bounds, w, h),
    }
}

/// Every view drawn to the end and labelled at its `cell_px`. The views go
/// side by side on rayon's threads: a job stamps its points one after
/// another, so views are the parallel unit. With `keep_text`, text comes
/// back as runs for a PDF instead of drawn into the pixels.
pub(crate) fn frames(views: &[(&Scene, Viewport, f32)], keep_text: bool) -> Vec<Frame> {
    let layers: Vec<Vec<Paint<'_>>> = views.iter().map(|(s, ..)| s.layers()).collect();
    let mut jobs: Vec<Job> = views
        .iter()
        .zip(&layers)
        .map(|((_, vp, _), l)| Job::new(*vp, l))
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
            scene.decorate(&mut frame, &vp, cell_px);
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
    let (cols, rows) = shape(scenes.len(), w as f32, h as f32);
    let (tw, th) = (w / cols, h / rows);
    let views: Vec<(&Scene, Viewport, f32)> = scenes
        .iter()
        .zip(cams)
        .map(|(&s, &cam)| (s, camera(cam, s.current().points.bounds, tw, th), cell_px))
        .collect();

    let [r, g, b] = color::BACKGROUND;
    let mut out = RgbaImage::from_pixel(w as u32, h as u32, image::Rgba([r, g, b, 255]));
    let font = Font::for_cell_height(cell_px, true);
    let lh = font.line_height() as i32;
    let ink = color::linear_rgb(color::INK);
    let mut kept = Vec::new();
    for (k, (mut frame, title)) in frames(&views, true).into_iter().zip(titles).enumerate() {
        let bg = frame.background();
        text::draw(
            &mut frame,
            font,
            title,
            lh / 2,
            th as i32 - lh - lh / 4,
            ink,
            bg,
        );
        let (x, y) = (k % cols * tw, k / cols * th);
        kept.extend(frame.take_text().into_iter().map(|mut t| {
            t.x += x as f32;
            t.y += y as f32;
            t
        }));
        image::imageops::replace(&mut out, &frame.to_image(), x as i64, y as i64);
    }
    // Hairlines between tiles.
    let [r, g, b] = color::MUTED;
    let rule = image::Rgba([r, g, b, 255]);
    for c in 1..cols {
        let x = (c * tw) as u32;
        (0..h as u32).for_each(|y| out.put_pixel(x.min(w as u32 - 1), y, rule));
    }
    for r in 1..rows {
        let y = (r * th) as u32;
        (0..w as u32).for_each(|x| out.put_pixel(x, y.min(h as u32 - 1), rule));
    }
    Page {
        img: out,
        text: kept,
    }
}

#[cfg(test)]
mod tests {
    use super::shape;

    #[test]
    fn the_grid_keeps_tiles_as_large_as_it_can() {
        assert_eq!(shape(1, 100.0, 100.0), (1, 1));
        assert_eq!(shape(4, 100.0, 100.0), (2, 2));
        assert_eq!(shape(3, 300.0, 100.0), (3, 1));
        assert_eq!(shape(3, 100.0, 300.0), (1, 3));
        assert_eq!(shape(5, 160.0, 100.0), (3, 2));
        assert_eq!(shape(0, 10.0, 10.0), (1, 1));
    }
}
