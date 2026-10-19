//! Fixed scenes drawn to the byte: every pixel and every kept text run of
//! the map, its labels, keys, marks and edges, pinned by a digest. A change
//! to how a map is drawn shows here first.

use super::data::{LabelKind, Labels, Points, Space, SpaceKind};
use super::tests::scene;
use super::*;

/// FNV-1a over `bytes`: stable across toolchains, unlike the std hasher.
pub(super) fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Size and digest of a picture.
pub(super) fn digest(img: &image::RgbaImage) -> (u32, u32, u64) {
    (img.width(), img.height(), fnv(img.as_raw().iter().copied()))
}

/// Digest of text kept as runs, every field to the bit.
pub(super) fn runs_digest(runs: &[legume_tui::map::text::TextRun]) -> u64 {
    fnv(runs
        .iter()
        .flat_map(|t| {
            let mut s = t.text.as_bytes().to_vec();
            for v in [t.x, t.y, t.w, t.line].into_iter().chain(t.ink).chain(t.bg) {
                s.extend(v.to_bits().to_le_bytes());
            }
            s.extend([u8::from(t.bold), u8::from(t.centred), u8::from(t.vertical)]);
            s
        })
        .collect::<Vec<_>>())
}

const W: usize = 240;
const H: usize = 180;

fn draw(s: &Scene, cell_px: f32) -> (u32, u32, u64) {
    let vp = render::fit(s.current().points.bounds, W, H);
    digest(
        &deck::frames(&[(s, vp, cell_px)], false)
            .remove(0)
            .to_image(),
    )
}

/// The fixture with its cell map replaced by `n` cells in five clusters,
/// loosely blobbed, so marks overlap and labels compete.
fn crowded(n: usize) -> Scene {
    let mut s = scene();
    let mut state = 12_345u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let centres = [[0.0, 0.0], [4.0, 1.0], [1.5, 4.0], [5.0, 4.5], [2.5, 2.0]];
    let mut names = Vec::new();
    let mut xy = Vec::new();
    let mut pairs = Vec::new();
    for i in 0..n {
        let g = i % centres.len();
        let [cx, cy] = centres[g];
        names.push(format!("c{i}").into_boxed_str());
        xy.push([cx + 2.0 * (next() - 0.5), cy + 2.0 * (next() - 0.5)]);
        pairs.push((format!("c{i}").into_boxed_str(), g as i64));
    }
    s.data.spaces[0] = Space {
        method: "umap".into(),
        kind: SpaceKind::Cells,
        points: std::sync::Arc::new(Points::new(names, xy)),
        backdrop: None,
        parent: None,
    };
    s.data.labels[0] = Labels::clusters(LabelKind::Cluster, pairs);
    s.groups = None;
    s.refresh_groups();
    s
}

/// The fixture's view of a clicked cell, its features placed on the cells.
fn with_near(mut s: Scene) -> Scene {
    let mut placed = Space {
        method: "umap".into(),
        kind: SpaceKind::FeaturesOnCells,
        points: std::sync::Arc::new(Points::new(
            ["g1", "g2", "g3"].map(Into::into).to_vec(),
            vec![[3.0, 0.0], [3.0, 2.0], [0.0, 2.0]],
        )),
        backdrop: Some(0),
        parent: None,
    };
    placed.backdrop = Some(0);
    s.data.spaces.push(placed);
    s.near = Some(features::Near {
        name: "c1".into(),
        centre: features::Centre::Cell,
        metric: features::Metric::Distance,
        features: vec![("g1".into(), 1.0), ("g2".into(), 0.5)],
        cells: vec![("c3".into(), 0.4)],
    });
    s
}

#[test]
fn a_cluster_map_draws_as_it_did() {
    let s = crowded(3000);
    assert_eq!(draw(&s, 16.0), (240, 180, 12731718632196819083));
}

#[test]
fn a_hovered_label_draws_as_it_did() {
    let mut s = crowded(3000);
    let vp = render::fit(s.current().points.bounds, W, H);
    draw(&s, 16.0);
    let (x, y) = vp.to_px(s.group_centre(2).unwrap());
    s.hover = s.group_at(&vp, x, y);
    assert_eq!(s.hover, Some(2));
    assert_eq!(draw(&s, 16.0), (240, 180, 13134139464734984117));
}

#[test]
fn styled_groups_draw_as_they_did() {
    let mut s = crowded(3000);
    let looks = [
        (style::Shape::Square, 1.0, 1.0),
        (style::Shape::Diamond, 0.6, 2.5),
        (style::Shape::Triangle, 1.0, 3.0),
        (style::Shape::Cross, 0.8, 2.0),
    ];
    for (g, (shape, alpha, size)) in looks.into_iter().enumerate() {
        s.restyle(g, |st| {
            st.shape = shape;
            st.alpha = alpha;
            st.size = size;
        });
    }
    s.restyle(4, |st| st.colour = Some([10, 120, 200]));
    assert_eq!(draw(&s, 16.0), (240, 180, 12142580646687436989));
    s.focus = Some(1);
    assert_eq!(draw(&s, 16.0), (240, 180, 14895850866872429048));
}

#[test]
fn large_text_draws_as_it_did() {
    // Past 32 px a cell's text is the 32 raster resampled up.
    let s = crowded(3000);
    assert_eq!(draw(&s, 48.0), (240, 180, 8862107923740463606));
}

#[test]
fn a_feature_map_with_its_key_draws_as_it_did() {
    let mut s = scene();
    s.set_space(1);
    s.colour_by(LabelKind::Markers);
    assert!(!s.legend().is_empty());
    assert_eq!(draw(&s, 16.0), (240, 180, 5319375523145005806));
}

#[test]
fn a_clicked_cell_draws_as_it_did() {
    let s = with_near(scene());
    assert_eq!(draw(&s, 16.0), (240, 180, 13247368101845531460));
    // The features over their backdrop of muted cells.
    let mut s = s;
    s.set_space(3);
    assert_eq!(draw(&s, 16.0), (240, 180, 4092230938843235177));
}

#[test]
fn activity_with_its_ramp_key_draws_as_it_did() {
    let mut s = crowded(3000);
    let n = s.current().points.xy.len();
    let values = (0..n)
        .map(|i| if i % 7 == 0 { 0.0 } else { (i % 50) as f32 })
        .collect();
    s.shown = Some(Shown {
        id: 1,
        space: 0,
        pick: Pick::One("GENE1".into()),
        source: Source::Observed,
        title: "GENE1 observed".into(),
        levels: Levels::from_values(values, true),
    });
    assert_eq!(draw(&s, 16.0), (240, 180, 12189296318815704741));
}

#[test]
fn text_kept_for_a_page_is_as_it_was() {
    let s = with_near(crowded(3000));
    let vp = render::fit(s.current().points.bounds, W, H);
    let mut frame = deck::frames(&[(&s, vp, 16.0)], true).remove(0);
    let runs = frame.take_text();
    assert_eq!(
        (digest(&frame.to_image()), runs.len(), runs_digest(&runs)),
        ((240, 180, 9351161338657952179), 7, 4663346714267551512)
    );
}

#[test]
fn a_grid_page_is_as_it_was() {
    let (a, b) = (crowded(2000), with_near(scene()));
    let titles = ["run A".to_string(), "run B".to_string()];
    let page = deck::render_grid(&[&a, &b], &[None, None], &titles, (W, H), 16.0);
    assert_eq!(
        (digest(&page.img), page.text.len(), runs_digest(&page.text)),
        ((240, 180, 11379143200818801882), 8, 16793650456510507046)
    );
}

#[test]
fn velocity_marks_draw_as_they_did() {
    let mut frame = render::Frame::blank(W, H, color::BACKGROUND);
    let lines = vec![
        vec![(10.0, 20.0), (60.0, 35.5), (130.0, 90.0), (300.0, 120.0)],
        vec![(200.0, 170.0), (180.0, 100.0), (185.5, -40.0)],
    ];
    velocity::draw_lines(&mut frame, &lines, color::oklch(0.30, 0.02, 250.0), 1.0);
    let arrows = [
        ((20.0, 150.0), (70.0, 120.0)),
        ((100.0, 60.0), (-30.0, 80.0)),
    ];
    velocity::draw(&mut frame, &arrows, color::highlight_ink(), 2.0);
    assert_eq!(digest(&frame.to_image()), (240, 180, 2108609922703495894));
}

/// Size and digest of the PDF of `pages`, as a save writes it.
fn pdf_digest(pages: &[pdf::Page], dpi: f32) -> (usize, u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("page.pdf");
    pdf::write(pages, dpi, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (bytes.len(), fnv(bytes))
}

#[test]
fn a_map_saved_as_a_pdf_is_as_it_was() {
    let (page, px) = headless_page(&[with_near(crowded(3000))], &[], false, 2.0, 150).unwrap();
    assert_eq!(px, (300, 278));
    assert_eq!(pdf_digest(&[page], 150.0), (42547, 4532757784813180276));
}

#[test]
fn a_grid_saved_as_a_pdf_is_as_it_was() {
    let titles = ["run A".to_string(), "run B".to_string()];
    let scenes = [crowded(2000), with_near(scene())];
    let (page, px) = headless_page(&scenes, &titles, false, 2.0, 150).unwrap();
    assert_eq!(px, (300, 113));
    assert_eq!(pdf_digest(&[page], 150.0), (14982, 1032533684822131051));
    // One run with a chart asked for takes a 4:3 page.
    let (page, px) = headless_page(&[scene()], &titles, true, 2.0, 150).unwrap();
    assert_eq!(px, (300, 225));
    assert_eq!(pdf_digest(&[page], 150.0), (1628, 9598306093650663253));
}

#[test]
fn views_saved_one_page_each_are_as_they_were() {
    let (a, b) = (crowded(2000), with_near(scene()));
    let views = [
        (&a, render::fit(a.current().points.bounds, W, H), 16.0),
        (&b, render::fit(b.current().points.bounds, H, W), 24.0),
    ];
    assert_eq!(
        pdf_digest(&deck::view_pages(&views), 144.0),
        (23939, 7457570728265228916)
    );
}

#[test]
fn a_page_too_large_or_empty_is_refused() {
    let s = [scene()];
    assert!(headless_page(&s, &[], false, 0.0, 300).is_err());
    assert!(headless_page(&s, &[], false, 7.0, 0).is_err());
    assert!(headless_page(&s, &[], false, 100.0, 1200).is_err());
}
