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

pub(super) fn scene() -> Scene {
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
        loads: activity::RunLoads::none(),
        times: None,
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
        manifests: Vec::new(),
        from: vec!["r.senna.json".into()],
        method: None,
        colour_by: None,
        graphics: Graphics::Blocks,
        export: PdfArgs {
            pdf: None,
            width: 7.0,
            dpi: 300,
        },
        space: None,
        feature: None,
        markers_of: None,
        observed: false,
        velocity: None,
        stream_density: None,
        focus: None,
        zoom_into: None,
        chart: None,
        relabel: false,
        lupin: None,
        watch: None,
        no_compute: true,
        extra: Vec::new(),
        cell_stat: None,
        session: Default::default(),
    };
    Scene::new(data, &args)
}

#[test]
fn the_pdf_flags_keep_their_names_and_defaults() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        view: ViewArgs,
    }
    let parse = |args: &[&str]| {
        let mut argv = vec!["senna", "-f", "r.senna.json"];
        argv.extend(args);
        Cli::try_parse_from(argv).unwrap().view.export
    };
    let pdf = parse(&["--pdf", "fig.pdf"]);
    assert_eq!(pdf.pdf.as_deref(), Some(std::path::Path::new("fig.pdf")));
    assert_eq!(
        pdf.size(),
        legume_tui::export::PageSize {
            width_in: 7.0,
            dpi: 300
        }
    );
    let pdf = parse(&["--pdf", "fig.pdf", "--width", "3.5", "--dpi", "600"]);
    assert_eq!((pdf.width, pdf.dpi), (3.5, 600));
    assert!(parse(&[]).pdf.is_none());
}

#[test]
fn batches_wait_for_the_data_files_and_are_taken_once() {
    let mut s = scene();
    let before = s.data.labels.len();
    // Not read yet: nothing to take, and asked again later.
    s.data.loads.files = std::sync::Arc::new(activity::RunFiles::new(|| Err("slow".into())));
    assert!(!s.take_batches());
    assert!(!s.batches_taken);
    // Read, but failed: taken (nothing added), not asked again.
    s.data.loads.files = std::sync::Arc::new(activity::RunFiles::ready(Err("gone".into())));
    assert!(!s.take_batches());
    assert!(s.batches_taken);
    assert_eq!(s.data.labels.len(), before);
}

#[test]
fn what_needed_a_read_not_done_yet_is_done_again_when_it_answers() {
    let mut s = scene();
    s.patient = false;
    s.data.run = Some((
        senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Bge, "r"),
        std::path::PathBuf::new(),
    ));
    let (gate, wait) = std::sync::mpsc::channel::<()>();
    s.data.loads.model = std::sync::Arc::new(activity::ModelTables::new(move || {
        let _ = wait.recv();
        Err("no model here".into())
    }));
    s.poll_loads();
    s.suggest();
    assert_eq!(s.note.take().as_deref(), Some(legume_tui::work::LOADING));
    assert!(s.loading_line().is_some_and(|l| l.contains("model tables")));
    // Nothing new yet: nothing done again.
    assert!(!s.poll_loads());
    gate.send(()).unwrap();
    while s.data.loads.model.try_get().is_none() {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(s.poll_loads());
    // The suggestion was asked again and got the model's answer.
    assert_eq!(s.note.take().as_deref(), Some("no model here"));
    assert!(s.loading_line().is_none());
}

#[test]
fn a_copied_view_shares_the_run_s_data_files() {
    let s = scene();
    let copy = s.duplicate();
    assert!(std::sync::Arc::ptr_eq(
        &s.data.loads.files,
        &copy.data.loads.files
    ));
    assert!(std::sync::Arc::ptr_eq(
        &s.data.loads.model,
        &copy.data.loads.model
    ));
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
fn another_round_leaves_relabel_mode_the_same_one_keeps_it_and_the_draft_stays() {
    for same in [false, true] {
        let label = if same {
            "the same round"
        } else {
            "another round"
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.senna.json");
        std::fs::write(&path, "{}").unwrap();
        let round = || {
            let m = senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Svd, "r");
            rounds::Round::load(&m, dir.path(), &path)
        };
        let mut s = scene();
        s.data.round = Some(round());
        s.enter_review().unwrap();
        s.review.as_mut().unwrap().draft.cluster(1).verdict = Some(review::Verdict::Keep {
            label: "CT1".into(),
            rationale: "clear".into(),
        });
        let at = s.review.as_ref().unwrap().cluster();

        let mut next = scene().data;
        if same {
            next.round = Some(round());
        }
        s.replace_data(next);
        let note = s.note.clone().unwrap_or_default();
        if same {
            let r = s.review.as_ref().expect("still relabelling");
            assert_eq!(r.cluster(), at, "{label}");
            assert!(r.draft.clusters[&1].verdict.is_some(), "{label}");
            assert!(note.contains("still relabelling"), "{label}: {note}");
        } else {
            assert!(s.review.is_none(), "{label}");
            assert!(
                note.contains("draft stays with r.senna.json"),
                "{label}: {note}"
            );
            assert!(!review::Draft::load(&path).is_empty(), "{label}");
        }
    }
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
    let vp = render::fit(s.current().points.bounds, 200, 200);
    let before = render_full(&s, vp, 16.0);
    s.near = Some(features::Near {
        name: "c1".into(),
        centre: features::Centre::Cell,
        metric: features::Metric::Distance,
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

#[test]
fn a_click_on_a_cluster_label_finds_that_cluster() {
    let s = scene();
    let vp = render::fit(s.current().points.bounds, 200, 200);
    // Only a frame that was drawn has labels to click.
    let (x, y) = vp.to_px(s.group_centre(1).unwrap());
    assert_eq!(s.group_at(&vp, x, y), None);
    render_full(&s, vp, 16.0);
    // C1 (cells c2, c3) is labelled at its centre; a click there is C1.
    assert_eq!(s.group_at(&vp, x, y), Some(1));
    // C0 is one cell in the frame's corner: its label would run off the
    // frame, so it is not drawn, and a click there is not C0.
    let (x0, y0) = vp.to_px(s.group_centre(0).unwrap());
    assert_ne!(s.group_at(&vp, x0, y0), Some(0));
    // Far from every label, nothing; nor on another camera's frame.
    assert_eq!(s.group_at(&vp, 100.0, 5.0), None);
    let moved = Viewport {
        cx: vp.cx + 1.0,
        ..vp
    };
    assert_eq!(s.group_at(&moved, x, y), None);
    // No grouping on screen (as `c` leaves it), no labels to click.
    let mut plain = scene();
    plain.cycle_colour();
    plain.cycle_colour();
    assert_eq!(plain.colour, None);
    render_full(&plain, vp, 16.0);
    assert_eq!(plain.group_at(&vp, x, y), None);
}

#[test]
fn a_label_under_the_pointer_is_framed() {
    let mut s = scene();
    s.show_labels = true;
    let vp = render::fit(s.current().points.bounds, 200, 200);
    let plain = render_full(&s, vp, 16.0);
    let [x0, y0, x1, _] = s.label_rect(1).unwrap();
    s.hover = Some(1);
    let framed = render_full(&s, vp, 16.0);
    // Along the frame's top edge, just outside the label's text.
    let changed = (x0 as u32..x1 as u32)
        .filter(|&x| plain.get_pixel(x, y0 as u32) != framed.get_pixel(x, y0 as u32))
        .count();
    assert!(changed > 0);
}

#[test]
fn a_clicked_cluster_is_drawn_with_edges_to_its_features() {
    let mut s = scene();
    let mut placed = space("umap", SpaceKind::FeaturesOnCells, &["g1", "g2", "g3"]);
    placed.points = std::sync::Arc::new(Points::new(
        ["g1", "g2", "g3"].map(Into::into).to_vec(),
        vec![[3.0, 2.0], [3.0, 0.0], [0.0, 2.0]],
    ));
    placed.backdrop = Some(0);
    s.data.spaces.push(placed);
    s.show_labels = false;
    let vp = render::fit(s.current().points.bounds, 200, 200);
    let before = render_full(&s, vp, 16.0);
    let centre = s.group_centre(1).unwrap();
    s.near = Some(features::Near {
        name: "C1".into(),
        centre: features::Centre::Group {
            space: 0,
            xy: centre,
        },
        metric: features::Metric::Distance,
        features: vec![("g1".into(), 1.0)],
        cells: Vec::new(),
    });
    let after = render_full(&s, vp, 16.0);
    // Halfway from the cluster's centre to g1 at (3, 2): the edge.
    let (x, y) = vp.to_px([0.5 * (centre[0] + 3.0), 0.5 * (centre[1] + 2.0)]);
    assert_ne!(
        before.get_pixel(x as u32, y as u32),
        after.get_pixel(x as u32, y as u32)
    );
}

#[test]
fn one_knob_scales_every_dot_and_text_together() {
    let mut s = scene();
    assert_eq!(s.scale, 1.0);
    let dot = |s: &Scene| s.layers().last().unwrap().size;
    let before = dot(&s);
    s.resize(1);
    assert!((s.scale - 1.25).abs() < 1e-6);
    assert!((dot(&s) - 1.25 * before).abs() < 1e-6);
    assert!(s.note.take().unwrap().contains("1.25"));
    // Text follows: the label box to click grows with it.
    let vp = render::fit(s.current().points.bounds, 200, 200);
    let (x, y) = vp.to_px(s.group_centre(1).unwrap());
    let reach = |s: &Scene| {
        render_full(s, vp, 16.0);
        (0..100)
            .map(|d| d as f32)
            .take_while(|&d| s.group_at(&vp, x + d, y) == Some(1))
            .count()
    };
    let small = {
        let mut t = scene();
        t.resize(-1);
        reach(&t)
    };
    assert!(reach(&s) > small);
    // Bounded both ways.
    for _ in 0..20 {
        s.resize(1);
    }
    assert!((s.scale - 4.0).abs() < 1e-6);
    for _ in 0..40 {
        s.resize(-1);
    }
    assert!((s.scale - 0.4).abs() < 1e-6);
    // A copy keeps its size.
    assert_eq!(s.duplicate().scale, s.scale);
}
