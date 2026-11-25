use super::search;
use super::*;

fn app() -> App {
    App::new(
        crate::view::tests::scene(),
        Picker::halfblocks(),
        "r.senna.json".into(),
        "lupin".into(),
    )
}

fn press(app: &mut App, c: char) {
    app.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
}

#[test]
fn a_chart_keeps_to_its_keys_and_esc_returns_to_the_map() {
    let mut a = app();
    a.scene.cycle_chart();
    assert!(a.scene.chart.is_some());
    let colour = a.scene.colour;
    press(&mut a, 'n');
    assert!(a.message.as_deref().unwrap().starts_with("on a chart"));
    assert!(a.scene.pick.is_none());
    press(&mut a, 'c');
    assert_ne!(a.scene.colour, colour);
    a.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(a.scene.chart.is_none());
}

#[test]
fn merge_mode_keeps_to_its_keys() {
    let mut a = app();
    let draft = crate::view::review::Draft::default();
    let mut review = crate::view::relabel::Review::new(Vec::new(), draft);
    review.merge = Some(crate::view::relabel::MergeSel {
        cursor: 0,
        chosen: std::collections::BTreeSet::new(),
        levels: Vec::new(),
        best: None,
    });
    a.scene.review = Some(review);
    let space = a.scene.space;
    for c in ['m', 'z', 'L', 'a'] {
        press(&mut a, c);
        assert!(
            a.message.as_deref().unwrap().starts_with("in merge mode"),
            "{c}"
        );
    }
    assert_eq!(a.scene.space, space);
    assert!(a.zooming.is_none());
}

#[test]
fn the_help_page_scrolls_and_closes_only_on_esc_q_or_question_mark() {
    let mut a = app();
    press(&mut a, '?');
    assert!(a.help);
    a.help_max.set(5);
    press(&mut a, 'j');
    a.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(a.help_scroll, 2);
    // A key of the map does nothing while the page is up.
    let colour = a.scene.colour;
    press(&mut a, 'c');
    assert!(a.help && a.scene.colour == colour);
    press(&mut a, 'q');
    assert!(!a.help && !a.quit);
}

#[test]
fn ctrl_c_quits_and_control_keys_type_nothing_in_search() {
    let mut a = app();
    a.modal = Some(Modal::Search(String::new(), Vec::new(), 0));
    press(&mut a, 'g');
    a.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
    match &a.modal {
        Some(Modal::Search(query, ..)) => assert_eq!(query, "g"),
        _ => panic!("search closed"),
    }
    a.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert!(a.quit);
}

#[test]
fn ctrl_e_asks_for_an_extra_file_and_e_alone_opens_the_menu() {
    let mut a = app();
    let ctrl_e = KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL);
    a.key(ctrl_e);
    assert!(a.message.as_deref().unwrap().starts_with("no run"));
    a.scene.data.run = Some((
        senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Svd, "r"),
        std::env::temp_dir(),
    ));
    a.key(ctrl_e);
    assert!(a.menu.is_none());
    let Some(m) = &a.modal else {
        panic!("no browser")
    };
    assert!(
        matches!(m, Modal::ExtraFile(_))
            && m.browser_lines(ratatui::layout::Rect::new(0, 0, 100, 22), 100)
                .is_some()
    );
    a.modal = None;
    press(&mut a, 'e');
    assert!(a.menu.is_some());
}

#[test]
pub(super) fn search_ranks_exact_then_symbol_then_prefix_then_substring() {
    let names: Vec<Box<str>> = ["ENSG1_GENE10", "GENE1", "XGENE1Y", "ENSG2_GENE1", "OTHER"]
        .map(Into::into)
        .to_vec();
    let lower: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
    let hits = search(&names, &lower, "gene1");
    assert_eq!(&*hits[0], "GENE1");
    assert_eq!(&*hits[1], "ENSG2_GENE1");
    assert_eq!(&*hits[2], "ENSG1_GENE10");
    assert_eq!(&*hits[3], "XGENE1Y");
    assert_eq!(hits.len(), 4);
    assert!(search(&names, &lower, "").is_empty());
}

#[test]
fn the_chart_status_line_names_every_key_a_chart_takes() {
    use crate::view::tui::help::{keys_of, Context};
    let line = keys_of(Context::Chart).join(" ");
    // `=` and `_` are `+` and `-` unshifted.
    for c in CHART_KEYS.chars().filter(|c| !matches!(c, '=' | '_')) {
        assert!(line.contains(c), "the Chart status line leaves out {c}");
    }
}
