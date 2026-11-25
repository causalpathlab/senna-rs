use super::*;
use legume_tui::save::{SaveKey, SaveTarget};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

/// What the save dialog offers over `n` views, this one named `view`.
fn target(n: usize, view: &str) -> Target {
    Target {
        n,
        view: view.into(),
        first: "run".into(),
        view_px: (800.0, 600.0),
        grid_px: (1200.0, 400.0),
    }
}

/// A gallery in `work` with a save of each of `names`, the last on top.
fn gallery(work: &std::path::Path, names: &[&str]) -> Gallery {
    let mut g = open_saved(&work.join(SAVED_DIR));
    let picture = image::RgbaImage::from_pixel(40, 30, image::Rgba([1, 2, 3, 255]));
    for name in names {
        let pdf = work.join(name);
        std::fs::write(&pdf, b"%PDF").unwrap();
        g.add(&[&pdf], "a view", Some(&picture), ()).unwrap();
    }
    g
}

#[test]
fn a_proposed_name_never_collides_with_a_saved_file() {
    let dir = tempfile::tempdir().unwrap();
    let stem = dir.path().join("run.view.umap.cells");
    let stem = stem.to_string_lossy();
    let t = target(1, &stem);
    let offered = |t: &Target| {
        SaveDialog::new(t, Scope::View, NUMBERING)
            .name()
            .to_string()
    };
    assert_eq!(offered(&t), stem);
    std::fs::write(format!("{stem}.pdf"), b"%PDF").unwrap();
    assert_eq!(offered(&t), format!("{stem}-2"));
    std::fs::write(format!("{stem}-2.pdf"), b"%PDF").unwrap();
    assert_eq!(offered(&t), format!("{stem}-3"));
}

#[test]
fn the_offered_name_opens_selected() {
    let t = target(1, "fig");
    // Typing replaces the offered name, and backspace clears it.
    let mut d = SaveDialog::new(&t, Scope::View, NUMBERING);
    d.key(key(KeyCode::Char('x')), &t);
    assert_eq!(d.name(), "x");
    let mut d = SaveDialog::new(&t, Scope::View, NUMBERING);
    d.key(key(KeyCode::Backspace), &t);
    assert_eq!(d.name(), "");
    // End keeps it to type on.
    let mut d = SaveDialog::new(&t, Scope::View, NUMBERING);
    d.key(key(KeyCode::End), &t);
    d.key(key(KeyCode::Char('x')), &t);
    assert_eq!(d.name(), "figx");
}

#[test]
fn replacing_a_file_takes_a_second_enter() {
    let dir = tempfile::tempdir().unwrap();
    let taken = dir.path().join("fig");
    std::fs::write(dir.path().join("fig.pdf"), b"%PDF").unwrap();
    let t = target(1, &taken.to_string_lossy());

    // A new name saves at once.
    let fresh = target(1, &dir.path().join("new").to_string_lossy());
    let mut d = SaveDialog::new(&fresh, Scope::View, NUMBERING);
    assert!(matches!(
        d.key(key(KeyCode::Enter), &fresh),
        SaveKey::Save(_)
    ));

    // An existing one asks first, then saves on the second enter.
    let mut d = SaveDialog::new(&t, Scope::View, NUMBERING);
    d.key(key(KeyCode::End), &t);
    for _ in 0.."-2".len() {
        d.key(key(KeyCode::Backspace), &t);
    }
    assert_eq!(d.name(), taken.to_string_lossy());
    assert_eq!(d.key(key(KeyCode::Enter), &t), SaveKey::Open);
    assert!(d.status().unwrap().contains("enter again replaces it"));
    let SaveKey::Save(req) = d.key(key(KeyCode::Enter), &t) else {
        panic!("the second enter saves");
    };
    assert_eq!(req.path, dir.path().join("fig.pdf"));
    assert_eq!((req.width_in, req.dpi, req.px), (7.0, 300, (2100, 1575)));

    // Another name in between asks again.
    let mut d = SaveDialog::new(&t, Scope::View, NUMBERING);
    d.key(key(KeyCode::End), &t);
    for _ in 0.."-2".len() {
        d.key(key(KeyCode::Backspace), &t);
    }
    assert_eq!(d.key(key(KeyCode::Enter), &t), SaveKey::Open);
    d.key(key(KeyCode::Char('x')), &t);
    d.key(key(KeyCode::Backspace), &t);
    assert_eq!(d.key(key(KeyCode::Enter), &t), SaveKey::Open);
}

#[test]
fn several_views_offer_a_grid_and_a_page_each() {
    let one = target(1, "run.view.umap.cells");
    assert_eq!(one.scopes(), [Scope::View]);
    let two = target(2, "run.view.umap.cells");
    assert_eq!(two.scopes(), Scope::ALL);
    assert_eq!(two.scope_name(Scope::Grid), "all 2 runs, one grid");
    assert_eq!(two.default_stem(Scope::Grid), "run.view.grid");
    assert_eq!(two.default_stem(Scope::Pages), "run.view.pages");
    assert_eq!(two.size_note(Scope::Pages), "(first of 2 pages)");
    assert_eq!(two.exts(Scope::View), ["pdf"]);
    // A grid is as wide as the grid on screen, a view as the view.
    assert_eq!(two.aspect(Scope::Grid), 3.0);
    assert_eq!(two.aspect(Scope::Pages), 800.0 / 600.0);
    // ← → step through the scopes, each with its own name.
    let mut d = SaveDialog::new(&two, Scope::View, NUMBERING);
    d.key(key(KeyCode::Right), &two);
    assert_eq!(d.scope(), Scope::Grid);
    assert_eq!(d.name(), "run.view.grid");
    assert_eq!(d.px(&two), (2100, 700));
}

#[test]
fn a_log_written_before_still_loads() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join(SAVED_DIR);
    std::fs::create_dir_all(store.join("thumbs")).unwrap();
    let pdf = work.path().join("a.pdf");
    std::fs::write(&pdf, b"%PDF").unwrap();
    let thumb = image::RgbaImage::from_pixel(24, 18, image::Rgba([1, 2, 3, 255]));
    thumb.save(store.join("thumbs").join("17.png")).unwrap();
    let log = serde_json::json!([{
        "path": pdf.canonicalize().unwrap(),
        "what": "this view · 7 in · 300 dpi",
        "when": 1_000_000,
        "thumb": "17.png",
    }]);
    std::fs::write(store.join("saved.json"), log.to_string()).unwrap();

    let mut g = open_saved(&store);
    assert_eq!(g.entries().len(), 1);
    let e = &g.entries()[0];
    assert_eq!(
        (e.name().as_str(), e.what.as_str()),
        ("a.pdf", "this view · 7 in · 300 dpi")
    );
    assert!(g.thumb(e).unwrap().exists());

    // A new save goes on top and the old one stays, as it was written.
    let b = work.path().join("b.pdf");
    std::fs::write(&b, b"%PDF").unwrap();
    g.add(&[&b], "a view", Some(&thumb), ()).unwrap();
    let names: Vec<String> = open_saved(&store)
        .entries()
        .iter()
        .map(|e| e.name())
        .collect();
    assert_eq!(names, ["b.pdf", "a.pdf"]);
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(store.join("saved.json")).unwrap()).unwrap();
    assert_eq!(written[1]["thumb"], "17.png");
    assert_eq!(written[1]["when"], 1_000_000);
}

#[test]
fn the_strip_scrolls_to_the_chosen_save_and_drops_it() {
    let work = tempfile::tempdir().unwrap();
    let names: Vec<String> = (0..10).map(|k| format!("{k}.pdf")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut deck = Deck::new(
        Vec::new(),
        Picker::halfblocks(),
        gallery(work.path(), &names),
    );
    // Over the grid, so what is said goes in a toast.
    deck.grid = true;
    let full = Rect::new(0, 0, 160, 50);
    let shown = deck.strip.shown(deck.strip_area(full).unwrap());
    assert!((1..10).contains(&shown), "{shown} of 10 fit");

    deck.focus_saved();
    assert_eq!(deck.strip.chosen(), Some(0));
    // Down past the last one shown scrolls it into view.
    for _ in 0..shown {
        deck.strip_key(full, key(KeyCode::Down));
    }
    assert_eq!(deck.strip.chosen(), Some(shown));
    assert_eq!(deck.strip.top(), 1);
    deck.strip_key(full, key(KeyCode::End));
    assert_eq!(deck.strip.chosen(), Some(9));
    assert_eq!(deck.strip.top(), 10 - shown);
    // The wheel stops at either end.
    assert!(!deck.strip.scroll(99, shown, &deck.gallery));
    assert!(deck.strip.scroll(0, shown, &deck.gallery));

    // x takes the last off the list but keeps its PDF; the choice stays
    // in range.
    deck.strip_key(full, key(KeyCode::End));
    let last = deck.gallery.entries()[9].path.clone();
    deck.strip_key(full, key(KeyCode::Char('x')));
    assert_eq!(deck.gallery.entries().len(), 9);
    assert!(last.exists());
    assert_eq!(deck.strip.chosen(), Some(8));
    // m asks where to move it.
    deck.strip_key(full, key(KeyCode::Char('m')));
    assert!(deck.moving.is_some());
    deck.moving = None;
    // Esc hands the keys back.
    deck.strip_key(full, key(KeyCode::Esc));
    assert_eq!(deck.strip.chosen(), None);
}

#[test]
fn the_strip_sits_above_the_status_lines_beside_the_views() {
    let work = tempfile::tempdir().unwrap();
    let mut deck = Deck::new(
        Vec::new(),
        Picker::halfblocks(),
        gallery(work.path(), &["a.pdf"]),
    );
    deck.grid = true;
    let full = Rect::new(0, 0, 160, 50);
    assert_eq!(
        deck.strip_area(full),
        Some(Rect::new(0, 0, 28, 50 - STATUS_LINES))
    );
    assert_eq!(deck.content(full), Rect::new(28, 0, 132, 50));
    // Hidden, or on a narrow screen, the views take it all.
    assert_eq!(deck.strip_area(Rect::new(0, 0, 80, 50)), None);
    deck.toggle_saved();
    assert_eq!(deck.content(full), full);
}

use super::attach_tests::{kitty, view};

#[test]
fn after_an_attach_every_tile_and_thumbnail_is_made_again() {
    let work = tempfile::tempdir().unwrap();
    let mut deck = drawn_deck(Picker::halfblocks(), work.path());
    let full = Rect::new(0, 0, 160, 50);

    deck.repick(Repick::Changed(kitty()));
    assert_eq!(deck.cell, (8.0, 16.0));
    assert!(deck.apps.iter().all(|a| a.cell == (8.0, 16.0)));
    assert!(deck.tiles.is_empty() && deck.strip.made() == 0);
    // The next round lays the tiles out again and draws them anew.
    deck.prepare_thumbs(full);
    deck.layout_grid(deck.content(full));
    assert_eq!(deck.tiles.len(), 2);
    assert!(deck.tiles.iter().all(|t| t.job.is_some()));
    assert_eq!(deck.strip.made(), 1);
    assert_eq!(
        deck.strip.painter().picker.protocol_type(),
        ratatui_image::picker::ProtocolType::Kitty
    );
}

/// A deck of two views made with `picker`: the first drawn whole, the
/// grid's tiles and a saved thumbnail too.
fn drawn_deck(picker: Picker, work: &std::path::Path) -> Deck {
    let apps = vec![view(), view()]
        .into_iter()
        .map(|mut a| {
            a.repick(picker.clone());
            a
        })
        .collect();
    let mut deck = Deck::new(apps, picker, gallery(work, &["a.pdf"]));
    let full = Rect::new(0, 0, 160, 50);
    let area = deck.content(full);
    deck.apps[0].layout(area, STATUS_LINES);
    while deck.apps[0].advance().unwrap() {}
    deck.grid = true;
    deck.prepare_thumbs(full);
    deck.layout_grid(deck.content(full));
    while deck.advance_tiles().unwrap() {}
    assert!(deck.apps[0].proto.is_some() && deck.apps[0].base.is_some());
    assert!(deck.tiles.iter().all(|t| t.proto.is_some()));
    assert!(deck.strip.made() > 0);
    deck
}

#[test]
fn a_terminal_like_the_last_keeps_the_pictures_and_kitty_s_are_sent_again() {
    // Block characters: nothing made again, all redrawn from the cells.
    let work = tempfile::tempdir().unwrap();
    let mut deck = drawn_deck(Picker::halfblocks(), work.path());
    let vp = deck.apps[0].vp;
    deck.repick(Repick::Same);
    assert!(deck.apps[0].job.is_none() && deck.apps[0].proto.is_some());
    assert_eq!(deck.apps[0].vp, vp);
    assert!(deck
        .tiles
        .iter()
        .all(|t| t.job.is_none() && t.proto.is_some()));
    assert!(deck.strip.made() > 0);

    // Kitty: nothing encoded yet; each view waits until it is next drawn.
    let work = tempfile::tempdir().unwrap();
    let mut deck = drawn_deck(kitty(), work.path());
    let vp = deck.apps[0].vp;
    deck.repick(Repick::Resend);
    assert!(deck.apps.iter().all(|a| a.proto.is_none()));
    // Then encoded again from the finished frame, no point drawn again.
    assert!(deck.apps[0].advance().unwrap());
    let app = &deck.apps[0];
    assert!(app.job.is_none(), "not drawn again");
    assert!(matches!(app.proto, Some(Protocol::Kitty(_))));
    assert_eq!(app.vp, vp);
    // The tiles and thumbnails are made again at the next round.
    assert!(deck.tiles.is_empty() && deck.strip.made() == 0);
    assert_eq!(deck.cell, (8.0, 16.0));
}

#[test]
fn d_on_a_gain_map_opens_a_view_per_donor_each_drawing_its_deviation() {
    let dir = tempfile::tempdir().unwrap();
    let mut first = view();
    first.scene = crate::view::stage2::tests::scene_with_stage2(dir.path());
    first.scene.set_pick(crate::view::Pick::One("g1".into()));
    let mut deck = Deck::new(vec![first], Picker::halfblocks(), gallery(dir.path(), &[]));
    // Not a gain map yet: said so, nothing opened.
    assert!(deck.deck_key(key(KeyCode::Char('D'))));
    assert_eq!(deck.apps.len(), 1);
    deck.apps[0].scene.toggle_gain();
    assert!(deck.deck_key(key(KeyCode::Char('D'))));
    let donors: Vec<Option<String>> = deck
        .apps
        .iter()
        .map(|a| match &a.scene.pick {
            Some(crate::view::Pick::Gain { donor, .. }) => donor.as_deref().map(String::from),
            _ => panic!("every view shows the gain map"),
        })
        .collect();
    assert_eq!(
        donors,
        [None, Some("A".into()), Some("B".into()), Some("C".into())]
    );
    assert!(deck.grid);
}
