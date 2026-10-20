use super::*;
use serde_json::json;
use std::sync::mpsc::{channel, Sender};

/// A view on the test scene that follows an agent, and the agent's end.
fn watching() -> (App, Sender<Heard>) {
    let mut app = App::new(
        crate::view::tests::scene(),
        Picker::halfblocks(),
        "r.senna.json".into(),
        "lupin".into(),
    );
    let (tx, rx) = channel();
    app.agent = Some(Agent::new(rx));
    app.layout(Rect::new(0, 0, 80, 24), 1);
    (app, tx)
}

fn view(focus: &str, bbox: Option<[f32; 4]>) -> Heard {
    Heard::View(json!({
        "run": "r.senna.json",
        "round": null,
        "map": "phate",
        "space": "cells",
        "colour_by": "cluster",
        "focus": focus,
        "feature": "",
        "observed": false,
        "bbox": bbox,
    }))
}

#[test]
fn a_view_the_agent_shows_is_shown_with_its_window() {
    let (mut app, tx) = watching();
    tx.send(view("C1", Some([0.0, 0.0, 2.0, 1.0]))).unwrap();
    assert!(app.hear_agent());
    assert_eq!(app.scene.current().method, "phate");
    assert_eq!(app.scene.focused_name().as_deref(), Some("C1"));
    let w = app.vp.unwrap().window();
    // The agent's window fits inside the map's shape, centred on it.
    assert!(
        (w[0] + w[2] - 2.0).abs() < 1e-4 && (w[1] + w[3] - 1.0).abs() < 1e-4,
        "{w:?}"
    );
    assert!(
        w[0] <= 0.0 && w[2] >= 2.0 && w[1] <= 0.0 && w[3] >= 1.0,
        "{w:?}"
    );
}

#[test]
fn changing_the_view_by_hand_pauses_following_until_w() {
    let (mut app, tx) = watching();
    tx.send(view("C1", None)).unwrap();
    app.hear_agent();
    let key = |c| Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
    app.handle(key('c'));
    assert!(!app.agent.as_ref().unwrap().following);
    let mine = app.scene.colour;

    tx.send(view("C0", None)).unwrap();
    app.hear_agent();
    assert_eq!(app.scene.colour, mine, "paused: the agent's view waits");

    app.handle(key('W'));
    assert!(app.agent.as_ref().unwrap().following);
    assert_eq!(app.scene.focused_name().as_deref(), Some("C0"));
}

#[test]
fn the_pane_lists_the_newest_activity_and_says_when_the_agent_ended() {
    let (mut app, tx) = watching();
    for k in 0..30 {
        tx.send(Heard::Line(format!("line {k}"))).unwrap();
    }
    drop(tx);
    app.hear_agent();
    let rows = app.agent.as_ref().unwrap().rows(3);
    assert_eq!(
        rows,
        [
            "agent · ended",
            "line 28",
            "line 29",
            "the agent's server ended"
        ]
    );
    assert!(app.side_lines().unwrap()[0].starts_with("agent"));
}
