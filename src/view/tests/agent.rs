use super::*;
use legume_mcp::watch::Meta;
use serde_json::json;
use std::io::Write;

/// A server's watch folder, as legume-mcp keeps it, with this process as
/// the server so it counts as running.
struct Server(tempfile::TempDir);

impl Server {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let meta = Meta {
            server: "senna".into(),
            version: "0".into(),
            pid: std::process::id(),
            project: None,
            started: String::new(),
        };
        std::fs::write(
            dir.path().join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.path().join("feed.jsonl"), "").unwrap();
        Self(dir)
    }

    fn say(&self, body: Body) {
        let event = watch::Event {
            t: "2026-10-20T14:03:07-07:00".into(),
            body,
        };
        let mut feed = std::fs::OpenOptions::new()
            .append(true)
            .open(self.0.path().join("feed.jsonl"))
            .unwrap();
        writeln!(feed, "{}", serde_json::to_string(&event).unwrap()).unwrap();
    }

    /// The agent shows `view`: its state file and its feed's event.
    fn show(&self, view: Value) {
        std::fs::write(self.0.path().join("state.json"), view.to_string()).unwrap();
        self.say(Body::State {
            id: "1".into(),
            state: view,
        });
    }
}

/// A view on the test scene that follows the agent of `server`.
fn watching(server: &Server) -> App {
    let mut app = App::new(
        crate::view::tests::scene(),
        Picker::halfblocks(),
        "r.senna.json".into(),
        "lupin".into(),
    );
    app.agent = Some(Agent::open(server.0.path()).unwrap());
    app.layout(Rect::new(0, 0, 80, 24), 1);
    app
}

fn view(colour_by: &str, focus: &str, bbox: Option<[f32; 4]>) -> Value {
    json!({
        "run": "r.senna.json",
        "round": null,
        "map": "phate",
        "space": "cells",
        "colour_by": colour_by,
        "focus": focus,
        "feature": "",
        "observed": false,
        "bbox": bbox,
        "added_by_a_later_senna": true,
    })
}

/// A tick of the view: the agent heard, the map framed.
fn tick(app: &mut App) -> bool {
    let heard = app.hear_agent();
    app.layout(Rect::new(0, 0, 80, 24), 1);
    heard
}

fn key(c: char) -> Event {
    Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
}

#[test]
fn the_agent_s_view_is_shown_with_its_window_through_a_resize() {
    let server = Server::new();
    server.show(view("cluster", "C1", Some([0.0, 0.0, 2.0, 1.0])));
    let mut app = watching(&server);
    assert!(tick(&mut app));
    assert_eq!(app.scene.current().method, "phate");
    assert_eq!(app.scene.focused_name().as_deref(), Some("C1"));
    let framed = |app: &App| {
        let w = app.vp.unwrap().window();
        (w[0] + w[2] - 2.0).abs() < 1e-4 && w[0] <= 0.0 && w[2] >= 2.0 && w[1] <= 0.0 && w[3] >= 1.0
    };
    assert!(framed(&app), "{:?}", app.vp.unwrap().window());
    // A resize frames the agent's window again, not the whole map.
    app.handle(Event::Resize(100, 30));
    app.layout(Rect::new(0, 0, 100, 30), 1);
    assert!(framed(&app), "{:?}", app.vp.unwrap().window());
    assert!(app.agent.as_ref().unwrap().following);
}

#[test]
fn changing_the_view_by_hand_pauses_following_and_w_resumes_it() {
    let server = Server::new();
    server.show(view("cluster", "C1", None));
    let mut app = watching(&server);
    tick(&mut app);
    // W while following changes nothing and keeps following.
    app.handle(key('W'));
    tick(&mut app);
    assert!(app.agent.as_ref().unwrap().following);

    app.handle(key('c'));
    assert!(!app.agent.as_ref().unwrap().following);
    let mine = app.scene.colour;
    server.show(view("cluster", "C0", None));
    tick(&mut app);
    assert_eq!(app.scene.colour, mine, "paused: the agent's view waits");

    app.handle(key('W'));
    tick(&mut app);
    assert!(app.agent.as_ref().unwrap().following);
    assert_eq!(app.scene.focused_name().as_deref(), Some("C0"));
}

#[test]
fn a_menu_open_holds_the_agent_s_view_until_it_closes() {
    let server = Server::new();
    server.show(view("cluster", "C1", None));
    let mut app = watching(&server);
    tick(&mut app);
    app.handle(key('e'));
    assert!(
        app.menu.is_some(),
        "e opens the style menu on the focused group"
    );
    // The agent now shows a grouping with fewer groups: the menu's row
    // would point past them.
    server.show(view("annotation", "", None));
    tick(&mut app);
    assert_eq!(
        app.scene.colour.map(|i| app.scene.data.labels[i].title()),
        Some("cluster")
    );
    app.handle(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
    tick(&mut app);
    assert_eq!(
        app.scene.colour.map(|i| app.scene.data.labels[i].title()),
        Some("annotation")
    );
}

#[test]
fn the_pane_shows_the_history_and_the_newest_lines_and_the_end() {
    let server = Server::new();
    for k in 0..20 {
        server.say(Body::Progress {
            id: "2".into(),
            message: format!("line {k}"),
        });
    }
    let mut app = watching(&server);
    let rows = app.agent.as_ref().unwrap().rows();
    assert_eq!(rows[0], "agent · following");
    assert_eq!(rows.len(), 1 + SHOWN);
    assert_eq!(rows.last().unwrap(), "14:03:07   line 19");
    server.say(Body::Ended { ok: true });
    tick(&mut app);
    assert_eq!(app.agent.as_ref().unwrap().rows()[0], "agent · ended");
    assert!(app.side_lines().unwrap()[0].starts_with("agent"));
}

#[test]
fn feed_events_read_as_one_short_line_each_with_their_time() {
    let t = "2026-10-20T14:03:07-07:00";
    let call = Body::Call {
        id: "3".into(),
        tool: "view_map".into(),
        args: json!({"focus": "C1"}),
    };
    assert_eq!(
        line(t, call).unwrap(),
        r#"14:03:07 → view_map {"focus":"C1"}"#
    );
    let done = Body::Result {
        id: "3".into(),
        ok: false,
        ms: 1500,
        text: "\n no group C9\nin cluster".into(),
    };
    assert_eq!(
        line(t, done).unwrap(),
        "14:03:07 ✗ 1.5s no group C9 in cluster"
    );
    let drawn = Body::Result {
        id: "4".into(),
        ok: true,
        ms: 100,
        text: "[image/png, 120639 bytes]\n{\n  \"map\": \"umap\"".into(),
    };
    assert_eq!(
        line(t, drawn).unwrap(),
        r#"14:03:07 ✓ 0.1s { "map": "umap""#
    );
    let ran = Body::Record {
        line: "if [ ! -e o.clusters.parquet ]; then\n  senna clustering\nfi\n".into(),
        command: Some("senna clustering".into()),
    };
    assert_eq!(line(t, ran).unwrap(), "14:03:07 $ senna clustering");
    let failed = Body::Record {
        line: "# did not succeed\n".into(),
        command: None,
    };
    assert_eq!(line(t, failed), None);
    let long = Body::Progress {
        id: "3".into(),
        message: "x".repeat(500),
    };
    assert_eq!(line(t, long).unwrap().chars().count(), LINE);
}
