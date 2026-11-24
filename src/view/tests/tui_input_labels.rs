use super::*;
use crate::view::places::Places;

/// A view of the test scene, its run at `dir/r`, laid out on a kitty
/// terminal and drawn.
fn drawn_in(dir: &std::path::Path) -> App {
    let mut scene = crate::view::tests::scene();
    scene.data.prefix = dir.join("r").to_string_lossy().into_owned();
    scene.places = Places::load(&scene.data.prefix);
    let mut a = App::new(
        scene,
        super::super::attach_tests::kitty(),
        "r.senna.json".into(),
        "lupin".into(),
    );
    a.sidebar = false;
    a.layout(Rect::new(0, 0, 100, 40), STATUS_LINES);
    draw(&mut a);
    a
}

fn draw(a: &mut App) {
    if a.job.is_none() && a.base.is_none() {
        a.restart();
    }
    while a.job.is_some() {
        a.advance().unwrap();
    }
}

fn mouse(a: &mut App, kind: MouseEventKind, (column, row): (u16, u16)) {
    a.mouse(MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    });
}

/// The cell at the centre of group `g`'s label.
fn label_cell(a: &App, g: u32) -> (u16, u16) {
    let [x0, y0, x1, y1] = a.scene.label_rect(g).expect("label drawn");
    (
        a.map.x + (0.5 * (x0 + x1) / a.cell.0) as u16,
        a.map.y + (0.5 * (y0 + y1) / a.cell.1) as u16,
    )
}

fn key(a: &mut App, code: KeyCode, mods: KeyModifiers) {
    a.key(KeyEvent::new(code, mods));
}

#[test]
fn a_dragged_label_stays_where_it_is_put_after_a_reload() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = drawn_in(dir.path());
    let before = a.vp;
    let (col, row) = label_cell(&a, 1);
    mouse(&mut a, MouseEventKind::Down(MouseButton::Left), (col, row));
    mouse(
        &mut a,
        MouseEventKind::Drag(MouseButton::Left),
        (col + 3, row + 2),
    );
    // Let go off the map, on the status lines: the drag ends all the
    // same.
    mouse(&mut a, MouseEventKind::Up(MouseButton::Left), (0, 39));
    assert!(!a.pointer.dragging());
    assert_eq!(a.vp, before, "the label moved, not the map");
    let em = a.scene.label_em(a.cell.1);
    let want = (3.0 * a.cell.0 / em, 2.0 * a.cell.1 / em);
    let got = a.scene.group_offsets().get(&1).unwrap();
    assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4);
    // A drag of the map pans, and a later one starts afresh.
    mouse(&mut a, MouseEventKind::Down(MouseButton::Left), (2, 2));
    mouse(&mut a, MouseEventKind::Drag(MouseButton::Left), (4, 2));
    mouse(&mut a, MouseEventKind::Up(MouseButton::Left), (4, 2));
    assert_ne!(a.vp, before);
    // A label pressed and let go where it was is clicked, not moved.
    a.vp = before;
    a.restart();
    draw(&mut a);
    let at = label_cell(&a, 1);
    mouse(&mut a, MouseEventKind::Down(MouseButton::Left), at);
    mouse(&mut a, MouseEventKind::Up(MouseButton::Left), at);
    assert_eq!(a.scene.focus, Some(1));
    let got = a.scene.group_offsets().get(&1).unwrap();
    assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4);
    // Opened again, the run's label is where it was put.
    let b = drawn_in(dir.path());
    let got = b.scene.group_offsets().get(&1).unwrap();
    assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4);
    assert!(b
        .scene
        .placed_labels()
        .iter()
        .any(|p| p.key == 1 && p.moved));
}

#[test]
fn shift_e_moves_labels_by_key_and_esc_leaves() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = drawn_in(dir.path());
    let space = a.scene.space;
    key(&mut a, KeyCode::Char('E'), KeyModifiers::SHIFT);
    let g = a.label_edit.and_then(|e| e.at).expect("a label chosen");
    assert_eq!(a.scene.editing, Some(g));
    assert!(a.context() == help::Context::Labels);
    key(&mut a, KeyCode::Right, KeyModifiers::NONE);
    assert_eq!(a.scene.group_offsets().get(&g), Some((0.5, 0.0)));
    key(&mut a, KeyCode::Up, KeyModifiers::SHIFT);
    assert_eq!(a.scene.group_offsets().get(&g), Some((0.5, -2.0)));
    // Kept at once.
    let back = Places::load(&a.scene.data.prefix);
    let name = a.scene.levels()[g as usize].to_string();
    assert_eq!(
        back.layout(&a.scene.layout_id()).get(&name),
        Some((0.5, -2.0))
    );
    // Tab chooses a label here, and does not step the view.
    key(&mut a, KeyCode::Tab, KeyModifiers::NONE);
    assert_eq!(a.scene.space, space);
    assert!(a.label_edit.is_some());
    a.label_edit = Some(LabelEdit { at: Some(g) });
    key(&mut a, KeyCode::Backspace, KeyModifiers::NONE);
    assert_eq!(a.scene.group_offsets().get(&g), None);
    // Other map keys wait for the mode to end.
    let colour = a.scene.colour;
    key(&mut a, KeyCode::Char('c'), KeyModifiers::NONE);
    assert_eq!(a.scene.colour, colour);
    key(&mut a, KeyCode::Esc, KeyModifiers::NONE);
    assert!(a.label_edit.is_none() && a.scene.editing.is_none());
    // Outside the mode, tab steps the view as ever.
    key(&mut a, KeyCode::Tab, KeyModifiers::NONE);
    assert_ne!(a.scene.space, space);
}

#[test]
fn e_with_no_labels_to_move_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = drawn_in(dir.path());
    a.scene.show_labels = false;
    a.restart();
    draw(&mut a);
    key(&mut a, KeyCode::Char('E'), KeyModifiers::SHIFT);
    assert!(a.label_edit.is_none());
    assert!(a.message.as_deref().unwrap().contains("no labels"));
}
