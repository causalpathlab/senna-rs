//! The strip of saved views beside the deck: opening, scrolling and
//! focusing it, and moving a saved view into a folder.

use super::*;
use legume_tui::gallery::strip::{self, Keymap};
use legume_tui::gallery::{Gallery, Policy};
use legume_tui::save::Numbering;
use legume_tui::Look;

/// Where the viewer keeps what it saved: `.senna-view/` in the directory
/// it runs from.
pub(in crate::view::tui) const SAVED_DIR: &str = ".senna-view";

/// A name offered for a save goes `-2`, `-3`, … past files already there.
pub(super) const NUMBERING: Numbering = Numbering::Suffix;

/// The saves logged in `dir`, kept as senna always has: a save whose file
/// is gone is dropped, and a log that cannot be read starts afresh.
pub(in crate::view::tui) fn open_saved(dir: &std::path::Path) -> Gallery {
    Gallery::open(dir, Policy::plain())
}

/// The strip's look: the page's, with its rule as muted as the map's.
pub(super) fn strip_look() -> Look {
    Look::from(CHROME).with_border(Style::default().fg(rgb(color::MUTED)))
}

impl Deck {
    /// The strip of saved figures, when it shows: on the left, above the
    /// status area (the grid's, or the view's at full width).
    pub(super) fn strip_area(&self, full: Rect) -> Option<Rect> {
        let status = if self.grid {
            STATUS_LINES
        } else {
            App::status_lines(full.width)
        };
        let body = Rect {
            height: full.height.saturating_sub(status),
            ..full
        };
        self.strip.area(body, &self.gallery)
    }

    /// Where the views or the grid go: beside the strip, when it shows.
    pub(super) fn content(&self, full: Rect) -> Rect {
        match self.strip_area(full) {
            Some(s) => Rect {
                x: full.x + s.width,
                width: full.width - s.width,
                ..full
            },
            None => full,
        }
    }

    pub(super) fn toggle_saved(&mut self) {
        self.strip.toggle();
        self.relayout();
    }

    /// Everything moves over, or the strip's pictures do: lay out and draw
    /// afresh, so no image lingers where it was.
    pub(super) fn relayout(&mut self) {
        self.area = Rect::default();
        self.drawn = None;
    }

    /// The strip scrolled from `top`: its pictures moved, so the screen is
    /// drawn afresh.
    pub(super) fn scrolled_from(&mut self, top: usize) {
        if self.strip.top() != top {
            self.drawn = None;
        }
    }

    /// Keys go to the strip, on the save at its top: it shows if hidden.
    pub(super) fn focus_saved(&mut self) {
        let shown = self.strip.is_shown();
        if !self.strip.focus(&self.gallery) {
            self.say("no figures saved here yet · s saves one".into());
            return;
        }
        if !shown {
            self.relayout();
        }
    }

    /// Say `msg` where the screen shows it: a toast over the grid, else the
    /// view's own.
    pub(super) fn say(&mut self, msg: String) {
        if self.grid {
            self.toast = Some((msg, std::time::Instant::now() + TOAST_FOR));
        } else {
            self.apps[self.at].pop(msg);
        }
    }

    /// A key while the strip has them: choose, move or remove a save.
    pub(super) fn strip_key(&mut self, full: Rect, k: KeyEvent) {
        let Some(area) = self.strip_area(full) else {
            self.strip.leave();
            return;
        };
        let top = self.strip.top();
        match self.strip.key(area, k, &self.gallery, &Keymap::default()) {
            strip::Action::Quit => self.quit = true,
            strip::Action::Relayout => self.relayout(),
            strip::Action::Move(at) => {
                let from = self.gallery.entries()[at].path.clone();
                self.moving = Some(MoveDialog {
                    to: crate::tui::shown(&from),
                    from,
                    status: None,
                });
            }
            strip::Action::Remove { at, delete } => {
                let e = self.gallery.entries()[at].clone();
                let msg = match self.gallery.remove(&e.path, delete) {
                    Ok(()) if delete => format!("deleted {}", e.path.display()),
                    Ok(()) => format!(
                        "{} taken off the list · the PDF stays (X deletes it too)",
                        e.name()
                    ),
                    Err(e) => format!("not removed: {e}"),
                };
                self.say(msg);
                // The views are laid out afresh only when the strip goes
                // with the last save; else the strip alone is drawn again.
                match self.strip.after_change(area, &self.gallery) {
                    strip::Action::Relayout => self.relayout(),
                    _ => self.drawn = None,
                }
            }
            _ => {}
        }
        self.scrolled_from(top);
    }

    /// A key in the move popup: type where to, enter moves.
    pub(super) fn move_key(&mut self, k: KeyEvent) {
        let Some(m) = self.moving.as_mut() else {
            return;
        };
        m.status = None;
        if crate::tui::edit_line(&mut m.to, &k) {
            return;
        }
        match k.code {
            KeyCode::Esc => self.moving = None,
            KeyCode::Enter => {
                let to = crate::tui::home(m.to.trim());
                if to.is_empty() {
                    m.status = Some("type where to".into());
                    return;
                }
                let to = std::path::PathBuf::from(to);
                let from = m.from.clone();
                match self.gallery.relocate(&from, &to) {
                    Ok(now) => {
                        self.moving = None;
                        self.say(format!("moved to {}", crate::tui::shown(&now)));
                        self.drawn = None;
                    }
                    Err(e) => {
                        if let Some(m) = self.moving.as_mut() {
                            m.status = Some(format!("not moved: {e}"));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Thumbnails for the saves the strip shows, made once each; none
    /// while it is hidden.
    pub(super) fn prepare_thumbs(&mut self, full: Rect) {
        if let Some(area) = self.strip_area(full) {
            self.strip.prepare(area, &self.gallery);
        }
    }
}

/// Moving a saved PDF: from where, to where as typed, and what went wrong.
pub(super) struct MoveDialog {
    pub(super) from: std::path::PathBuf,
    pub(super) to: String,
    pub(super) status: Option<String>,
}

pub(super) fn draw_move(f: &mut ratatui::Frame, m: &MoveDialog) {
    let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
    let lines = vec![
        Line::from(Span::styled(
            format!(" Move {}", files::name(&m.from)),
            bold,
        )),
        Line::from(Span::styled(
            format!(" now {}", crate::tui::shown(&m.from)),
            hint(),
        )),
        Line::from(""),
        Line::from(Span::styled(format!(" to  {}▏", m.to), selected())),
        Line::from(""),
        match &m.status {
            Some(s) => Line::from(Span::styled(format!(" {s}"), bold)),
            None => Line::from(Span::styled(
                " a folder (ending in /) keeps the name · folders are made as needed",
                hint(),
            )),
        },
        Line::from(" type the path (ctrl+u clears)   enter move   esc cancel"),
    ];
    CHROME.popup(f, f.area(), lines, 76, At::Middle);
}
