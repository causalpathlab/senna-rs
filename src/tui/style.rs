//! How `senna view` and `senna run` look: the page's colours, the styles
//! of text, menus and hints, and bordered popups over the screen.

use legume_tui::map::theme::Theme;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};

/// senna's page, the map's and the screen's: legume-tui's paper.
pub const THEME: Theme = Theme::PAPER;
/// Page background: a warm light gray.
pub const BACKGROUND: [u8; 3] = THEME.background;
/// Rules, and in the view points with no group or outside the focused one.
pub const MUTED: [u8; 3] = THEME.muted;
/// Menus, popups and the status line: near-black, to read at a glance.
pub const TEXT: [u8; 3] = THEME.text;
/// Key hints and other secondary lines beside `TEXT`: lighter, still clear.
pub const HINT: [u8; 3] = THEME.hint;

pub(crate) fn rgb(c: [u8; 3]) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

/// Bold text.
pub(crate) fn bold() -> Style {
    Style::default().add_modifier(ratatui::style::Modifier::BOLD)
}

/// The page's colours: ink on the map's background.
pub(crate) fn page() -> Style {
    Style::default().bg(rgb(BACKGROUND)).fg(rgb(TEXT))
}

/// Key hints and other secondary lines: a step lighter than the text.
pub(crate) fn hint() -> Style {
    Style::default().fg(rgb(HINT))
}

/// The line under a menu's cursor: a dark bar, so where you are is plain.
pub(crate) fn selected() -> Style {
    Style::default()
        .bg(rgb(TEXT))
        .fg(rgb(BACKGROUND))
        .patch(bold())
}

/// The first of `n` rows to show in a window `rows` tall so that `row`
/// sits in its middle where it can.
pub(crate) fn first_row(row: usize, rows: usize, n: usize) -> usize {
    row.saturating_sub(rows / 2).min(n.saturating_sub(rows))
}

/// `rows` scrolled to keep row `at` in sight within `room` rows; the first
/// and last rows shown say how many are cut off above and below.
pub(crate) fn scrolled(rows: Vec<Line<'static>>, at: usize, room: usize) -> Vec<Line<'static>> {
    let room = room.max(1);
    let first = first_row(at, room, rows.len());
    let below = rows.len().saturating_sub(first + room);
    let mut shown: Vec<Line> = rows.into_iter().skip(first).take(room).collect();
    let more =
        |n: usize, arrow: &str| Line::from(Span::styled(format!(" {arrow} {n} more"), hint()));
    if shown.len() > 1 {
        if first > 0 {
            shown[0] = more(first + 1, "↑");
        }
        if below > 0 {
            let last = shown.len() - 1;
            shown[last] = more(below + 1, "↓");
        }
    }
    shown
}

/// The rows of a page that fit `room` from `scroll`, its furthest scroll put
/// in `max` for the keys that move it, and where it stands for its footer:
/// `"more below · "`, `"end · "`, or `""` when it all fits.
pub(crate) fn page_window(
    body: Vec<Line<'static>>,
    room: usize,
    scroll: usize,
    max: &std::cell::Cell<usize>,
) -> (Vec<Line<'static>>, &'static str) {
    let room = room.max(1);
    let furthest = body.len().saturating_sub(room);
    max.set(furthest);
    let skip = scroll.min(furthest);
    let stands = if furthest == 0 {
        ""
    } else if skip < furthest {
        "more below · "
    } else {
        "end · "
    };
    (body.into_iter().skip(skip).take(room).collect(), stands)
}

/// Where a popup sits over its area.
#[derive(Clone, Copy)]
pub(crate) enum At {
    Top,
    Middle,
}

/// A bordered popup of `lines` over `area`, at most `max_w` columns wide
/// and as tall as its lines once wrapped. When they do not fit, the last
/// line (a popup's keys) keeps the bottom row, so it is never cut off.
pub(crate) fn popup(
    f: &mut ratatui::Frame,
    area: Rect,
    mut lines: Vec<Line<'_>>,
    max_w: u16,
    at: At,
    border: [u8; 3],
) {
    let w = max_w.min(area.width);
    let inner = usize::from(w.saturating_sub(2)).max(1);
    let rows: usize = lines
        .iter()
        .map(|l| {
            let text: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
            super::wrapped_rows(&text, inner)
        })
        .sum();
    let h = (rows.min(usize::from(u16::MAX)) as u16)
        .saturating_add(2)
        .min(area.height);
    let y = match at {
        At::Top => area.y + 1.min(area.height - h),
        At::Middle => area.y + (area.height - h) / 2,
    };
    let r = Rect::new(area.x + (area.width - w) / 2, y, w, h);
    f.render_widget(Clear, r);
    let block = Block::bordered()
        .border_style(Style::default().fg(rgb(border)))
        .style(page());
    let body = block.inner(r);
    f.render_widget(block, r);
    let wrap = ratatui::widgets::Wrap { trim: false };
    if rows > usize::from(body.height) && lines.len() > 1 && body.height > 1 {
        let last = lines.pop().unwrap_or_default();
        let top = Rect {
            height: body.height - 1,
            ..body
        };
        let bottom = Rect {
            y: body.y + body.height - 1,
            height: 1,
            ..body
        };
        f.render_widget(Paragraph::new(lines).wrap(wrap), top);
        f.render_widget(Paragraph::new(last), bottom);
    } else {
        f.render_widget(Paragraph::new(lines).wrap(wrap), body);
    }
}

/// A one-line notice at the top of `area`: lupin answered, a file saved.
pub(crate) fn toast(f: &mut ratatui::Frame, area: Rect, text: &str) {
    let line = Line::from(Span::styled(text.to_string(), bold())).centered();
    let w = text.chars().count() as u16 + 4;
    popup(f, area, vec![line], w, At::Top, TEXT);
}

#[cfg(test)]
#[path = "tests/style.rs"]
mod tests;
