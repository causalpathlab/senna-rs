//! How `senna view` and `senna run` look: the page's colours, the styles
//! of text, menus and hints, and bordered popups over the screen.

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};

/// Page background: a warm light gray.
pub const BACKGROUND: [u8; 3] = [246, 245, 242];
/// Rules, and in the view points with no group or outside the focused one.
pub const MUTED: [u8; 3] = [200, 199, 196];
/// Menus, popups and the status line: near-black, to read at a glance.
pub const TEXT: [u8; 3] = [30, 30, 32];
/// Key hints and other secondary lines beside `TEXT`: lighter, still clear.
pub const HINT: [u8; 3] = [104, 104, 104];

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

/// Where a popup sits over its area.
#[derive(Clone, Copy)]
pub(crate) enum At {
    Top,
    Middle,
}

/// A bordered popup of `lines` over `area`, at most `max_w` columns wide
/// and as tall as its lines.
pub(crate) fn popup(
    f: &mut ratatui::Frame,
    area: Rect,
    lines: Vec<Line<'_>>,
    max_w: u16,
    at: At,
    border: [u8; 3],
) {
    let w = max_w.min(area.width);
    let h = (lines.len() as u16 + 2).min(area.height);
    let y = match at {
        At::Top => area.y + 1.min(area.height - h),
        At::Middle => area.y + (area.height - h) / 2,
    };
    let r = Rect::new(area.x + (area.width - w) / 2, y, w, h);
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(lines)
            .wrap(ratatui::widgets::Wrap { trim: false })
            .block(Block::bordered().border_style(Style::default().fg(rgb(border))))
            .style(page()),
        r,
    );
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
