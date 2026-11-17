//! How `senna view` and `senna run` look: the page's colours, the styles
//! of text, menus and hints, and bordered popups over the screen.

use legume_tui::chrome::Chrome;
use legume_tui::map::theme::Theme;
use ratatui::style::Style;

/// senna's page, the map's and the screen's: legume-tui's paper.
pub const THEME: Theme = Theme::PAPER;
/// Page background: a warm light gray.
pub const BACKGROUND: [u8; 3] = THEME.background;
/// Rules, and in the view points with no group or outside the focused one.
pub const MUTED: [u8; 3] = THEME.muted;
/// Menus, popups and the status line: near-black, to read at a glance.
pub const TEXT: [u8; 3] = THEME.text;

/// The screen's styles and popups: legume-tui's, on senna's page.
pub(crate) const CHROME: Chrome = Chrome::themed(THEME);

pub(crate) use legume_tui::chrome::{first_row, page_window, rgb, At};

/// Bold text.
pub(crate) fn bold() -> Style {
    CHROME.bold()
}

/// The page's colours: ink on the map's background.
pub(crate) fn page() -> Style {
    CHROME.page()
}

/// Key hints and other secondary lines: a step lighter than the text.
pub(crate) fn hint() -> Style {
    CHROME.hint()
}

/// The line under a menu's cursor: a dark bar, so where you are is plain.
pub(crate) fn selected() -> Style {
    CHROME.selected()
}

#[cfg(test)]
#[path = "tests/style.rs"]
mod tests;
