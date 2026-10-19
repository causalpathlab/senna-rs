//! Colour for `senna view`: legume-tui's map colour on senna's page (its
//! paper theme and categorical palette).
//!
//! Every colour inside the renderer is linear RGB in `[0, 1]`; conversion to
//! 8-bit sRGB happens once per pixel at composite time, through senna's
//! table (`to_srgb_senna`), so the pixels stay those senna always drew.

pub use legume_tui::map::colour::{
    activity_ramp, highlight_ink, oklch, to_linear as linear_rgb, to_srgb_senna, Lin as Rgb,
};

/// The page, its text and points with no group: shared with `senna run`.
pub use crate::tui::style::{BACKGROUND, MUTED, TEXT, THEME};
/// Text drawn on the map (labels, titles).
pub const INK: [u8; 3] = THEME.ink;

/// Point colour of category `i` of `n`: the page's palette.
#[must_use]
pub fn category(i: usize, n: usize) -> Rgb {
    THEME.palette.colour(i, n)
}

/// Label ink for points drawn in `c`: the same hue, darker and a little
/// less saturated, so text stays legible on the light page.
#[must_use]
pub fn ink_for(c: Rgb) -> Rgb {
    THEME.ink_for(c)
}
