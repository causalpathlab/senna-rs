//! Anti-aliased text drawn straight into the raster, so labels sit on the map
//! under every graphics protocol, block-character fallback included.
//!
//! Glyphs come pre-rasterized from Noto Sans Mono. Each label gets a halo in
//! the page colour, which keeps it legible over dense points without a box.

use super::color::Rgb;
use noto_sans_mono_bitmap::{get_raster, get_raster_width, FontWeight, RasterHeight};

#[derive(Clone, Copy)]
pub struct Font {
    pub height: RasterHeight,
    pub weight: FontWeight,
}

impl Font {
    /// A label size matched to the terminal's cell height.
    #[must_use]
    pub fn for_cell_height(px: f32, bold: bool) -> Self {
        let height = if px >= 30.0 {
            RasterHeight::Size24
        } else if px >= 22.0 {
            RasterHeight::Size20
        } else {
            RasterHeight::Size16
        };
        let weight = if bold {
            FontWeight::Bold
        } else {
            FontWeight::Regular
        };
        Self { height, weight }
    }

    #[must_use]
    pub fn advance(&self) -> usize {
        get_raster_width(self.weight, self.height)
    }

    #[must_use]
    pub fn line_height(&self) -> usize {
        self.height.val()
    }

    /// Pixel width of `s`.
    #[must_use]
    pub fn width(&self, s: &str) -> usize {
        s.chars().count() * self.advance()
    }
}

/// A linear-light canvas the text is blended into.
pub trait Canvas {
    fn size(&self) -> (usize, usize);
    /// Blend `c` over pixel `(x, y)` at opacity `a`.
    fn blend(&mut self, x: usize, y: usize, c: Rgb, a: f32);
}

/// Draw `s` with its top-left corner at `(x, y)`, over a `halo` of the page
/// colour `bg`.
pub fn draw<C: Canvas>(canvas: &mut C, font: Font, s: &str, x: i32, y: i32, ink: Rgb, bg: Rgb) {
    let (w, h) = canvas.size();
    let adv = font.advance() as i32;
    let glyphs: Vec<_> = s
        .chars()
        .map(|ch| {
            get_raster(ch, font.weight, font.height)
                .or_else(|| get_raster('?', font.weight, font.height))
        })
        .collect();

    // Halo: the glyph coverage dilated by 2 px, laid down in the page colour
    // first so the ink never has to fight the points beneath it.
    const HALO: i32 = 2;
    for pass in 0..2 {
        for (k, g) in glyphs.iter().enumerate() {
            let Some(g) = g else { continue };
            let gx = x + k as i32 * adv;
            for (row, line) in g.raster().iter().enumerate() {
                for (col, &v) in line.iter().enumerate() {
                    if v < 24 {
                        continue;
                    }
                    let a = f32::from(v) / 255.0;
                    let (px, py) = (gx + col as i32, y + row as i32);
                    if pass == 0 {
                        for dy in -HALO..=HALO {
                            for dx in -HALO..=HALO {
                                if dx * dx + dy * dy > HALO * HALO + 1 {
                                    continue;
                                }
                                let (qx, qy) = (px + dx, py + dy);
                                if qx >= 0 && qy >= 0 && (qx as usize) < w && (qy as usize) < h {
                                    canvas.blend(qx as usize, qy as usize, bg, 0.8 * a);
                                }
                            }
                        }
                    } else if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                        canvas.blend(px as usize, py as usize, ink, a);
                    }
                }
            }
        }
    }
}
