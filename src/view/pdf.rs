//! PDF export: each page is the points as one raster image, with the text
//! over it kept as text (Helvetica, one of the standard fonts every reader
//! has), so labels stay sharp, selectable and editable.

use super::color::{self, Rgb};
use super::text::TextRun;
use image::RgbaImage;
use pdf_writer::types::{LineJoinStyle, TextRenderingMode};
use pdf_writer::{Content, Filter, Finish, Name, Pdf, Rect, Ref, Str};
use rayon::prelude::*;

/// One page: the raster, and the text to set over it (pixel coordinates).
pub(crate) struct Page {
    pub img: RgbaImage,
    pub text: Vec<TextRun>,
}

/// Point size of text whose raster line is `line` pixels tall, and where its
/// baseline sits below the top of that line, as fractions of it.
const EM_PER_LINE: f32 = 0.74;
const BASELINE: f32 = 0.78;

/// Write `pages` to `path`, `dpi` raster pixels to the inch.
pub(crate) fn write(pages: &[Page], dpi: f32, path: &std::path::Path) -> anyhow::Result<()> {
    std::fs::write(path, document(pages, dpi))?;
    Ok(())
}

fn document(pages: &[Page], dpi: f32) -> Vec<u8> {
    let (catalog, tree, regular, bold) = (Ref::new(1), Ref::new(2), Ref::new(3), Ref::new(4));
    // Page, content and image of page k.
    let ids = |k: usize| {
        let b = 5 + 3 * k as i32;
        (Ref::new(b), Ref::new(b + 1), Ref::new(b + 2))
    };
    let mut pdf = Pdf::new();
    pdf.catalog(catalog).pages(tree);
    pdf.pages(tree)
        .kids((0..pages.len()).map(|k| ids(k).0))
        .count(pages.len() as i32);
    for (id, name) in [(regular, &b"Helvetica"[..]), (bold, b"Helvetica-Bold")] {
        pdf.type1_font(id)
            .base_font(Name(name))
            .encoding_predefined(Name(b"WinAnsiEncoding"));
    }

    // The rasters are the bulk of the file: pack them side by side. Level 3
    // is several times faster than the default and barely larger on points.
    let packed: Vec<Vec<u8>> = pages
        .par_iter()
        .map(|page| {
            let mut rgb = Vec::with_capacity(page.img.as_raw().len() / 4 * 3);
            rgb.extend(page.img.pixels().flat_map(|p| [p[0], p[1], p[2]]));
            miniz_oxide::deflate::compress_to_vec_zlib(&rgb, 3)
        })
        .collect();
    let pt = 72.0 / dpi;
    for (k, page) in pages.iter().enumerate() {
        let (page_id, content_id, image_id) = ids(k);
        let (w, h) = (page.img.width(), page.img.height());
        let (wpt, hpt) = (w as f32 * pt, h as f32 * pt);

        let mut p = pdf.page(page_id);
        p.media_box(Rect::new(0.0, 0.0, wpt, hpt))
            .parent(tree)
            .contents(content_id);
        let mut res = p.resources();
        res.x_objects().pair(Name(b"Im"), image_id);
        res.fonts()
            .pair(Name(b"F1"), regular)
            .pair(Name(b"F2"), bold);
        res.finish();
        p.finish();

        let mut im = pdf.image_xobject(image_id, &packed[k]);
        im.filter(Filter::FlateDecode);
        im.width(w as i32).height(h as i32).bits_per_component(8);
        im.color_space().device_rgb();
        im.finish();

        let mut c = Content::new();
        c.save_state()
            .transform([wpt, 0.0, 0.0, hpt, 0.0, 0.0])
            .x_object(Name(b"Im"))
            .restore_state();
        c.begin_text().set_line_join(LineJoinStyle::RoundJoin);
        for run in &page.text {
            let size = run.line * EM_PER_LINE * pt;
            let bytes = win_ansi(&run.text);
            let width = helvetica_width(&bytes, run.bold) * size;
            let mut x = run.x * pt;
            if run.centred {
                x += 0.5 * (run.w * pt - width);
            }
            // Turned a quarter left, the baseline is a vertical line
            // `BASELINE` of a line in from the column's left.
            let matrix = if run.vertical {
                [
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    (run.x + BASELINE * run.line) * pt,
                    hpt - run.y * pt,
                ]
            } else {
                let y = hpt - (run.y + BASELINE * run.line) * pt;
                [1.0, 0.0, 0.0, 1.0, x, y]
            };
            let font = Name(if run.bold { b"F2" } else { b"F1" });
            c.set_font(font, size);
            // The halo: a stroke in the page colour under the letters.
            let [r, g, b] = srgb(run.bg);
            c.set_stroke_rgb(r, g, b)
                .set_line_width(0.22 * size)
                .set_text_rendering_mode(TextRenderingMode::Stroke)
                .set_text_matrix(matrix)
                .show(Str(&bytes));
            let [r, g, b] = srgb(run.ink);
            c.set_fill_rgb(r, g, b)
                .set_text_rendering_mode(TextRenderingMode::Fill)
                .set_text_matrix(matrix)
                .show(Str(&bytes));
        }
        c.end_text();
        pdf.stream(content_id, &c.finish());
    }
    pdf.finish()
}

fn srgb(c: Rgb) -> [f32; 3] {
    let enc = color::encoder();
    c.map(|v| f32::from(enc.encode(v)) / 255.0)
}

/// `s` in WinAnsi, the standard fonts' encoding: ASCII and Latin-1 as they
/// are, a few typographic marks moved, anything else a `?`.
fn win_ansi(s: &str) -> Vec<u8> {
    s.chars()
        .map(|ch| match ch {
            ' '..='~' | '\u{a0}'..='\u{ff}' => ch as u8,
            '•' => 0x95,
            '–' => 0x96,
            '—' => 0x97,
            '…' => 0x85,
            '‘' => 0x91,
            '’' => 0x92,
            '“' => 0x93,
            '”' => 0x94,
            _ => b'?',
        })
        .collect()
}

/// Width of WinAnsi `bytes` in Helvetica at size 1 (from the fonts' AFM).
fn helvetica_width(bytes: &[u8], bold: bool) -> f32 {
    let table = if bold { &HELVETICA_BOLD } else { &HELVETICA };
    let units: u32 = bytes
        .iter()
        .map(|&b| match b {
            32..=126 => u32::from(table[usize::from(b - 32)]),
            0x95 => 350,
            0x96 => 556,
            0x97 => 1000,
            0x85 => 1000,
            _ => 556,
        })
        .sum();
    units as f32 / 1000.0
}

/// Advance widths of ASCII 32..=126, in thousandths of the size.
#[rustfmt::skip]
const HELVETICA: [u16; 95] = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
];

#[rustfmt::skip]
const HELVETICA_BOLD: [u16; 95] = [
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_stays_text_and_the_raster_is_one_image() {
        let img = RgbaImage::from_pixel(40, 20, image::Rgba([10, 20, 30, 255]));
        let run = TextRun {
            text: "T cell · CD8".into(),
            x: 2.0,
            y: 3.0,
            w: 30.0,
            line: 10.0,
            bold: true,
            centred: true,
            vertical: false,
            ink: [0.1; 3],
            bg: [0.9; 3],
        };
        let bytes = document(
            &[Page {
                img,
                text: vec![run],
            }],
            144.0,
        );
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.starts_with("%PDF-"));
        assert!(s.contains("/Helvetica-Bold"));
        // "T cell · CD8" in WinAnsi, as a hex string: halo, then letters.
        assert_eq!(s.matches("<542063656C6C20B720434438> Tj").count(), 2);
        assert!(s.contains("/Subtype /Image"));
        // 40 × 20 px at 144 dpi is 20 × 10 pt.
        assert!(s.contains("/MediaBox [0 0 20 10]"));
    }

    #[test]
    fn widths_follow_the_font() {
        assert!((helvetica_width(b"ii", false) - 0.444).abs() < 1e-6);
        assert!(helvetica_width(b"W", true) > helvetica_width(b"i", true));
        assert_eq!(win_ansi("a·b✓"), vec![b'a', 0xb7, b'b', b'?']);
    }
}
