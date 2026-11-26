//! `view_export`: the current view written as a PNG and/or a PDF, either as
//! the viewer draws it or, with any panel argument, as a figure panel
//! (`crate::view::panel`): an exact size in inches, points and labels in
//! points, a white page or none, names on the map, in a legend or alone.

use super::absolute;
use super::view::{Open, MAX_SIDE};
use crate::view::{deck, panel};
use anyhow::{anyhow, ensure, Context, Result};
use image::codecs::png::CompressionType;
use legume_mcp::{Content, ToolSpec};
use legume_tui::export::{Background, Labels, PageSize, Panel};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;

/// Width and resolution of an exported figure.
const EXPORT_WIDTH: usize = 2400;
const EXPORT_DPI: u32 = 300;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExportArgs {
    png: Option<PathBuf>,
    pdf: Option<PathBuf>,
    svg: Option<PathBuf>,
    width: Option<usize>,
    #[serde(default)]
    overwrite: bool,
    width_in: Option<f32>,
    height_in: Option<f32>,
    dpi: Option<u32>,
    point_pt: Option<f32>,
    point_alpha: Option<f32>,
    background: Option<Background>,
    font_pt: Option<f32>,
    labels: Option<Labels>,
    legend: Option<PathBuf>,
    /// Groups by name or cluster id, as `view_map`'s focus takes them.
    #[serde(default)]
    hide: Vec<Value>,
}

impl ExportArgs {
    /// Whether any panel argument is given.
    fn panel(&self) -> bool {
        self.width_in.is_some()
            || self.height_in.is_some()
            || self.dpi.is_some()
            || self.point_pt.is_some()
            || self.point_alpha.is_some()
            || self.background.is_some()
            || self.font_pt.is_some()
            || self.labels.is_some()
            || self.legend.is_some()
            || !self.hide.is_empty()
    }
}

impl Open {
    pub(super) fn export(&mut self, a: ExportArgs) -> Result<Vec<Content>> {
        ensure!(
            a.png.is_some() || a.pdf.is_some() || a.svg.is_some() || a.legend.is_some(),
            "pass png, pdf, svg, legend or more than one"
        );
        // Figures only: no other file is written, overwrite or not.
        let named = [(&a.png, "png"), (&a.pdf, "pdf"), (&a.svg, "svg")];
        for (path, ext) in named.iter().filter_map(|(p, e)| Some((p.as_ref()?, *e))) {
            ensure!(
                path.extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case(ext)),
                "{} is not a .{ext} file",
                path.display()
            );
        }
        if let Some(path) = &a.legend {
            ensure!(
                panel::Kind::of(path).is_some(),
                "legend {} is not a .png, .pdf or .svg file",
                path.display()
            );
        }
        for path in a.png.iter().chain(&a.pdf).chain(&a.svg).chain(&a.legend) {
            ensure!(
                a.overwrite || !path.exists(),
                "{} exists; pass overwrite to replace it",
                path.display()
            );
        }
        if a.panel() {
            return self.export_panel(&a);
        }
        let vp = self.viewport(super::view::width(a.width, EXPORT_WIDTH)?);
        legume_tui::export::fits((vp.w, vp.h), EXPORT_DPI as f32).map_err(|e| anyhow!(e))?;
        // Text as large on the page as `senna view --pdf` sets it.
        let size = PageSize {
            width_in: vp.w as f32 / EXPORT_DPI as f32,
            dpi: EXPORT_DPI,
        };
        let cell_px = size.cell_px();
        let mut wrote = Vec::new();
        if let Some(path) = &a.png {
            std::fs::write(path, self.draw(vp, cell_px, CompressionType::Default)?)
                .with_context(|| format!("writing {}", path.display()))?;
            wrote.push(absolute(path));
        }
        if a.pdf.is_some() || a.svg.is_some() {
            let pages = deck::view_pages(&[(&self.scene, vp, cell_px)]);
            if let Some(path) = &a.pdf {
                legume_tui::export::write_pdf(&pages, EXPORT_DPI as f32, path)
                    .with_context(|| format!("writing {}", path.display()))?;
                wrote.push(absolute(path));
            }
            if let Some(path) = &a.svg {
                let face = legume_tui::export::Face::Helvetica;
                legume_tui::export::write_svg(&pages[0], EXPORT_DPI as f32, face, path)
                    .with_context(|| format!("writing {}", path.display()))?;
                wrote.push(absolute(path));
            }
        }
        Ok(vec![Content::Json(
            json!({"wrote": wrote, "image": [vp.w, vp.h]}),
        )])
    }

    /// The view as the figure panel `a` asks for: `width_in` (else the
    /// pixel width at the panel's dpi) by `height_in` (else the window's
    /// shape).
    fn export_panel(&self, a: &ExportArgs) -> Result<Vec<Content>> {
        let dpi = a.dpi.unwrap_or(Panel::DPI);
        let width_in = match a.width_in {
            Some(w) => w,
            None => super::view::width(a.width, EXPORT_WIDTH)? as f32 / dpi as f32,
        };
        let bounds = self.window.unwrap_or(self.scene.fit_bounds());
        let height_in = a
            .height_in
            .unwrap_or_else(|| panel::shaped_height(width_in, bounds));
        let mut p = Panel::new(width_in, height_in).with_dpi(dpi);
        p.point_pt = a.point_pt;
        p.point_alpha = a.point_alpha;
        p.font_pt = a.font_pt;
        p.background = a.background.unwrap_or_default();
        p.labels = a.labels.unwrap_or_default();
        p.hidden = a
            .hide
            .iter()
            .map(super::view::group_name)
            .collect::<Result<_>>()?;
        let files = panel::Files {
            png: a.png.as_deref(),
            pdf: a.pdf.as_deref(),
            svg: a.svg.as_deref(),
            legend: a.legend.as_deref(),
        };
        let saved = panel::save(&self.scene, &p, self.window, &files)?;
        let wrote: Vec<PathBuf> = saved.wrote.iter().map(|p| absolute(p)).collect();
        let mut out = json!({
            "wrote": wrote,
            "image": [saved.px.0, saved.px.1],
            "inches": [p.width_in, p.height_in],
            "dpi": p.dpi,
            "font_pt": p.font_pt(),
        });
        if let Some(note) = saved.note {
            out["note"] = json!(note);
        }
        Ok(vec![Content::Json(out)])
    }
}

pub(super) fn spec() -> ToolSpec {
    let num = |d: &str| json!({"type": "number", "exclusiveMinimum": 0, "description": d});
    ToolSpec::new(
        "view_export",
        "Write the current view as a PNG, a one-page PDF and/or an SVG, text sized for print; \
         in a PDF or SVG the points are one raster image and every label and key's text \
         stays text. Colours, scale, range, palette, draw order, missing cells and the key \
         come from view_map's colour_scale; focus from its focus. \
         Any of width_in, height_in, dpi, point_pt, point_alpha, background, font_pt, \
         labels, legend or hide makes it a figure panel instead: exactly width_in × \
         height_in inches with the map fitted inside (aspect kept, centred), no title, \
         white or no background, dots and every label sized in points (PDF text in \
         Arial). Refuses to replace a file unless overwrite.",
        json!({
            "type": "object",
            "properties": {
                "png": {"type": "string", "description": "PNG path"},
                "pdf": {"type": "string", "description": "PDF path"},
                "svg": {"type": "string", "description": "SVG path: the points one embedded image, text as SVG text"},
                "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": format!("Image width in pixels [default: {EXPORT_WIDTH}]; a panel's width_in, when that is not given")},
                "overwrite": {"type": "boolean", "description": "Replace files that exist"},
                "width_in": num("Panel width in inches (e.g. 1.8)"),
                "height_in": num("Panel height in inches [default: the view's shape]"),
                "dpi": {"type": "integer", "minimum": 1, "description": format!("Panel raster resolution [default: {}]", Panel::DPI)},
                "point_pt": num("Point diameter in points [default: by how crowded the map is]"),
                "point_alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "description": "Point opacity [default: as styled]"},
                "background": {"type": "string", "enum": ["white", "none", "theme"], "description": "Under the points: white, none (transparent PNG, no PDF fill) or the viewer's page [default: white]"},
                "font_pt": num("Size of every label in points [default: fit to the panel, 7.2 at 1.8 in]"),
                "labels": {"type": "string", "enum": ["map", "legend", "none"], "description": "Group names on the map (with a halo), in a legend at the right, or none [default: map]; none also leaves out a number's key: the panel alone"},
                "legend": {"type": "string", "description": "Also write the legend alone (a number's key, with its ticks, when cells are coloured by one), a key shared by a row of panels (.png, .pdf or .svg), the panel's height"},
                "hide": {"type": "array", "items": {"type": ["string", "integer"]}, "description": "Groups of the grouping shown left out of the panel, by name or cluster id (C10, 10, K10); the map refits to the rest"},
            },
            "additionalProperties": false,
        }),
    )
}
