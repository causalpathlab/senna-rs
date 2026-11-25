//! `view_stage2`: the run's stage 2 as numbers, and its donor tiles, for an
//! agent; the maps and charts are `view_map`'s (`stage2`, `chart`).

use super::view::{png, Open, CELL_PX, MAX_SIDE, WIDTH};
use crate::view::deck;
use crate::view::Pick;
use anyhow::{anyhow, ensure, Result};
use image::codecs::png::CompressionType;
use legume_mcp::{Content, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Value};

/// Rows a list gives without `top`.
const TOP: usize = 30;

/// Donor tiles drawn at most, as `D` draws them.
const MAX_TILES: usize = 12;

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum What {
    Genes,
    Gene,
    Volcano,
    Tiles,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Stage2Args {
    what: What,
    gene: Option<String>,
    donor: Option<String>,
    top: Option<usize>,
    width: Option<usize>,
}

impl Open {
    pub(super) fn stage2(&mut self, a: Stage2Args) -> Result<Vec<Content>> {
        let scene = &self.scene;
        ensure!(scene.has_stage2(), crate::view::stage2::NO_STAGE2);
        let top = a.top.unwrap_or(TOP).max(1);
        let gene = || {
            a.gene
                .clone()
                .or_else(|| scene.pick.as_ref().and_then(Pick::feature).map(Into::into))
                .ok_or_else(|| anyhow!("which gene: pass gene (or show one with view_map)"))
        };
        let json = match a.what {
            What::Genes => scene.stage2_genes(a.donor.as_deref(), top),
            What::Gene => scene.stage2_gene(&gene()?),
            What::Volcano => scene.stage2_volcano(a.donor.as_deref(), top),
            What::Tiles => return self.tiles(&gene()?, a.width),
        };
        Ok(vec![Content::Json(json.map_err(|e| anyhow!(e))?)])
    }

    /// The gain map of `gene` as each donor's deviation drawn at every
    /// cell's state, a tile per donor (the first [`MAX_TILES`]) in one image.
    fn tiles(&self, gene: &str, width: Option<usize>) -> Result<Vec<Content>> {
        let width = width.unwrap_or(WIDTH);
        ensure!(
            (1..=MAX_SIDE).contains(&width),
            "width must be 1 to {MAX_SIDE}"
        );
        let donors = self.scene.stage2_donors().map_err(|e| anyhow!(e))?.to_vec();
        let mut scenes = Vec::new();
        for donor in donors.iter().take(MAX_TILES) {
            let mut s = self.scene.duplicate();
            s.chart = None;
            s.set_pick(Pick::Gain {
                gene: gene.into(),
                donor: Some(donor.clone()),
            });
            if s.current_shown().is_none() {
                return Err(anyhow!(s
                    .note
                    .take()
                    .unwrap_or_else(|| "nothing to draw".into())));
            }
            scenes.push(s);
        }
        let refs: Vec<&crate::view::Scene> = scenes.iter().collect();
        let titles: Vec<String> = donors
            .iter()
            .take(MAX_TILES)
            .map(ToString::to_string)
            .collect();
        let height = (width * 3 / 4).min(MAX_SIDE);
        let page = deck::render_grid(
            &refs,
            &vec![None; refs.len()],
            &titles,
            (width, height),
            CELL_PX,
            false,
        );
        let mut out = vec![Content::Png(png(page.img, CompressionType::Best)?)];
        let mut said = json!({"gene": gene, "donors": titles});
        if donors.len() > MAX_TILES {
            said["note"] = Value::from(format!(
                "the first {MAX_TILES} of {} donors; view_map stage2 gain with donor shows any",
                donors.len()
            ));
        }
        out.push(Content::Json(said));
        Ok(out)
    }
}

pub(super) fn spec() -> ToolSpec {
    ToolSpec::new(
        "view_stage2",
        "The open run's stage 2 (senna donor-deviation; view_open says whether it has one) as \
         numbers, over the grouping view_map shows (colour_by): genes ranks the genes by their \
         largest batch-jackknife |z| over each donor's cells in each group (by |log gain| with \
         one batch), each with the donor and group where it departs most; gene gives one \
         gene's log gain and z for each donor in each group, at the mean state of the donor's \
         cells there (a donor needs 10 cells in a group); volcano gives the strongest (gene, \
         donor) pairs at each donor's state in the focused group (view_map focus), else the \
         whole view; tiles draws the gene's gain map once per donor, that donor's deviation at \
         every cell's state, as one image. A log gain is the fitted log ratio of the gene's \
         activity in the donor to matched cells of other donors; read it through its z, not \
         through being nonzero. view_map stage2 gain draws one gene's map, and chart gains / \
         volcano draws these as charts.",
        json!({
            "type": "object",
            "properties": {
                "what": {"enum": ["genes", "gene", "volcano", "tiles"]},
                "gene": {"type": "string", "description": "For gene and tiles; the feature view_map shows when left out"},
                "donor": {"type": "string", "description": "For genes and volcano: this donor's only"},
                "top": {"type": "integer", "minimum": 1, "description": format!("For genes and volcano: how many [default: {TOP}]")},
                "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": format!("For tiles: image width in pixels [default: {WIDTH}]")},
            },
            "required": ["what"],
            "additionalProperties": false,
        }),
    )
}
