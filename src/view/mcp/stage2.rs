//! `run_donor_deviation`: stage 2 on a finished embedding run, which of its
//! states are specific to a donor and which genes drive them.

use super::flags;
use super::jobs::{job_property, planned, start_server, Argv, Task};
use super::parse;
use crate::view::{Pick, Scene};
use anyhow::Result;
use legume_mcp::{Call, Content, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// A stage-2 tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_donor_deviation" => run_donor_deviation(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

/// What of the run's stage 2 colours the cells, or none of it.
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Stage2Arg {
    Gain,
    Exclusivity,
    Off,
}

impl Stage2Arg {
    pub(super) fn of(pick: Option<&Pick>) -> Self {
        match pick {
            Some(Pick::Gain { .. }) => Self::Gain,
            Some(Pick::Exclusive) => Self::Exclusivity,
            _ => Self::Off,
        }
    }
}

/// The donor a gain map draws, `""` for each cell's own.
pub(super) fn donor_of(pick: Option<&Pick>) -> &str {
    match pick {
        Some(Pick::Gain { donor: Some(d), .. }) => d,
        _ => "",
    }
}

/// `stage2` and `donor` on the scene, after `feature`: a gain map of the
/// feature shown (`donor`'s deviation, `""` each cell's own), exclusivity,
/// or off (the feature's activity again); `donor` alone changes the donor
/// of the gain map shown.
pub(super) fn apply_stage2(scene: &mut Scene, stage2: Option<Stage2Arg>, donor: Option<&str>) {
    let donor_now = || donor.map(|d| (!d.is_empty()).then(|| d.into()));
    match (stage2, scene.pick.clone()) {
        (Some(Stage2Arg::Gain), pick) => {
            let Some(gene) = pick.as_ref().and_then(Pick::feature).map(Into::into) else {
                return;
            };
            let kept = match pick {
                Some(Pick::Gain { donor, .. }) => donor,
                _ => None,
            };
            let donor = donor_now().unwrap_or(kept);
            scene.set_pick(Pick::Gain { gene, donor });
        }
        (Some(Stage2Arg::Exclusivity), _) => scene.set_pick(Pick::Exclusive),
        (Some(Stage2Arg::Off), Some(Pick::Gain { gene, .. })) => scene.set_pick(Pick::One(gene)),
        (Some(Stage2Arg::Off), Some(Pick::Exclusive)) => {
            scene.clear_pick();
        }
        (None, Some(Pick::Gain { gene, .. })) => {
            if let Some(donor) = donor_now() {
                scene.set_pick(Pick::Gain { gene, donor });
            }
        }
        _ => {}
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DonorDeviationCall {
    from: String,
    out: Option<String>,
    #[serde(default)]
    flags: Map<String, Value>,
    #[serde(default)]
    background: bool,
}

/// `senna donor-deviation --from {from} -o {out} {flags}`, `flags` the
/// words of its `flags`: the per-(gene, donor) deviations first, then their
/// batch jackknife and the state tables.
pub(super) fn donor_deviation_task(a: &DonorDeviationCall, flags: &[String], out: &str) -> Task {
    use crate::donor_deviation::outputs::{table_path, TABLES};
    let made = TABLES.map(|t| (t, table_path(out, t).into()));
    Task {
        argv: Argv::new("donor-deviation")
            .flag("--from", Some(&a.from))
            .flag("-o", Some(out))
            .rest(flags),
        made: made.to_vec(),
        summary: format!("senna donor-deviation of {}", a.from),
    }
}

/// Named as the other run tools name theirs ([`planned`]): `out` when given,
/// else `{run}_stage2` in the results folder, a name no running job holds.
fn run_donor_deviation(a: DonorDeviationCall, cx: &Call) -> Result<Vec<Content>> {
    let flags = flags::words("donor-deviation", &a.flags)?;
    let (_, task) = planned(cx, a.out.as_deref(), Some(&a.from), "stage2", |out| {
        Ok(donor_deviation_task(&a, &flags, out))
    })?;
    let done = start_server(task, a.background, cx)?;
    Ok(vec![Content::Json(done)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![ToolSpec::new(
        "run_donor_deviation",
        "Stage 2 on a finished run that recorded its cell → pseudobulk partition on one gene axis, as `senna donor-deviation \
         --from {from} -o {out}`: which of its states are specific to a donor, and which genes \
         drive them. The run's cell embedding is held fixed; cells are pooled into pseudobulk \
         samples (finest pseudobulk × batch) and each sample is contrasted, gene by gene, with \
         its nearest samples of other donors, on raw counts with each batch's robust gene factor \
         taken out. Writes {out}.donor_gene_deviation.parquet (per gene and donor: intercept and \
         state_axis coefficients; a cell's log gain is its donor's intercept + Σ_k coefficient_k · \
         its state_axis_k), {out}.donor_gene_deviation_without_batch.parquet (batch jackknife \
         replicates, for a gain's standard error), {out}.donor_sample_states.parquet (a large \
         matched distance marks a state no other donor has) and {out}.donor_cell_states.parquet. \
         Needs at least three donors; without flags donor-of-batch each batch is its own donor. Nothing \
         is cut off: read gains through their jackknife z, not through being nonzero. The run's \
         manifest records the tables, so view_map's stage2 draws them on its map.",
        json!({
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Run manifest (or prefix) of a finished run that recorded its cell → pseudobulk partition on one gene axis"},
                "out": {"type": "string", "description": "Output prefix [default: today's results folder, by the run's name]"},
                "flags": flags::property("senna donor-deviation", r#"{"donor-of-batch": "donors.tsv", "device": "cuda"}"#),
                "background": job_property(),
            },
            "required": ["from"],
            "additionalProperties": false,
        }),
    )]
}

#[cfg(test)]
#[path = "../tests/mcp_stage2.rs"]
mod tests;
