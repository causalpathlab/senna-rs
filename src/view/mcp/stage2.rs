//! `run_donor_deviation`: stage 2 on a finished embedding run, which of its
//! states are specific to a donor and which genes drive them.

use super::jobs::{job_property, start_server, Argv, Task};
use super::parse;
use anyhow::Result;
use legume_mcp::{Call, Content, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Value};

/// A stage-2 tool's call; `None` for another family's.
pub(super) fn call(name: &str, args: &Value, cx: &Call) -> Result<Option<Vec<Content>>> {
    Ok(Some(match name {
        "run_donor_deviation" => run_donor_deviation(parse(args.clone())?, cx)?,
        _ => return Ok(None),
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DonorDeviationCall {
    from: String,
    out: Option<String>,
    donor_of_batch: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
}

/// `senna donor-deviation --from {from} -o {out}`: the per-(gene, donor)
/// deviations first, then their batch jackknife and the state tables.
pub(super) fn donor_deviation_task(a: &DonorDeviationCall) -> Task {
    // senna's own default, named here so the files can be checked.
    let out = a
        .out
        .clone()
        .unwrap_or_else(|| crate::donor_deviation::default_out(&a.from));
    let made = [
        "donor_gene_deviation",
        "donor_gene_deviation_without_batch",
        "donor_sample_states",
        "donor_cell_states",
    ]
    .map(|t| (t, format!("{out}.{t}.parquet").into()));
    Task {
        argv: Argv::new("donor-deviation")
            .flag("--from", Some(&a.from))
            .flag("-o", Some(&out))
            .flag("--donor-of-batch", a.donor_of_batch.as_deref())
            .rest(&a.args),
        made: made.to_vec(),
        summary: format!("senna donor-deviation of {}", a.from),
    }
}

fn run_donor_deviation(a: DonorDeviationCall, cx: &Call) -> Result<Vec<Content>> {
    let done = start_server(donor_deviation_task(&a), a.background, cx)?;
    Ok(vec![Content::Json(done)])
}

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![ToolSpec::new(
        "run_donor_deviation",
        "Stage 2 on a finished run that recorded its cell → pseudobulk partition (not ode), as `senna donor-deviation \
         --from {from} -o {out}`: which of its states are specific to a donor, and which genes \
         drive them. The run's cell embedding is held fixed; cells are pooled into pseudobulk \
         samples (finest pseudobulk × batch) and each sample is contrasted, gene by gene, with \
         its nearest samples of other donors, on raw counts with each batch's robust gene factor \
         taken out. Writes {out}.donor_gene_deviation.parquet (per gene and donor: intercept and \
         state_axis coefficients; a cell's log gain is its donor's intercept + Σ_k coefficient_k · \
         its state_axis_k), {out}.donor_gene_deviation_without_batch.parquet (batch jackknife \
         replicates, for a gain's standard error), {out}.donor_sample_states.parquet (a large \
         matched distance marks a state no other donor has) and {out}.donor_cell_states.parquet. \
         Needs at least three donors; without donor_of_batch each batch is its own donor. Nothing \
         is cut off: read gains through their jackknife z, not through being nonzero.",
        json!({
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Run manifest (or prefix) of a run that recorded its cell → pseudobulk partition (bge, the topic family, svd, vae; not ode)"},
                "out": {"type": "string", "description": "Output prefix [default: from's prefix with .stage2]"},
                "donor_of_batch": {"type": "string", "description": "Two-column file, each batch and its donor (--donor-of-batch) [default: each batch is its own donor]"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Further flags, as run_help (donor-deviation) lists them, e.g. --device cuda"},
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
