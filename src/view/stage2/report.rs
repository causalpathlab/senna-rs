//! Stage 2 as numbers, for an agent (MCP `view_stage2`): the genes ranked,
//! one gene's gain by group × donor, and the strongest (gene, donor) pairs
//! at one group's state. The same computations the views draw.

use super::gains::{self, DonorState};
use super::*;
use serde_json::{json, Value};

impl Scene {
    /// The grouping on screen's name for group `g` (`all`: no grouping).
    fn group_name(&self, g: u32) -> String {
        self.levels()
            .get(g as usize)
            .map_or_else(|| "all".to_string(), ToString::to_string)
    }

    /// Each donor's mean state in each group on screen (the whole view as
    /// one group, `all`, with no grouping), with stage 2.
    pub(super) fn donor_states(&self) -> Result<(&Stage2, Vec<DonorState>), String> {
        let s = self.stage2()?;
        let rows = self.rows_in_stage2(s, self.space);
        let states = gains::donor_group_states(s, &rows, self.groups(), self.levels().len());
        Ok((s, states))
    }

    /// The genes ranked as `i` ranks them (`donor`'s only, when given), the
    /// first `top`, each with where it departs most.
    pub(crate) fn stage2_genes(&self, donor: Option<&str>, top: usize) -> Result<Value, String> {
        let (s, states) = self.donor_states()?;
        let d = s.donor(donor)?;
        let genes: Vec<Value> = gains::rank_genes(s, &states, d, top)
            .into_iter()
            .map(|r| {
                json!({
                    "gene": s.genes[r.gene],
                    "score": r.score,
                    "donor": s.donors[r.donor],
                    "group": self.group_name(r.group),
                    "log_gain": r.gain.log_gain,
                    "z": finite(r.gain.z),
                })
            })
            .collect();
        Ok(json!({
            "ranked_by": if s.has_jackknife() { "|z|" } else { "|log gain| (one batch: no jackknife)" },
            "over": self.grouping_title(),
            "min_cells": gains::MIN_CELLS_IN_GROUP,
            "genes": genes,
        }))
    }

    /// `gene`'s log gain and z for each donor in each group on screen, at
    /// the mean state of the donor's cells there (the gain heatmap's cells).
    pub(crate) fn stage2_gene(&self, gene: &str) -> Result<Value, String> {
        let (s, states) = self.donor_states()?;
        let g = s.gene(gene)?;
        let rows: Vec<Value> = states
            .iter()
            .map(|st| {
                let gain = s.gain(g, st.donor, &st.mean);
                json!({
                    "group": self.group_name(st.group),
                    "donor": s.donors[st.donor],
                    "cells": st.cells,
                    "log_gain": gain.log_gain,
                    "z": finite(gain.z),
                })
            })
            .collect();
        Ok(json!({
            "gene": s.genes[g],
            "over": self.grouping_title(),
            "min_cells": gains::MIN_CELLS_IN_GROUP,
            "z_full": gains::Z_FULL,
            "cells": rows,
        }))
    }

    /// The volcano's strongest `top` (gene, donor) pairs (`donor`'s only,
    /// when given), at the state of each donor's cells in the focused group,
    /// else in the whole view.
    pub(crate) fn stage2_volcano(&self, donor: Option<&str>, top: usize) -> Result<Value, String> {
        let s = self.stage2()?;
        let d = s.donor(donor)?;
        let v = self.volcano()?;
        let pairs: Vec<Value> = v
            .strongest(d, top)
            .into_iter()
            .map(|(g, donor, log_gain, z)| {
                json!({"gene": s.genes[g], "donor": s.donors[donor], "log_gain": log_gain, "z": finite(z)})
            })
            .collect();
        Ok(json!({"at": v.at(), "pairs": pairs}))
    }

    /// The grouping the stage-2 numbers are over.
    pub(super) fn grouping_title(&self) -> String {
        self.colour.map_or_else(
            || "the cells in view as one group (all)".to_string(),
            |li| self.data.labels[li].title().to_string(),
        )
    }
}

/// `v`, or null when not a number (no jackknife).
fn finite(v: f32) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        Value::Null
    }
}
