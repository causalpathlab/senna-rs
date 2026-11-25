//! Stage 2 (`senna donor-deviation`) on the run's map, once it recorded
//! itself in the manifest:
//!
//! - **gain map** (`i`): each cell coloured by a gene's log gain in its own
//!   donor, on the diverging ramp; cells whose jackknife `|z|` is under
//!   [`gains::Z_FULL`] fade toward muted, so only supported departures
//!   stand out. `I` steps to one donor's deviation drawn at every cell's
//!   state ("if this were donor D2"), and back.
//! - **exclusivity map** (`U`): each cell coloured by its sample's distance
//!   to the nearest matched sample of another donor: large where no other
//!   donor has the state.
//! - **gene ranking** (`i` with no gene shown): genes by their largest
//!   `|z|` over the donors' states in the grouping on screen, in the
//!   sidebar; the arrows step through them on the gain map.

mod charts;
mod gains;
mod report;
mod tables;

pub(crate) use charts::{draw_gain_heat, draw_volcano, GainHeat, Volcano};
pub(crate) use tables::Stage2;

use super::features::Ask;
use super::*;
use activity::Levels;
use std::collections::HashMap;
use std::sync::Arc;

/// Genes listed by a ranking.
const RANKED: usize = 100;

/// What a run without stage 2 is told when asked for it.
pub(crate) const NO_STAGE2: &str = "this run has no stage 2 (run `senna donor-deviation --from` \
     it, then open the run again)";

/// Each point's row in stage 2's per-cell tables, by space: matched once.
pub(crate) type Stage2Rows = std::cell::RefCell<HashMap<usize, Arc<Vec<Option<u32>>>>>;

impl Scene {
    /// Whether the run recorded a stage 2.
    pub fn has_stage2(&self) -> bool {
        self.data.has_stage2()
    }

    /// Stage 2, waited for when the scene is patient.
    fn stage2(&self) -> Result<&Stage2, String> {
        if !self.has_stage2() {
            return Err(NO_STAGE2.into());
        }
        self.data.loads.stage2.answer(self.patient)
    }

    /// Stage 2's row of each point of `space` (cached).
    fn rows_in_stage2(&self, s: &Stage2, space: usize) -> Arc<Vec<Option<u32>>> {
        self.stage2_rows
            .borrow_mut()
            .entry(space)
            .or_insert_with(|| Arc::new(s.cells(&self.data.spaces[space].points.names)))
            .clone()
    }

    /// Stage 2's donors, in its tables' order.
    pub(crate) fn stage2_donors(&self) -> Result<&[Box<str>], String> {
        self.stage2().map(|s| s.donors.as_slice())
    }

    /// The gene a volcano rings: the one shown, when stage 2 is read.
    pub(super) fn ringed_gene(&self) -> Option<usize> {
        let f = self.pick.as_ref().and_then(Pick::feature)?;
        let s = self.data.loads.stage2.try_get()?.as_ref().ok()?;
        s.gene(f).ok()
    }

    /// The gain map of the gene shown, or back to its activity; with no
    /// gene shown, rank the genes and show the first.
    pub fn toggle_gain(&mut self) {
        if !self.has_stage2() {
            self.note = Some(NO_STAGE2.into());
            return;
        }
        match self.pick.clone() {
            Some(Pick::Gain { gene, .. }) => self.set_pick(Pick::One(gene)),
            Some(Pick::One(gene)) => self.set_pick(Pick::Gain { gene, donor: None }),
            _ => self.rank_gains(None),
        }
    }

    /// The next (`delta` 1) or previous donor whose deviation the gain map
    /// draws at every cell, through each cell's own donor between the last
    /// and the first.
    pub fn step_donor(&mut self, delta: i64) {
        let Some(Pick::Gain { gene, donor }) = self.pick.clone() else {
            self.note = Some("show a gain map first (i)".into());
            return;
        };
        let donor = match self.stage2_donors() {
            Ok(donors) => {
                // Position 0 is each cell's own donor, then the donors.
                let n = donors.len() as i64 + 1;
                let at = donor
                    .as_ref()
                    .and_then(|d| donors.iter().position(|x| x == d))
                    .map_or(0, |i| i as i64 + 1);
                let next = (at + delta).rem_euclid(n) as usize;
                (next > 0).then(|| donors[next - 1].clone())
            }
            Err(e) => {
                self.note = Some(e);
                return;
            }
        };
        self.set_pick(Pick::Gain { gene, donor });
    }

    /// The exclusivity map, or stop.
    pub fn toggle_exclusive(&mut self) {
        if self.pick == Some(Pick::Exclusive) {
            self.clear_pick();
        } else if !self.has_stage2() {
            self.note = Some(NO_STAGE2.into());
        } else {
            self.set_pick(Pick::Exclusive);
        }
    }

    /// Rank the genes by their largest jackknife `|z|` over each donor's
    /// states in the grouping on screen (`donor`'s only, when given), list
    /// them in the sidebar and show the first on the gain map. Done once
    /// stage 2 is read, when it is not yet.
    pub fn rank_gains(&mut self, donor: Option<Box<str>>) {
        if self.current().axis() != Axis::Cells {
            self.note = Some("stage 2 is drawn on a cell view (tab)".into());
            return;
        }
        let space = self.space;
        let over = self.colour.map_or_else(
            || "each donor's cells".to_string(),
            |li| format!("each donor in each {}", self.data.labels[li].title()),
        );
        let ranked = self.stage2().and_then(|s| {
            let d = s.donor(donor.as_deref())?;
            let rows = self.rows_in_stage2(s, space);
            let states = gains::donor_group_states(s, &rows, self.groups(), self.levels().len());
            Ok(gains::rank_genes(s, &states, d, RANKED)
                .into_iter()
                .map(|r| (s.genes[r.gene].clone(), r.score))
                .collect::<Vec<_>>())
        });
        let list = match ranked {
            Ok(list) => list,
            Err(e) => {
                if e == legume_tui::work::LOADING {
                    self.redo = Some(Ask::RankGains(donor));
                }
                self.note = Some(e);
                return;
            }
        };
        let Some(first) = list.first().map(|(g, _)| g.clone()) else {
            self.note = Some(format!(
                "no gene departs in {over} (each needs {} of a donor's cells)",
                gains::MIN_CELLS_IN_GROUP
            ));
            return;
        };
        let whose = donor.as_deref().unwrap_or("any donor");
        self.suggestions = Some(Suggestions {
            space,
            title: format!("genes by largest |z| of {whose}, over {over}"),
            list,
        });
        self.set_pick(Pick::Gain { gene: first, donor });
    }

    /// The gain heatmap of the feature shown, over the grouping on screen.
    pub(super) fn gain_heat(&self) -> Result<GainHeat, String> {
        let gene = self
            .pick
            .as_ref()
            .and_then(Pick::feature)
            .ok_or("show a gene first (i ranks them, / finds one)")?;
        let li = self
            .colour
            .ok_or("colour the cells by a grouping first (c)")?;
        let groups = self.groups().ok_or("no grouping on screen")?;
        let s = self.stage2()?;
        let rows = self.rows_in_stage2(s, self.space);
        charts::gain_heat(
            s,
            gene,
            &rows,
            groups,
            self.levels(),
            self.data.labels[li].title().to_string(),
        )
    }

    /// The volcano at the focused group's state, else the view's.
    pub(super) fn volcano(&self) -> Result<Volcano, String> {
        let s = self.stage2()?;
        let mut rows = self.rows_in_stage2(s, self.space).to_vec();
        let at = match (self.focus, self.focused_name(), self.groups()) {
            (Some(f), Some(name), Some(groups)) => {
                for (r, &g) in rows.iter_mut().zip(groups) {
                    if g != f {
                        *r = None;
                    }
                }
                format!("in {name}")
            }
            _ => "over the cells in view ([ ] focuses a group)".to_string(),
        };
        charts::volcano(s, &rows, at)
    }

    /// A stage-2 pick's levels on `space`, and its key's title.
    pub(super) fn stage2_levels(
        &self,
        pick: &Pick,
        space: usize,
    ) -> Result<(Levels, String), String> {
        let s = self.stage2()?;
        let rows = self.rows_in_stage2(s, space);
        match pick {
            Pick::Gain { gene, donor } => {
                let g = s.gene(gene)?;
                let d = s.donor(donor.as_deref())?;
                let (values, weight) = gains::cell_gains(s, g, d, &rows)
                    .iter()
                    .map(|r| r.map_or((f32::NAN, 0.0), |r| (r.log_gain, gains::weight(r.z))))
                    .unzip();
                let levels = Levels::diverging(values, weight);
                let whose = donor
                    .as_deref()
                    .map_or_else(|| "own donor".to_string(), |d| format!("as {d}"));
                let title = format!(
                    "{} log gain · {whose} · ±{:.2} · faded |z| < {}",
                    s.genes[g],
                    levels.hi,
                    gains::Z_FULL
                );
                Ok((levels, title))
            }
            _ => {
                let values = rows
                    .iter()
                    .map(|r| r.map_or(f32::NAN, |c| s.exclusivity_of(c)))
                    .collect();
                Ok((
                    Levels::from_values(values, false),
                    "exclusivity · distance to the nearest other donor's sample".to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/stage2.rs"]
pub(crate) mod tests;
