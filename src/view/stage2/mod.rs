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
mod tables;

pub(crate) use charts::{draw_gain_heat, draw_volcano, GainHeat, Volcano};
pub(crate) use tables::Stage2;

use super::*;
use activity::Levels;

/// Genes listed by a ranking.
const RANKED: usize = 100;

/// What a run without stage 2 is told when asked for it.
pub(crate) const NO_STAGE2: &str =
    "this run has no stage 2 (run `senna donor-deviation --from` it first)";

impl Scene {
    /// Whether the run recorded a stage 2.
    pub fn has_stage2(&self) -> bool {
        self.data
            .run
            .as_ref()
            .is_some_and(|(m, _)| !m.stage2.is_empty())
    }

    /// Stage 2, waited for when the scene is patient.
    fn stage2(&self) -> Result<&Stage2, String> {
        self.data.loads.stage2.answer(self.patient)
    }

    /// Stage 2's donors, in its tables' order.
    pub(crate) fn stage2_donors(&self) -> Result<Vec<Box<str>>, String> {
        if !self.has_stage2() {
            return Err(NO_STAGE2.into());
        }
        self.stage2().map(|s| s.donors.clone())
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
        let donors = match self.stage2() {
            Ok(s) => s.donors.clone(),
            Err(e) => {
                self.note = Some(e);
                return;
            }
        };
        // Position 0 is each cell's own donor, then the donors in order.
        let n = donors.len() as i64 + 1;
        let at = donor
            .as_ref()
            .and_then(|d| donors.iter().position(|x| x == d))
            .map_or(0, |i| i as i64 + 1);
        let next = (at + delta).rem_euclid(n) as usize;
        let donor = (next > 0).then(|| donors[next - 1].clone());
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
    /// them in the sidebar and show the first on the gain map.
    pub fn rank_gains(&mut self, donor: Option<Box<str>>) {
        if self.current().axis() != Axis::Cells {
            self.note = Some("stage 2 is drawn on a cell view (tab)".into());
            return;
        }
        let space = self.space;
        let names = &self.data.spaces[space].points.names;
        let group_of: Vec<Option<u32>> = match self.groups() {
            Some(g) => g.iter().map(|&g| (g != data::NONE).then_some(g)).collect(),
            None => vec![Some(0); names.len()],
        };
        let over = self.colour.map_or_else(
            || "each donor's cells".to_string(),
            |li| format!("each donor in each {}", self.data.labels[li].title()),
        );
        let s = match self.stage2() {
            Ok(s) => s,
            Err(e) => {
                self.note = Some(e);
                return;
            }
        };
        let d = match donor.as_deref().map(|d| s.donor(d).ok_or(d)) {
            Some(Err(d)) => {
                self.note = Some(format!("no donor {d} in stage 2"));
                return;
            }
            Some(Ok(d)) => Some(d),
            None => None,
        };
        let rows = s.cells(names);
        let states = gains::donor_group_states(s, &rows, &group_of);
        let list = gains::rank_genes(s, &states, d, RANKED);
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
        let group_of: Vec<Option<u32>> = groups
            .iter()
            .map(|&g| (g != data::NONE).then_some(g))
            .collect();
        let s = self.stage2()?;
        let rows = s.cells(&self.current().points.names);
        charts::gain_heat(
            s,
            gene,
            &rows,
            &group_of,
            self.levels(),
            self.data.labels[li].title().to_string(),
        )
    }

    /// The volcano at the focused group's state, else the view's.
    pub(super) fn volcano(&self) -> Result<Volcano, String> {
        let s = self.stage2()?;
        let names = &self.current().points.names;
        let mut rows = s.cells(names);
        let at = match (self.focused_name(), self.groups()) {
            (Some(name), Some(groups)) => {
                let f = self.focus.expect("a focus named");
                for (r, &g) in rows.iter_mut().zip(groups) {
                    if g != f {
                        *r = None;
                    }
                }
                format!("in {name}")
            }
            _ => "over the cells in view ([ ] focuses a group)".to_string(),
        };
        let shown = self.pick.as_ref().and_then(Pick::feature);
        charts::volcano(s, &rows, at, shown)
    }

    /// Draw a stage-2 pick on `space`. Returns whether `pick` was one.
    pub(super) fn show_stage2(&mut self, pick: &Pick, space: usize, source: Source) -> bool {
        let names = &self.data.spaces[space].points.names;
        let made = match pick {
            Pick::Gain { gene, donor } => self.stage2().and_then(|s| {
                let g = s
                    .gene(gene)
                    .ok_or_else(|| format!("{gene} has no stage-2 deviation"))?;
                let d = match donor {
                    Some(name) => Some(
                        s.donor(name)
                            .ok_or_else(|| format!("no donor {name} in stage 2"))?,
                    ),
                    None => None,
                };
                let read = gains::cell_gains(s, g, d, &s.cells(names));
                let (values, weight) = read
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
            }),
            Pick::Exclusive => self.stage2().map(|s| {
                let values = s
                    .cells(names)
                    .iter()
                    .map(|r| r.map_or(f32::NAN, |c| s.exclusivity_of(c)))
                    .collect();
                (
                    Levels::from_values(values, false),
                    "exclusivity · distance to the nearest other donor's sample".to_string(),
                )
            }),
            _ => return false,
        };
        match made {
            Ok((levels, title)) => {
                self.shown_ids += 1;
                self.shown = Some(Shown {
                    id: self.shown_ids,
                    space,
                    pick: pick.clone(),
                    source,
                    title,
                    levels,
                });
            }
            Err(e) => {
                self.shown = None;
                self.note = Some(e);
            }
        }
        true
    }
}

#[cfg(test)]
#[path = "../tests/stage2.rs"]
pub(crate) mod tests;
