//! What `mung clones` found, told so the user can decide whether the fits
//! after it keep donor-private clones apart (`--cnv-clones`).

use rustc_hash::FxHashMap;
use senna::clone_strata::CloneTable;
use std::collections::BTreeMap;

/// One donor-private clone.
#[derive(Clone, Debug, PartialEq)]
pub struct Clonal {
    pub stratum: usize,
    pub cells: usize,
    /// The donor most of its cells come from, and their share.
    pub donor: String,
    pub share: f32,
    /// Mean donor purity and posterior malignancy, when the table has them.
    pub purity: Option<f32>,
    pub p_malig: Option<f32>,
}

/// One donor: its cells, and those in its clones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Donor {
    pub name: String,
    pub cells: usize,
    pub in_clones: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub cells: usize,
    pub clones: Vec<Clonal>,
    pub donors: Vec<Donor>,
}

/// The clones and donors of `t` in one pass: clones by stratum, donors
/// as first seen.
#[must_use]
pub fn of(t: &CloneTable) -> Summary {
    #[derive(Default)]
    struct Acc {
        cells: usize,
        /// Cells per donor, by donor slot.
        by_donor: FxHashMap<usize, usize>,
        purity: f32,
        p_malig: f32,
    }
    let mut slot: FxHashMap<&str, usize> = FxHashMap::default();
    let mut donors: Vec<Donor> = Vec::new();
    let mut strata: BTreeMap<usize, Acc> = BTreeMap::new();
    for (i, (d, &s)) in t.donor.iter().zip(&t.stratum).enumerate() {
        let k = *slot.entry(d).or_insert_with(|| {
            donors.push(Donor {
                name: d.to_string(),
                cells: 0,
                in_clones: 0,
            });
            donors.len() - 1
        });
        donors[k].cells += 1;
        if s == 0 {
            continue;
        }
        donors[k].in_clones += 1;
        let a = strata.entry(s).or_default();
        a.cells += 1;
        *a.by_donor.entry(k).or_default() += 1;
        a.purity += t.purity.as_ref().map_or(0.0, |v| v[i]);
        a.p_malig += t.p_malig.as_ref().map_or(0.0, |v| v[i]);
    }
    let clones = strata
        .into_iter()
        .map(|(stratum, a)| {
            // The largest donor; the first seen among equals.
            let (k, n) = a
                .by_donor
                .iter()
                .max_by_key(|(k, n)| (**n, std::cmp::Reverse(**k)))
                .map_or((0, 0), |(k, n)| (*k, *n));
            let n_cells = a.cells as f32;
            Clonal {
                stratum,
                cells: a.cells,
                donor: donors[k].name.clone(),
                share: n as f32 / n_cells,
                purity: t.purity.as_ref().map(|_| a.purity / n_cells),
                p_malig: t.p_malig.as_ref().map(|_| a.p_malig / n_cells),
            }
        })
        .collect();
    Summary {
        cells: t.stratum.len(),
        clones,
        donors,
    }
}

fn percent(part: usize, whole: usize) -> String {
    format!("{:.0}%", 100.0 * part as f32 / whole.max(1) as f32)
}

impl Summary {
    /// Cells in any clone.
    #[must_use]
    pub fn in_clones(&self) -> usize {
        self.clones.iter().map(|c| c.cells).sum()
    }

    /// The summary as lines of text: the overall split, the donors, the
    /// clones, then what keeping them means.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        let held = self.in_clones();
        let mut out = vec![format!(
            "{} cells: {} mix freely, {} ({}) in {} donor-private clone{}",
            self.cells,
            self.cells - held,
            held,
            percent(held, self.cells),
            self.clones.len(),
            if self.clones.len() == 1 { "" } else { "s" }
        )];
        out.push(String::new());
        let w = self
            .donors
            .iter()
            .map(|d| d.name.chars().count())
            .max()
            .unwrap_or(5)
            .clamp(5, 24);
        out.push(format!(
            "{:<w$}  {:>8}  {:>9}",
            "donor", "cells", "in clones"
        ));
        for d in &self.donors {
            out.push(format!(
                "{:<w$}  {:>8}  {:>9}",
                d.name,
                d.cells,
                format!("{} {}", d.in_clones, percent(d.in_clones, d.cells))
            ));
        }
        if !self.clones.is_empty() {
            out.push(String::new());
            out.push(format!(
                "{:>5}  {:>8}  {:<w$}  {:>6}  {:>6}  {:>7}",
                "clone", "cells", "donor", "share", "purity", "p_malig"
            ));
            let num = |x: Option<f32>| x.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
            for c in &self.clones {
                out.push(format!(
                    "{:>5}  {:>8}  {:<w$}  {:>6}  {:>6}  {:>7}",
                    c.stratum,
                    c.cells,
                    c.donor,
                    format!("{:.2}", c.share),
                    num(c.purity),
                    num(c.p_malig)
                ));
            }
        }
        out
    }

    /// A word on what keeping the clones would do.
    #[must_use]
    pub fn advice(&self) -> &'static str {
        if self.clones.is_empty() {
            "no donor-private clone: --cnv-clones would change nothing"
        } else {
            "keeping them stops batch correction from mixing a donor's clone into other donors' cells"
        }
    }
}

#[cfg(test)]
#[path = "tests/clones.rs"]
mod tests;
