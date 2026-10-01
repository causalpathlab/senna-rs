//! What `mung clones` found, told so the user can decide whether the fits
//! after it keep donor-private clones apart (`--cnv-clones`).

use senna::clone_strata::CloneTable;

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

fn mean(of: Option<&[f32]>, at: &[usize]) -> Option<f32> {
    let v = of?;
    (!at.is_empty()).then(|| at.iter().map(|&i| v[i]).sum::<f32>() / at.len() as f32)
}

/// The clones and donors of `t`, clones by stratum, donors as first seen.
#[must_use]
pub fn of(t: &CloneTable) -> Summary {
    let mut donors: Vec<Donor> = Vec::new();
    for (d, &s) in t.donor.iter().zip(&t.stratum) {
        let at = match donors.iter().position(|x| *x.name == **d) {
            Some(k) => k,
            None => {
                donors.push(Donor {
                    name: d.to_string(),
                    cells: 0,
                    in_clones: 0,
                });
                donors.len() - 1
            }
        };
        donors[at].cells += 1;
        donors[at].in_clones += usize::from(s > 0);
    }
    let mut strata: Vec<usize> = t.stratum.iter().copied().filter(|&s| s > 0).collect();
    strata.sort_unstable();
    strata.dedup();
    let clones = strata
        .into_iter()
        .map(|stratum| {
            let at: Vec<usize> = (0..t.stratum.len())
                .filter(|&i| t.stratum[i] == stratum)
                .collect();
            let mut by: Vec<(&str, usize)> = Vec::new();
            for &i in &at {
                match by.iter_mut().find(|(d, _)| *d == &*t.donor[i]) {
                    Some(b) => b.1 += 1,
                    None => by.push((&t.donor[i], 1)),
                }
            }
            let (donor, n) = by
                .iter()
                .max_by_key(|(_, n)| *n)
                .copied()
                .unwrap_or(("", 0));
            Clonal {
                stratum,
                cells: at.len(),
                donor: donor.to_string(),
                share: n as f32 / at.len().max(1) as f32,
                purity: mean(t.purity.as_deref(), &at),
                p_malig: mean(t.p_malig.as_deref(), &at),
            }
        })
        .collect();
    Summary {
        cells: t.cell.len(),
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
