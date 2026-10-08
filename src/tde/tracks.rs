//! Row-grammar track assignment for `senna tde`.
//!
//! A tde feature axis is one gene axis carrying two count tracks: every row is
//! `{gene}/count/spliced` or `{gene}/count/unspliced`, read by
//! [`data_beans::aux::feature_rows::intern_count_rows`]. The track and the
//! gene come from the row itself, never from a file name or load order. Any
//! other row (another modality, a subunit, or the pooled `{gene}/count/total`
//! some producers also write) is rejected, naming up to ten offending rows and
//! the total.

use data_beans::aux::feature_rows::{intern_count_rows, UnparsedRowPolicy};

/// The row-grammar-derived plan for one feature axis.
#[derive(Clone, Debug)]
pub(crate) struct TrackPlan {
    /// Dense gene id per row, assigned in first-seen row order over both
    /// tracks.
    pub row_gene: Vec<u32>,
    /// `row_unspliced[r]` iff row `r` is an unspliced row; every other row is
    /// a spliced (base) row.
    pub row_unspliced: Vec<bool>,
    /// Gene keys in id order.
    pub gene_names: Vec<Box<str>>,
}

/// Assign every row of a feature axis to its track and gene.
pub(crate) fn assign_tracks(feature_names: &[Box<str>]) -> anyhow::Result<TrackPlan> {
    let map = intern_count_rows(feature_names, UnparsedRowPolicy::Reject);
    if !map.unparsed.is_empty() {
        let total = map.unparsed.len();
        let preview: Vec<&str> = map
            .unparsed
            .iter()
            .take(10)
            .map(|&r| feature_names[r].as_ref())
            .collect();
        let more = match total - preview.len() {
            0 => String::new(),
            n => format!(" (+{n} more)"),
        };
        anyhow::bail!(
            "{total} feature row(s) are not `{{gene}}/count/spliced` or \
             `{{gene}}/count/unspliced` rows: {preview:?}{more}"
        );
    }
    anyhow::ensure!(
        map.row_is_nascent.iter().any(|&u| !u),
        "no base rows: at least one `{{gene}}/count/spliced` row is required"
    );
    Ok(TrackPlan {
        row_gene: map.row_to_gene,
        row_unspliced: map.row_is_nascent,
        gene_names: map.gene_names,
    })
}

/// The spliced axis a [`TrackPlan`] pairs up: per gene with a spliced row,
/// ascending by that row.
pub(crate) struct Pairs {
    /// The spliced rows.
    pub spliced: Vec<usize>,
    /// Each gene's unspliced row, if it has one.
    pub unspliced: Vec<Option<usize>>,
    /// Each gene's name.
    pub genes: Vec<Box<str>>,
}

impl TrackPlan {
    /// Rows on the unspliced track (`true`) or the spliced one, ascending.
    pub(crate) fn rows(&self, unspliced: bool) -> Vec<usize> {
        (0..self.row_unspliced.len())
            .filter(|&r| self.row_unspliced[r] == unspliced)
            .collect()
    }

    /// Each spliced gene with its unspliced row: what
    /// `graph_embedding_util::split_divergence` cuts the axis by.
    pub(crate) fn pair_rows(&self) -> anyhow::Result<Pairs> {
        let n_genes = self.gene_names.len();
        let mut spliced_of = vec![None; n_genes];
        let mut unspliced_of = vec![None; n_genes];
        for (r, (&g, &u)) in self.row_gene.iter().zip(&self.row_unspliced).enumerate() {
            let slot = if u {
                &mut unspliced_of[g as usize]
            } else {
                &mut spliced_of[g as usize]
            };
            anyhow::ensure!(
                slot.replace(r).is_none(),
                "gene {} has more than one row on one track",
                self.gene_names[g as usize]
            );
        }
        let spliced = self.rows(false);
        let genes: Vec<usize> = spliced.iter().map(|&r| self.row_gene[r] as usize).collect();
        Ok(Pairs {
            unspliced: genes.iter().map(|&g| unspliced_of[g]).collect(),
            genes: genes.iter().map(|&g| self.gene_names[g].clone()).collect(),
            spliced,
        })
    }
}

#[cfg(test)]
#[path = "tracks/tests.rs"]
mod tests;
