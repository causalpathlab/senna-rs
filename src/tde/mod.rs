//! `senna tde`: track divergence embedding, over the shared
//! `graph_embedding_util` engine and `senna bge`'s driver.
//!
//! The input axis is one gene axis carrying two count tracks,
//! `{gene}/count/spliced` and `{gene}/count/unspliced`.
//! [`tracks::assign_tracks`] reads that grammar off the row names alone
//! (never a file name or load order). The spliced rows alone train the cell
//! state and the gene table; each gene's unspliced share in each cell is read
//! against its spliced reads, through the gene's direction in that space and
//! its steady state, which gives each cell a velocity in that space
//! (`{out}.cell_velocity.parquet`, `{out}.feature_divergence.parquet`,
//! `{out}.divergence_loading.parquet`).

pub(crate) mod args;
/// Pooled per-gene HVG projection weights over a [`tracks::TrackPlan`]
/// (every track of a gene shares one selection decision, weight lands only
/// on the base row).
pub(crate) mod hvg;
/// Multi-file loading: gives each file a sample id, loads them into one
/// `UnifiedData`, and assigns the [`tracks::TrackPlan`].
pub(crate) mod load;
/// The `senna tde` run. Binary entry: [`run::run_tde`].
pub mod run;
/// Row-grammar track assignment: turns a feature axis of
/// `{gene}/count/{spliced,unspliced}` rows into a [`tracks::TrackPlan`].
pub(crate) mod tracks;

/// Synthetic fixture builders shared by `tde::run::tests` and
/// `predict::tests`.
#[cfg(test)]
pub(crate) mod test_fixtures;
