//! `senna tde`: track divergence embedding, over the shared
//! `graph_embedding_util` engine and `senna bge`'s driver.
//!
//! The input axis is one gene axis carrying two count tracks,
//! `{gene}/count/spliced` and `{gene}/count/unspliced`.
//! [`tracks::assign_tracks`] reads that grammar off the row names alone
//! (never a file name or load order). The spliced rows alone train the cell
//! state and the gene table; each gene's unspliced share in each pseudobulk
//! and cell is read against its spliced reads, as a displacement of the unit
//! within that space and a per-gene steady state
//! (`{out}.{pb,cell,feature}_divergence.parquet`,
//! `{out}.divergence_encoder.safetensors`).

pub(crate) mod args;
/// Pooled per-gene HVG projection weights over a [`tracks::TrackPlan`]
/// (every track of a gene shares one selection decision, weight lands only
/// on the base row).
pub(crate) mod hvg;
/// Multi-file input resolution and loading: matches files by sample id,
/// loads them into one `UnifiedData`, and assigns the [`tracks::TrackPlan`].
pub(crate) mod load;
/// Loading of the co-embedded feature table for annotate / lineage lives in
/// [`crate::marker_embedding`] (not β).
/// The `senna tde` run. Binary entry: [`run::run_tde`].
pub mod run;
pub mod sample_id;
/// Row-grammar track assignment: turns a feature axis of
/// `{gene}/{modality}/{channel}` rows into a [`tracks::TrackPlan`] /
/// `graph_embedding_util::fit::TrackSpec`. Also reads the axis of a run of
/// the retired joint `senna gem`.
pub(crate) mod tracks;

/// Synthetic fixture builders shared by `tde::run::tests` and
/// `predict::tests`.
#[cfg(test)]
pub(crate) mod test_fixtures;
