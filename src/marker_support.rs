//! Gene-name matching shared with marker files.
//!
//! Annotation itself lives in `lupin annotate`; senna keeps only the name
//! matcher its own consumers use.

/// Flexible gene-name matching, delegated to the shared implementation in
/// `data_beans` so every consumer agrees on symbol / alias / case
/// normalization.
pub use data_beans::utilities::name_matching::flexible_name_match as flexible_gene_match;
