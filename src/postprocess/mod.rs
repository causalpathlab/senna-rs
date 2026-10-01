mod fit_layout_common;
pub(crate) mod fit_layout_features;
mod fit_layout_phate;
mod fit_layout_tree;
mod fit_layout_tsne;
mod fit_layout_umap;
pub(crate) mod viz_prep;

// Only `senna view` lays out a subset of cells from here.
#[cfg(feature = "view")]
pub(crate) use fit_layout_common::{latent_layout_features, DEFAULT_TRIM_MADS};

pub use fit_layout_phate::*;
pub use fit_layout_tree::*;
pub use fit_layout_tsne::*;
pub use fit_layout_umap::*;
