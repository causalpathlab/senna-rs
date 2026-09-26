//! `senna fne` (Feature Network Embedding): PyTorch-BigGraph over a typed
//! feature graph (see `graph_embedding_util::fne`).
//!
//! Where `bge` builds a bipartite (cell × feature) graph from expression
//! counts, `fne` *consumes* graphs and emits per-feature embeddings alone,
//! no cells involved. The nodes are typed — genes, and whatever else the
//! edge files name (cell types, ontology terms, genomic windows, words) —
//! and every input source is one relation of the graph. The result is a
//! word2vec-shaped `feature_embedding.parquet` over every node, with the
//! gene rows a direct input to `--{freeze,init,lora}-feature-embedding` on
//! `senna bge`, `senna simba` and the masked models.
//!
//! Thin by design: the recipe lives in the shared crate; this module owns
//! the command line, the readers that turn edge files into a typed graph,
//! and the artifacts. `links` reads published association and enhancer–gene
//! resources (GWAS Catalog, GTEx, eQTL Catalogue, ABC / ENCODE-rE2G,
//! OpenGWAS) onto the same region windows.

mod args;
mod graph;
mod links;
mod output;
mod run;

pub use args::FneArgs;
pub use run::fit_fne;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "links_tests.rs"]
mod links_tests;
