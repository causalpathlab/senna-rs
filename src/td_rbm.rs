//! `senna td-rbm`: every cell's time, its states and its velocity in a gene
//! embedding, from which gene and which track (unspliced or spliced) every
//! read of a two-track count backend is on: a restricted Boltzmann machine
//! with no encoder.
//!
//! The visible units are a cell's reads over (gene, track); the hidden units
//! a categorical (lineage, time) unit on a grid and Gaussian state units.
//! Every lineage's trajectory `z(τ)` runs through the genes' embedding space
//! `ρ`; its derivative `ż` is
//! the velocity, and every gene's unspliced share reads it through the gene's
//! own embedding by the splicing half of the ODE (`β u = ds/dt + γ s`,
//! [`model`]). State units shift which genes a cell reads, never its
//! unspliced share. A cell's time and states given its reads are exact and
//! independent ([`rbm`]); the energy is fitted by persistent contrastive
//! divergence ([`train`]).
//!
//! The energy is the replicated softmax's (Salakhutdinov and Hinton, *NIPS*
//! (2009)), trained by persistent contrastive divergence (Tieleman, *ICML*
//! (2008)). Unspliced derived from a spliced trajectory follows UniTVelo
//! (Gao, Qiao and Huang, *Nature Communications* (2022)); a velocity in a
//! latent space read out to genes linearly, LatentVelo (Farrell, Mani and
//! Goyal, bioRxiv (2022)); the gene embedding, the embedded topic model
//! (Dieng, Ruiz and Blei, *TACL* (2020)).

pub mod args;
pub mod counts;
pub mod curve;
pub mod init;
pub mod levels;
pub mod model;
pub mod rbm;
pub mod run;
#[cfg(test)]
#[path = "td_rbm/tests/sim.rs"]
pub mod sim;
pub mod start;
#[cfg(test)]
#[path = "td_rbm/tests/util.rs"]
pub(crate) mod test_util;
pub mod train;
pub mod tree;
