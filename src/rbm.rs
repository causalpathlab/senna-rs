//! `senna rbm`: cells as the hidden units of a restricted Boltzmann machine
//! over their reads, fitted on pseudobulks and read out exactly on cells.

pub mod args;
pub mod blocks;
pub mod eval;
pub mod hidden;
pub mod machine;
pub mod run;
pub mod train;
pub mod weights;
