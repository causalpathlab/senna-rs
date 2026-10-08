//! `senna tde`, stage 1: splicing kinetics of gene modules fitted on the finest
//! pseudobulks of a finished base fit.
//!
//! The base fit places the pseudobulks; this stage reads each pseudobulk's
//! unspliced and spliced reads gene by gene and asks, for every read, which
//! track it is on. The answer comes from one splicing ODE per gene module and
//! one time `τ` per pseudobulk ([`kinetics`]); nothing here moves the base
//! embedding.

pub mod args;
pub mod counts;
pub mod fit;
pub mod init;
pub mod joint;
pub mod kinetics;
pub mod levels;
pub mod modules;
pub mod run;
#[cfg(test)]
pub mod sim;
#[cfg(test)]
mod test_util;
