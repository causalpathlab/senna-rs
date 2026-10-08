//! `senna ode`: a time for every unit and the splicing kinetics of gene
//! modules, from which track (unspliced or spliced) and which module every
//! read of a two-track count backend is on.
//!
//! Cells of a finished base fit are grouped into multilevel pseudobulks by
//! their base states; each gene module gets one switch-and-splicing ODE
//! ([`kinetics`]) and every pseudobulk a time `τ`, fitted coarse to fine
//! ([`fit`]). Nothing here moves the base embedding: the times and rates are
//! inputs for a dynamic embedding to build on.
//!
//! How it is trained is its own:
//! - every read is classified rather than counted: which track it is on given
//!   its gene (a binomial that conditions on each gene's total, so levels are
//!   never modelled twice), and which module it is from, an exact
//!   factorization of the reads with no count model and no ambient or
//!   detection terms to fit;
//! - pseudobulks of the base fit, not cells, carry the time, fitted coarse to
//!   fine over the levels of a multilevel collapse, each level pulled toward
//!   its parents and its neighbours;
//! - times and switch times move by grid search as well as by gradients, so
//!   a unit can jump between arcs and a module between "off inside the
//!   window" and "off after it", which no gradient reaches;
//! - the direction of time is the likelihood's call, both orientations fitted
//!   to the end, with no root or labels.
//!
//! The module structure itself (transcription a sum of modules that switch on
//! and off at shared times and relax toward their targets, one time per cell
//! shared by every module, a closed-form linear ODE) is cell2fate's:
//! Aivazidis et al., "Cell2fate infers RNA velocity modules to improve cell
//! fate prediction", *Nature Methods* (2025).

pub mod args;
pub mod counts;
pub mod fit;
pub mod init;
pub mod kinetics;
pub mod levels;
pub mod modules;
pub mod run;
#[cfg(test)]
#[path = "ode/tests/sim.rs"]
pub mod sim;
#[cfg(test)]
#[path = "ode/tests/util.rs"]
mod test_util;
