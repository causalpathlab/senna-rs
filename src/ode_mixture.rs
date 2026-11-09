//! A mixture of ODEs: every pseudobulk on one lineage's trajectory at one
//! time, the trajectory the splicing ODE driven by the lineage's
//! transcription. Built bottom up; see each module.

pub mod band;
pub mod blocks;
pub mod coarsen;
pub mod command;
pub mod counts;
pub mod fit;
pub mod init;
pub mod kernel;
pub mod kinetics;
pub mod likelihood;
pub mod model;
pub mod output;
pub mod readout;
pub mod refine;
#[cfg(test)]
pub mod sim;

#[cfg(test)]
mod tests;
