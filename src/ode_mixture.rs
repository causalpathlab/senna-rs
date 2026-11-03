//! A mixture of ODEs: every pseudobulk on one lineage's trajectory at one
//! time, the trajectory the splicing ODE driven by the lineage's
//! transcription. Built bottom up; see each module.

pub mod blocks;
pub mod fit;
pub mod kernel;
pub mod kinetics;
pub mod likelihood;
pub mod model;
#[cfg(test)]
pub mod sim;

#[cfg(test)]
mod tests;
