//! A mixture of ODEs: every pseudobulk on one lineage's trajectory at one
//! time, the trajectory the splicing ODE driven by the lineage's
//! transcription. Built bottom up; see each module.

pub mod kernel;
pub mod kinetics;
#[cfg(test)]
pub mod sim;

#[cfg(test)]
mod tests;
