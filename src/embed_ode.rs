//! An embedding of pseudobulks, genes and cells shaped by an ode run: every
//! pseudobulk a point `x_p` that fits its reads through a full softmax over
//! the genes, `softmax_g(x_pᵀ w_g + b_g)`, and sits near the curve `μ_ℓ` of
//! the lineage and position the ode run put it at. Built bottom up; see each
//! module.

pub mod command;
pub mod curves;
pub mod inputs;
pub mod model;
pub mod place;

#[cfg(test)]
mod tests;
