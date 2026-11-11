//! Shared helpers for the unified `senna predict` subcommand.
//!
//! - `decoder_only_inference_*`: optimize θ logits against the frozen decoder
//!   from a uniform start, no encoder involvement.
//! - `predictive_llik_*`: wrap decoder forwards to return just the per-cell
//!   log-likelihood tensor.

use candle_core::{Device, Result as CandleResult, Tensor, Var};
use candle_nn::ops;
use legume_numeric::candle::traits::DecoderModuleT;
use senna::embed_common::*;

/// Latent inference mode chosen by the caller from CLI flags.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LatentMode {
    /// Forward pass through the encoder only (default, fastest).
    Encoder,
    /// Encoder warm-start, then decoder-side gradient on θ anchored to encoder by L2.
    EncoderRefine,
    /// Skip encoder; init θ uniform `log(1/K)`, optimize against frozen decoder.
    DecoderOnly,
}

/// Decoder-only latent inference for the dense topic model.
///
/// Initializes θ logits to zeros (uniform distribution after `log_softmax`) and
/// runs `num_steps` of plain gradient descent against the frozen decoder's
/// likelihood. No regularization — the data should pin θ on its own.
pub fn decoder_only_inference_dense<Dec: DecoderModuleT>(
    decoder: &Dec,
    x_nd: &Tensor,
    n_topics: usize,
    learning_rate: f64,
    num_steps: usize,
    dev: &Device,
) -> CandleResult<Tensor> {
    let n = x_nd.dim(0)?;
    let z_init = Tensor::zeros((n, n_topics), candle_core::DType::F32, dev)?;
    let z_var = Var::from_tensor(&z_init)?;

    for _step in 0..num_steps {
        let log_z = ops::log_softmax(z_var.as_tensor(), 1)?;
        let (_recon, llik) = decoder.forward_with_llik(&log_z, x_nd, &|_, _| unreachable!())?;
        let loss = llik.mean_all()?.neg()?;
        let grad = loss.backward()?;
        let z_grad = grad.get(z_var.as_tensor()).unwrap();
        let updated = (z_var.as_tensor() - (z_grad * learning_rate)?)?;
        z_var.set(&updated)?;
    }

    ops::log_softmax(z_var.as_tensor(), 1)
}

/// Genes per slice when scoring a dense block. Only the width of the
/// likelihood's temporaries; decoders that cannot slice ignore it and take the
/// dense path they always did.
pub const SCORE_GENE_CHUNK: usize = 4096;

/// Per-cell predictive log-likelihood for the dense topic model.
///
/// Goes through [`DecoderModuleT::llik_gene_chunked`] rather than
/// `forward_with_llik`: inference wants the scalar, not the `[N, D]`
/// reconstruction that comes with it, and at whole-transcriptome width
/// holding that reconstruction (plus the likelihood chain's temporaries) is
/// what turns scoring into an OOM. Decoders that have not been taught to
/// slice fall back to exactly the previous behaviour.
pub fn predictive_llik_dense<Dec: DecoderModuleT>(
    decoder: &Dec,
    log_z_nk: &Tensor,
    x_nd: &Tensor,
) -> CandleResult<Tensor> {
    decoder.llik_gene_chunked(log_z_nk, x_nd, SCORE_GENE_CHUNK)
}
