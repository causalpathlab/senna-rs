//! `senna resolve-topics`: topics for a finished embedding run that has none
//! (`simba`, `tde`, `bge --skip-etm`), made the way `senna bge` resolves its
//! own: one topic per cell cluster, from the run's cell embedding `Z`, gene
//! embedding `ρ` and clusters. No training; a few matrix products.
//!
//!   α [K,H] = each cluster's L2-normalised centroid in `Z`,
//!   θ [N,K] = each cell's softmax over ⟨z, α_k⟩,
//!   β [D,K] = `log_softmax` over genes of ρ·(α−ᾱ)ᵀ.
//!
//! Writes `{out}.latent.parquet` (log θ), `{out}.softmax_dictionary.parquet`
//! (β) and `{out}.topic_embedding.parquet` (α), and records the first two in
//! the manifest, so `senna view` draws a structure plot and colours by topic,
//! and `lupin plot-topic` reads them. Topic `T{c}` is cluster `c`.

use crate::bge::resolve_etm::topics_from_clusters;
use senna::embed_common::*;
use senna::run_manifest::{self, RunManifest};
use std::path::Path;

#[derive(Args, Debug)]
pub struct ResolveTopicsArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest ({prefix}.senna.json) with a cell and a gene embedding, and cell clusters"
    )]
    pub from: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix (default: the manifest's own, so the tables go beside the run)"
    )]
    pub out: Option<Box<str>>,
}

pub fn resolve_topics(args: &ResolveTopicsArgs) -> anyhow::Result<()> {
    let path = Path::new(args.from.as_ref());
    let (mut m, dir) = RunManifest::load(path)?;
    anyhow::ensure!(
        m.outputs.latent.is_none(),
        "{} already has a latent (topics, or its own cell factors); nothing to resolve",
        args.from
    );
    let z_rel = m
        .outputs
        .cell_embedding
        .clone()
        .ok_or_else(|| anyhow::anyhow!("{} records no cell embedding", args.from))?;
    let clusters_rel = m.cluster.clusters.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "{} has no cell clusters yet; run `senna clustering --from {}` first",
            args.from,
            args.from
        )
    })?;
    let read = |p: &str| Mat::from_parquet_with_row_names(p, Some(0));
    let z = read(&run_manifest::resolve(&dir, &z_rel).to_string_lossy())?;
    let (rho_path, _) = run_manifest::resolve_feature_embedding_for(&m, &dir)?;
    let rho = read(&rho_path)?;
    let clusters = read(&run_manifest::resolve(&dir, &clusters_rel).to_string_lossy())?;

    // Each cell's cluster, the ids renumbered densely in order.
    let of: rustc_hash::FxHashMap<&str, i64> = clusters
        .rows
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            let v = clusters.mat[(i, 0)];
            (v.is_finite() && v >= 0.0).then_some((n.as_ref(), v as i64))
        })
        .collect();
    let mut ids: Vec<i64> = of.values().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let keep: Vec<usize> = (0..z.rows.len())
        .filter(|&i| of.contains_key(z.rows[i].as_ref()))
        .collect();
    anyhow::ensure!(!keep.is_empty(), "no cell of the embedding has a cluster");
    if keep.len() < z.rows.len() {
        log::warn!(
            "{} of {} cells have no cluster and are left out of the topics",
            z.rows.len() - keep.len(),
            z.rows.len()
        );
    }
    let labels: Vec<usize> = keep
        .iter()
        .map(|&i| {
            let c = of[z.rows[i].as_ref()];
            ids.binary_search(&c).expect("every id was collected")
        })
        .collect();
    let zk = z.mat.select_rows(keep.iter());
    let cells: Vec<Box<str>> = keep.iter().map(|&i| z.rows[i].clone()).collect();

    let (alpha, log_theta, beta) = topics_from_clusters(&zk, &rho.mat, &labels)?;

    let topics: Vec<Box<str>> = ids.iter().map(|c| format!("T{c}").into()).collect();
    let h_names = axis_id_names("h", zk.ncols());
    let out = args.out.as_deref().map_or_else(
        || run_manifest::derive_out_prefix(args.from.as_ref()),
        String::from,
    );
    let latent = format!("{out}.latent.parquet");
    let dictionary = format!("{out}.softmax_dictionary.parquet");
    log_theta.to_parquet_with_names(&latent, (Some(&cells), Some("cell")), Some(&topics))?;
    beta.to_parquet_with_names(&dictionary, (Some(&rho.rows), Some("gene")), Some(&topics))?;
    alpha.to_parquet_with_names(
        &format!("{out}.topic_embedding.parquet"),
        (Some(&topics), Some("topic")),
        Some(&h_names),
    )?;
    m.outputs.latent = Some(run_manifest::rel_to_manifest(&dir, &latent));
    m.outputs.softmax_dictionary = Some(run_manifest::rel_to_manifest(&dir, &dictionary));
    m.save(path)?;
    info!(
        "resolve-topics: {} topics over {} cells and {} genes → {out}.{{latent,softmax_dictionary,topic_embedding}}.parquet",
        topics.len(),
        cells.len(),
        rho.rows.len()
    );
    Ok(())
}
