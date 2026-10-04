use crate::hvg::HvgCliArgs;
use crate::topic::common::{
    load_and_project, load_cnv_cell_strata, LoadProjectArgs, ProjectedData,
};
use data_beans::sparse_data_visitors::VisitColumnsOps;
use senna::embed_common::*;

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct SvdArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed, comma- or space-separated,\n\
                     and are concatenated column-wise on a shared feature set."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet         gene × component loadings\n  \
                     {out}.latent.parquet             cell × component scores\n  \
                     {out}.delta.parquet              per-batch effects (if --batch-files)\n  \
                     {out}.adjusted.zarr              batch-adjusted backend (if --save-adjusted)\n  \
                     {out}.selected_features.txt      selected HVG names (if HVG enabled)\n  \
                     {out}.cell_proj.parquet          cached random projection\n  \
                     {out}.feature_weights.parquet    per-gene weights (if any)\n  \
                     {out}.senna.json                 run manifest for `senna layout/plot --from`"
    )]
    out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell.\n\
                     The cells come in the same order as its matching data file.\n\
                     Example: batch1.tsv,batch2.tsv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    collapse: crate::refine_weighting::CollapseArgs,

    /// The parent's carried pseudobulks, when `senna update` chose to reuse
    /// them instead of re-reading its cells. Derived per invocation, so it is
    /// neither a CLI flag nor part of the recorded configuration.
    #[arg(skip)]
    #[serde(skip)]
    pb_reference: Option<senna::pb_reference::ReferenceInput>,

    /// The parent run this one continues, set by `senna update` — svd has no
    /// weights to warm-start, so unlike the other families this is not a CLI
    /// flag; it only chains the emitted reference's generation counter.
    #[arg(skip)]
    #[serde(skip)]
    init_from: Option<Box<str>>,

    /// `senna revise`'s labels: the fit then votes on its genes before
    /// solving. Set by revise, never a flag; the weights it leaves are
    /// recorded instead.
    #[arg(skip)]
    #[serde(skip)]
    pub(crate) vote: Option<crate::svd::revise::Vote>,

    #[arg(
        long,
        value_name = "FILE",
        help = "Per-gene weights applied before the SVD (gene, weight)",
        long_help = "Scales each gene's standardised log expression before the solve,\n\
                     so the components weigh it accordingly; a gene not listed keeps\n\
                     weight 1. `senna revise` writes these as\n\
                     {out}.feature_weights.parquet and records them here."
    )]
    feature_weights: Option<Box<str>>,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)",
        hide = true
    )]
    block_size: Option<usize>,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column-sum normalization scale",
        long_help = "Target library size after per-cell normalization."
    )]
    // pub(crate) for `impute`, which replays this scale from the recorded
    // train_args to reproduce the projection transform.
    pub(crate) column_sum_norm: f32,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent components (K)"
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training",
        hide = true
    )]
    preload_data: bool,

    #[arg(long, help = "Write the batch-adjusted data to a new zarr backend")]
    save_adjusted: bool,

    #[command(flatten)]
    hvg: HvgCliArgs,

    #[command(flatten)]
    qc: QcArgs,
}

pub fn fit_svd(args: &SvdArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let proj_dim = args.collapse.proj_dim.max(args.n_latent_topics);
    let ProjectedData {
        mut data_vec,
        batch_membership,
        proj_kn,
        selected_features,
        output_keep_idx,
    } = load_and_project(&LoadProjectArgs {
        data_files: &args.data_files,
        batch_files: &args.batch_files,
        preload: args.preload_data,
        proj_dim,
        block_size: args.block_size,
        max_features: args.hvg.n_hvg,
        feature_list_file: args.hvg.feature_list_file.as_deref(),
        must_train_file: args.hvg.must_train_features.as_deref(),
        ignore_batch: args.collapse.ignore_batch,
        qc: args.qc.to_config(),
        qc_block_size: args.block_size,
        qc_report_out: args.qc.qc_report.as_deref(),
        feature_mask_fn: None,
        pb_reference: args.pb_reference.as_ref(),
        row_alignment: data_beans::sparse_io_vector::RowAlignment::default(),
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::default(),
        feature_kind: None,
    })?;

    // Per-gene weights: the recorded ones, times a revision's votes, which
    // need no collapse and so are cast before it. A revision's weights are
    // this run's own and are recorded by absolute path, so a replay from
    // anywhere solves in the same space.
    let gene_names = data_vec.row_names()?;
    let mut feature_weights = args
        .feature_weights
        .as_deref()
        .map(|f| crate::svd::revise::read_feature_weights(f, &gene_names))
        .transpose()?;
    let mut weights_file = args.feature_weights.as_deref().map(absolute).transpose()?;
    if let Some(vote) = args.vote.as_ref() {
        let partition = vote.partition.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "senna revise of an svd run needs --pb-from: the run whose partition the \
                 labels were judged on"
            )
        })?;
        let votes = crate::svd::revise::feature_votes(
            &data_vec,
            partition,
            &vote.peer,
            args.column_sum_norm,
        )?;
        let mut w = feature_weights
            .take()
            .unwrap_or_else(|| vec![1.0; votes.len()]);
        w.iter_mut().zip(&votes).for_each(|(a, b)| *a *= b);
        let path = format!("{}.feature_weights.parquet", args.out);
        crate::svd::revise::write_feature_weights(&path, &gene_names, &w, &votes)?;
        weights_file = Some(absolute(&path)?);
        feature_weights = Some(w);
    }

    // 3. Batch-adjusted collapsing (pseudobulk)
    //
    // `collapse_columns_multilevel` would do, but it is a thin wrapper that
    // computes the cell → pb membership and then throws it away — and
    // `--emit-pb-reference` needs it, to know how many cells each carried
    // column stands for. Same work, one more return value.
    //
    // data_beans::alg hands levels back finest-FIRST, and svd consumes only
    // the finest: take element 0 of each and drop the coarser tail now — a
    // retained level is up to six `[D, S]` planes of dead weight.
    let ml_params = MultilevelParams {
        knn_pb_samples: args.collapse.knn_cells,
        num_levels: args.collapse.num_levels,
        sort_dim: args.collapse.sort_dim,
        num_opt_iter: args.collapse.iter_opt,
        refine: args.collapse.pb_refine.to_params(),
        output_calibration: legume_numeric::param::traits::CalibrateTarget::All,
        // See `topic::common::load_and_collapse` — greedy correction
        // against the carried reference when one is loaded.
        anchor_batches: args
            .pb_reference
            .is_some()
            .then(|| vec![senna::pb_reference::REFERENCE_BATCH.into()]),
        bulk_batches: args.collapse.mixture_batch.clone(),
        observe_panels: true,
        keep_finest_stats: false,
        pb_tree: args.collapse.pb_tree_params(),
        strata: None,
    };
    let mut multilevel = if let Some(path) = args.collapse.cnv_clones.as_deref() {
        let cell_to_stratum = load_cnv_cell_strata(path, &data_vec)?;
        data_beans::alg::collapse_data::collapse_columns_multilevel_with_strata(
            &mut data_vec,
            &proj_kn,
            &batch_membership,
            &ml_params,
            &cell_to_stratum,
        )?
    } else {
        data_beans::alg::collapse_data::collapse_columns_multilevel_with_hierarchy(
            &mut data_vec,
            &proj_kn,
            &batch_membership,
            &ml_params,
        )?
    };
    anyhow::ensure!(!multilevel.levels.is_empty(), "collapse returned no levels");
    let collapse_out = multilevel.levels.swap_remove(0);
    let finest_membership = multilevel.cell_to_pb_per_level.swap_remove(0);
    drop(multilevel);

    // 4. batch-adjusted data
    let batch_dp = collapse_out.mu_residual.as_ref();

    if let Some(delta_dp) = batch_dp.map(legume_numeric::param::traits::Inference::posterior_mean) {
        info!("{} x {}", delta_dp.nrows(), delta_dp.ncols());

        if args.save_adjusted {
            info!("Generating batch-adjusted data...");

            let triplets = triplets_adjusted_by_pseudobulk(&data_vec, delta_dp)?;

            let mtx_shape = (data_vec.num_rows(), data_vec.num_columns(), triplets.len());

            let backend_file = args.out.to_string() + ".adjusted.zarr";
            let backend = SparseIoBackend::Zarr;
            remove_file(&backend_file)?;

            let mut adjusted_data = create_sparse_from_triplets(
                &triplets,
                mtx_shape,
                Some(&backend_file),
                Some(&backend),
            )?;

            adjusted_data.register_row_names_vec(&data_vec.row_names()?);
            adjusted_data.register_column_names_vec(&data_vec.column_names()?);

            info!("Batch-adjusted backend: {backend_file}");
        }
    }

    // Borrowed, not moved: `--emit-pb-reference` reads the rest of
    // `collapse_out` further down, and a partial move would put it out of reach.
    if let Some(batch_db) = collapse_out.delta.as_ref() {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_melted_parquet(
            &outfile,
            (Some(&gene_names), Some("gene")),
            (batch_names.as_deref(), Some("batch")),
        )?;
    }

    // 5. Nystrom projection
    let x_dn = match collapse_out.mu_adjusted.as_ref() {
        Some(adj) => adj,
        None => &collapse_out.mu_observed,
    };

    let nystrom_out = do_nystrom_proj(
        x_dn.posterior_log_mean().clone(),
        batch_dp.map(legume_numeric::param::traits::Inference::posterior_mean),
        &data_vec,
        args.n_latent_topics,
        args.column_sum_norm,
        args.block_size,
        feature_weights.as_deref(),
    )?;

    let cell_names = data_vec.column_names()?;

    // SVD reuses the topic models' `T{c}` convention so `lupin plot
    // --colour-by topic` reads the latent.parquet identically regardless
    // of upstream (`senna topic` or `senna svd`).
    senna::output_helpers::save_latent(
        &args.out,
        &nystrom_out.latent_nk,
        &cell_names,
        output_keep_idx.as_deref(),
    )?;
    senna::output_helpers::save_dictionary(&args.out, &nystrom_out.dictionary_dk, &gene_names)?;

    {
        let pb_gene_gp: Mat = x_dn.posterior_mean().clone();
        senna::output_helpers::save_pb_gene(&args.out, &pb_gene_gp, &gene_names)?;
    }

    let column_weight = data_vec.column_multiplicities();

    // Save selected feature list if feature selection was applied
    if let Some(sel) = &selected_features {
        use legume_numeric::matrix::common_io::write_lines;
        let feature_file = args.out.to_string() + ".selected_features.txt";
        write_lines(&sel.selected_names, &feature_file)?;
        info!(
            "Saved {} selected features to {}",
            sel.selected_names.len(),
            feature_file
        );
    }

    crate::postprocess::viz_prep::write_cell_proj(
        &args.out,
        &proj_kn,
        &cell_names,
        output_keep_idx.as_deref(),
    )?;

    let pb_reference_suffix = senna::pb_reference::emit_if_requested(
        args.collapse.emits_pb_reference(),
        &args.out,
        &collapse_out,
        Some(std::slice::from_ref(&finest_membership)),
        column_weight,
        &gene_names,
        args.init_from.as_deref(),
        args.pb_reference.as_ref(),
    )?;

    let input: Vec<String> = args
        .data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    let mut train_args = senna::run_manifest::record_train_args(args)?;
    train_args.args["feature_weights"] = serde_json::to_value(&weights_file)?;
    senna::run_manifest::write_run_manifest(&senna::run_manifest::RunDescription {
        train_args: Some(train_args),
        kind: senna::run_manifest::RunKind::Svd,
        prefix: &args.out,
        data_input: &input,
        data_multiome: None,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: true,
        pb_gene_suffix: Some("pb_gene.parquet"),
        pb_reference_suffix,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: None,
        feature_coembedding_suffix: None,
        carried: None,
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: None,
        cell_embedding_suffix: None,
        cell_encoder_suffix: None,
        feature_contrast_suffix: None,
        feature_contrast_bias_suffix: None,
        track_encoder_suffixes: vec![],
        // SVD produces no topic / cluster labels on its own; users
        // typically run `senna clustering` next, so the viz column
        // `cluster` is the natural default.
        default_colour_by: "cluster",
        has_latent: true,
        has_cell_to_pb: false,
        has_pb_tree: false,
    })?;

    Ok(())
}

/// Multiply row `g` of `m` by `w[g]`; no weights, no change.
fn scale_rows(m: &mut Mat, w: Option<&[f32]>) {
    if let Some(w) = w {
        for (mut row, &wg) in m.row_iter_mut().zip(w) {
            row *= wg;
        }
    }
}

/// `path` made absolute, so a recorded path does not depend on the cwd.
fn absolute(path: &str) -> anyhow::Result<Box<str>> {
    Ok(std::path::absolute(path)?.to_string_lossy().into())
}

struct NystromParam<'a> {
    basis_dk: &'a Mat,
    delta_dp: Option<&'a Mat>,
    column_sum_norm: f32,
}

struct NystromOut {
    pub dictionary_dk: Mat,
    pub latent_nk: Mat,
}

/// Nystrom projection for fast latent representation
///
/// # Arguments
/// * `xx_dn` - feature x sample matrix
/// * `delta_db` - feature x batch batch effect matrix
/// * `full_data_vec` - full sparse data vector
/// * `rank` - matrix factorization rank
/// * `column_sum_norm` - column sum normalization scale
/// * `block_size` - online learning block size
/// * `feature_weights` - per-gene weights `w`, if any
///
/// With weights the solve is on `W·X`, and the dictionary is `W·U`: a cell
/// `x` then projects as `x^T W U S^-1`, here and in every reader of the
/// dictionary (`predict`, `impute`), with no weight file to carry along.
fn do_nystrom_proj(
    log_xx_dn: Mat,
    delta_dp: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    rank: usize,
    column_sum_norm: f32,
    block_size: Option<usize>,
    feature_weights: Option<&[f32]>,
) -> anyhow::Result<NystromOut> {
    let mut log_xx_dn = log_xx_dn;

    log_xx_dn.scale_columns_inplace();
    scale_rows(&mut log_xx_dn, feature_weights);

    let (mut u_dk, s_k, _) = log_xx_dn.rsvd(rank)?;
    scale_rows(&mut u_dk, feature_weights);
    let basis_dk = nystrom_basis(&u_dk, &s_k);

    info!(
        "Constructed {} x {} projection matrix",
        u_dk.nrows(),
        u_dk.ncols()
    );

    let ntot = full_data_vec.num_columns();
    let kk = rank;

    let nystrom_param = NystromParam {
        basis_dk: &basis_dk,
        delta_dp,
        column_sum_norm,
    };

    let mut proj_kn = Mat::zeros(kk, ntot);

    full_data_vec.visit_columns_by_block(
        &nystrom_proj_visitor,
        &nystrom_param,
        &mut proj_kn,
        block_size,
    )?;

    let z_nk = proj_kn.transpose();

    Ok(NystromOut {
        dictionary_dk: u_dk,
        latent_nk: z_nk,
    })
}

fn nystrom_proj_visitor(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    proj_basis: &NystromParam,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let basis_dk = proj_basis.basis_dk;

    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    let pseudobulk = match proj_basis.delta_dp {
        Some(_) => Some(full_data_vec.get_group_membership(lb..ub)?),
        None => None,
    };
    nystrom_preprocess_columns(
        &mut x_dn,
        proj_basis.column_sum_norm,
        proj_basis.delta_dp.zip(pseudobulk.as_deref()),
    );

    let chunk = (x_dn.transpose() * basis_dk).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in nystrom");

    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

/// The per-chunk transform every column passes through before hitting the
/// Nyström basis. Shared with `senna impute`'s dictionary projection so the
/// training and query-side chains cannot drift.
///
/// The chunk must STAY SPARSE: `scale_columns_inplace` on a CSC standardizes
/// over a column's stored entries only, so running the same chain on a
/// densified copy would standardize against the zeros too and land in a
/// different space.
///
/// `delta` pairs the per-group divisor with the GROUP membership of exactly
/// `x_dn`'s columns (it indexes the divisor's columns) — build it with
/// `delta_dp.zip(membership.as_deref())` so both come or neither does.
pub(crate) fn nystrom_preprocess_columns(
    x_dn: &mut nalgebra_sparse::CscMatrix<f32>,
    column_sum_norm: f32,
    delta: Option<(&Mat, &[usize])>,
) {
    x_dn.normalize_columns_inplace();
    *x_dn *= column_sum_norm;
    if let Some((delta_dp, pseudobulk)) = delta {
        x_dn.adjust_by_division_of_selected_inplace(delta_dp, pseudobulk);
    }
    x_dn.log1p_inplace();
    x_dn.scale_columns_inplace();
}

/// Adjust the original data by eliminating batch effects `delta_db`
/// (`d x b`) from each column. We will directly call
/// `get_batch_membership` in `data_vec`.
///
/// # Arguments
/// * `data_vec` - sparse data vector
/// * `delta_dp` - row/feature by pseudobulk average effect matrix
///
/// # Returns
/// * `triplets` - we can feed this vector to create a new backend
fn triplets_adjusted_by_pseudobulk(
    data_vec: &SparseIoVec,
    delta_dp: &Mat,
) -> anyhow::Result<Vec<(u64, u64, f32)>> {
    let mut triplets = vec![];
    data_vec.visit_columns_by_block(&adjust_triplets_visitor, delta_dp, &mut triplets, None)?;
    Ok(triplets)
}

#[allow(clippy::type_complexity)]
fn adjust_triplets_visitor(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    delta_dp: &Mat,
    triplets: Arc<Mutex<&mut Vec<(u64, u64, f32)>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;

    let pbs = full_data_vec.get_group_membership(lb..ub)?;
    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    x_dn.adjust_by_division_of_selected_inplace(delta_dp, &pbs);

    let new_triplets = x_dn
        .triplet_iter()
        .filter_map(|(i, j, &x_ij)| {
            let x_ij = x_ij.round();
            if x_ij < 1_f32 {
                None
            } else {
                Some((i as u64, (j + lb) as u64, x_ij))
            }
        })
        .collect::<Vec<_>>();

    let mut triplets = triplets.lock().expect("lock triplets");
    triplets.extend(new_triplets);
    Ok(())
}

impl crate::update::Updatable for SvdArgs {
    fn rebase(&mut self, r: crate::update::Rebase) {
        self.data_files = r.data_files;
        self.batch_files = r.batch_files;
        self.out = r.out;
        self.pb_reference = r.reference;
        self.init_from = Some(r.init_from);
        // A revision votes on genes over the partition its labels were
        // judged on; svd collapses on its own for everything else.
        self.vote = r.peer.map(|peer| crate::svd::revise::Vote {
            peer,
            partition: r.pb_from.pb_from,
        });
        // `svd` has no weights and no epoch loop. `update` rejects `--epochs`
        // for an svd parent before reaching this.
    }
}
