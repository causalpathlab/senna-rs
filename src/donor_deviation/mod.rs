//! Stage 2 on a finished run (`senna donor-deviation`): which of its states
//! are specific to a donor, and which genes drive them.
//!
//! Stage 1 (a bge, topic-family, svd, vae or rbm run that recorded its
//! cell → pseudobulk partition) placed every cell in a space shared by every
//! donor; stage 2 reads that space in a Euclidean view of its kind
//! (`inputs::euclidean_view`). Stage 2
//! keeps that space frozen and asks, gene by gene, whether a donor's
//! pseudobulk samples carry the gene more than the samples of OTHER donors
//! that stage 1 puts in the same state. It is a contrast with two kinds of
//! negatives:
//!
//! 1. matched other-donor samples (`matched_other_donor_units`): the
//!    anchor's nearest samples of other donors in the stage-1 embedding
//!    (approximate kNN), so the shared state cancels and only the donor's
//!    own departure can win;
//! 2. same-section samples, in their robust form (`same_section_gene_factor`):
//!    a gene's per-batch factor, so anything constant within a batch (depth,
//!    the per-(gene, batch) fold stage 1 divided by, a donor-wide shift)
//!    cancels.
//!
//! A unit is a pseudobulk sample: the cells of one finest pseudobulk in one
//! batch, their raw counts averaged per cell (so a large and a small sample
//! are on one scale and neither dominates a contrast set by its size). The
//! counts are NOT divided by stage 1's batch fold: a fold estimated through
//! the collapse can take up part of a donor's own population, which is what
//! stage 2 looks for; the robust section factor removes the shared shift
//! instead.
//!
//! The per-gene fit (`contrast_fit`) gives `deviation[gene, donor]`, a linear
//! function of the unit's state coordinates (group lasso per donor row, so
//! it is 0 unless the data need it). Stage 2 decides nothing: it writes the
//! deviations and each unit's state coordinates and matched distances, from
//! which every gain, state summary or ranking follows.
//!
//! The fit core (`contrast_fit`, `batched_contrast_fit`, `section_jackknife`,
//! `same_section_gene_factor`, `fit_genes`, `matched_other_donor_units`) is
//! pinto's `apply donor-deviation` (pinto commit 7bda66d), whose units are
//! cell pairs; here they are pseudobulk samples. Nothing in it depends on
//! what a unit is.

mod batched_contrast_fit;
mod contrast_fit;
mod fit_genes;
mod inputs;
mod matched_other_donor_units;
pub(crate) mod outputs;
mod same_section_gene_factor;
mod samples;
mod section_jackknife;
#[cfg(test)]
mod tests;

use contrast_fit::{ContrastFitSettings, UnitStates};
use log::warn;
use rand::SeedableRng;
use senna::embed_common::*;

/// `senna donor-deviation`'s long help.
pub const DONOR_DEVIATION_LONG_ABOUT: &str =
    "Stage 2 on a finished run: which of its states are specific to a donor,\n\
     and which genes drive them. Takes any run that recorded its cell →\n\
     pseudobulk partition, except ode (two tracks of counts). Cells are\n\
     compared as `senna critique` compares them: an embedding by angle, θ by\n\
     Hellinger distance, signed scores z-scored.\n\n\
     The run's cell table is held fixed. Cells are grouped into pseudobulk\n\
     samples (the run's finest pseudobulk × batch, counts averaged per cell).\n\
     For each gene, every sample is contrasted with its nearest samples of\n\
     OTHER donors in the embedding (the same shared state elsewhere), on raw\n\
     counts with each batch's robust gene factor taken out (what is constant\n\
     within a batch: depth, the run's batch fold, a donor-wide shift). A donor\n\
     whose samples carry the gene more than their matches gets a deviation, a\n\
     linear function of the state (group lasso: 0 unless the data need it).\n\
     Needs at least three donors.\n\n\
     TYPICAL USE:\n\
     \x20 senna bge s1.zarr.zip,...,s6.zarr.zip -b b1,...,b6 -o stage1\n\
     \x20 senna donor-deviation --from stage1.senna.json --donor-of-batch donors.tsv -o stage2\n\n\
     Without --donor-of-batch each batch is its own donor.\n\n\
     Outputs (no cutoffs; everything else is derived from them):\n\
     \x20 {out}.donor_gene_deviation.parquet  per (gene, donor): intercept and\n\
     \x20     state_axis_1.. coefficients\n\
     \x20 {out}.donor_gene_deviation_without_batch.parquet  per (gene, donor),\n\
     \x20     the coefficients without each batch b ({coef}.without_{b}):\n\
     \x20     jackknife replicates; a quantity's variance is\n\
     \x20     (B - 1) / B · Σ_b (q_b - mean)²\n\
     \x20 {out}.donor_sample_states.parquet  per sample: batch, donor, cells,\n\
     \x20     anchor, matched_other_donors, matched distances, state_axis_1..\n\
     \x20 {out}.donor_cell_states.parquet    per cell: sample, batch, donor,\n\
     \x20     state_axis_1..\n\
     \x20 {out}.donor_deviation.json         settings and donors\n\n\
     A cell's log gain for a gene (the fitted log ratio of the gene in its\n\
     state over other donors' matched samples) is its donor's\n\
     intercept + Σ_k state_axis_k(gene, donor) · state_axis_k(cell).\n\
     A large matched distance marks a state no other donor has: the\n\
     deviation cannot see such a state, the distance does.";

#[derive(Args, Debug, Clone)]
pub struct DonorDeviationArgs {
    #[arg(
        long,
        help = "Stage-1 run (its .senna.json or prefix) whose cell embedding is held fixed"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        help = "Two-column file: each batch of the run and its donor [default: each batch is its own donor]",
        long_help = "Two-column text file, one batch per line: the batch label as\n\
                     the run's batch files name it and its donor, separated by a\n\
                     tab, a comma or spaces. A `batch donor` header is skipped.\n\
                     Without it each batch is its own donor. At least three\n\
                     donors are needed."
    )]
    pub donor_of_batch: Option<Box<str>>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix [default: the run's prefix with .stage2]"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Data files [default: those the run's manifest records]"
    )]
    pub data: Option<Vec<Box<str>>>,

    #[arg(
        long,
        short = 'b',
        value_delimiter = ',',
        help = "Batch label files, one per data file [default: the run's]"
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Matched other-donor samples per anchor sample (the contrast's negatives)"
    )]
    pub matched_samples: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Other donors an anchor's matched samples must come from",
        long_help = "An anchor sample takes part only when its matched samples come\n\
                     from at least this many donors other than its own: a set\n\
                     drawn from one other donor compares two donors, not a donor\n\
                     with the shared state. Every sample is still read out."
    )]
    pub min_other_donors: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Leading axes of the stage-1 embedding a deviation may vary along"
    )]
    pub state_axes: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Group-lasso penalty on each (gene, donor) deviation"
    )]
    pub group_lasso_penalty: f32,

    #[arg(
        long,
        default_value_t = 50_000,
        help = "Anchor samples the contrast uses at most (sampled; every cell is read out)"
    )]
    pub max_anchor_samples: usize,

    #[arg(
        long,
        default_value_t = 500,
        help = "FISTA iterations per gene at most"
    )]
    pub max_iterations: usize,

    #[arg(long, default_value_t = 1, help = "Seed for the anchor sample")]
    pub seed: u64,

    #[command(flatten)]
    pub compute: senna::embed_common::DeviceArgs,

    #[arg(
        long,
        default_value_t = 0.8,
        help = "Fraction of free GPU memory a block of genes may take",
        hide = true
    )]
    pub gpu_mem_fraction: f32,
}

/// The command line's default output prefix: the run's, with `.stage2`
/// (the MCP tool names its results in the session's results folder).
fn default_out(from: &str) -> String {
    format!("{}.stage2", senna::run_manifest::derive_out_prefix(from))
}

pub fn run_donor_deviation(args: &DonorDeviationArgs) -> anyhow::Result<()> {
    let out: String = match args.out.as_deref() {
        Some(out) => out.to_string(),
        None => default_out(&args.from),
    };
    anyhow::ensure!(
        args.matched_samples >= 1 && args.max_anchor_samples >= 1,
        "--matched-samples and --max-anchor-samples must be at least 1"
    );
    let inputs::Stage1Inputs {
        mut cells,
        data,
        cell_of_column,
    } = inputs::read_stage1_inputs(args)?;
    let number_of_donors = cells.donor_names.len();

    // state coordinates on the cells; a sample sits at its cells' mean
    let (state_coordinates_of_cell, width) =
        matched_other_donor_units::state_coordinates_of_units(&cells.embedding, args.state_axes);
    let mut samples = samples::pb_samples(&cells, &state_coordinates_of_cell, width);
    // the cells' table has given their coordinates and samples; only the
    // read-out (names, batches, donors, coordinates) is needed from here on
    cells.embedding = Mat::zeros(0, 0);
    let counts = samples::pool_sample_counts_by_gene(&data, &cell_of_column, &samples)?;
    drop(data);
    let number_of_samples = samples.number_of_samples();
    info!(
        "{} cells in {number_of_samples} pseudobulk samples (median {} cells), {number_of_donors} \
         donors",
        cells.cell_names.len(),
        median_cells(&samples.cells_in_sample)
    );

    let matched = matched_other_donor_units::find_matched_other_donor_units(
        &samples.embedding_of_sample,
        &samples.donor_index_of_sample,
        number_of_donors,
        args.matched_samples,
    )?;
    samples.embedding_of_sample = Mat::zeros(0, 0);
    let eligible = matched
        .anchors_spanning_other_donors(&samples.donor_index_of_sample, args.min_other_donors);
    if eligible.len() < number_of_samples {
        info!(
            "{} of {number_of_samples} samples anchor no set: their matches come from fewer \
             than {} other donors",
            number_of_samples - eligible.len(),
            args.min_other_donors
        );
    }
    anyhow::ensure!(
        !eligible.is_empty(),
        "no sample's matches span {} other donors; lower --min-other-donors or raise \
         --matched-samples",
        args.min_other_donors
    );
    let anchor_samples = sample_anchor_units(&eligible, args.max_anchor_samples, args.seed);
    let units_matching_unit = matched.units_matching_each_unit(number_of_samples);
    let sample_states = UnitStates {
        state_coordinates: &samples.state_coordinates_of_sample,
        coordinates_per_unit: width,
        donor_index_of_unit: &samples.donor_index_of_sample,
        number_of_donors,
    };
    let sections = index_sections(&samples.batch_of_sample);
    let settings = ContrastFitSettings {
        group_lasso_penalty: args.group_lasso_penalty,
        max_iterations: args.max_iterations,
        gradient_mapping_tolerance: GRADIENT_MAPPING_TOLERANCE,
    };
    info!(
        "Fitting {} genes: {} anchors, {} matched each, {width} state coordinates, {} batches",
        counts.gene_names.len(),
        anchor_samples.len(),
        args.matched_samples,
        sections.section_names.len()
    );
    let activity_inputs = fit_genes::ActivityInputs {
        matched: &matched,
        units_matching_unit: &units_matching_unit,
        sections: &sections,
    };
    let fitted_genes = fit_all_genes(
        args,
        &counts.nonzero_samples_of_gene,
        &activity_inputs,
        &anchor_samples,
        &sample_states,
        &settings,
    )?;
    let fits_at_cap: usize = fitted_genes
        .iter()
        .map(|fit| fit.fits_at_iteration_cap)
        .sum();
    if fits_at_cap > 0 {
        warn!(
            "{fits_at_cap} fits (full or jackknife) reached --max-iterations {} before \
             converging; raise it for their deviations to be at the optimum",
            args.max_iterations
        );
    }

    outputs::write_donor_deviation_outputs(
        &out,
        &outputs::DonorDeviationResults {
            gene_names: &counts.gene_names,
            section_names: &sections.section_names,
            cells: &cells,
            state_coordinates_of_cell: &state_coordinates_of_cell,
            width,
            samples: &samples,
            matched: &matched,
            anchor_samples: &anchor_samples,
            fitted_genes: &fitted_genes,
        },
    )?;
    outputs::write_settings_record(
        &out,
        args,
        &cells.donor_names,
        &settings,
        number_of_samples,
        anchor_samples.len(),
    )?;
    outputs::record_in_stage1_manifest(&args.from, &out)
}

/// Every gene's full fit and batch jackknife, on CPU threads or in blocks of
/// genes on a device.
fn fit_all_genes(
    args: &DonorDeviationArgs,
    nonzero_samples_of_gene: &[Vec<(u32, f32)>],
    activity_inputs: &fit_genes::ActivityInputs,
    anchor_samples: &[u32],
    sample_states: &UnitStates,
    settings: &ContrastFitSettings,
) -> anyhow::Result<Vec<fit_genes::FittedGene>> {
    Ok(match args.compute.device {
        ComputeDevice::Cpu => fit_genes::fit_genes_on_cpu_threads(
            nonzero_samples_of_gene,
            activity_inputs,
            anchor_samples,
            sample_states,
            settings,
        ),
        ComputeDevice::Cuda | ComputeDevice::Metal => {
            let device = args.compute.to_device()?;
            let sections = activity_inputs.sections;
            let shared = batched_contrast_fit::SharedContrastTensors::new(
                anchor_samples,
                activity_inputs.matched,
                sample_states,
                &sections.section_index_of_unit,
                sections.section_names.len(),
                &device,
            )?;
            let genes_per_block = fit_genes::genes_per_block(&shared, args.gpu_mem_fraction);
            info!(
                "On {}: {genes_per_block} genes per block",
                args.compute.device
            );
            fit_genes::fit_genes_in_blocks(
                nonzero_samples_of_gene,
                activity_inputs,
                &shared,
                settings,
                genes_per_block,
            )?
        }
    })
}

fn median_cells(cells_in_sample: &[usize]) -> usize {
    let mut sorted = cells_in_sample.to_vec();
    sorted.sort_unstable();
    sorted.get(sorted.len() / 2).copied().unwrap_or(0)
}

/// The fit stops when its gradient mapping is below this in norm (the loss
/// is normalized by the gene's total activity, so this is scale-free).
const GRADIENT_MAPPING_TOLERANCE: f32 = 1e-4;

/// The units' sections (batches), indexed in order of first appearance.
pub(crate) struct IndexedSections {
    pub section_index_of_unit: Vec<usize>,
    pub number_of_units_in_section: Vec<usize>,
    pub section_names: Vec<Box<str>>,
}

pub(crate) fn index_sections(section_of_unit: &[Box<str>]) -> IndexedSections {
    let mut index_of_name: rustc_hash::FxHashMap<&str, usize> = Default::default();
    let mut section_names: Vec<Box<str>> = Vec::new();
    let mut number_of_units_in_section: Vec<usize> = Vec::new();
    let section_index_of_unit = section_of_unit
        .iter()
        .map(|name| {
            let index = *index_of_name.entry(name).or_insert_with(|| {
                section_names.push(name.clone());
                number_of_units_in_section.push(0);
                section_names.len() - 1
            });
            number_of_units_in_section[index] += 1;
            index
        })
        .collect();
    IndexedSections {
        section_index_of_unit,
        number_of_units_in_section,
        section_names,
    }
}

/// At most `max_anchor_units` of the `eligible` units, sampled (sorted).
pub(crate) fn sample_anchor_units(
    eligible: &[u32],
    max_anchor_units: usize,
    seed: u64,
) -> Vec<u32> {
    if eligible.len() <= max_anchor_units {
        return eligible.to_vec();
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut anchors: Vec<u32> =
        rand::seq::index::sample(&mut rng, eligible.len(), max_anchor_units)
            .into_iter()
            .map(|i| eligible[i])
            .collect();
    anchors.sort_unstable();
    anchors
}
