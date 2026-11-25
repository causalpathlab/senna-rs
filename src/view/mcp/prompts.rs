//! senna's MCP prompts: recipes for the work that runs across senna's
//! tools and lupin's, from a fit to its cell types and from those to a
//! trajectory.

use legume_mcp::Prompt;

const ANNOTATE: &str = "\
Take {data} to cell types with the user, step by step; every choice below is \
theirs, so propose it with its reason and ask. Their marker panel: {markers}.
1. Fit: senna run_fit (method {method}) on the data. view_open the run and \
look at the map with the user.
2. Cluster: senna run_step clustering from the run's manifest into a new out \
(lupin annotates clusters; run_help clustering lists the methods: leiden \
with a resolution, or kmeans with a number of clusters).
3. Annotate: lupin annotate_run from the run's manifest with the marker \
panel (without one, lupin annotate_ask, and label the clusters it lists). \
Read the round with lupin annotate_review and annotate_genes.
4. Look: senna view_open the round (or view_map round on the run open) so \
its cell types are a grouping; view_markers a type and view_map focus it.
5. Revise: where a review shows a cluster mixing two types, cluster again \
finer (leiden at a higher resolution, or more clusters) into a new out and \
annotate that; where two clusters are one type, merge or relabel them with \
lupin annotate_relabel, after annotate_why on the choice. Rounds are never \
overwritten.
Close with notebook (kind conclusion): the types, the evidence for each, \
and what is still open.";

const TRAJECTORY: &str = "\
Order the cells of {run} (an annotated run: a lupin round of it, or view_open \
it to find one) along a trajectory with the user. The root is the user's to \
choose; their guess: {root}.
1. Types: senna view_open the round (or view_map round) so its cell types \
are the grouping; view_markers each candidate root type, and ask lupin \
annotate_why which type is least differentiated (stem or progenitor markers).
2. Root: senna view_cells with groups the chosen type writes its cells.
3. Fit: senna run_fit ode on the run's data ({data}) with flags root-cells \
set to that list and a new out. Its result's root says which cells it grew \
from and what they are; a needs_input is put to the user.
4. Check: senna view_open the ODE run and view_map time: time should run \
from the root type to the known mature types; view_map velocity draws the \
field.
5. Compare: lupin trajectory_run on the round with roots the same type, and \
trajectory_read; where the two orders disagree, say so.
Close with notebook (kind conclusion): the root, why it was chosen, and \
whether the fits agree.";

/// The prompts senna's server offers.
pub(super) fn prompts() -> Vec<Prompt> {
    vec![
        Prompt::new(
            "annotate",
            "Fit, cluster and annotate",
            "From data to cell types: a senna fit, its clustering, lupin's annotation \
             loop, and the round looked at here.",
            ANNOTATE,
        )
        .arg("data", "The data files to fit", true)
        .arg(
            "markers",
            "The marker panel (cell type, gene per line)",
            false,
        )
        .arg("method", "The fit (topic, bge, svd, …)", false),
        Prompt::new(
            "trajectory",
            "Root and fit a trajectory",
            "From an annotated run to an ODE trajectory: a root type chosen with \
             lupin, its cells written with view_cells, senna ode rooted there, and \
             lupin's trajectory to compare.",
            TRAJECTORY,
        )
        .arg(
            "run",
            "The annotated run's manifest, or its lupin round",
            true,
        )
        .arg(
            "data",
            "The two-track (spliced and unspliced) data files for senna ode",
            true,
        )
        .arg("root", "The root type the user has in mind", false),
    ]
}
