//! `senna fne`'s command-line surface. Training defaults are PBG's own
//! (the settings SIMBA's `pbg_train` uses) at the workspace's embedding
//! dimension, so a bare invocation is the published recipe.

use data_beans::aux::feature_names::FeatureNameKindArg;
use senna::embed_common::*;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "senna::embed_common::clap_defaults")]
pub struct FneArgs {
    // ── Gene–gene edges ──────────────────────────────────────────────
    #[arg(
        value_delimiter = ',',
        help_heading = "Gene–gene edges",
        help = "Homogeneous gene–gene pair file(s): gene gene [weight]; relation = file stem; PPI QC/SNN/PPR apply",
        long_help = "Zero or more positional paths, comma-separated or space-separated.\n\
                     Use these for a homogeneous gene–gene network (e.g. physical PPI).\n\
                     Each file is whitespace/comma/tab-delimited;\n\
                     every line is a pair of gene names,\n\
                     with an optional third column holding a per-edge weight (confidence ≥ 0).\n\
                     Lines starting with `#` are skipped;\n\
                     self-loops are dropped and a repeated pair keeps its largest weight.\n\
                     Every file is its own relation, named `gene:gene/<file stem>`\n\
                     (the file name without its .tsv/.csv/.gz extensions).\n\
                     Default scoring polarity is friend (Dot attract). That is a geometry\n\
                     default, not a biological claim — override any stem with\n\
                     `--relation-polarity gene:gene/<stem>=enemy` if you want linked\n\
                     endpoints to repel. For mixed relation kinds in one file, use\n\
                     `--named-pairs` instead.\n\
                     PPI QC and derived SNN/PPR relations apply here (see --ppi-*).\n\
                     \n\
                     Examples:\n\
                       senna fne string.tsv --membership cell_type=markers.tsv -o out\n\
                       senna fne string.tsv gi.tsv \\\n\
                         --relation-polarity gene:gene/gi=enemy -o out"
    )]
    pub(crate) networks: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Gene–gene edges",
        help = "Mixed gene–gene file(s): gene gene relation [weight]; no PPI derive",
        long_help = "Named gene–gene edge lists, comma-separated or repeated.\n\
                     Every line is `gene <TAB> gene <TAB> relation [<TAB> weight]`.\n\
                     The relation token becomes `gene:gene/<relation>` across the file;\n\
                     weight is confidence only (≥ 0) — never a polarity sign.\n\
                     Every relation defaults to friend (attract). You choose which\n\
                     relations should repel via `--relation-polarity` (full ids as logged);\n\
                     FNE does not map genetic / physical / DepMap labels to polarity for you.\n\
                     No PPI QC / SNN / PPR is derived from these files; for that, put the\n\
                     physical network on the positional NETWORKS argument.\n\
                     \n\
                     DepMap / CRISPR (no --depmap flag): sparsify first, then feed here.\n\
                       Co-essentiality → e.g. relation `coessential` (usually leave friend).\n\
                       Other derived graphs → their own relation tokens; set polarity yourself.\n\
                       Gene × cell-line essentials → prefer `--membership cell_line=...`.\n\
                     \n\
                     Examples:\n\
                       senna fne --named-pairs biogrid.tsv \\\n\
                         --relation-polarity gene:gene/genetic_sl=enemy \\\n\
                         --gmt hallmark.gmt -o out\n\
                       senna fne --named-pairs coessential.tsv \\\n\
                         --membership cell_line=essentials.tsv -o out"
    )]
    pub(crate) named_pairs: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = 0,
        help_heading = "Gene–gene edges",
        help = "PPI QC: drop pair edges whose endpoints share fewer than this many neighbours (0 = off)",
        long_help = "Quality control on the positional pair files, before anything else:\n\
                     drop an edge whose two endpoints share fewer than this many neighbours.\n\
                     An interaction with no corroborating shared partner is likely a noisy hit.\n\
                     0 keeps every edge. Does not apply to `--named-pairs`."
    )]
    pub(crate) ppi_min_shared_neighbors: usize,

    #[arg(
        long,
        default_value_t = 0,
        help_heading = "Gene–gene edges",
        help = "PPI QC: cap each gene's degree, keeping the neighbours with the most shared partners (0 = off)",
        long_help = "Cap the degree of every gene in the positional pair files:\n\
                     a hub keeps only its neighbours with the most shared partners,\n\
                     and an edge survives when either endpoint keeps it.\n\
                     Hubs otherwise dominate the embedding, since every batch samples them.\n\
                     0 keeps every edge. Does not apply to `--named-pairs`."
    )]
    pub(crate) ppi_max_degree: usize,

    #[arg(
        long,
        default_value_t = 0,
        help_heading = "Gene–gene edges",
        help = "PPI QC: iteratively drop genes with fewer edges than this (k-core; 0 = off)"
    )]
    pub(crate) ppi_min_degree: usize,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Gene–gene edges",
        help = "Do not derive the second-order (shared-neighbour) relation from the pair files",
        long_help = "By default every positional pair file also yields the relation `gene:gene/<stem>/snn`:\n\
                     for each gene, its --ppi-snn-k co-interactors that are not direct partners,\n\
                     ranked and weighted by the Jaccard overlap of their neighbourhoods\n\
                     (shared partners over the union), which discounts the hubs a scale-free\n\
                     network makes everyone share by chance.\n\
                     The first-order relation only ever sees direct interactions;\n\
                     this one lets co-interactors pull together. This flag leaves it out.\n\
                     Does not apply to `--named-pairs`."
    )]
    pub(crate) no_ppi_snn: bool,

    #[arg(
        long,
        default_value_t = 10,
        help_heading = "Gene–gene edges",
        help = "Co-interactors kept per gene in the shared-neighbour relation"
    )]
    pub(crate) ppi_snn_k: usize,

    #[arg(
        long,
        default_value_t = 1,
        help_heading = "Gene–gene edges",
        help = "Fewest shared neighbours for a pair to count as co-interactors"
    )]
    pub(crate) ppi_snn_min_shared: usize,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Gene–gene edges",
        help = "Do not derive the diffusion (personalized-PageRank) relation from the pair files",
        long_help = "By default every positional pair file also yields the relation `gene:gene/<stem>/ppr`:\n\
                     for each gene, its --ppi-ppr-k strongest targets under a random walk with restart\n\
                     (personalized PageRank, forward-push approximation),\n\
                     weighted by the PageRank mass relative to the gene's strongest target.\n\
                     This flag leaves it out. Does not apply to `--named-pairs`."
    )]
    pub(crate) no_ppi_ppr: bool,

    #[arg(
        long,
        default_value_t = 10,
        help_heading = "Gene–gene edges",
        help = "Targets kept per gene in the diffusion relation"
    )]
    pub(crate) ppi_ppr_k: usize,

    #[arg(
        long,
        default_value_t = 0.15,
        help_heading = "Gene–gene edges",
        help = "Restart probability of the diffusion random walk"
    )]
    pub(crate) ppi_ppr_restart: f64,

    // ── Membership and annotations ───────────────────────────────────
    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Membership and annotations",
        help = "Typed edge list(s): lhs_type, lhs, rhs_type, rhs [, weight]",
        long_help = "Typed edge files, comma-separated or repeated.\n\
                     Every line is `lhs_type <TAB> lhs <TAB> rhs_type <TAB> rhs [<TAB> weight]`,\n\
                     e.g. `gene TP53 term GO:0006915 1.0` or `gene TP53 word apoptosis 0.61`.\n\
                     Rows sharing a type pair form one relation named `lhs_type:rhs_type`,\n\
                     across every file.\n\
                     A relation whose two types coincide is undirected.\n\
                     Names of type `gene` are canonicalised like the positional files;\n\
                     every other type is matched exactly.\n\
                     \n\
                     Do NOT use this for mixed gene–gene interaction kinds (PPI vs genetic):\n\
                     all gene↔gene rows collapse into one `gene:gene` relation. Use\n\
                     `--named-pairs` for that. Default polarity is friend (affiliation Dot)."
    )]
    pub(crate) edges: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Membership and annotations",
        help = "Membership file(s), `type=path`: gene <TAB> label → gene:<type> edges (friend)",
        long_help = "Membership / affiliation files, `type=path`, comma-separated or repeated,\n\
                     e.g. `cell_type=markers.tsv` or `cell_line=essentials.tsv`.\n\
                     Every row is `gene <TAB> label`, tab or comma delimited;\n\
                     a header and `#` rows are skipped.\n\
                     The labels become nodes of the given type\n\
                     and the rows the relation `gene:<type>/<file stem>`.\n\
                     Default polarity is friend: the gene aligns with its label vector;\n\
                     co-members meet through the hub (not gene–gene enmity).\n\
                     \n\
                     DepMap CRISPR GeneEffect / Chronos: threshold or top-k essentials per\n\
                     line, write gene–cell_line rows, pass as `--membership cell_line=...`.\n\
                     Do not pass dense matrices directly."
    )]
    pub(crate) membership: Vec<Box<str>>,

    #[arg(
        long,
        help_heading = "Membership and annotations",
        help = "GO annotations (GAF, .gaf or .gaf.gz); needs --obo",
        long_help = "A GO annotation file.\n\
                     Every gene→term row becomes a gene:term edge (friend polarity),\n\
                     propagated up the ontology (is_a + part_of, the true-path rule) when --obo is given,\n\
                     so a gene annotated to a leaf also links to every ancestor.\n\
                     Terms outside --min/--max-gene-set are dropped."
    )]
    pub(crate) gaf: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Membership and annotations",
        help = "Ontology file(s) (OBO) for --gaf propagation, term:term edges and term text",
        long_help = "OBO ontologies (go-basic.obo, cl-basic.obo, efo.obo), comma-separated or repeated.\n\
                     Besides propagating --gaf annotations (with the ontology that holds GO,\n\
                     or the first one given),\n\
                     each hierarchy joins the graph as the relations `term:term/is_a` and `term:term/part_of`\n\
                     over the terms kept (friend polarity),\n\
                     and its names and definitions feed --export-text.\n\
                     `--obo efo.obo` gives the GWAS Catalog's EFO / MONDO trait terms their hierarchy.\n\
                     \n\
                     Example:\n\
                       senna fne --gaf goa_human.gaf.gz --obo go-basic.obo,efo.obo \\\n\
                         --links gwas-catalog:associations.tsv -o out"
    )]
    pub(crate) obo: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Membership and annotations",
        help = "Drop IEA (electronic) annotations from --gaf"
    )]
    pub(crate) no_iea: bool,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Membership and annotations",
        help = "Gene-set file(s) (GMT): each set becomes a term node (friend)",
        long_help = "MSigDB-style GMT files, comma-separated or repeated.\n\
                     Every line is `term <TAB> description <TAB> gene...`;\n\
                     the set's genes link to the term node in the relation `gene:term/<file stem>`\n\
                     (friend polarity).\n\
                     Sets outside --min/--max-gene-set are dropped;\n\
                     the description feeds --export-text."
    )]
    pub(crate) gmt: Vec<Box<str>>,

    #[arg(
        long,
        help_heading = "Regions and links",
        value_name = "TSV",
        help = "eQTL Catalogue dataset_metadata*.tsv: names each dataset's context (sample_group) and relation",
        long_help = "The eQTL Catalogue's dataset table (`dataset_metadata_r8.tsv` in\n\
                     github.com/eQTL-Catalogue/eQTL-Catalogue-resources, data_tables/).\n\
                     A `--links eqtl-catalogue:QTD000356.credible_sets.tsv.gz` file then takes its\n\
                     `sample_group` as its context (`blood`, `macrophage_Salmonella`, `CD4+_Th2`) and\n\
                     `<study>_<sample_group>[_<quant>]` as its relation stem (`GTEx_v10_blood`).\n\
                     Without it the dataset id names both."
    )]
    pub(crate) eqtl_metadata: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Regions and links",
        help = "--links: do not tie links to their tissue / cell type (no `context` nodes)",
        long_help = "By default a link measured in a tissue or cell type also ties its windows and its gene\n\
                     to a `context` node, in `region:context/<stem>` and `gene:context/<stem>`\n\
                     (the link's weight): GTEx takes the tissue from the file name (`Whole_Blood`),\n\
                     the eQTL Catalogue the dataset's sample_group (--eqtl-metadata),\n\
                     ABC / ENCODE-rE2G the CellType column, and `preset@context:path` names it for\n\
                     any source (e.g. `gtex@blood:Whole_Blood.v10.eQTLs.signif_pairs.parquet`, which\n\
                     also merges it with the eQTL Catalogue's `blood`). Labels are matched exactly.\n\
                     This flag leaves the context out."
    )]
    pub(crate) no_links_context: bool,

    #[arg(
        long,
        default_value_t = 1.0,
        help_heading = "Regions and links",
        help = "--links: keep a window/gene→context edge only if it reaches ≤ this share of all contexts (1 = keep all)",
        long_help = "After every --links source is read, keep a window→context or gene→context edge\n\
                     only when that window (or gene) reaches at most max(1, ⌈share · N⌉) of the N context\n\
                     nodes, counted over every source. A window active in most cell types says nothing\n\
                     about any one of them, and such windows are most of the E2G release (the median\n\
                     window is active in ~23 of 659 cell types, the median gene in 424).\n\
                     1 (default) keeps every edge; 0.05 keeps the specific ones."
    )]
    pub(crate) links_context_max_share: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help_heading = "Regions and links",
        help = "--links abc / e2g: keep enhancer-gene links scoring at least this"
    )]
    pub(crate) links_min_score: f64,

    #[arg(
        long,
        default_value_t = 5,
        help_heading = "Membership and annotations",
        help = "Smallest gene set (GAF/GMT term) kept"
    )]
    pub(crate) min_gene_set: usize,

    #[arg(
        long,
        default_value_t = 500,
        help_heading = "Membership and annotations",
        help = "Largest gene set (GAF/GMT term) kept; 0 = no cap"
    )]
    pub(crate) max_gene_set: usize,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Membership and annotations",
        help = "Plain region-gene link file(s): region <TAB> gene [<TAB> score]; see --links for published formats",
        long_help = "Genomic region→gene links (eQTL, peak-to-gene, ABC), comma-separated or repeated.\n\
                     A region is `chr:start-end`, `chr_start_end`, or a position `chr:pos` / `chr_pos`;\n\
                     `chr` prefixes are dropped.\n\
                     Each region is tiled onto fixed windows (--region-window)\n\
                     and links its gene from every window it overlaps,\n\
                     in the relation `region:gene/<file stem>` (friend polarity);\n\
                     the score column is the edge weight."
    )]
    pub(crate) region_gene: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = 10000,
        help_heading = "Regions and links",
        help = "Window size (bp) every region input is tiled onto; 0 keeps regions as given",
        long_help = "Window size (bp) the regions of --region-gene and --links are tiled onto:\n\
                     fixed windows `[i·w, (i+1)·w)` on 1-based coordinates.\n\
                     One window size serves every region input, so sources that hit the same window\n\
                     share one `region` node. 0 keeps each region as given\n\
                     (only sensible when every source names the same intervals)."
    )]
    pub(crate) region_window: i64,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Regions and links",
        value_name = "[PRESET:]PATH",
        help = "Association / enhancer-gene resource(s) in their published formats → region:<type>/<stem>",
        long_help = "Genomic link resources read in their published formats, repeatable\n\
                     (comma-separated too). `PRESET:PATH`, or a bare PATH whose preset is detected\n\
                     from its header. Every locus is tiled onto --region-window windows, and each\n\
                     source becomes the relation `region:<type>/<file stem>` (friend polarity).\n\
                     \n\
                     Presets:\n\
                       gwas-catalog    GWAS Catalog associations (v1.0.2, \"with ontology annotations\"):\n\
                                       CHR_ID/CHR_POS → `term` nodes of the MAPPED_TRAIT_URI ids\n\
                                       (EFO:…, MONDO:…; add --obo efo.obo for their hierarchy)\n\
                       gtex            GTEx signif pairs (.parquet or .txt.gz): variant_id → gene_id\n\
                       eqtl-catalogue  eQTL Catalogue SuSiE credible sets: variant → gene_id, weight = pip\n\
                       abc             ABC / ENCODE-rE2G predictions: chr,start,end → TargetGene,\n\
                                       weight = ABC.Score or Score (alias: encode-re2g)\n\
                       opengwas        a GWAS-VCF (ES:SE:LP) or a directory of them:\n\
                                       records with LP ≥ --links-min-lp → `trait` node per VCF sample\n\
                       e2g             the E2G parquet release directory (gs://e2g: cell_types.parquet,\n\
                                       enhancers.parquet, enhancer_gene_predictions/): enhancer → gene,\n\
                                       weight = score, one relation per model, context = cell type name\n\
                       positions       a VCF or a table of loci (locus/variant column, or chr+pos[+end]),\n\
                                       optional label/trait and weight columns → `trait` nodes\n\
                                       (the file stem when there is no label)\n\
                     \n\
                     `PRESET@CONTEXT:PATH` names the tissue / cell type of the whole file\n\
                     (see --no-links-context).\n\
                     \n\
                     Every file's genome build must match --genome-build.\n\
                     \n\
                     Example:\n\
                       senna fne --links gwas-catalog:gwas_catalog_v1.0.2-associations.tsv \\\n\
                         --links eqtl-catalogue:QTD000356.credible_sets.tsv.gz \\\n\
                         --links gtex:Whole_Blood.v10.eQTLs.signif_pairs.parquet \\\n\
                         --links abc:ENCODE_rE2G_K562.tsv.gz \\\n\
                         --gene-gff gencode.v39.basic.annotation.gtf.gz --obo efo.obo -o out/g2r"
    )]
    pub(crate) links: Vec<Box<str>>,

    #[arg(
        long,
        help_heading = "Regions and links",
        help = "--links: keep rows with p ≤ this (GWAS Catalog, GTEx, eQTL Catalogue, positions)"
    )]
    pub(crate) links_max_pvalue: Option<f64>,

    #[arg(
        long,
        default_value_t = 0.0,
        help_heading = "Regions and links",
        help = "--links eqtl-catalogue: keep credible-set variants with PIP ≥ this"
    )]
    pub(crate) links_min_pip: f64,

    #[arg(
        long,
        default_value_t = 7.30103,
        help_heading = "Regions and links",
        help = "--links opengwas: keep GWAS-VCF records with LP (−log10 p) ≥ this (default: p ≤ 5e-8)"
    )]
    pub(crate) links_min_lp: f64,

    #[arg(
        long,
        default_value_t = 5,
        help_heading = "Regions and links",
        help = "--links: drop trait/term nodes reached by fewer than this many distinct windows (≤1 = keep all)",
        long_help = "Drop the `term` / `trait` nodes that the --links sources reach through fewer than\n\
                     this many distinct region windows, and the windows left with no edge after that.\n\
                     A trait backed by one or two loci carries almost no signal for the embedding but\n\
                     still takes a row and crowds nearest-neighbour lists (about half of the\n\
                     GWAS Catalog's terms, holding ~2% of its edges, fall under 5).\n\
                     Terms that also carry a non-link edge (a --gaf / --gmt annotation, an --edges row)\n\
                     are kept; the ontology hierarchy does not count. 0 or 1 keeps every trait."
    )]
    pub(crate) links_min_trait_windows: usize,

    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "",
        value_name = "COLUMN",
        help_heading = "Regions and links",
        help = "--links: one relation per group, `region:<type>/<stem>/<group>` (ABC default: CellType)",
        long_help = "Split every --links source into one relation per group value,\n\
                     `region:<type>/<stem>/<group>`. Without a COLUMN the preset's own group is used:\n\
                     ABC / ENCODE-rE2G split by CellType, OpenGWAS by trait (VCF sample).\n\
                     With a COLUMN, that column of any table source."
    )]
    pub(crate) links_split_by_group: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Regions and links",
        help = "--links gwas-catalog: also link each hit to its MAPPED_GENEs (region:gene/<stem>/mapped_gene)",
        long_help = "Also turn the GWAS Catalog's MAPPED_GENE column into the relation\n\
                     `region:gene/<stem>/mapped_gene`. Off by default: the mapped genes are the\n\
                     nearest or overlapping genes, a positional annotation rather than evidence,\n\
                     and they would teach the embedding proximity that the windows already encode."
    )]
    pub(crate) gwas_mapped_gene: bool,

    #[arg(
        long,
        default_value = "GRCh38",
        help_heading = "Regions and links",
        help = "Genome build of the run (GRCh38 or GRCh37); every region input must match",
        long_help = "The reference assembly every region input is expected to be on.\n\
                     A source whose build is declared (GWAS Catalog, eQTL Catalogue: GRCh38) or\n\
                     detected (a `_b37` variant id, a VCF header, `hg19` in the file name) and differs\n\
                     is an error: windows of different builds do not name the same DNA.\n\
                     There is no in-tool liftover; lift the file over first."
    )]
    pub(crate) genome_build: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Regions and links",
        help = "Tile a source anyway when its genome build differs from --genome-build (warns)"
    )]
    pub(crate) allow_build_mismatch: bool,

    #[arg(
        long,
        help_heading = "Regions and links",
        help = "GFF/GTF mapping Ensembl gene ids to symbols across every input",
        long_help = "A GENCODE / Ensembl GFF or GTF (gzip or not). Its `gene` rows map\n\
                     version-stripped Ensembl ids onto gene symbols before the --feature-name-kind rule,\n\
                     for every input, so `ENSG00000141510.17` from GTEx and `TP53` from the GWAS Catalog\n\
                     or a PPI become one node. Without it, versions are still stripped, and a run that\n\
                     mixes ids and symbols warns."
    )]
    pub(crate) gene_gff: Option<Box<str>>,

    #[arg(
        long,
        help_heading = "Membership and annotations",
        help = "Write `feature <TAB> type <TAB> name <TAB> text` for every node with text",
        long_help = "Export the text the inputs carry (OBO names and definitions, GMT descriptions)\n\
                     as `feature <TAB> type <TAB> name <TAB> text`,\n\
                     one row per node that has any.\n\
                     This is the input of the text encoder that turns descriptions into gene:word edges."
    )]
    pub(crate) export_text: Option<Box<str>>,

    // ── Relation overrides ───────────────────────────────────────────
    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Relation overrides",
        help = "Relation weight override(s), `name=weight` (full relation id)",
        long_help = "Loss weight of a relation, `name=weight`, comma-separated or repeated.\n\
                     Names are the full ids the run logs,\n\
                     e.g. `gene:gene/biogrid=2`, `gene:term/goa_human=0.5` or `gene:word=0.5`.\n\
                     Every relation defaults to 1."
    )]
    pub(crate) relation_weight: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Relation overrides",
        help = "Passes over a relation's edges per epoch, `name=k`",
        long_help = "How many times a relation's training edges are drawn per epoch, `name=k`,\n\
                     comma-separated or repeated; every relation defaults to 1.\n\
                     Use the full relation id as logged.\n\
                     Batches are drawn in proportion to the edges left,\n\
                     so a small relation beside a large one (a marker panel beside a PPI)\n\
                     gets few updates per epoch and its nodes stay near their init;\n\
                     repeating it restores its share without changing the loss weight."
    )]
    pub(crate) relation_repeat: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help_heading = "Relation overrides",
        help = "Scoring polarity override(s), `full_id=friend|enemy` (you choose; no biology defaults)",
        long_help = "Polarity of a relation, `name=friend|enemy`, comma-separated or repeated.\n\
                     Use the full relation id as logged (e.g. `gene:gene/genetic_sl=enemy`).\n\
                     friend (default): Dot softmax-NCE — linked endpoints attract.\n\
                     enemy: anti-Dot softmax-NCE — linked endpoints repel.\n\
                     These names are geometric only. FNE never infers polarity from the\n\
                     relation token (physical vs genetic, positive vs negative GI, etc.);\n\
                     you decide which relations should attract or repel.\n\
                     Weight stays confidence (≥ 0); do not encode polarity as a signed weight.\n\
                     \n\
                     Example (illustrative — pick friend/enemy per your experiment):\n\
                       --relation-polarity gene:gene/genetic_sl=enemy"
    )]
    pub(crate) relation_polarity: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = graph_embedding_util::EmbeddingDim::Fixed(128),
        value_name = "H|auto",
        alias = "dim-embedding",
        help = "Embedding dimension H (auto = the width of a given feature embedding)"
    )]
    pub(crate) embedding_dim: graph_embedding_util::EmbeddingDim,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) train: crate::pbg_train_args::PbgTrainArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub(crate) feature_embedding: crate::feature_embedding_args::FeatureEmbeddingArgs,

    #[arg(
        long,
        default_value_t = 1,
        hide = true,
        help = "Held-out edges per relation at least, when the fraction is positive"
    )]
    pub(crate) eval_min_per_relation: usize,

    #[arg(
        long,
        value_enum,
        default_value = "gene",
        help = "How gene names are matched across inputs",
        long_help = "How gene names are matched across inputs.\n\
                     `gene` (default) takes the last `_`-separated token as canonical,\n\
                     so `ENSG00000_TGFB1` and `TGFB1` merge into one node.\n\
                     `exact` matches names as given. The other rules are `senna masked-topic`'s."
    )]
    pub(crate) feature_name_kind: FeatureNameKindArg,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Produces:\n  \
                     {out}.feature_embedding.parquet  N × H per-feature embeddings, every node type\n  \
                     {out}.feature_types.parquet      the node type of every row\n  \
                     {out}.relations.parquet          one row per relation: types, polarity, weight, counts\n  \
                     {out}.log_likelihood.parquet     per-epoch train and eval loss\n  \
                     {out}.senna.json                 run manifest"
    )]
    pub(crate) out: Box<str>,
}

impl FneArgs {
    pub(crate) fn name_kind(&self) -> data_beans::aux::feature_names::FeatureNameKind {
        self.feature_name_kind.resolve_or_gene()
    }
}
