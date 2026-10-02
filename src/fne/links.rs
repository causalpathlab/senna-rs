//! `--links [preset:]path`: association and enhancer–gene resources read in
//! their published formats and tiled onto the shared region windows.
//!
//! Every source becomes one relation `region:<type>/<stem>` (or one per
//! group with `--links-split-by-group`). The region side is always a window
//! of `--region-window` bp, so a GWAS hit, an eQTL credible-set variant and
//! an enhancer interval that fall in the same window meet in one region
//! node, whichever file they came from.
//!
//! | preset           | rows                                  | rhs node                       | weight |
//! |------------------|---------------------------------------|--------------------------------|--------|
//! | `gwas-catalog`   | `CHR_ID`, `CHR_POS`                   | `term` (EFO/MONDO ids of `MAPPED_TRAIT_URI`), else `trait` | 1 |
//! | `gtex`           | `variant_id` (`chr_pos_ref_alt_b38`)  | `gene` (`gene_id`)             | 1      |
//! | `eqtl-catalogue` | `variant` (`chr_pos_ref_alt`)         | `gene` (`gene_id`)             | `pip`  |
//! | `abc`            | `chr`, `start`, `end` (BED)           | `gene` (`TargetGene`)          | `ABC.Score` / `Score` |
//! | `opengwas`       | GWAS-VCF records (`LP` filtered)      | `trait` (the VCF sample id)    | 1      |
//! | `positions`      | a VCF, or a table of loci             | `trait` (a label column, else the file stem) | a weight column, else 1 |
//!
//! A link measured in a tissue or cell type also ties its windows and its
//! gene to a `context` node, in `region:context/<stem>` and
//! `gene:context/<stem>` (same weight as the link): GTEx takes the tissue
//! from the file name, the eQTL Catalogue its `sample_group` from
//! `--eqtl-metadata`, ABC the `CellType` column, and `preset@context:path`
//! names it for any source. Traits then sit near the contexts whose
//! regulatory windows they share.
//!
//! Coordinates are 1-based throughout: a BED start (0-based) is shifted by
//! one so an interval and a variant at the same base land in the same
//! window.

use super::graph::{
    curie_from_underscore, weight_value, NodeText, TypedGraphBuilder, GENE_TYPE, REGION_TYPE,
    TERM_TYPE,
};
use genomic_data::coordinates::{chr_stripped, PeakCoord};
use genomic_data::e2g::E2gRelease;
use genomic_data::variant::{parse_variant_id, GenomeBuild};
use genomic_data::vcf::VcfReader;
use legume_numeric::matrix::common_io::file_stem;
use legume_numeric::matrix::table::TableReader;
use log::{info, warn};
use rustc_hash::FxHashMap;
use std::path::Path;

/// Node type of a trait that has no ontology id.
pub(crate) const TRAIT_TYPE: &str = "trait";
/// Node type of the tissue / cell type / condition a link was measured in.
pub(crate) const CONTEXT_TYPE: &str = "context";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preset {
    GwasCatalog,
    Gtex,
    EqtlCatalogue,
    Abc,
    OpenGwas,
    Positions,
    E2g,
}

impl Preset {
    const ALL: [Preset; 7] = [
        Preset::GwasCatalog,
        Preset::Gtex,
        Preset::EqtlCatalogue,
        Preset::Abc,
        Preset::OpenGwas,
        Preset::Positions,
        Preset::E2g,
    ];

    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Preset::GwasCatalog => "gwas-catalog",
            Preset::Gtex => "gtex",
            Preset::EqtlCatalogue => "eqtl-catalogue",
            Preset::Abc => "abc",
            Preset::OpenGwas => "opengwas",
            Preset::Positions => "positions",
            Preset::E2g => "e2g",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        let s = s.to_ascii_lowercase();
        let s = match s.as_str() {
            "encode-re2g" | "re2g" => "abc",
            "eqtl" => "eqtl-catalogue",
            "gwas" => "gwas-catalog",
            other => other,
        };
        Self::ALL.into_iter().find(|p| p.as_str() == s)
    }
}

/// One `--links` value: an optional preset and the path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LinkSpec {
    pub preset: Option<Preset>,
    /// `preset@context:path`: the context of every link in the file.
    pub context: Option<Box<str>>,
    pub path: Box<str>,
}

impl LinkSpec {
    /// `preset[@context]:path` when the text before the first `:` is a
    /// preset name (with an optional `@context`), else the whole value is a
    /// path (so a path that contains a colon still works without a preset).
    pub(crate) fn parse(spec: &str) -> anyhow::Result<Self> {
        let spec = spec.trim();
        if let Some((head, tail)) = spec.split_once(':') {
            let (name, context) = match head.split_once('@') {
                Some((n, c)) => (n, Some(c.trim())),
                None => (head, None),
            };
            if let Some(p) = Preset::parse(name) {
                anyhow::ensure!(
                    !tail.is_empty(),
                    "--links `{spec}`: no path after `{head}:`"
                );
                anyhow::ensure!(
                    context.is_none_or(|c| !c.is_empty()),
                    "--links `{spec}`: empty context after `@`"
                );
                return Ok(Self {
                    preset: Some(p),
                    context: context.map(Box::from),
                    path: tail.into(),
                });
            }
        }
        anyhow::ensure!(!spec.is_empty(), "--links: empty value");
        Ok(Self {
            preset: None,
            context: None,
            path: spec.into(),
        })
    }
}

/// Filters and switches shared by every `--links` source.
#[derive(Clone, Debug)]
pub(crate) struct LinkOpts {
    pub window: i64,
    /// Keep rows with p ≤ this (GWAS Catalog, GTEx, eQTL Catalogue, and a
    /// `positions` table with a p column).
    pub max_pvalue: Option<f64>,
    /// Keep eQTL Catalogue credible-set variants with PIP ≥ this.
    pub min_pip: f64,
    /// Keep GWAS-VCF records with `LP` (−log10 p) ≥ this.
    pub min_lp: f64,
    /// `Some("")` = split by the preset's own group column (ABC: `CellType`);
    /// `Some(col)` = split by that column; `None` = one relation per source.
    pub split_by_group: Option<Box<str>>,
    /// GWAS Catalog: also link each hit to its `MAPPED_GENE`s.
    pub gwas_mapped_gene: bool,
    pub genome_build: GenomeBuild,
    pub allow_build_mismatch: bool,
    /// ABC / E2G: keep links scoring at least this.
    pub min_score: f64,
    /// Tie links to their tissue / cell type (`--no-links-context` clears).
    pub context: bool,
    /// eQTL Catalogue `dataset_id` → its dataset (`--eqtl-metadata`).
    pub eqtl_datasets: FxHashMap<Box<str>, EqtlDataset>,
}

impl LinkOpts {
    /// The `--links` settings of a run, checked, with `--eqtl-metadata` read.
    pub(crate) fn from_args(args: &super::FneArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(
            args.links_max_pvalue.is_none_or(|p| p > 0.0 && p <= 1.0),
            "fne: --links-max-pvalue must lie in (0, 1]"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&args.links_min_pip),
            "fne: --links-min-pip must lie in [0, 1]"
        );
        anyhow::ensure!(
            args.links_context_max_share > 0.0 && args.links_context_max_share <= 1.0,
            "fne: --links-context-max-share must lie in (0, 1]"
        );
        Ok(Self {
            window: args.region_window,
            max_pvalue: args.links_max_pvalue,
            min_pip: args.links_min_pip,
            min_lp: args.links_min_lp,
            min_score: args.links_min_score,
            split_by_group: args.links_split_by_group.clone(),
            gwas_mapped_gene: args.gwas_mapped_gene,
            genome_build: args.genome_build.parse()?,
            allow_build_mismatch: args.allow_build_mismatch,
            context: !args.no_links_context,
            eqtl_datasets: match &args.eqtl_metadata {
                Some(path) => read_eqtl_metadata(path)?,
                None => Default::default(),
            },
        })
    }
}

/// One row of the eQTL Catalogue's `dataset_metadata` table.
#[derive(Clone, Debug)]
pub(crate) struct EqtlDataset {
    /// `sample_group`: tissue or cell type, with its condition when not
    /// naive (`blood`, `macrophage_Salmonella`, `CD4+_Th2`).
    pub sample_group: Box<str>,
    /// `<study_label>_<sample_group>`, plus `_<quant_method>` when not
    /// gene expression: the relation stem in place of `QTD000356`.
    pub stem: Box<str>,
}

/// Read the eQTL Catalogue's `dataset_metadata*.tsv`.
pub(crate) fn read_eqtl_metadata(path: &str) -> anyhow::Result<FxHashMap<Box<str>, EqtlDataset>> {
    let t = TableReader::open(path)?;
    let cols = t.select(&["dataset_id", "study_label", "sample_group", "quant_method"])?;
    let mut out = FxHashMap::default();
    for row in t.rows(&cols)? {
        let row = row?;
        let (id, study, group, quant) =
            (row[0].trim(), row[1].trim(), row[2].trim(), row[3].trim());
        if id.is_empty() || group.is_empty() {
            continue;
        }
        let mut stem = format!("{study}_{group}");
        if !quant.is_empty() && quant != "ge" {
            stem = format!("{stem}_{quant}");
        }
        out.insert(
            id.into(),
            EqtlDataset {
                sample_group: group.into(),
                stem: group_token(&stem).into(),
            },
        );
    }
    info!("fne: --eqtl-metadata {path}: {} datasets", out.len());
    Ok(out)
}

/// Where a source's build came from, for the mismatch message.
enum BuildEvidence {
    /// The format fixes it (GWAS Catalog, eQTL Catalogue are GRCh38).
    Declared(GenomeBuild),
    /// Read off the file (a `_b37` variant id, a VCF header, a file name).
    Detected(GenomeBuild),
    Unknown,
}

/// Refuse a source whose build is not the run's, unless allowed. `found`
/// is the build the file itself states (read off it as `how`); without one
/// the file name is searched for `hg19` / `GRCh38` / `b37`….
pub(crate) fn check_build(
    path: &str,
    what: &str,
    found: Option<GenomeBuild>,
    how: &str,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    let (evidence, how) = match found {
        Some(b) => (Some(b), how),
        None => (GenomeBuild::detect(&file_stem(path)), "the file name"),
    };
    let ev = match evidence {
        Some(b) => BuildEvidence::Detected(b),
        None => BuildEvidence::Unknown,
    };
    check_build_evidence(path, what, ev, how, opts)
}

fn check_build_evidence(
    path: &str,
    what: &str,
    ev: BuildEvidence,
    how: &str,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    let (b, kind) = match ev {
        BuildEvidence::Declared(b) => (b, "declared"),
        BuildEvidence::Detected(b) => (b, "detected"),
        BuildEvidence::Unknown => {
            info!(
                "fne: {path}: no genome build found; assuming it matches --genome-build {}",
                opts.genome_build
            );
            return Ok(());
        }
    };
    if b == opts.genome_build {
        return Ok(());
    }
    let msg = format!(
        "fne: {path} ({what}) is {b} ({kind} from {how}) but the run is --genome-build {}; \
         lift the file over first (there is no in-tool liftover), or pass \
         --allow-build-mismatch to tile it anyway",
        opts.genome_build
    );
    anyhow::ensure!(opts.allow_build_mismatch, msg);
    warn!("{msg} — continuing because of --allow-build-mismatch");
    Ok(())
}

/// Append an optional column to a row request; its position in the rows.
fn push_col(cols: &mut Vec<usize>, c: Option<usize>) -> Option<usize> {
    c.map(|j| {
        cols.push(j);
        cols.len() - 1
    })
}

/// Pick the preset of a source from its shape: a directory or a GWAS-VCF
/// is `opengwas`, another VCF `positions`; a table by its header.
pub(crate) fn detect_preset(path: &str) -> anyhow::Result<Preset> {
    if Path::new(path).is_dir() {
        return Ok(if E2gRelease::is_release(Path::new(path)) {
            Preset::E2g
        } else {
            Preset::OpenGwas
        });
    }
    if is_vcf_path(path) {
        let r = VcfReader::open(path)?;
        return Ok(if r.is_gwas_vcf() {
            Preset::OpenGwas
        } else {
            Preset::Positions
        });
    }
    let t = TableReader::open(path)?;
    Ok(if t.has_columns(&["CHR_ID", "CHR_POS"]) {
        Preset::GwasCatalog
    } else if t.has_columns(&["molecular_trait_id", "variant", "pip"]) {
        Preset::EqtlCatalogue
    } else if t.has_columns(&["variant_id"])
        && t.find_column(&["gene_id", "phenotype_id"]).is_some()
    {
        Preset::Gtex
    } else if t.has_columns(&["TargetGene"]) {
        Preset::Abc
    } else {
        Preset::Positions
    })
}

fn is_vcf_path(path: &str) -> bool {
    let p = path.to_ascii_lowercase();
    [".vcf", ".vcf.gz", ".vcf.bgz"]
        .iter()
        .any(|e| p.ends_with(e))
}

/// A group value made safe for a relation id: whitespace and `/` become `_`.
fn group_token(g: &str) -> String {
    g.trim()
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == '/' {
                '_'
            } else {
                c
            }
        })
        .collect()
}

/// Per-source counts for the log line.
#[derive(Default)]
struct Tally {
    rows: usize,
    kept: usize,
    no_locus: usize,
    filtered: usize,
    no_target: usize,
}

impl Tally {
    fn log(&self, path: &str, preset: Preset, builder: &TypedGraphBuilder, rels: &[usize]) {
        let names: Vec<String> = rels
            .iter()
            .map(|&r| {
                let (e, w) = builder.relation_size(r);
                let name = builder.relation_name(r);
                if builder.relation_lhs_is(r, REGION_TYPE) {
                    format!("`{name}` ({e} edges over {w} windows)")
                } else {
                    format!("`{name}` ({e} edges)")
                }
            })
            .collect();
        let shown = if names.len() > 4 {
            format!("{} … and {} more", names[..4].join(", "), names.len() - 4)
        } else {
            names.join(", ")
        };
        info!(
            "fne: {path} [{}]: {} rows, {} kept ({} filtered, {} without a locus, {} without a target) → {shown}",
            preset.as_str(),
            self.rows,
            self.kept,
            self.filtered,
            self.no_locus,
            self.no_target
        );
    }
}

/// The relation roster of one source, split by group when asked.
struct Relations {
    base: String,
    lhs_type: &'static str,
    rhs_type: &'static str,
    /// Relation ids in first-use order, for the log.
    ids: Vec<usize>,
    /// Group value → relation id (`""` = the unsplit relation), so a row
    /// costs one lookup rather than a formatted name.
    by_group: FxHashMap<Box<str>, usize>,
}

impl Relations {
    fn new(rhs_type: &'static str, stem: &str) -> Self {
        Self::between(REGION_TYPE, rhs_type, stem)
    }

    fn between(lhs_type: &'static str, rhs_type: &'static str, stem: &str) -> Self {
        Self {
            base: format!("{lhs_type}:{rhs_type}/{stem}"),
            lhs_type,
            rhs_type,
            ids: Vec::new(),
            by_group: FxHashMap::default(),
        }
    }

    fn get(&mut self, b: &mut TypedGraphBuilder, group: Option<&str>) -> usize {
        let g = group.map_or("", str::trim);
        if let Some(&r) = self.by_group.get(g) {
            return r;
        }
        let name = if g.is_empty() {
            self.base.clone()
        } else {
            format!("{}/{}", self.base, group_token(g))
        };
        let r = b.relation(&name, self.lhs_type, self.rhs_type);
        if !self.ids.contains(&r) {
            self.ids.push(r);
        }
        self.by_group.insert(g.into(), r);
        r
    }
}

/// The context side of one source: `region:context/<stem>` and
/// `gene:context/<stem>`, fed alongside every link.
struct Contexts {
    on: bool,
    /// `preset@context`: overrides whatever the file says.
    fixed: Option<Box<str>>,
    region: Relations,
    gene: Relations,
}

impl Contexts {
    fn new(stem: &str, fixed: Option<&str>, on: bool) -> Self {
        Self {
            on,
            fixed: fixed.map(Box::from),
            region: Relations::new(CONTEXT_TYPE, stem),
            gene: Relations::between(GENE_TYPE, CONTEXT_TYPE, stem),
        }
    }

    /// Tie a link's windows (and its gene, if any) to `found`, or to the
    /// fixed context when one was named; no context, no edge.
    fn link(
        &mut self,
        b: &mut TypedGraphBuilder,
        wins: &[u32],
        gene: Option<u32>,
        found: Option<&str>,
        w: f32,
    ) {
        if !self.on {
            return;
        }
        let ctx = match self.fixed.as_deref().or(found).map(str::trim) {
            Some(c) if !c.is_empty() => c,
            _ => return,
        };
        let c = b.node_id(CONTEXT_TYPE, ctx);
        let r = self.region.get(b, None);
        b.link_windows(r, wins, c, w);
        if let Some(g) = gene {
            let r = self.gene.get(b, None);
            b.add_edge(r, g, c, w);
        }
    }
}

/// Where one source's links go: its relations (split by group), its context
/// edges, and the window buffer every link resolves its locus into.
struct Sink {
    rels: Relations,
    ctxs: Contexts,
    wins: Vec<u32>,
}

impl Sink {
    fn new(rhs_type: &'static str, stem: &str, fixed_ctx: Option<&str>, opts: &LinkOpts) -> Self {
        Self {
            rels: Relations::new(rhs_type, stem),
            ctxs: Contexts::new(stem, fixed_ctx, opts.context),
            wins: Vec::new(),
        }
    }

    /// One link: the windows of `locus` → `rhs` in the group's relation,
    /// plus the windows (and `rhs`, when a gene) → the context.
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        b: &mut TypedGraphBuilder,
        locus: &PeakCoord,
        window: i64,
        group: Option<&str>,
        rhs: &str,
        w: f32,
        found_ctx: Option<&str>,
    ) {
        b.windows(locus, window, &mut self.wins);
        let r = self.rels.get(b, group);
        let j = b.node_id(self.rels.rhs_type, rhs);
        b.link_windows(r, &self.wins, j, w);
        let gene = (self.rels.rhs_type == GENE_TYPE).then_some(j);
        self.ctxs.link(b, &self.wins, gene, found_ctx, w);
    }

    /// Every relation this source touched, for the log.
    fn ids(&self) -> Vec<usize> {
        [&self.rels, &self.ctxs.region, &self.ctxs.gene]
            .iter()
            .flat_map(|r| r.ids.iter().copied())
            .collect()
    }
}

/// Read one `--links` source into the graph.
pub(crate) fn add_links(
    b: &mut TypedGraphBuilder,
    spec: &LinkSpec,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    let path: &str = &spec.path;
    let preset = match spec.preset {
        Some(p) => p,
        None => {
            let p = detect_preset(path)?;
            info!("fne: {path}: detected --links preset `{}`", p.as_str());
            p
        }
    };
    let ctx = spec.context.as_deref();
    match preset {
        Preset::GwasCatalog => add_gwas_catalog(b, path, ctx, opts),
        Preset::Gtex => add_gtex(b, path, ctx, opts),
        Preset::EqtlCatalogue => add_eqtl_catalogue(b, path, ctx, opts),
        Preset::Abc => add_abc(b, path, ctx, opts),
        Preset::OpenGwas => add_opengwas(b, path, ctx, opts),
        Preset::Positions => add_positions(b, path, ctx, opts),
        Preset::E2g => add_e2g(b, path, ctx, opts),
    }
}

/// The group column of a table source: `--links-split-by-group` names it,
/// or (empty) asks for the preset's default.
fn group_column(
    t: &TableReader,
    opts: &LinkOpts,
    default: Option<&str>,
) -> anyhow::Result<Option<usize>> {
    match opts.split_by_group.as_deref() {
        None => Ok(None),
        Some("") => match default {
            Some(col) => Ok(Some(t.select(&[col])?[0])),
            None => {
                warn!(
                    "fne: {}: --links-split-by-group has no default group column for this source; \
                     name one (--links-split-by-group COLUMN)",
                    t.path()
                );
                Ok(None)
            }
        },
        Some(col) => match t.column_index(col) {
            Some(j) => Ok(Some(j)),
            None => {
                warn!(
                    "fne: {}: no column `{col}` to split by; one relation for the file",
                    t.path()
                );
                Ok(None)
            }
        },
    }
}

fn parse_p(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok().filter(|p| p.is_finite())
}

fn passes_p(p: &str, opts: &LinkOpts) -> bool {
    match opts.max_pvalue {
        None => true,
        Some(max) => parse_p(p).is_some_and(|p| p <= max),
    }
}

/// `http://www.ebi.ac.uk/efo/EFO_0004340` → `EFO:0004340`: the id OBO files
/// use, so an `--obo efo.obo` hierarchy joins the GWAS term nodes.
pub(crate) fn ontology_id_from_uri(uri: &str) -> Option<Box<str>> {
    let last = uri.trim().rsplit('/').next()?.trim();
    (!last.is_empty()).then(|| curie_from_underscore(last).map_or(last.into(), Into::into))
}

/// `MAPPED_GENE` cells list genes as `A, B` (overlapping), `A - B`
/// (flanking an intergenic hit) or `A x B` (interactions).
fn split_mapped_genes(s: &str) -> Vec<&str> {
    s.split([',', ';'])
        .flat_map(|p| p.split(" - "))
        .flat_map(|p| p.split(" x "))
        .map(str::trim)
        .filter(|g| !g.is_empty() && *g != "NR" && *g != "NA")
        .collect()
}

fn gwas_catalog_loci(chr: &str, pos: &str) -> Vec<PeakCoord> {
    let split = |s: &str| -> Vec<String> {
        s.split([';', 'x'])
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect()
    };
    let (chrs, poss) = (split(chr), split(pos));
    if chrs.len() != poss.len() {
        return Vec::new();
    }
    chrs.iter()
        .zip(&poss)
        .filter_map(|(c, p)| {
            let p: i64 = p.parse().ok()?;
            Some(PeakCoord {
                chr: chr_stripped(c).into(),
                start: p,
                end: p + 1,
            })
        })
        .collect()
}

fn add_gwas_catalog(
    b: &mut TypedGraphBuilder,
    path: &str,
    fixed_ctx: Option<&str>,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    check_build_evidence(
        path,
        "GWAS Catalog",
        BuildEvidence::Declared(GenomeBuild::GRCh38),
        "the GWAS Catalog, which maps every association to GRCh38",
        opts,
    )?;
    let t = TableReader::open(path)?;
    let stem = file_stem(path);
    let base = t.select(&["CHR_ID", "CHR_POS"])?;
    let uri = t.column_index("MAPPED_TRAIT_URI");
    let (rhs_type, label_col) = match uri {
        Some(j) => (TERM_TYPE, j),
        None => {
            warn!(
                "fne: {path}: no MAPPED_TRAIT_URI column (use the release \"with ontology \
                 annotations\"); traits become `trait` nodes named by DISEASE/TRAIT"
            );
            (TRAIT_TYPE, t.select(&["DISEASE/TRAIT"])?[0])
        }
    };
    let name_col = t.column_index("MAPPED_TRAIT");
    let p_col = t.column_index("P-VALUE");
    if opts.max_pvalue.is_some() && p_col.is_none() {
        warn!("fne: {path}: no P-VALUE column; --links-max-pvalue is not applied");
    }
    let gene_col = if opts.gwas_mapped_gene {
        Some(t.select(&["MAPPED_GENE"])?[0])
    } else {
        None
    };
    let group_col = group_column(&t, opts, None)?;
    let mut cols = vec![base[0], base[1], label_col];
    let name_at = push_col(&mut cols, name_col);
    let p_at = push_col(&mut cols, p_col);
    let gene_at = push_col(&mut cols, gene_col);
    let group_at = push_col(&mut cols, group_col);

    let mut sink = Sink::new(rhs_type, &stem, fixed_ctx, opts);
    // The mapped genes share the hits' windows; their context is the hits'.
    let mut gene_sink = Sink::new(GENE_TYPE, &format!("{stem}/mapped_gene"), None, opts);
    gene_sink.ctxs.on = false;
    let mut n = Tally::default();
    for row in t.rows(&cols)? {
        let row = row?;
        n.rows += 1;
        let loci = gwas_catalog_loci(&row[0], &row[1]);
        if loci.is_empty() {
            n.no_locus += 1;
            continue;
        }
        if let Some(k) = p_at {
            if !passes_p(&row[k], opts) {
                n.filtered += 1;
                continue;
            }
        }
        let labels: Vec<Box<str>> = match rhs_type {
            TERM_TYPE => row[2].split(',').filter_map(ontology_id_from_uri).collect(),
            _ => {
                let l = row[2].trim();
                if l.is_empty() {
                    vec![]
                } else {
                    vec![l.into()]
                }
            }
        };
        let genes = gene_at
            .map(|k| split_mapped_genes(&row[k]))
            .unwrap_or_default();
        if labels.is_empty() && genes.is_empty() {
            n.no_target += 1;
            continue;
        }
        n.kept += 1;
        let group = group_at.map(|k| row[k].as_ref());
        if !labels.is_empty() {
            // Names ride along in the same order when the counts agree (a
            // trait name can itself contain a comma, so a mismatch is left
            // unnamed rather than misnamed).
            let names: Vec<&str> = name_at
                .map(|k| row[k].split(',').map(str::trim).collect())
                .unwrap_or_default();
            for (i, id) in labels.iter().enumerate() {
                for l in &loci {
                    sink.emit(b, l, opts.window, group, id, 1.0, None);
                }
                if rhs_type == TERM_TYPE && names.len() == labels.len() && !names[i].is_empty() {
                    b.set_text(
                        TERM_TYPE,
                        id,
                        NodeText {
                            name: Some(names[i].into()),
                            text: None,
                        },
                    );
                }
            }
        }
        for g in &genes {
            for l in &loci {
                gene_sink.emit(b, l, opts.window, group, g, 1.0, None);
            }
        }
    }
    let mut all = sink.ids();
    all.extend(gene_sink.ids());
    n.log(path, Preset::GwasCatalog, b, &all);
    Ok(())
}

/// A GTEx `phenotype_id` of an sQTL / apaQTL carries its gene last,
/// `chr1:14829:14970:clu_1:ENSG00000227232.5`.
fn gtex_gene(s: &str) -> &str {
    s.rsplit(':').next().unwrap_or(s).trim()
}

fn add_gtex(
    b: &mut TypedGraphBuilder,
    path: &str,
    fixed_ctx: Option<&str>,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    let t = TableReader::open(path)?;
    let stem = file_stem(path);
    let v = t.select(&["variant_id"])?[0];
    let g = t
        .find_column(&["gene_id", "phenotype_id"])
        .ok_or_else(|| anyhow::anyhow!("{path}: no gene_id or phenotype_id column"))?;
    let p = t.find_column(&["pval_nominal", "pvalue", "pval"]);
    let group_col = group_column(&t, opts, None)?;
    let mut cols = vec![v, g];
    let p_at = push_col(&mut cols, p);
    let group_at = push_col(&mut cols, group_col);
    let mut sink = Sink::new(GENE_TYPE, &stem, fixed_ctx, opts);
    // GTEx names its files `<Tissue>.v10.…`: the tissue is the context.
    let tissue = stem.split('.').next().unwrap_or(&stem).to_string();
    let mut n = Tally::default();
    let mut build_checked = false;
    for row in t.rows(&cols)? {
        let row = row?;
        n.rows += 1;
        let Some(var) = parse_variant_id(&row[0]) else {
            n.no_locus += 1;
            continue;
        };
        if !build_checked {
            check_build(path, "GTEx", var.build, "the variant id suffix", opts)?;
            build_checked = true;
        }
        if let Some(k) = p_at {
            if !passes_p(&row[k], opts) {
                n.filtered += 1;
                continue;
            }
        }
        let gene = gtex_gene(&row[1]);
        if gene.is_empty() {
            n.no_target += 1;
            continue;
        }
        n.kept += 1;
        let group = group_at.map(|k| row[k].as_ref());
        sink.emit(
            b,
            &var.locus(),
            opts.window,
            group,
            gene,
            1.0,
            Some(&tissue),
        );
    }
    n.log(path, Preset::Gtex, b, &sink.ids());
    Ok(())
}

fn add_eqtl_catalogue(
    b: &mut TypedGraphBuilder,
    path: &str,
    fixed_ctx: Option<&str>,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    check_build_evidence(
        path,
        "eQTL Catalogue",
        BuildEvidence::Declared(GenomeBuild::GRCh38),
        "the eQTL Catalogue, which is uniformly processed on GRCh38",
        opts,
    )?;
    let t = TableReader::open(path)?;
    // `QTD000356.credible_sets` → dataset QTD000356: its sample group is
    // the context and `<study>_<sample_group>` the relation stem.
    let file = file_stem(path);
    let dataset = file.split('.').next().unwrap_or(&file);
    let (stem, found_ctx): (String, String) = match opts.eqtl_datasets.get(dataset) {
        Some(d) => (d.stem.to_string(), d.sample_group.to_string()),
        None => {
            if opts.context && fixed_ctx.is_none() {
                info!(
                    "fne: {path}: dataset `{dataset}` is not in --eqtl-metadata; \
                     its context is named by the dataset id"
                );
            }
            (file.clone(), dataset.to_string())
        }
    };
    let base = t.select(&["variant", "gene_id", "pip"])?;
    let p = t.column_index("pvalue");
    let group_col = group_column(&t, opts, None)?;
    let mut cols = base.clone();
    let p_at = push_col(&mut cols, p);
    let group_at = push_col(&mut cols, group_col);
    let mut sink = Sink::new(GENE_TYPE, &stem, fixed_ctx, opts);
    let mut n = Tally::default();
    for row in t.rows(&cols)? {
        let row = row?;
        n.rows += 1;
        let Some(locus) = super::graph::outside_locus(&row[0]) else {
            n.no_locus += 1;
            continue;
        };
        let pip = weight_value(&row[2]).unwrap_or(0.0);
        let p_ok = p_at.is_none_or(|k| passes_p(&row[k], opts));
        if (pip as f64) < opts.min_pip || !p_ok {
            n.filtered += 1;
            continue;
        }
        let gene = row[1].trim();
        if gene.is_empty() {
            n.no_target += 1;
            continue;
        }
        n.kept += 1;
        let group = group_at.map(|k| row[k].as_ref());
        sink.emit(b, &locus, opts.window, group, gene, pip, Some(&found_ctx));
    }
    n.log(path, Preset::EqtlCatalogue, b, &sink.ids());
    Ok(())
}

fn add_abc(
    b: &mut TypedGraphBuilder,
    path: &str,
    fixed_ctx: Option<&str>,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    check_build(path, "ABC / ENCODE-rE2G", None, "", opts)?;
    let t = TableReader::open(path)?;
    let stem = file_stem(path);
    let base = t.select(&["chr", "start", "end", "TargetGene"])?;
    let score = t.find_column(&["ABC.Score", "Score", "score"]);
    if score.is_none() {
        warn!("fne: {path}: no ABC.Score / Score column; every link weighs 1");
    }
    let default_group = t.column_index("CellType").map(|_| "CellType");
    let group_col = group_column(&t, opts, default_group)?;
    let mut cols = base.clone();
    let s_at = push_col(&mut cols, score);
    let group_at = push_col(&mut cols, group_col);
    let cell_at = push_col(&mut cols, t.column_index("CellType"));
    let mut sink = Sink::new(GENE_TYPE, &stem, fixed_ctx, opts);
    let mut n = Tally::default();
    for row in t.rows(&cols)? {
        let row = row?;
        n.rows += 1;
        let (Ok(start), Ok(end)) = (row[1].trim().parse::<i64>(), row[2].trim().parse::<i64>())
        else {
            n.no_locus += 1;
            continue;
        };
        if end <= start || row[0].trim().is_empty() {
            n.no_locus += 1;
            continue;
        }
        // BED is 0-based half-open; shift to the 1-based frame of the
        // variant sources.
        let region = PeakCoord {
            chr: chr_stripped(row[0].trim()).into(),
            start: start + 1,
            end: end + 1,
        };
        let gene = row[3].trim();
        if gene.is_empty() {
            n.no_target += 1;
            continue;
        }
        let w = match s_at {
            Some(k) => match weight_value(&row[k]) {
                Some(w) if w as f64 >= opts.min_score => w,
                _ => {
                    n.filtered += 1;
                    continue;
                }
            },
            None => 1.0,
        };
        n.kept += 1;
        let group = group_at.map(|k| row[k].as_ref());
        let cell = cell_at.map(|k| row[k].as_ref());
        sink.emit(b, &region, opts.window, group, gene, w, cell);
    }
    n.log(path, Preset::Abc, b, &sink.ids());
    Ok(())
}

/// The VCFs of an `opengwas` source: the file itself, or every
/// `*.vcf[.gz|.bgz]` in a directory, sorted.
fn opengwas_files(path: &str) -> anyhow::Result<Vec<String>> {
    if !Path::new(path).is_dir() {
        return Ok(vec![path.to_string()]);
    }
    let mut files: Vec<String> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().into_owned())
        .filter(|p| is_vcf_path(p))
        .collect();
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "{path}: no .vcf / .vcf.gz files in the directory"
    );
    Ok(files)
}

fn add_opengwas(
    b: &mut TypedGraphBuilder,
    path: &str,
    fixed_ctx: Option<&str>,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    let stem = file_stem(path.trim_end_matches('/'));
    let mut sink = Sink::new(TRAIT_TYPE, &stem, fixed_ctx, opts);
    let mut n = Tally::default();
    for file in opengwas_files(path)? {
        let r = VcfReader::open(&file)?;
        anyhow::ensure!(
            r.is_gwas_vcf(),
            "{file}: not a GWAS-VCF (no ES/SE/LP FORMAT fields); use `positions:` for a plain VCF"
        );
        check_build(&file, "GWAS-VCF", r.header().build, "the VCF header", opts)?;
        let traits: Vec<Box<str>> = if r.header().samples.is_empty() {
            vec![file_stem(&file).into()]
        } else {
            r.header().samples.clone()
        };
        for rec in r {
            let rec = rec?;
            n.rows += 1;
            let locus = PeakCoord {
                chr: rec.chr.clone(),
                start: rec.pos,
                end: rec.pos + 1,
            };
            let mut any = false;
            for (s, tr) in traits.iter().enumerate() {
                let lp = rec.gwas(s).lp;
                if lp.is_some_and(|lp| lp >= opts.min_lp) {
                    let group = opts.split_by_group.is_some().then_some(tr.as_ref());
                    sink.emit(b, &locus, opts.window, group, tr, 1.0, None);
                    any = true;
                }
            }
            if any {
                n.kept += 1;
            } else {
                n.filtered += 1;
            }
        }
    }
    n.log(path, Preset::OpenGwas, b, &sink.ids());
    Ok(())
}

fn add_positions(
    b: &mut TypedGraphBuilder,
    path: &str,
    fixed_ctx: Option<&str>,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    let stem = file_stem(path);
    let mut sink = Sink::new(TRAIT_TYPE, &stem, fixed_ctx, opts);
    let mut n = Tally::default();
    if is_vcf_path(path) {
        let r = VcfReader::open(path)?;
        check_build(
            path,
            "positions VCF",
            r.header().build,
            "the VCF header",
            opts,
        )?;
        for rec in r {
            let rec = rec?;
            n.rows += 1;
            n.kept += 1;
            let locus = PeakCoord {
                chr: rec.chr.clone(),
                start: rec.pos,
                end: rec.pos + 1,
            };
            sink.emit(b, &locus, opts.window, None, &stem, 1.0, None);
        }
        n.log(path, Preset::Positions, b, &sink.ids());
        return Ok(());
    }

    check_build(path, "positions", None, "", opts)?;
    let t = TableReader::open(path)?;
    let locus_col = t.find_column(&[
        "locus",
        "variant",
        "variant_id",
        "region",
        "snp",
        "SNP",
        "id",
    ]);
    let chr_col = t.find_column(&["chr", "chrom", "CHR", "CHROM", "chromosome", "Chromosome"]);
    let pos_col = t.find_column(&[
        "pos",
        "position",
        "POS",
        "BP",
        "bp",
        "start",
        "base_pair_location",
    ]);
    let end_col = t.find_column(&["end", "END", "stop"]);
    let label_col = t.find_column(&["trait", "label", "name", "group", "set"]);
    let weight_col = t.find_column(&["weight", "score", "pip", "PIP"]);
    let p_col = t.find_column(&["pvalue", "p_value", "pval", "P", "p"]);

    // A headerless list: the "header" is itself a locus row.
    let headerless = locus_col.is_none()
        && (chr_col.is_none() || pos_col.is_none())
        && t.header()
            .first()
            .is_some_and(|h| super::graph::outside_locus(h).is_some());
    enum Shape {
        Locus(usize),
        ChrPos(usize, usize, Option<usize>),
    }
    let (shape, label, weight) = if headerless {
        let w = t.header().len();
        (Shape::Locus(0), (w > 1).then_some(1), (w > 2).then_some(2))
    } else if let Some(l) = locus_col {
        (Shape::Locus(l), label_col, weight_col)
    } else if let (Some(c), Some(p)) = (chr_col, pos_col) {
        (Shape::ChrPos(c, p, end_col), label_col, weight_col)
    } else {
        anyhow::bail!(
            "{path}: a positions table needs a `locus` / `variant` column or `chr` + `pos` columns \
             (header: {})",
            t.header().join(", ")
        );
    };
    let group_col = group_column(&t, opts, None)?;
    let mut cols: Vec<usize> = match shape {
        Shape::Locus(l) => vec![l],
        Shape::ChrPos(c, p, e) => {
            let mut v = vec![c, p];
            v.extend(e);
            v
        }
    };
    let n_loc = cols.len();
    let label_at = push_col(&mut cols, label);
    let weight_at = push_col(&mut cols, weight);
    let p_at = push_col(&mut cols, if headerless { None } else { p_col });
    let group_at = push_col(&mut cols, group_col);

    let mut handle = |row: &[Box<str>], n: &mut Tally, b: &mut TypedGraphBuilder| {
        n.rows += 1;
        let locus = if n_loc == 1 {
            super::graph::outside_locus(&row[0])
        } else {
            let pos = row[1].trim().parse::<i64>().ok();
            let end = if n_loc == 3 {
                row[2].trim().parse::<i64>().ok()
            } else {
                pos.map(|p| p + 1)
            };
            match (pos, end) {
                (Some(p), Some(e)) if e > p && !row[0].trim().is_empty() => Some(PeakCoord {
                    chr: chr_stripped(row[0].trim()).into(),
                    start: p,
                    end: e,
                }),
                _ => None,
            }
        };
        let Some(locus) = locus else {
            n.no_locus += 1;
            return;
        };
        if let Some(k) = p_at {
            if !passes_p(&row[k], opts) {
                n.filtered += 1;
                return;
            }
        }
        let label: &str = match label_at.map(|k| row[k].trim()) {
            Some(l) if !l.is_empty() => l,
            _ => &stem,
        };
        let w = match weight_at {
            Some(k) => match weight_value(&row[k]) {
                Some(w) => w,
                None => {
                    n.filtered += 1;
                    return;
                }
            },
            None => 1.0,
        };
        n.kept += 1;
        let group = group_at.map(|k| row[k].as_ref());
        sink.emit(b, &locus, opts.window, group, label, w, None);
    };
    if headerless {
        let first: Vec<Box<str>> = cols.iter().map(|&j| t.header()[j].clone()).collect();
        handle(&first, &mut n, b);
    }
    for row in t.rows(&cols)? {
        handle(&row?, &mut n, b);
    }
    n.log(path, Preset::Positions, b, &sink.ids());
    Ok(())
}

/// The E2G parquet release directory ([`E2gRelease`]): one relation per
/// model, each link tied to its sample's cell type name, so replicate
/// samples of one cell type share a context.
fn add_e2g(
    b: &mut TypedGraphBuilder,
    path: &str,
    fixed_ctx: Option<&str>,
    opts: &LinkOpts,
) -> anyhow::Result<()> {
    check_build_evidence(
        path,
        "E2G",
        BuildEvidence::Declared(GenomeBuild::GRCh38),
        "the E2G release, which predicts on GRCh38",
        opts,
    )?;
    let release = E2gRelease::open(Path::new(path))?;
    let stem = file_stem(path.trim_end_matches('/'));
    let mut sink = Sink::new(GENE_TYPE, &stem, fixed_ctx, opts);
    let mut n = Tally::default();
    // Prediction files are per chromosome: reuse one region and re-box its
    // chromosome only when it changes.
    let mut region = PeakCoord {
        chr: "".into(),
        start: 0,
        end: 0,
    };
    release.for_each_link(|l| {
        n.rows += 1;
        let Some((start, end)) = l.span else {
            n.no_locus += 1;
            return;
        };
        let w = match l
            .score
            .filter(|&w| w.is_finite() && w >= opts.min_score.max(0.0))
        {
            Some(w) => w as f32,
            None => {
                n.filtered += 1;
                return;
            }
        };
        if l.gene.is_empty() {
            n.no_target += 1;
            return;
        }
        n.kept += 1;
        let chr = chr_stripped(l.chr);
        if region.chr.as_ref() != chr {
            region.chr = chr.into();
        }
        (region.start, region.end) = (start, end);
        sink.emit(
            b,
            &region,
            opts.window,
            Some(l.model),
            l.gene,
            w,
            l.cell_type,
        );
    })?;
    n.log(path, Preset::E2g, b, &sink.ids());
    Ok(())
}
