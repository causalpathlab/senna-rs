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

use super::graph::{TypedGraphBuilder, GENE_TYPE, REGION_TYPE};
use formats::{
    add_abc, add_e2g, add_eqtl_catalogue, add_gtex, add_gwas_catalog, add_opengwas, add_positions,
};
use genomic_data::coordinates::PeakCoord;
use genomic_data::e2g::E2gRelease;
use genomic_data::variant::GenomeBuild;
use genomic_data::vcf::VcfReader;
use legume_numeric::matrix::common_io::file_stem;
use legume_numeric::matrix::table::TableReader;
use log::{info, warn};
use rustc_hash::FxHashMap;
use std::path::Path;

mod formats;

#[cfg(test)]
pub(crate) use formats::ontology_id_from_uri;

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
    /// naive (`TISSUE1`, `CT1_STIM1`, `CT2`).
    pub sample_group: Box<str>,
    /// `<study_label>_<sample_group>`, plus `_<quant_method>` when not
    /// gene expression: the relation stem in place of `QTD000001`.
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
