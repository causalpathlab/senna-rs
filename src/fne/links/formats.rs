//! One reader per `--links` preset, each in its published format: the GWAS
//! Catalog, GTEx, the eQTL Catalogue, ABC, OpenGWAS VCFs, plain positions and
//! ENCODE E2G.

use super::{
    check_build, check_build_evidence, group_column, is_vcf_path, push_col, BuildEvidence,
    LinkOpts, Preset, Sink, Tally, TRAIT_TYPE,
};
use crate::fne::graph::{
    curie_from_underscore, weight_value, NodeText, TypedGraphBuilder, GENE_TYPE, TERM_TYPE,
};
use genomic_data::coordinates::{chr_stripped, PeakCoord};
use genomic_data::e2g::E2gRelease;
use genomic_data::variant::{parse_variant_id, GenomeBuild};
use genomic_data::vcf::VcfReader;
use legume_numeric::matrix::common_io::file_stem;
use legume_numeric::matrix::table::TableReader;
use log::{info, warn};
use std::path::Path;

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

pub(super) fn add_gwas_catalog(
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
/// `chr1:14829:14970:clu_1:ENSG00000000150.5`.
fn gtex_gene(s: &str) -> &str {
    s.rsplit(':').next().unwrap_or(s).trim()
}

pub(super) fn add_gtex(
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

pub(super) fn add_eqtl_catalogue(
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
    // `QTD000001.credible_sets` → dataset QTD000001: its sample group is
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
        let Some(locus) = crate::fne::graph::outside_locus(&row[0]) else {
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

pub(super) fn add_abc(
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

pub(super) fn add_opengwas(
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

pub(super) fn add_positions(
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
            .is_some_and(|h| crate::fne::graph::outside_locus(h).is_some());
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
            crate::fne::graph::outside_locus(&row[0])
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
pub(super) fn add_e2g(
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
