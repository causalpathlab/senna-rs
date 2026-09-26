//! `--links`: every preset reads its published format onto the shared
//! region windows, builds are checked, and gene names from ids and symbols
//! meet in one node.

use super::graph::{canonical_term_id, TypedGraph, TypedGraphBuilder};
use super::links::{add_links, detect_preset, ontology_id_from_uri, LinkOpts, LinkSpec, Preset};
use super::{fit_fne, FneArgs};
use clap::Parser;
use data_beans::aux::feature_names::FeatureNameKind;
use genomic_data::variant::GenomeBuild;
use legume_numeric::matrix::common_io::open_buf_writer;
use legume_numeric::matrix::parquet::{read_parquet_string_columns_by_name, write_table, Column};
use std::io::Write;
use std::path::Path;

#[derive(Parser)]
struct Wrap<A: clap::Args> {
    #[command(flatten)]
    args: A,
}

fn parse_args<A: clap::Args>(argv: &[&str]) -> A {
    Wrap::<A>::parse_from(argv).args
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p.to_string_lossy().into_owned()
}

fn write_gz(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name).to_string_lossy().into_owned();
    let mut w = open_buf_writer(&p).unwrap();
    w.write_all(body.as_bytes()).unwrap();
    w.flush().unwrap();
    drop(w);
    p
}

fn opts() -> LinkOpts {
    LinkOpts {
        window: 10_000,
        max_pvalue: None,
        min_pip: 0.0,
        min_lp: 7.30103,
        split_by_group: None,
        gwas_mapped_gene: false,
        genome_build: GenomeBuild::GRCh38,
        allow_build_mismatch: false,
        // Off for the per-preset edge lists; the context tests turn it on.
        context: false,
        eqtl_datasets: Default::default(),
        min_score: 0.0,
    }
}

fn with_context() -> LinkOpts {
    LinkOpts {
        context: true,
        ..opts()
    }
}

fn builder() -> TypedGraphBuilder {
    TypedGraphBuilder::new(FeatureNameKind::Gene { delim: '_' })
}

fn link(b: &mut TypedGraphBuilder, spec: &str, o: &LinkOpts) -> anyhow::Result<()> {
    add_links(b, &LinkSpec::parse(spec)?, o)
}

/// `(relation, lhs, rhs, weight)` of every edge, sorted.
fn edges(g: &TypedGraph) -> Vec<(String, String, String, f32)> {
    let mut v: Vec<_> = (0..g.edges.len())
        .map(|i| {
            (
                g.relations.get(g.edges.rel[i] as usize).name.to_string(),
                g.node_names[g.edges.lhs[i] as usize].to_string(),
                g.node_names[g.edges.rhs[i] as usize].to_string(),
                g.edges.edge_weight(i),
            )
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn e(r: &str, l: &str, rhs: &str, w: f32) -> (String, String, String, f32) {
    (r.into(), l.into(), rhs.into(), w)
}

fn nodes_of(g: &TypedGraph, ty: &str) -> Vec<String> {
    let t = g.types.index_of(ty).unwrap();
    g.types
        .range(t)
        .map(|i| g.node_names[i as usize].to_string())
        .collect()
}

const GWAS_HEADER: &str = "DATE ADDED TO CATALOG\tDISEASE/TRAIT\tCHR_ID\tCHR_POS\tMAPPED_GENE\tSNPS\tP-VALUE\tMAPPED_TRAIT\tMAPPED_TRAIT_URI\n";

fn gwas_fixture(dir: &Path) -> String {
    let body = format!(
        "{GWAS_HEADER}\
         2020\tBMI\t17\t7675000\tTP53\trs1\t2E-9\tbody mass index\thttp://www.ebi.ac.uk/efo/EFO_0004340\n\
         2020\tT2D and BMI\t1;1\t15000;25000\tA - B\trs2;rs3\t1E-6\ttype 2 diabetes mellitus, body mass index\thttp://purl.obolibrary.org/obo/MONDO_0005148, http://www.ebi.ac.uk/efo/EFO_0004340\n\
         2020\tno position\t\t\tC\trs4\t1E-20\theight\thttp://www.ebi.ac.uk/efo/EFO_0004339\n"
    );
    write(dir, "gwas_catalog.tsv", &body)
}

#[test]
fn link_specs_take_a_known_preset_prefix_and_leave_other_colons_in_the_path() {
    let s = LinkSpec::parse("gwas-catalog:a/b.tsv").unwrap();
    assert_eq!(
        (s.preset, s.path.as_ref()),
        (Some(Preset::GwasCatalog), "a/b.tsv")
    );
    let s = LinkSpec::parse("encode-re2g:x.tsv.gz").unwrap();
    assert_eq!(s.preset, Some(Preset::Abc));
    let s = LinkSpec::parse("gtex@blood:x.parquet").unwrap();
    assert_eq!(
        (s.preset, s.context.as_deref(), s.path.as_ref()),
        (Some(Preset::Gtex), Some("blood"), "x.parquet")
    );
    assert!(LinkSpec::parse("gtex@:x.parquet").is_err());
    let s = LinkSpec::parse("C:/data/x.tsv").unwrap();
    assert_eq!((s.preset, s.path.as_ref()), (None, "C:/data/x.tsv"));
    assert!(LinkSpec::parse("gtex:").is_err());
    assert_eq!(
        ontology_id_from_uri("http://www.ebi.ac.uk/efo/EFO_0004340").as_deref(),
        Some("EFO:0004340")
    );
    assert_eq!(
        ontology_id_from_uri(" http://purl.obolibrary.org/obo/MONDO_0005148").as_deref(),
        Some("MONDO:0005148")
    );
}

#[test]
fn efo_spellings_of_its_own_ids_become_obo_foundry_curies() {
    assert_eq!(canonical_term_id("efo:EFO_0004340"), "EFO:0004340");
    assert_eq!(canonical_term_id("EFO:0004340"), "EFO:0004340");
    assert_eq!(canonical_term_id("MONDO:0005148"), "MONDO:0005148");
    assert_eq!(canonical_term_id("GO:0008150"), "GO:0008150");
    assert_eq!(
        canonical_term_id("http://dbpedia.org/resource/X_Y"),
        "http://dbpedia.org/resource/X_Y"
    );
    assert_eq!(canonical_term_id("efo:OBI_0000070"), "efo:OBI_0000070");
}

#[test]
fn gwas_catalog_rows_link_windows_to_their_ontology_terms() {
    let dir = tempfile::tempdir().unwrap();
    let p = gwas_fixture(dir.path());
    assert_eq!(detect_preset(&p).unwrap(), Preset::GwasCatalog);
    let mut b = builder();
    link(&mut b, &p, &opts()).unwrap();
    let g = b.finish().unwrap();
    let r = "region:term/gwas_catalog";
    assert_eq!(
        edges(&g),
        vec![
            e(r, "17:7670000-7680000", "EFO:0004340", 1.0),
            e(r, "1:10000-20000", "EFO:0004340", 1.0),
            e(r, "1:10000-20000", "MONDO:0005148", 1.0),
            e(r, "1:20000-30000", "EFO:0004340", 1.0),
            e(r, "1:20000-30000", "MONDO:0005148", 1.0),
        ],
        "a `;` row is one hit per locus; the row with no position is skipped"
    );
    let term = g.types.index_of("term").unwrap();
    let names: Vec<_> = g
        .texts
        .iter()
        .filter(|(i, _)| g.types.range(term).contains(i))
        .map(|(i, t)| {
            (
                g.node_names[*i as usize].to_string(),
                t.name.clone().unwrap().to_string(),
            )
        })
        .collect();
    assert!(names.contains(&("MONDO:0005148".into(), "type 2 diabetes mellitus".into())));

    // p filter and the opt-in mapped genes.
    let mut o = opts();
    o.max_pvalue = Some(5e-8);
    o.gwas_mapped_gene = true;
    let mut b = builder();
    link(&mut b, &p, &o).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(
        edges(&g),
        vec![
            e(
                "region:gene/gwas_catalog/mapped_gene",
                "17:7670000-7680000",
                "TP53",
                1.0
            ),
            e(r, "17:7670000-7680000", "EFO:0004340", 1.0),
        ]
    );
}

#[test]
fn gtex_parquet_pairs_link_variants_to_version_stripped_genes_and_check_the_build() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir
        .path()
        .join("Whole_Blood.v10.eQTLs.signif_pairs.parquet");
    let p = p.to_str().unwrap();
    let s = |v: &[&str]| v.iter().map(|x| Box::from(*x)).collect::<Vec<Box<str>>>();
    write_table(
        p,
        &[
            (
                "gene_id".into(),
                Column::Str(&s(&["ENSG00000141510.17", "ENSG00000136997.21"])),
            ),
            (
                "variant_id".into(),
                Column::Str(&s(&["chr17_7675001_A_G_b38", "chr8_127736000_C_T_b38"])),
            ),
            ("pval_nominal".into(), Column::F32(&[1e-10, 1e-3])),
        ],
    )
    .unwrap();
    assert_eq!(detect_preset(p).unwrap(), Preset::Gtex);
    let mut b = builder();
    link(&mut b, &format!("gtex:{p}"), &opts()).unwrap();
    let g = b.finish().unwrap();
    let r = "region:gene/Whole_Blood.v10.eQTLs.signif_pairs";
    assert_eq!(
        edges(&g),
        vec![
            e(r, "17:7670000-7680000", "ENSG00000141510", 1.0),
            e(r, "8:127730000-127740000", "ENSG00000136997", 1.0),
        ]
    );
    let mut o = opts();
    o.max_pvalue = Some(1e-5);
    let mut b = builder();
    link(&mut b, &format!("gtex:{p}"), &o).unwrap();
    assert_eq!(b.finish().unwrap().edges.len(), 1, "p filter");

    // A GRCh37 file (v7 ids end in _b37) is refused unless allowed.
    let old = write(
        dir.path(),
        "v7.signif_variant_gene_pairs.txt",
        "variant_id\tgene_id\tpval_nominal\n1_15000_A_G_b37\tENSG00000000001.1\t1e-9\n",
    );
    let err = link(&mut builder(), &old, &opts()).unwrap_err().to_string();
    assert!(
        err.contains("GRCh37") && err.contains("--allow-build-mismatch"),
        "{err}"
    );
    let mut o = opts();
    o.allow_build_mismatch = true;
    link(&mut builder(), &old, &o).unwrap();
}

const EQTL_HEADER: &str = "molecular_trait_id\tgene_id\tcs_id\tvariant\trsid\tcs_size\tpip\tpvalue\tbeta\tse\tz\tcs_min_r2\tregion\n";

#[test]
fn eqtl_catalogue_credible_sets_weigh_links_by_pip() {
    let dir = tempfile::tempdir().unwrap();
    let body = format!(
        "{EQTL_HEADER}\
         ENSG00000141510\tENSG00000141510\tcs1\tchr17_7675500_C_T\trs1\t2\t0.9\t1e-12\t0.5\t0.1\t5\t0.9\tchr17:1-2\n\
         ENSG00000141510\tENSG00000141510\tcs1\tchr17_7677000_G_A\trs2\t2\t0.05\t1e-10\t0.5\t0.1\t5\t0.9\tchr17:1-2\n"
    );
    let p = write_gz(dir.path(), "QTD000356.credible_sets.tsv.gz", &body);
    assert_eq!(detect_preset(&p).unwrap(), Preset::EqtlCatalogue);
    let mut b = builder();
    link(&mut b, &p, &opts()).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(
        edges(&g),
        vec![e(
            "region:gene/QTD000356.credible_sets",
            "17:7670000-7680000",
            "ENSG00000141510",
            0.9
        )],
        "two variants in one window keep the larger PIP"
    );
    let mut o = opts();
    o.min_pip = 0.95;
    let mut b = builder();
    link(&mut b, &p, &o).unwrap();
    assert!(b.finish().is_err(), "every row below --links-min-pip");
}

const ABC_BODY: &str = "#chr\tstart\tend\tname\tclass\tTargetGene\tCellType\tScore\n\
                        chr17\t7669999\t7670500\te1\tgenic\tTP53\tK562\t0.7\n\
                        chr17\t7679000\t7681000\te2\tgenic\tTP53\tHepG2\t0.2\n";

#[test]
fn abc_intervals_shift_to_one_based_windows_and_split_by_cell_type() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_gz(dir.path(), "ENCODE_rE2G_K562.tsv.gz", ABC_BODY);
    assert_eq!(detect_preset(&p).unwrap(), Preset::Abc);
    let mut b = builder();
    link(&mut b, &format!("abc:{p}"), &opts()).unwrap();
    let g = b.finish().unwrap();
    let r = "region:gene/ENCODE_rE2G_K562";
    assert_eq!(
        edges(&g),
        vec![
            // BED start 7669999 is base 7670000: one window, not two.
            e(r, "17:7670000-7680000", "TP53", 0.7),
            e(r, "17:7680000-7690000", "TP53", 0.2),
        ]
    );
    let mut o = opts();
    o.split_by_group = Some("".into());
    let mut b = builder();
    link(&mut b, &format!("abc:{p}"), &o).unwrap();
    let g = b.finish().unwrap();
    let names: Vec<&str> = g.relations.iter().map(|r| r.name.as_ref()).collect();
    assert_eq!(
        names,
        vec![
            "region:gene/ENCODE_rE2G_K562/K562",
            "region:gene/ENCODE_rE2G_K562/HepG2"
        ]
    );

    let hg19 = write(dir.path(), "ABC_hg19_K562.tsv", ABC_BODY);
    let err = link(&mut builder(), &format!("abc:{hg19}"), &opts())
        .unwrap_err()
        .to_string();
    assert!(err.contains("GRCh37"), "{err}");
}

fn gwas_vcf(sample: &str, assembly: &str) -> String {
    format!(
        "##fileformat=VCFv4.2\n\
         ##FORMAT=<ID=ES,Number=A,Type=Float,Description=\"e\">\n\
         ##FORMAT=<ID=SE,Number=A,Type=Float,Description=\"s\">\n\
         ##FORMAT=<ID=LP,Number=A,Type=Float,Description=\"l\">\n\
         ##contig=<ID=1,assembly={assembly}>\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n\
         1\t15000\trs1\tA\tG\t.\tPASS\t.\tES:SE:LP\t0.1:0.01:9.2\n\
         1\t55000\trs2\tA\tG\t.\tPASS\t.\tES:SE:LP\t0.1:0.05:2.0\n"
    )
}

#[test]
fn opengwas_vcfs_keep_records_above_the_lp_threshold_one_trait_per_sample() {
    let dir = tempfile::tempdir().unwrap();
    let vcfs = dir.path().join("opengwas");
    std::fs::create_dir(&vcfs).unwrap();
    write_gz(&vcfs, "ebi-a-1.vcf.gz", &gwas_vcf("ebi-a-1", "GRCh38"));
    write(&vcfs, "ebi-a-2.vcf", &gwas_vcf("ebi-a-2", "GRCh38"));
    write(&vcfs, "README.txt", "not a vcf");
    let d = vcfs.to_str().unwrap();
    assert_eq!(detect_preset(d).unwrap(), Preset::OpenGwas);
    let mut b = builder();
    link(&mut b, d, &opts()).unwrap();
    let g = b.finish().unwrap();
    let r = "region:trait/opengwas";
    assert_eq!(
        edges(&g),
        vec![
            e(r, "1:10000-20000", "ebi-a-1", 1.0),
            e(r, "1:10000-20000", "ebi-a-2", 1.0),
        ]
    );
    let old = write(
        dir.path(),
        "ieu-a-2.vcf",
        &gwas_vcf("ieu-a-2", "HG19/GRCh37"),
    );
    let err = link(&mut builder(), &format!("opengwas:{old}"), &opts())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("GRCh37") && err.contains("VCF header"),
        "{err}"
    );
}

#[test]
fn position_lists_accept_vcfs_headerless_loci_and_chr_pos_tables() {
    let dir = tempfile::tempdir().unwrap();
    let vcf = write(
        dir.path(),
        "leads.vcf",
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\nchr2\t5\t.\tA\tC\n",
    );
    assert_eq!(detect_preset(&vcf).unwrap(), Preset::Positions);
    let bare = write(dir.path(), "finemap.txt", "chr2:15000\n2_25000_A_G\n");
    let table = write(
        dir.path(),
        "sets.tsv",
        "chrom\tpos\ttrait\tweight\n2\t35000\tLDL\t0.5\n2\tx\tLDL\t1\n",
    );
    let mut b = builder();
    for p in [&vcf, &bare, &table] {
        link(&mut b, p, &opts()).unwrap();
    }
    let g = b.finish().unwrap();
    assert_eq!(
        edges(&g),
        vec![
            e("region:trait/finemap", "2:10000-20000", "finemap", 1.0),
            e("region:trait/finemap", "2:20000-30000", "finemap", 1.0),
            e("region:trait/leads", "2:0-10000", "leads", 1.0),
            e("region:trait/sets", "2:30000-40000", "LDL", 0.5),
        ]
    );
}

#[test]
fn a_gwas_hit_an_eqtl_and_an_enhancer_in_one_window_share_one_region_node_and_ids_meet_symbols() {
    let dir = tempfile::tempdir().unwrap();
    let gwas = gwas_fixture(dir.path());
    let eqtl = write_gz(
        dir.path(),
        "eqtl.tsv.gz",
        &format!(
            "{EQTL_HEADER}ENSG00000141510.17\tENSG00000141510.17\tcs1\tchr17_7675500_C_T\trs1\t1\t0.8\t1e-12\t0.5\t0.1\t5\t0.9\tchr17:1-2\n"
        ),
    );
    let abc = write(dir.path(), "abc.tsv", ABC_BODY);
    let gff = write(
        dir.path(),
        "genes.gtf",
        "chr17\tHAVANA\tgene\t7661779\t7687538\t.\t-\t.\tgene_id \"ENSG00000141510.17\"; gene_name \"TP53\";\n",
    );
    let mut b = builder();
    b.set_gene_symbols(genomic_data::gff::load_ensembl_symbol_map(&gff).unwrap());
    for p in [&gwas, &eqtl, &abc] {
        link(&mut b, p, &opts()).unwrap();
    }
    let g = b.finish().unwrap();
    let win = "17:7670000-7680000";
    let regions = nodes_of(&g, "region");
    assert_eq!(regions.iter().filter(|r| r.as_str() == win).count(), 1);
    let genes = nodes_of(&g, "gene");
    assert_eq!(
        genes,
        vec!["TP53"],
        "ENSG00000141510.17 and TP53 are one node"
    );
    let at_win: Vec<(String, String)> = edges(&g)
        .into_iter()
        .filter(|(_, l, _, _)| l == win)
        .map(|(r, _, rhs, _)| (r, rhs))
        .collect();
    assert_eq!(
        at_win,
        vec![
            ("region:gene/abc".into(), "TP53".into()),
            ("region:gene/eqtl".into(), "TP53".into()),
            ("region:term/gwas_catalog".into(), "EFO:0004340".into()),
        ]
    );
}

#[test]
fn clap_takes_repeated_links_and_obo_and_the_ten_kb_window_by_default() {
    let a: FneArgs = parse_args(&["fne", "-o", "x"]);
    assert_eq!(a.region_window, 10_000);
    assert_eq!(a.links_min_trait_windows, 5);
    assert_eq!(a.genome_build.as_ref(), "GRCh38");
    assert!(a.links.is_empty() && a.obo.is_empty() && a.links_split_by_group.is_none());
    let a: FneArgs = parse_args(&[
        "fne",
        "--links",
        "gwas-catalog:a.tsv",
        "--links",
        "abc:b.tsv,c.tsv",
        "--obo",
        "efo.obo",
        "--obo",
        "go-basic.obo",
        "--links-split-by-group",
        "--gwas-mapped-gene",
        "-o",
        "x",
    ]);
    assert_eq!(a.links.len(), 3);
    assert_eq!(a.obo.len(), 2);
    assert_eq!(a.links_split_by_group.as_deref(), Some(""));
    assert!(a.gwas_mapped_gene);
    let a: FneArgs = parse_args(&["fne", "--links-split-by-group", "tissue", "-o", "x"]);
    assert_eq!(a.links_split_by_group.as_deref(), Some("tissue"));
}

#[test]
fn fne_trains_on_links_end_to_end_and_names_their_relations() {
    let dir = tempfile::tempdir().unwrap();
    let gwas = gwas_fixture(dir.path());
    let abc = write_gz(dir.path(), "ENCODE_rE2G_K562.tsv.gz", ABC_BODY);
    let ppi = write(dir.path(), "ppi.tsv", "TP53\tMDM2\nTP53\tATM\nMDM2\tATM\n");
    let obo = write(
        dir.path(),
        "mini.obo",
        "format-version: 1.2\n\n[Term]\nid: EFO:0004340\nname: body mass index\nis_a: EFO:0004339 ! height\n\n\
         [Term]\nid: efo:EFO_0004339\nname: height\n\n[Term]\nid: MONDO:0005148\nname: type 2 diabetes mellitus\nis_a: EFO:0004340\n",
    );
    let out = dir.path().join("g2r").to_string_lossy().into_owned();
    let bad: FneArgs = parse_args(&[
        "fne",
        "--links",
        &gwas,
        "--genome-build",
        "GRCh37",
        "-o",
        &out,
    ]);
    let err = fit_fne(&bad).unwrap_err().to_string();
    assert!(err.contains("GRCh38") && err.contains("GRCh37"), "{err}");

    let args: FneArgs = parse_args(&[
        "fne",
        &ppi,
        "--links",
        &format!("gwas-catalog:{gwas}"),
        "--links",
        &format!("abc:{abc}"),
        "--links-split-by-group",
        "--links-min-trait-windows",
        "2",
        "--obo",
        &obo,
        "--embedding-dim",
        "4",
        "-i",
        "3",
        "--batch-size",
        "4",
        "--num-batch-negs",
        "2",
        "--num-uniform-negs",
        "2",
        "--weight-decay",
        "0",
        "--eval-fraction",
        "0",
        "--no-ppi-snn",
        "--no-ppi-ppr",
        "-o",
        &out,
    ]);
    fit_fne(&args).unwrap();
    let rels = read_parquet_string_columns_by_name(
        &format!("{out}.relations.parquet"),
        &["relation", "lhs_type", "rhs_type"],
    )
    .unwrap();
    let names: Vec<&str> = rels[0].iter().map(AsRef::as_ref).collect();
    assert_eq!(
        names,
        vec![
            "gene:gene/ppi",
            "region:term/gwas_catalog",
            "region:gene/ENCODE_rE2G_K562/K562",
            "region:context/ENCODE_rE2G_K562",
            "gene:context/ENCODE_rE2G_K562",
            "region:gene/ENCODE_rE2G_K562/HepG2",
            "term:term/is_a",
        ],
        "the ontology's hierarchy reaches the GWAS terms; empty part_of is dropped"
    );
    assert_eq!(rels[1][1].as_ref(), "region");
    assert_eq!(rels[2][1].as_ref(), "term");
    let types = read_parquet_string_columns_by_name(
        &format!("{out}.feature_types.parquet"),
        &["feature", "type"],
    )
    .unwrap();
    let type_of = |n: &str| {
        let i = types[0].iter().position(|f| f.as_ref() == n).unwrap();
        types[1][i].to_string()
    };
    assert_eq!(type_of("17:7670000-7680000"), "region");
    assert_eq!(type_of("EFO:0004340"), "term");
    assert_eq!(type_of("TP53"), "gene");
    assert_eq!(type_of("K562"), "context");
}

#[test]
fn traits_reached_by_too_few_windows_are_dropped_with_their_orphan_windows() {
    let dir = tempfile::tempdir().unwrap();
    // EFO:0004340 reaches 3 windows, MONDO:0005148 reaches 2 (both on
    // chr1, which EFO:0004340 shares); a GMT set also names MONDO:0005148
    // in the second builder, which keeps it.
    let gwas = gwas_fixture(dir.path());
    let mut b = builder();
    link(&mut b, &gwas, &opts()).unwrap();
    assert_eq!(b.prune_sparse_link_nodes("term", 3), (1, 0));
    let g = b.finish().unwrap();
    assert_eq!(nodes_of(&g, "term"), vec!["EFO:0004340"]);
    assert_eq!(
        nodes_of(&g, "region").len(),
        3,
        "every window still links EFO:0004340"
    );

    let mut b = builder();
    link(&mut b, &gwas, &opts()).unwrap();
    assert_eq!(
        b.prune_sparse_link_nodes("term", 4),
        (2, 3),
        "all terms go, then all windows"
    );

    let gmt = write(
        dir.path(),
        "sets.gmt",
        "MONDO:0005148\tT2D genes\tTCF7L2\tKCNJ11\n",
    );
    let mut b = builder();
    link(&mut b, &gwas, &opts()).unwrap();
    b.add_gene_sets(
        &data_beans::aux::gene_sets::read_gmt(&gmt).unwrap(),
        "sets",
        1,
        0,
    );
    assert_eq!(
        b.prune_sparse_link_nodes("term", 3),
        (0, 0),
        "an annotated term stays"
    );
}

fn gtex_fixture(dir: &Path) -> String {
    write(
        dir,
        "Whole_Blood.v10.eQTLs.signif_pairs.tsv",
        "variant_id\tgene_id\tpval_nominal\nchr17_7675001_A_G_b38\tENSG00000141510.17\t1e-10\n",
    )
}

fn eqtl_fixture(dir: &Path) -> String {
    write(
        dir,
        "QTD000356.credible_sets.tsv",
        &format!(
            "{EQTL_HEADER}ENSG00000141510\tENSG00000141510\tcs1\tchr17_7675500_C_T\trs1\t1\t0.8\t1e-12\t0.5\t0.1\t5\t0.9\tchr17:1-2\n"
        ),
    )
}

#[test]
fn eqtl_links_tie_their_windows_and_genes_to_the_tissue_they_were_measured_in() {
    let dir = tempfile::tempdir().unwrap();
    let gtex = gtex_fixture(dir.path());
    let eqtl = eqtl_fixture(dir.path());
    let meta = write(
        dir.path(),
        "dataset_metadata_r8.tsv",
        "study_id\tdataset_id\tstudy_label\tsample_group\ttissue_id\ttissue_label\tcondition_label\tsample_size\tquant_method\tpmid\tstudy_type\n\
         QTS000015\tQTD000356\tGTEx_v10\tblood\tUBERON_0000178\tblood\tnaive\t853\tge\t1\tbulk\n",
    );
    let mut o = with_context();
    o.eqtl_datasets = super::links::read_eqtl_metadata(&meta).unwrap();
    let mut b = builder();
    link(&mut b, &format!("gtex:{gtex}"), &o).unwrap();
    link(&mut b, &format!("eqtl-catalogue:{eqtl}"), &o).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(nodes_of(&g, "context"), vec!["Whole_Blood", "blood"]);
    let win = "17:7670000-7680000";
    let gtex_rel = "Whole_Blood.v10.eQTLs.signif_pairs";
    assert_eq!(
        edges(&g),
        vec![
            e(
                "gene:context/GTEx_v10_blood",
                "ENSG00000141510",
                "blood",
                0.8
            ),
            e(
                &format!("gene:context/{gtex_rel}"),
                "ENSG00000141510",
                "Whole_Blood",
                1.0
            ),
            e("region:context/GTEx_v10_blood", win, "blood", 0.8),
            e(
                &format!("region:context/{gtex_rel}"),
                win,
                "Whole_Blood",
                1.0
            ),
            e("region:gene/GTEx_v10_blood", win, "ENSG00000141510", 0.8),
            e(
                &format!("region:gene/{gtex_rel}"),
                win,
                "ENSG00000141510",
                1.0
            ),
        ],
        "the metadata names the eQTL Catalogue relation and its context"
    );

    // `@blood` merges GTEx's tissue with the Catalogue's; without metadata
    // the dataset id names the context; --no-links-context drops it all.
    let mut b = builder();
    link(&mut b, &format!("gtex@blood:{gtex}"), &with_context()).unwrap();
    link(&mut b, &format!("eqtl-catalogue:{eqtl}"), &with_context()).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(nodes_of(&g, "context"), vec!["blood", "QTD000356"]);
    let mut b = builder();
    link(&mut b, &gtex, &opts()).unwrap();
    assert!(b.finish().unwrap().types.index_of("context").is_none());
}

#[test]
fn abc_links_take_their_context_from_each_rows_cell_type() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(dir.path(), "abc.tsv", ABC_BODY);
    let mut b = builder();
    link(&mut b, &p, &with_context()).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(nodes_of(&g, "context"), vec!["K562", "HepG2"]);
    let ctx: Vec<_> = edges(&g)
        .into_iter()
        .filter(|(r, ..)| r.starts_with("gene:context"))
        .collect();
    assert_eq!(
        ctx,
        vec![
            e("gene:context/abc", "TP53", "HepG2", 0.2),
            e("gene:context/abc", "TP53", "K562", 0.7),
        ]
    );
}

/// A two-sample E2G release: HepG2 twice (replicates) and one scE2G cell type.
fn e2g_fixture(dir: &Path) -> String {
    let root = dir.join("e2g");
    std::fs::create_dir_all(root.join("enhancer_gene_predictions")).unwrap();
    let s = |v: &[&str]| v.iter().map(|x| Box::from(*x)).collect::<Vec<Box<str>>>();
    let at = |f: &str| root.join(f).to_string_lossy().into_owned();
    write_table(
        &at("cell_types.parquet"),
        &[
            (
                "id".into(),
                Column::Str(&s(&["HepG2_A", "HepG2_B", "cardio_1"])),
            ),
            (
                "name".into(),
                Column::Str(&s(&["HepG2", "HepG2", "Ventricular cardiomyocytes"])),
            ),
        ],
    )
    .unwrap();
    write_table(
        &at("enhancers.parquet"),
        &[
            ("id".into(), Column::I32(&[1, 2])),
            ("chromosome".into(), Column::Str(&s(&["17", "17"]))),
            ("start".into(), Column::I32(&[7669999, 7690000])),
            ("end".into(), Column::I32(&[7670500, 7690500])),
            ("class".into(), Column::Str(&s(&["intergenic", "genic"]))),
        ],
    )
    .unwrap();
    write_table(
        &at("enhancer_gene_predictions/chr17.parquet"),
        &[
            (
                "chromosome".into(),
                Column::Str(&s(&["17", "17", "17", "17"])),
            ),
            ("enhancer_id".into(), Column::I32(&[1, 1, 2, 2])),
            (
                "target_gene_name".into(),
                Column::Str(&s(&["TP53", "TP53", "", "TP53"])),
            ),
            (
                "target_gene_id".into(),
                Column::Str(&s(&[
                    "ENSG00000141510",
                    "ENSG00000141510",
                    "ENSG00000141510",
                    "ENSG00000141510",
                ])),
            ),
            ("score".into(), Column::F32(&[0.9, 0.4, 0.3, 0.2])),
            (
                "model".into(),
                Column::Str(&s(&["ENCODE-rE2G", "ENCODE-rE2G", "ENCODE-rE2G", "scE2G"])),
            ),
            (
                "cell_type_id".into(),
                Column::Str(&s(&["HepG2_A", "HepG2_B", "HepG2_B", "cardio_1"])),
            ),
        ],
    )
    .unwrap();
    root.to_string_lossy().into_owned()
}

#[test]
fn e2g_releases_link_enhancers_per_model_and_name_contexts_by_cell_type() {
    let dir = tempfile::tempdir().unwrap();
    let e2g = e2g_fixture(dir.path());
    assert_eq!(detect_preset(&e2g).unwrap(), Preset::E2g);
    let mut b = builder();
    link(&mut b, &e2g, &with_context()).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(
        nodes_of(&g, "context"),
        vec!["HepG2", "Ventricular cardiomyocytes"],
        "replicate samples share their cell type"
    );
    let (w1, w2) = ("17:7670000-7680000", "17:7690000-7700000");
    let genes: Vec<_> = edges(&g)
        .into_iter()
        .filter(|(r, ..)| r.starts_with("region:gene"))
        .collect();
    assert_eq!(
        genes,
        vec![
            // BED start 7669999 is base 7670000; replicates keep the max score;
            // an empty symbol falls back to the gene id (its own node without
            // --gene-gff).
            e("region:gene/e2g/ENCODE-rE2G", w1, "TP53", 0.9),
            e("region:gene/e2g/ENCODE-rE2G", w2, "ENSG00000141510", 0.3),
            e("region:gene/e2g/scE2G", w2, "TP53", 0.2),
        ]
    );
    let mut o = with_context();
    o.min_score = 0.35;
    let mut b = builder();
    link(&mut b, &e2g, &o).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(nodes_of(&g, "region"), vec![w1], "--links-min-score");
}

#[test]
fn context_edges_of_windows_shared_by_many_contexts_are_dropped_on_request() {
    let dir = tempfile::tempdir().unwrap();
    // Three cell types; window 7670000 is active in all three, window
    // 7680000 in one.
    let abc = write(
        dir.path(),
        "abc.tsv",
        "chr\tstart\tend\tTargetGene\tCellType\tScore\n\
         chr17\t7670000\t7670100\tTP53\tA\t0.5\n\
         chr17\t7670000\t7670100\tTP53\tB\t0.5\n\
         chr17\t7670000\t7670100\tMDM2\tC\t0.5\n\
         chr17\t7685000\t7685100\tMYC\tC\t0.5\n",
    );
    let mut b = builder();
    link(&mut b, &abc, &with_context()).unwrap();
    // limit = ⌈0.5 · 3⌉ = 2: the window in 3 contexts goes, TP53 (in A, B)
    // stays, MDM2 and MYC (one each) stay.
    let out = b.prune_shared_context_edges("context", 0.5);
    let dropped: Vec<(String, usize, usize)> = out
        .into_iter()
        .map(|(t, d, k)| (t.to_string(), d, k))
        .collect();
    assert_eq!(
        dropped,
        vec![("region".into(), 3, 1), ("gene".into(), 0, 4)]
    );
    let g = b.finish().unwrap();
    let ctx: Vec<_> = edges(&g)
        .into_iter()
        .filter(|(r, ..)| r.starts_with("region:context"))
        .map(|(_, l, r, _)| (l, r))
        .collect();
    assert_eq!(
        ctx,
        vec![("17:7680000-7690000".to_string(), "C".to_string())]
    );
}
