//! Tests for the latent-sharpness summary.

use super::{latent_sharpness, Mat};
use legume_numeric::matrix::traits::IoOps;

/// The effective topic count and mean max weight, one row per case:
/// - `flat`: k effective topics, max weight 1/k;
/// - `one-hot`: one effective topic, max weight 1;
/// - `renormalized`: rows are renormalized, so unnormalized weights and
///   proportions agree;
/// - `empty`: NaN, not zero;
/// - `non-finite`: a diverged latent must report NaN, not `+inf`. `f32::max`
///   drops a NaN operand, so the guarded row sum has to check finiteness
///   explicitly.
#[test]
fn latent_sharpness_is_the_effective_topic_count_and_mean_max_weight() {
    let k = 8;
    let mut one_hot = Mat::zeros(10, k);
    for i in 0..10 {
        one_hot[(i, i % k)] = 1.0;
    }
    let proportions = Mat::from_row_slice(2, 3, &[0.2, 0.3, 0.5, 0.2, 0.3, 0.5]);
    let mut diverged = Mat::from_element(3, 4, 0.25);
    diverged[(1, 2)] = f32::NAN;
    let cases = [
        (
            "flat",
            Mat::from_element(10, k, 1.0 / k as f32),
            (k as f32, 1.0 / k as f32),
        ),
        ("one-hot", one_hot, (1.0, 1.0)),
        (
            "renormalized",
            &proportions * 7.0,
            latent_sharpness(&proportions),
        ),
        ("empty", Mat::zeros(0, 4), (f32::NAN, f32::NAN)),
        ("non-finite", diverged, (f32::NAN, f32::NAN)),
    ];
    let near = |got: f32, want: f32, tol: f32| {
        if want.is_nan() {
            got.is_nan()
        } else {
            (got - want).abs() < tol
        }
    };
    for (label, theta, (want_eff, want_mx)) in cases {
        let (eff, mx) = latent_sharpness(&theta);
        assert!(near(eff, want_eff, 1e-3), "{label}: effective topics {eff}");
        assert!(near(mx, want_mx, 1e-6), "{label}: max weight {mx}");
    }
}

#[test]
fn zero_rows_survive_l2_normalization_untouched() {
    let mut m = Mat::from_row_slice(2, 2, &[3.0, 4.0, 0.0, 0.0]);
    super::l2_normalize_rows_inplace(&mut m);
    assert!((m.row(0).norm() - 1.0).abs() < 1e-6);
    assert!(m.row(1).iter().all(|&x| x == 0.0));
}

//////////////////////////////////////////////////////
// Dense bulk tables: header detection + orientation //
//////////////////////////////////////////////////////

fn bulk_fixture() -> (Vec<Box<str>>, Vec<Box<str>>, Mat) {
    let genes: Vec<Box<str>> = ["g0", "g1", "g2"].iter().map(|s| Box::from(*s)).collect();
    let samples: Vec<Box<str>> = ["s0", "s1"].iter().map(|s| Box::from(*s)).collect();
    let mat = Mat::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    (genes, samples, mat)
}

fn write_tsv(path: &std::path::Path, header: bool) {
    let (genes, samples, mat) = bulk_fixture();
    let mut s = String::new();
    if header {
        s.push_str(&format!("gene\t{}\n", samples.join("\t")));
    }
    for (i, g) in genes.iter().enumerate() {
        let vals: Vec<String> = mat.row(i).iter().map(|v| format!("{v}")).collect();
        s.push_str(&format!("{g}\t{}\n", vals.join("\t")));
    }
    std::fs::write(path, s).expect("write tsv");
}

/// Regression: `read_mat` reads text with no header, so a header line was parsed
/// as counts and the `.expect` in the parser panicked. A headered TSV is the
/// ordinary case for a bulk table and has to carry its sample names through.
#[test]
fn a_headered_tsv_keeps_its_sample_names() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    write_tsv(&path, true);
    let (genes, samples, mat) = bulk_fixture();
    let got =
        super::read_labeled_mat(path.to_str().unwrap(), super::HeaderArg::Auto).expect("read");
    assert_eq!(got.rows, genes);
    assert_eq!(got.cols, samples);
    assert_eq!(got.mat, mat);
}

#[test]
fn a_headerless_tsv_still_reads_its_genes_and_values() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    write_tsv(&path, false);
    let (genes, _, mat) = bulk_fixture();
    let got =
        super::read_labeled_mat(path.to_str().unwrap(), super::HeaderArg::Auto).expect("read");
    assert_eq!(got.rows, genes);
    assert_eq!(got.mat, mat);
    assert_eq!(
        got.cols.len(),
        2,
        "one label per sample even without a header"
    );
    assert_ne!(got.cols[0], got.cols[1]);
}

#[test]
fn parquet_and_headered_tsv_read_the_same_table() {
    let dir = tempfile::tempdir().expect("tmp");
    let tsv = dir.path().join("bulk.tsv");
    let pq = dir.path().join("bulk.parquet");
    write_tsv(&tsv, true);
    let (genes, samples, mat) = bulk_fixture();
    mat.to_parquet_with_names(
        pq.to_str().unwrap(),
        (Some(&genes), Some("gene")),
        Some(&samples),
    )
    .expect("write parquet");
    let a = super::read_labeled_mat(tsv.to_str().unwrap(), super::HeaderArg::Auto).expect("tsv");
    let b = super::read_labeled_mat(pq.to_str().unwrap(), super::HeaderArg::Auto).expect("parquet");
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.mat, b.mat);
}

fn labels(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

fn model_genes() -> Vec<Box<str>> {
    labels(&["GENES", "GENEJ", "GENEO", "GENEQ", "GENEM"])
}

/// How a bulk table's axes are read against the model's genes, one row per
/// case (`Err` holds what the message must name):
/// - genes on rows is genes × samples, genes on columns samples × genes;
/// - the bridge is the shared canonicalizer, not string equality: an
///   `ENSG…_SYMBOL` axis must be recognised as genes against a bare-symbol
///   model, on either axis;
/// - a forced orientation wins over the evidence;
/// - nothing matching is a naming failure, and the error has to show what the
///   file holds (both axes and the model) so the user can see why;
/// - sample IDs that are themselves gene symbols make both axes score: an
///   ambiguity to report, naming the override, never a coin flip.
#[test]
fn orientation_follows_the_axis_holding_the_models_genes() {
    use super::Orientation::{GenesBySamples, SamplesByGenes};
    let genes = labels(&["GENES", "GENEJ", "GENEO", "NOTINMODEL"]);
    let ensg = labels(&[
        "ENSG00000000130_GENES",
        "ENSG00000000140_GENEJ",
        "ENSG00000000120_GENEO",
    ]);
    let three = labels(&["GENES", "GENEJ", "GENEO"]);
    let samples = labels(&["s0", "s1"]);
    let cases = [
        ("genes on rows", &genes, &samples, None, Ok(GenesBySamples)),
        (
            "genes on columns",
            &samples,
            &genes,
            None,
            Ok(SamplesByGenes),
        ),
        ("ENSG on rows", &ensg, &samples, None, Ok(GenesBySamples)),
        ("ENSG on columns", &samples, &ensg, None, Ok(SamplesByGenes)),
        (
            "forced",
            &three,
            &samples,
            Some(SamplesByGenes),
            Ok(SamplesByGenes),
        ),
        (
            "no axis matching",
            &labels(&["r0", "r1"]),
            &labels(&["c0", "c1"]),
            None,
            Err(&["r0", "c0", "GENES"][..]),
        ),
        (
            "both axes matching",
            &three,
            &labels(&["GENEQ", "GENEM"]),
            None,
            Err(&["--bulk-orientation"][..]),
        ),
    ];
    for (label, rows, cols, forced, want) in cases {
        let got = super::resolve_orientation(rows, cols, &model_genes(), forced);
        match want {
            Ok(o) => assert_eq!(got.expect(label), o, "{label}"),
            Err(needles) => {
                let err = got.expect_err(label).to_string();
                for n in needles {
                    assert!(err.contains(n), "{label}: {err}");
                }
            }
        }
    }
}

/// `oriented` is the only place a table is ever turned; genes end on rows.
#[test]
fn oriented_transposes_a_samples_by_genes_table_once() {
    let (genes, samples, mat) = bulk_fixture();
    let t = legume_numeric::matrix::traits::MatWithNames {
        rows: samples.clone(),
        cols: genes.clone(),
        mat: mat.transpose(),
    };
    let got = super::oriented(t, super::Orientation::SamplesByGenes);
    assert_eq!(got.rows, genes);
    assert_eq!(got.cols, samples);
    assert_eq!(got.mat, mat);
}

////////////////////////////////////////////
// read_bulk_data_aligned (deconvolve)     //
////////////////////////////////////////////

fn write_labeled_tsv(path: &std::path::Path, rows: &[Box<str>], cols: &[Box<str>], mat: &Mat) {
    let mut s = format!("id\t{}\n", cols.join("\t"));
    for (i, r) in rows.iter().enumerate() {
        let vals: Vec<String> = mat.row(i).iter().map(|v| format!("{v}")).collect();
        s.push_str(&format!("{r}\t{}\n", vals.join("\t")));
    }
    std::fs::write(path, s).expect("write tsv");
}

/// The deconvolve entry point on an ordinary headered TSV: the sample names
/// come through and the counts land on the reference rows.
#[test]
fn a_headered_bulk_tsv_aligns_to_the_reference() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    let genes = labels(&["GENES", "GENEJ"]);
    let samples = labels(&["s0", "s1"]);
    let mat = Mat::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    write_labeled_tsv(&path, &genes, &samples, &mat);
    let reference = labels(&["GENEJ", "GENES", "GENEO"]);
    let out = super::read_bulk_data_aligned(
        &[path.to_str().unwrap().into()],
        &reference,
        &super::BulkTableOpts::default(),
    )
    .expect("aligned");
    assert_eq!(out.genes, reference);
    assert_eq!(out.samples, samples);
    // GENEJ is reference row 0, GENES row 1, GENEO row 2 (unobserved → 0).
    assert_eq!(
        out.data,
        Mat::from_row_slice(3, 2, &[3.0, 4.0, 1.0, 2.0, 0.0, 0.0])
    );
}

/// A samples × genes file is turned, not refused: the data was fine.
#[test]
fn a_transposed_bulk_file_aligns_to_the_reference() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk_t.tsv");
    let genes = labels(&["GENES", "GENEJ"]);
    let samples = labels(&["s0", "s1"]);
    let mat = Mat::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]); // genes × samples
    write_labeled_tsv(&path, &samples, &genes, &mat.transpose());
    let reference = labels(&["GENEJ", "GENES", "GENEO"]);
    let out = super::read_bulk_data_aligned(
        &[path.to_str().unwrap().into()],
        &reference,
        &super::BulkTableOpts::default(),
    )
    .expect("aligned after transposing");
    assert_eq!(out.samples, samples);
    assert_eq!(
        out.data,
        Mat::from_row_slice(3, 2, &[3.0, 4.0, 1.0, 2.0, 0.0, 0.0])
    );
}

/////////////////////////////////////////
// The header blind spot and overrides   //
/////////////////////////////////////////

/// Sample IDs that are all numbers are indistinguishable BY TYPE from a count
/// row, so `auto` reads them as data; the documented way out is `--bulk-header
/// yes`, which has to take that line as the names.
#[test]
fn a_header_of_numeric_sample_ids_is_data_under_auto_and_names_under_yes() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("ids.tsv");
    std::fs::write(&path, "gene\t2019\t2020\ng0\t1\t2\n").unwrap();
    let p = path.to_str().unwrap();
    let auto = super::read_labeled_mat(p, super::HeaderArg::Auto).expect("auto");
    assert_eq!(
        auto.rows,
        labels(&["gene", "g0"]),
        "auto cannot see this header"
    );
    let yes = super::read_labeled_mat(p, super::HeaderArg::Yes).expect("yes");
    assert_eq!(yes.rows, labels(&["g0"]));
    assert_eq!(yes.cols, labels(&["2019", "2020"]));
}

/// `--bulk-header no` on a file that plainly has one must be an error, not the
/// parser's panic on `s0`.
#[test]
fn forcing_no_header_on_a_headered_file_is_an_error_not_a_panic() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    write_tsv(&path, true);
    let err = match super::read_labeled_mat(path.to_str().unwrap(), super::HeaderArg::No) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("must refuse"),
    };
    assert!(err.contains("--bulk-header"), "{err}");
    assert!(err.contains("s0"), "the offending fields are shown: {err}");
}

/// The parquet name column is found by type, not by position.
#[test]
fn a_parquet_whose_name_column_is_not_first_still_reads_genes() {
    use legume_numeric::matrix::parquet::{write_table, Column};
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("t.parquet");
    let p = path.to_str().unwrap();
    let genes = labels(&["g0", "g1"]);
    write_table(
        p,
        &[
            ("s0".into(), Column::F32(&[1.0, 2.0])),
            ("gene".into(), Column::Str(&genes)),
            ("s1".into(), Column::F32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    let got = super::read_labeled_mat(p, super::HeaderArg::Auto).expect("read");
    assert_eq!(got.rows, genes);
    assert_eq!(got.cols, labels(&["s0", "s1"]));
    assert_eq!(got.mat, Mat::from_row_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]));
}

/// A parquet written without its index has no names to align on. Say so;
/// do not stringify the first sample's counts into "gene names".
#[test]
fn a_parquet_with_no_string_column_is_refused() {
    use legume_numeric::matrix::parquet::{write_table, Column};
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("nonames.parquet");
    let p = path.to_str().unwrap();
    write_table(
        p,
        &[
            ("s0".into(), Column::F32(&[1.0, 2.0])),
            ("s1".into(), Column::F32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    let err = match super::read_labeled_mat(p, super::HeaderArg::Auto) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("must refuse"),
    };
    assert!(err.contains("name column"), "{err}");
}

#[test]
fn match_key_folds_gene_case_but_keeps_a_locus_as_is() {
    use super::{is_locus, match_key};
    use data_beans::aux::feature_names::FeatureNameKind;
    let exact = FeatureNameKind::Exact;
    assert_eq!(&*match_key(&exact, "Genej"), &*match_key(&exact, "GENEJ"));
    // A locus keeps its case: by the shared rule it can tell scaffolds apart.
    assert!(is_locus("chrX:0-100"));
    assert_ne!(
        &*match_key(&exact, "chrX:0-100"),
        &*match_key(&exact, "chrx:0-100")
    );
    // Under the locus rule both spellings of one peak share a key.
    let mixed = FeatureNameKind::Mixed;
    assert_eq!(
        &*match_key(&mixed, "chrX:0-100"),
        &*match_key(&mixed, "X:0-100")
    );
    assert!(!is_locus("GENE1"));
}
