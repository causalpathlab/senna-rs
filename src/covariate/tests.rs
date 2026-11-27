//! senna's covariate wiring on a real backend: the planted S, G2/M and
//! resting cells (cell-cycle tests) binned, the model, scores and batch
//! files written whole, and the recorded model read back for predict. The
//! scoring and binning themselves are tested in legume-genomic-types.

use super::outputs::{column_layout, Line};
use super::{
    loadable, model_covariates, model_path, Blocks, CovariateArgs, CovariateModel, CovariateOutput,
};
use crate::cell_cycle::tests::planted;

fn args(names: &[&str]) -> CovariateArgs {
    CovariateArgs {
        covariates: names.iter().map(|n| Box::from(*n)).collect(),
        ..CovariateArgs::default()
    }
}

#[test]
fn a_fit_writes_its_model_scores_and_batch_files_and_predict_reads_them() -> anyhow::Result<()> {
    let (dir, data, _) = planted(true)?;
    let n = data.num_columns();
    let batch: Vec<Box<str>> = vec!["b0".into(); n];
    let proj = nalgebra::DMatrix::from_fn(4, n, |d, i| ((i * 7 + d * 13) % 11) as f32);
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let m = model_covariates(
        &data,
        &batch,
        &proj,
        &args(&["cycle"]),
        16,
        Some(CovariateOutput {
            out: &out,
            batch_files: None,
            ignore_batch: false,
        }),
        || anyhow::bail!("no column was set aside: nothing to read again"),
    )?
    .expect("the cycle is modelled");
    // One batch file, every column a line, each its cell's label.
    let files = m.batch_files.expect("written");
    assert_eq!(files.len(), 1);
    let lines: Vec<String> = std::fs::read_to_string(files[0].as_ref())?
        .lines()
        .map(String::from)
        .collect();
    assert_eq!(lines.len(), n);
    assert!(lines.iter().zip(&m.labels).all(|(l, m)| l == m.as_ref()));
    assert!(std::path::Path::new(&format!("{out}.covariates.parquet")).is_file());
    // The model predict reads bins the fitted cells' binned cells alike.
    let model = CovariateModel::read(std::path::Path::new(&model_path(&out)))?.expect("recorded");
    let again = model.labels(&Blocks(&data), &batch, None, 16)?;
    for (a, l) in again.iter().zip(&m.labels) {
        if l.contains('|') {
            assert_eq!(a, l);
        }
    }
    Ok(())
}

#[test]
fn every_column_is_a_line_and_nothing_is_read_again_when_none_was_set_aside() -> anyhow::Result<()>
{
    let (_dir, data, _) = planted(true)?;
    let layout = column_layout(&data, None, false, || {
        anyhow::bail!("no column was set aside: nothing to read again")
    })?;
    assert_eq!(layout.len(), 1);
    assert!(layout[0]
        .iter()
        .enumerate()
        .all(|(i, l)| matches!(l, Line::Cell(c) if *c == i)));
    Ok(())
}

#[test]
fn multiome_and_a_carried_reference_skip_all_and_refuse_a_list() {
    assert!(!loadable(&args(&["all"]), true, false).unwrap());
    assert!(!loadable(&args(&["all"]), false, true).unwrap());
    assert!(loadable(&args(&["cycle"]), true, false).is_err());
    assert!(loadable(&args(&["all"]), false, false).unwrap());
    assert!(!loadable(&args(&["none"]), false, false).unwrap());
}

#[test]
fn a_reused_run_gives_its_covariate_labels_once_it_recorded_a_model() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let prefix = dir.path().join("src").to_string_lossy().into_owned();
    let mut m = senna::run_manifest::RunManifest::new(senna::run_manifest::RunKind::Svd, &prefix);
    let files =
        |m: &senna::run_manifest::RunManifest| super::recorded_batch_files(m, dir.path(), &prefix);
    std::fs::write(model_path(&prefix), "{}")?;
    assert!(files(&m).is_none(), "no batch files recorded");
    m.data.batch = vec!["src.covariate.batch.0.txt".into()];
    assert_eq!(
        files(&m).expect("recorded"),
        [Box::from(
            dir.path()
                .join("src.covariate.batch.0.txt")
                .to_string_lossy()
        )]
    );
    std::fs::remove_file(model_path(&prefix))?;
    assert!(files(&m).is_none(), "no model recorded");
    Ok(())
}

#[test]
fn a_run_fitted_with_the_former_cell_cycle_flag_is_refused_for_new_cells() -> anyhow::Result<()> {
    let (dir, data, _) = planted(true)?;
    let prefix = dir.path().join("old").to_string_lossy().into_owned();
    let batch: Vec<Box<str>> = vec!["b0".into(); data.num_columns()];
    assert_eq!(super::query_batches(&prefix, &data, batch.clone())?, batch);
    std::fs::write(format!("{prefix}.cell_cycle_model.json"), "{}")?;
    assert!(super::query_batches(&prefix, &data, batch).is_err());
    Ok(())
}

#[test]
fn a_two_track_backend_is_scored_on_its_spliced_rows() {
    let names: Vec<Box<str>> = [
        "A/count/spliced",
        "A/count/unspliced",
        "B/count/spliced",
        "B/count/unspliced",
    ]
    .into_iter()
    .map(Box::from)
    .collect();
    assert_eq!(super::scored_rows(&names), Some(vec![0, 2]));
    let spliced_only: Vec<Box<str>> = ["A/count/spliced", "B/count/spliced"]
        .into_iter()
        .map(Box::from)
        .collect();
    assert_eq!(super::scored_rows(&spliced_only), None);
    let plain: Vec<Box<str>> = ["A", "B"].into_iter().map(Box::from).collect();
    assert_eq!(super::scored_rows(&plain), None);
}
