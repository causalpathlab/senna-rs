//! A fit reusing an earlier run's preprocessing (`--pb-from`): its rows,
//! cells, HVG selection and projection, and with covariates its labels.

use super::{read_args, LoadProjectArgs, ProjectedData};
use data_beans::alg::hvg::HvgSelection;
use log::info;
use senna::senna_input::{read_data_on_shared_rows, SparseDataWithBatch};

/// The run at `run`'s kept preprocessing ([`crate::topic::prep`]) applied to
/// this one's files: read without QC, the rows and cells QC kept there kept
/// here, its HVG selection and projection taken as they are. `None` when it
/// kept none, or it does not fit this run's files and sketch width.
pub(super) fn reuse_preprocessing(
    args: &LoadProjectArgs,
    run: &str,
    covariates_on: bool,
) -> anyhow::Result<Option<ProjectedData>> {
    let (m, dir) = senna::run_manifest::load_for(run)?;
    let prefix = senna::run_manifest::resolve(&dir, &m.prefix);
    let prefix = prefix.to_string_lossy();
    let Some(prep) = crate::topic::prep::Prep::read(&prefix)? else {
        return Ok(None);
    };
    if prep.proj_kn.nrows() != args.proj_dim {
        return Ok(None);
    }
    // With covariates, its projection was made under its `batch|cvK` labels:
    // reused only with them.
    let covariate_files = if covariates_on {
        let files = crate::covariate::recorded_batch_files(&m, &dir, &prefix);
        if files.is_none() {
            info!("--pb-from {run}: it recorded no covariate model; its preprocessing not reused");
            return Ok(None);
        }
        files
    } else {
        None
    };
    let mut read = read_args(args, None);
    if covariate_files.is_some() {
        read.batch_files.clone_from(&covariate_files);
    }
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: mut batch_membership,
        ..
    } = read_data_on_shared_rows(read)?;
    let by_name = |names: &[Box<str>], kept: &[Box<str>]| -> Vec<bool> {
        let kept: std::collections::HashSet<&str> = kept.iter().map(AsRef::as_ref).collect();
        names.iter().map(|n| kept.contains(n.as_ref())).collect()
    };
    let rows = data_vec.row_names()?;
    if rows != prep.rows {
        data_vec.mask_rows(&by_name(&rows, &prep.rows))?;
        if data_vec.row_names()? != prep.rows {
            return Ok(None);
        }
    }
    let cells = data_vec.column_names()?;
    if cells != prep.cells {
        let keep = by_name(&cells, &prep.cells);
        data_vec.mask_columns(&keep)?;
        batch_membership = data_beans::qc_lib::filter_by_keep(&batch_membership, &keep);
        if data_vec.column_names()? != prep.cells {
            return Ok(None);
        }
    }
    if args.ignore_batch && covariate_files.is_none() {
        senna::senna_input::collapse_to_single_batch(&mut batch_membership);
    }
    if let (Some(out), Some(_)) = (args.out, &covariate_files) {
        crate::covariate::copy_recorded(&prefix, out)?;
    }
    let selected_features = prep.hvg.map(|names| {
        let at: std::collections::HashMap<&str, usize> = prep
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.as_ref(), i))
            .collect();
        let selected_indices: Vec<usize> = names
            .iter()
            .filter_map(|n| at.get(n.as_ref()).copied())
            .collect();
        let index_map = selected_indices
            .iter()
            .enumerate()
            .map(|(new_i, &old_i)| (old_i, new_i))
            .collect();
        HvgSelection {
            selected_indices,
            selected_names: names,
            index_map,
        }
    });
    info!(
        "Preprocessing of {run}: {} cells, {} rows, its HVG selection and projection reused",
        prep.cells.len(),
        prep.rows.len()
    );
    Ok(Some(ProjectedData {
        batch_files: covariate_files.or_else(|| args.batch_files.clone()),
        data_vec,
        batch_membership,
        proj_kn: prep.proj_kn,
        selected_features,
        output_keep_idx: prep.output,
    }))
}

/// Keep `done`'s preprocessing at `args.out` for a later fit (not with
/// carried pseudobulks, whose columns are not cells).
pub(super) fn keep_preprocessing(
    args: &LoadProjectArgs,
    done: &ProjectedData,
) -> anyhow::Result<()> {
    let Some(out) = args.out.filter(|_| args.pb_reference.is_none()) else {
        return Ok(());
    };
    crate::topic::prep::Prep {
        cells: done.data_vec.column_names()?,
        rows: done.data_vec.row_names()?,
        hvg: done
            .selected_features
            .as_ref()
            .map(|s| s.selected_names.clone()),
        output: done.output_keep_idx.clone(),
        proj_kn: done.proj_kn.clone(),
    }
    .write(out)
}
