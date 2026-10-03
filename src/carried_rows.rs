//! The rows of a given feature table that pin nothing on a run's feature
//! axis, carried through unchanged into the run's own ρ output (see
//! `feature_preset` in the binary): built once by the preset loader, appended
//! by [`crate::run_manifest::write_run_manifest`] for every engine.

use data_beans::aux::feature_types::{
    feature_rows, feature_types_path, is_data_feature_type, read_feature_types,
    write_feature_types, FeatureType, GENE_TYPE,
};
use data_beans::aux::frozen_features::FrozenFeatureHost;
use graph_embedding_util as ge;
use legume_numeric::matrix::dmatrix_util::concatenate_vertical;
use legume_numeric::matrix::parquet::{peek_parquet_field_names, read_parquet_string_column};
use legume_numeric::matrix::traits::IoOps;
use log::info;
use nalgebra::DMatrix;
use rustc_hash::FxHashSet;

/// Which rows of a table (`names`, in order) may match a data feature, and
/// `types` (the types table written beside it) when it describes those rows:
/// by their types, a gene or a genomic window — not a term, word or cell
/// type that may share a gene's name. Without types, or with a types table
/// written for another table (an older run's under the same prefix, which
/// `what` names in a warning), every row may match and is taken as a gene.
#[must_use]
pub fn matchable_rows<'a>(
    types: Option<&'a [FeatureType]>,
    names: &[Box<str>],
    what: &str,
) -> (Vec<bool>, Option<&'a [FeatureType]>) {
    if let Some(t) = types {
        if let Some(marks) = feature_rows(t, names) {
            return (marks, Some(t));
        }
        log::warn!(
            "the types table beside {what} lists other rows (left by another run?); \
             every row of {what} is taken as a gene"
        );
    }
    (vec![true; names.len()], None)
}

/// Remove `{out_prefix}.feature_types.parquet` when it is older than the
/// run's ρ at `rho_path` and does not list its rows: another run's, left
/// under this prefix, which would otherwise be read as describing this one.
pub fn clear_stale_types(out_prefix: &str, rho_path: &str) -> anyhow::Result<()> {
    let path = feature_types_path(out_prefix);
    if !older(&path, rho_path) {
        return Ok(());
    }
    let Some(types) = read_feature_types(out_prefix)? else {
        return Ok(());
    };
    let rows = read_parquet_string_column(rho_path, 0)?;
    if feature_rows(&types, &rows).is_none() {
        std::fs::remove_file(&path)?;
        info!("Removed {path}: it describes an earlier run's table, not {rho_path}");
    }
    Ok(())
}

/// Whether file `a` was last written before file `b`.
fn older(a: &str, b: &str) -> bool {
    let modified = |p: &str| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    matches!((modified(a), modified(b)), (Some(x), Some(y)) if x < y)
}

/// The rows of a given table that pin nothing on this run's feature axis,
/// kept to write back beside the trained table. Only meaningful when the
/// matched rows stay in the table's space (freeze, lora): then the trained
/// rows and these sit in one space, and the output is the full table.
#[derive(Clone, Debug, PartialEq)]
pub struct CarriedRows {
    pub names: Vec<Box<str>>,
    /// One per row, from the source run's types table (`gene` without one).
    pub types: Vec<Box<str>>,
    /// `[n, H]`, in source order.
    pub rows: DMatrix<f32>,
    /// The file the rows came from, for the log.
    pub source: String,
}

impl CarriedRows {
    /// Every source row the match left unused whose name (raw, or canonical
    /// under `kind`) is not a feature of `target_names` either — a duplicate
    /// canonical row of a matched gene, or a word that also names a feature,
    /// would otherwise come out twice. `None` when nothing is left over, and
    /// `None` with a log line when `pins` is false: then the trained rows
    /// leave the table's space and the leftover rows would not belong beside
    /// them. `flag` names the caller's mode for that line; `src_types` is the
    /// source's types, one per source row in order ([`types_of_rows`]; empty
    /// when it has none: all genes).
    pub fn from_unmatched(
        pins: bool,
        flag: &str,
        host: &FrozenFeatureHost,
        target_names: &[Box<str>],
        kind: &ge::FeatureNameKind,
        src_types: &[FeatureType],
        dictionary_path: &str,
    ) -> anyhow::Result<Option<Self>> {
        if !pins {
            info!(
                "{flag}: the rows of {dictionary_path} that matched nothing are not carried \
                 through, since the trained rows leave the table's space"
            );
            return Ok(None);
        }
        anyhow::ensure!(
            src_types.is_empty() || src_types.len() == host.src_names.len(),
            "{dictionary_path}: {} types for {} rows",
            src_types.len(),
            host.src_names.len()
        );
        let used: FxHashSet<usize> = host.keep_src_indices.iter().copied().collect();
        let taken_exact: FxHashSet<&Box<str>> = target_names.iter().collect();
        let taken: FxHashSet<Box<str>> = target_names
            .iter()
            .flat_map(|n| [n.clone(), kind.canonicalize(n)])
            .collect();
        let idx: Vec<usize> = host
            .src_names
            .iter()
            .enumerate()
            // A row named as a feature of the axis would come out twice. A
            // gene or region row counts by its canonical name too; a term,
            // word or cell type is its own row, so only by its exact name.
            .filter(|(i, n)| {
                let data = src_types
                    .get(*i)
                    .is_none_or(|(_, t)| is_data_feature_type(t));
                let matched_or_taken = used.contains(i)
                    || taken_exact.contains(*n)
                    || data && taken.contains(&kind.canonicalize(n));
                !matched_or_taken
            })
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() {
            return Ok(None);
        }
        let names: Vec<Box<str>> = idx.iter().map(|&i| host.src_names[i].clone()).collect();
        // By position: a cell type and a gene may share a name.
        let types: Vec<Box<str>> = idx
            .iter()
            .map(|&i| {
                src_types
                    .get(i)
                    .map_or_else(|| GENE_TYPE.into(), |(_, t)| t.clone())
            })
            .collect();
        let rows = host.src_e_feat.select_rows(idx.iter());
        Ok(Some(Self {
            names,
            types,
            rows,
            source: dictionary_path.to_string(),
        }))
    }

    /// Append these rows to the ρ table the run wrote at `{out_prefix}.{suffix}`,
    /// keeping its row axis and column names, and write
    /// `{out_prefix}.feature_types.parquet` over every row: the run's own rows
    /// keep the types it wrote, or are all `gene` when it wrote none.
    ///
    /// A carried row whose name the run wrote itself (a term or cell type both
    /// fne graphs hold, say) is superseded by the run's own, trained row. This
    /// is a second filter on purpose: [`Self::from_unmatched`] can only see the
    /// feature axis the match ran on, and a run may write more rows than that.
    pub fn append_to(&self, out_prefix: &str, suffix: &str) -> anyhow::Result<()> {
        let rho_path = &format!("{out_prefix}.{suffix}");
        let fields = peek_parquet_field_names(rho_path)?;
        let row_axis = fields
            .first()
            .ok_or_else(|| anyhow::anyhow!("{rho_path}: no columns"))?;
        let table = DMatrix::<f32>::from_parquet(rho_path)?;
        let (n, h) = (table.mat.nrows(), table.mat.ncols());
        anyhow::ensure!(
            h == self.rows.ncols(),
            "{rho_path} is {h} wide but the carried rows of {} are {}",
            self.source,
            self.rows.ncols()
        );
        let own: FxHashSet<&str> = table.rows.iter().map(AsRef::as_ref).collect();
        let keep: Vec<usize> = (0..self.names.len())
            .filter(|&i| !own.contains(self.names[i].as_ref()))
            .collect();
        let superseded = self.names.len() - keep.len();
        if keep.is_empty() {
            info!(
                "Nothing of {} to carry through: the run wrote all {superseded} of its unmatched rows itself",
                self.source
            );
            return Ok(());
        }
        // A types table an earlier run left here is replaced; one written
        // since the table that disagrees with it is a writer's fault.
        let types_path = feature_types_path(out_prefix);
        let own_types: Vec<Box<str>> = match read_feature_types(out_prefix)? {
            Some(rows) if feature_rows(&rows, &table.rows).is_some() => {
                rows.into_iter().map(|(_, t)| t).collect()
            }
            Some(_) => {
                anyhow::ensure!(
                    older(&types_path, rho_path),
                    "{types_path}: rows disagree with {rho_path}"
                );
                log::warn!("{types_path} describes an earlier run's table; replacing it");
                vec![GENE_TYPE.into(); n]
            }
            None => vec![GENE_TYPE.into(); n],
        };

        let m = keep.len();
        let mat = concatenate_vertical(&[table.mat, self.rows.select_rows(keep.iter())])?;
        let mut names = table.rows;
        names.extend(keep.iter().map(|&i| self.names[i].clone()));
        mat.to_parquet_with_names(rho_path, (Some(&names), Some(row_axis)), Some(&table.cols))?;
        let mut types = own_types;
        types.extend(keep.iter().map(|&i| self.types[i].clone()));
        write_feature_types(out_prefix, &names, &types)?;
        let n_gene = keep
            .iter()
            .filter(|&&i| self.types[i].as_ref() == GENE_TYPE)
            .count();
        info!(
            "Carried {m} rows of {} through unchanged into {rho_path} ({n_gene} gene, {} other; \
             {superseded} superseded by the run's own rows); every row's type is in {}",
            self.source,
            m - n_gene,
            feature_types_path(out_prefix)
        );
        Ok(())
    }
}
