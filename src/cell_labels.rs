//! Cells matched by name across tables, and the label make-up of groups of
//! cells: shared by the critique, `senna cell-cycle`'s run diagnostics and
//! any reader that joins a per-cell table to a run's cells.

use rustc_hash::FxHashMap;

/// Position of every source name in `target`, or `usize::MAX` when absent.
/// Identical lists match by position, repeats included. Otherwise a name
/// that would be matched is refused when it repeats on either side, since no
/// order says which row it means; a repeat that matches nothing is harmless.
pub fn tolerant_align(source: &[Box<str>], target: &[Box<str>]) -> anyhow::Result<Vec<usize>> {
    if source == target {
        return Ok((0..source.len()).collect());
    }
    // A name repeated in `target` maps to `REPEATED`, refused only if looked up.
    const REPEATED: usize = usize::MAX - 1;
    let mut at: FxHashMap<&str, usize> =
        FxHashMap::with_capacity_and_hasher(target.len(), Default::default());
    for (i, n) in target.iter().enumerate() {
        at.entry(n.as_ref())
            .and_modify(|j| *j = REPEATED)
            .or_insert(i);
    }
    let mut seen = vec![false; target.len()];
    source
        .iter()
        .map(|n| match at.get(n.as_ref()).copied() {
            None => Ok(usize::MAX),
            Some(REPEATED) => Err(repeated(n)),
            Some(j) if std::mem::replace(&mut seen[j], true) => Err(repeated(n)),
            Some(j) => Ok(j),
        })
        .collect()
}

fn repeated(name: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "cell name {name} appears more than once, so cells cannot be matched by name; \
         make the names unique (e.g. suffix each sample's barcodes)"
    )
}

/// Per group (a pseudobulk, a cluster), the fraction of its labelled cells
/// carrying each label; `None` for a group with no labelled cell.
pub fn label_composition(
    pb_of_cell: &[usize],
    label_of_cell: &[Option<u32>],
    n_pb: usize,
    n_labels: usize,
) -> Vec<Option<Vec<f32>>> {
    let mut count = vec![vec![0u32; n_labels]; n_pb];
    for (&pb, label) in pb_of_cell.iter().zip(label_of_cell) {
        if let (true, Some(k)) = (pb != usize::MAX, label) {
            count[pb][*k as usize] += 1;
        }
    }
    count
        .into_iter()
        .map(|c| {
            let total: u32 = c.iter().sum();
            (total > 0).then(|| c.iter().map(|&n| n as f32 / total as f32).collect())
        })
        .collect()
}
