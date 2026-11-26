//! Scores on synthetic counts: planted S, G2/M and resting cells come back
//! in their phases, and a dictionary without the genes is skipped.

use super::score::{score, Phase, Skipped};
use data_beans::sparse_io::open_sparse_matrix_by_path;
use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::cell_cycle::{CycleGenes, CyclePhase};
use senna::sparse_output::{write_backend, Entries};

/// Genes: the Tirosh sets then 400 others; 30 cells each of S, G2/M and
/// resting (G1).
pub(super) fn planted(
    with_cycle_genes: bool,
) -> anyhow::Result<(tempfile::TempDir, SparseIoVec, Vec<Phase>)> {
    let genes = CycleGenes::tirosh_2016();
    let s: Vec<&str> = genes.symbols(CyclePhase::S).collect();
    let g: Vec<&str> = genes.symbols(CyclePhase::G2M).collect();
    let mut rows: Vec<Box<str>> = Vec::new();
    if with_cycle_genes {
        rows.extend(s.iter().chain(&g).map(|n| Box::from(*n)));
    }
    let n_cycle = rows.len();
    rows.extend((0..400).map(|i| format!("OTHER{i}").into_boxed_str()));
    let mut truth = Vec::new();
    let mut trip: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..90u64 {
        let phase = [Phase::S, Phase::G2M, Phase::G1][(c / 30) as usize];
        truth.push(phase);
        // The others span the cycle genes' range of expression, so each
        // cycle gene's bin holds ordinary genes to draw controls from.
        for k in 0..400u64 {
            let v = 1.0 + ((k + c) % 8) as f32;
            trip.push((n_cycle as u64 + k, c, v));
        }
        if with_cycle_genes {
            // A count of 1 in every cell, 8 in the gene's own phase.
            for r in 0..n_cycle {
                let own = match phase {
                    Phase::S => r < s.len(),
                    Phase::G2M => r >= s.len(),
                    Phase::G1 | Phase::Undecided => false,
                };
                trip.push((r as u64, c, if own { 8.0 } else { 1.0 }));
            }
        }
    }
    let cols: Vec<Box<str>> = (0..90).map(|c| format!("c{c}").into_boxed_str()).collect();
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("x.zarr.zip");
    let eff = write_backend(
        path.to_str().unwrap(),
        Entries::Triplets(trip),
        &rows,
        &cols,
    )?;
    let mut data = SparseIoVec::new();
    data.push(
        std::sync::Arc::from(open_sparse_matrix_by_path(&eff)?),
        None,
    )?;
    Ok((dir, data, truth))
}

#[test]
fn planted_phases_are_called_and_their_genes_learned() -> anyhow::Result<()> {
    let (_dir, data, truth) = planted(true)?;
    let genes = CycleGenes::tirosh_2016();
    let s = score(&data, &genes, 16, true, 1)?.expect("scored");
    let right = s.phase.iter().zip(&truth).filter(|(a, b)| a == b).count();
    assert!(right >= 85, "{right} of 90 phases right");
    // Each S cell scores higher on S than on G2/M, and the reverse.
    for (j, t) in truth.iter().enumerate() {
        match t {
            Phase::S => assert!(s.s[j] > s.g2m[j]),
            Phase::G2M => assert!(s.g2m[j] > s.s[j]),
            Phase::G1 => assert!(s.s[j] < 0.0 && s.g2m[j] < 0.0),
            Phase::Undecided => unreachable!("no cell is planted undecided"),
        }
    }
    // Learned: every set gene correlates with its own score.
    let (r_s, r_g2m) = s.learned.as_ref().expect("learned");
    assert!(s.s_rows.iter().all(|&g| r_s[g] > 0.3));
    assert!(s.g2m_rows.iter().all(|&g| r_g2m[g] > 0.3));
    Ok(())
}

#[test]
fn a_dictionary_without_the_genes_is_skipped_not_failed() -> anyhow::Result<()> {
    let (_dir, data, _) = planted(false)?;
    let out = score(&data, &CycleGenes::tirosh_2016(), 16, false, 1)?;
    assert_eq!(out.err(), Some(Skipped { s: 0, g2m: 0 }));
    Ok(())
}

#[test]
fn cycle_bins_become_batch_labels_written_whole() -> anyhow::Result<()> {
    use super::covariate::{column_layout, cycle_batches, CycleCovariateArgs, CycleMode, Line};
    let (dir, data, truth) = planted(true)?;
    let n = truth.len();
    let batch: Vec<Box<str>> = (0..n)
        .map(|i| if i % 2 == 0 { "b0" } else { "b1" }.into())
        .collect();
    // A projection with no structure of its own: one group of cells.
    let proj = nalgebra::DMatrix::from_fn(4, n, |d, i| ((i * 7 + d * 13) % 11) as f32);
    let args = CycleCovariateArgs {
        mode: CycleMode::All,
        cycle_bins_max: 10,
        ..Default::default()
    };
    let cb = cycle_batches(&data, &batch, &proj, &args, 16)?.expect("the cycle is modelled");
    assert!(cb.model.k >= 2, "k = {}", cb.model.k);
    assert!(cb
        .labels
        .iter()
        .all(|l| l.starts_with("b0|cc") || l.starts_with("b1|cc")));
    // S cells and resting cells do not share a bin.
    assert_ne!(
        cb.labels[0].split('|').nth(1),
        cb.labels[60].split('|').nth(1)
    );

    // Every loaded cell at its place in its file; nothing set aside, so
    // nothing is read again.
    let cells = data.column_names()?;
    let layout = column_layout(&data, None, false, || {
        anyhow::bail!("no column was set aside: nothing to read again")
    })?;
    assert_eq!(layout.len(), 1);
    assert!(layout[0]
        .iter()
        .enumerate()
        .all(|(i, l)| matches!(l, Line::Cell(c) if *c == i)));

    // A column QC set aside is written with its plain label.
    let mut lines: Vec<Line> = (0..n).map(Line::Cell).collect();
    lines.push(Line::Plain("b0".into()));
    let layout = vec![lines];
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let files = cb.write(&out, &cells, &layout)?;
    let lines: Vec<String> = std::fs::read_to_string(files[0].as_ref())?
        .lines()
        .map(String::from)
        .collect();
    assert_eq!(lines.len(), n + 1);
    assert_eq!(lines[0], cb.labels[0].as_ref());
    assert_eq!(lines[n], "b0");
    assert!(std::path::Path::new(&format!("{out}.cell_cycle_model.json")).is_file());

    // Labels an earlier cycle fit wrote are binned afresh, not refused.
    let again: Vec<Box<str>> = cb.labels.clone();
    let cb2 = cycle_batches(&data, &again, &proj, &args, 16)?.expect("modelled again");
    assert_eq!(cb2.labels, cb.labels);
    Ok(())
}
