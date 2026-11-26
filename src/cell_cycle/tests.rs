//! Scores on synthetic counts: planted S, G2/M and resting cells come back
//! in their phases, and a dictionary without the genes is skipped.

use super::score::{score, Phase, Skipped};
use data_beans::sparse_io::open_sparse_matrix_by_path;
use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::cell_cycle::{CycleGenes, CyclePhase};
use senna::sparse_output::{write_backend, Entries};

/// Genes: the Tirosh sets then 400 others; 30 cells each of S, G2/M and
/// resting (G1).
fn planted(with_cycle_genes: bool) -> anyhow::Result<(tempfile::TempDir, SparseIoVec, Vec<Phase>)> {
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
