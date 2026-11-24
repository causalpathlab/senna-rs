use crate::ode_mixture::ratio_pbs::{
    cell_offset, ratio_pbs, residuals, sigmoid, with_the_rest, RatioArgs,
};
use crate::ode_mixture::sim::{simulate, SimConfig};
use data_beans::aux::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use data_beans::aux::feature_names::FeatureNameKind;
use data_beans::sparse_io_vector::{ColumnAlignment, RowAlignment};
use nalgebra::DMatrix;
use senna::sparse_output::{write_backend, Entries};

/// A cell whose every gene is shifted by one logit is all offset, no
/// residual: its overall unspliced share says nothing of the dynamics.
#[test]
fn a_shift_of_every_gene_is_the_cells_offset() {
    let eta = [-1.0f32, 0.0, 0.5, 1.5];
    let entries: Vec<(u32, f32, f32)> = eta
        .iter()
        .enumerate()
        .map(|(g, &e)| {
            let p = sigmoid(e + 2.0);
            (g as u32, 100.0 * p, 100.0 * (1.0 - p))
        })
        .collect();
    assert!((cell_offset(&entries, &eta) - 2.0).abs() < 1e-3);
    assert!(residuals(&entries, &eta)
        .iter()
        .all(|&(_, z)| z.abs() < 1e-2));
}

/// A gene out of step with the cell's others stands out, the rest do not.
#[test]
fn a_gene_out_of_step_has_the_residual() {
    let eta = [0.0f32; 5];
    let mut entries: Vec<(u32, f32, f32)> = (0..5).map(|g| (g, 50.0, 50.0)).collect();
    entries[3] = (3, 90.0, 10.0);
    let z = residuals(&entries, &eta);
    assert!(z[3].1 > 4.0, "{z:?}");
    assert!(
        z.iter().filter(|e| e.0 != 3).all(|e| e.1.abs() < 2.5),
        "{z:?}"
    );
}

/// Simulated cells (pseudobulks of a cell's depth) on two lineages, a fifth
/// of them with every unspliced read four times over (a nuclear share):
/// the pseudobulks hold cells of one time and lineage, and the nuclear
/// cells among the others rather than apart.
#[test]
fn splicing_groups_cells_by_time_and_lineage_not_by_their_share() {
    let sim = simulate(&SimConfig {
        n_genes: 300,
        n_pb: 3000,
        depth: 1500.0,
        lineages: 2,
        fine: 200,
        seed: 3,
        ..SimConfig::default()
    })
    .unwrap();
    let (n, g) = (sim.n_pb, sim.n_genes);
    let nuclear: Vec<bool> = (0..n).map(|c| c % 5 == 0).collect();
    // Rows: every gene's spliced then unspliced; columns: cells.
    let m = DMatrix::<f32>::from_fn(2 * g, n, |r, c| {
        let (gene, unspliced) = (r / 2, r % 2 == 1);
        if unspliced {
            let u = sim.u[c * g + gene] as f32;
            if nuclear[c] {
                4.0 * u
            } else {
                u
            }
        } else {
            sim.s[c * g + gene] as f32
        }
    });
    let rows: Vec<Box<str>> = (0..2 * g)
        .map(|r| {
            let track = if r % 2 == 1 { "unspliced" } else { "spliced" };
            format!("g{}/count/{track}", r / 2).into()
        })
        .collect();
    let cols: Vec<Box<str>> = (0..n).map(|c| format!("c{c}").into()).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("cells.zarr.zip")
        .to_string_lossy()
        .into_owned();
    write_backend(&path, Entries::Dense(&m), &rows, &cols).unwrap();
    let data = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: vec![path.into()],
        feature_kind: Some(FeatureNameKind::Exact),
        row_alignment: RowAlignment::Intersect,
        column_alignment: ColumnAlignment::Disjoint,
        ..Default::default()
    })
    .unwrap()
    .data;
    let n_pbs = 150;
    let mut row_totals = vec![0f64; data.num_rows()];
    let x = data.read_columns_csc(0..n).unwrap();
    for (&r, &v) in x.row_indices().iter().zip(x.values()) {
        row_totals[r] += f64::from(v);
    }
    let out = ratio_pbs(
        &data,
        None,
        &vec![true; n],
        &vec![0; n],
        &row_totals,
        RatioArgs {
            n_pbs,
            block_size: 500,
            seed: 1,
        },
    )
    .unwrap();
    let truth = &sim.truth;
    let time: Vec<f64> = (0..n)
        .map(|c| truth.time[c] / truth.t_len[truth.lineage[c]])
        .collect();
    let p_count = out.n_cells.len();
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); p_count];
    for (c, &p) in out.member.iter().enumerate() {
        groups[p].push(c);
    }
    let sd = |xs: &[f64]| {
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
    };
    let within: f64 = groups
        .iter()
        .map(|gr| sd(&gr.iter().map(|&c| time[c]).collect::<Vec<_>>()) * gr.len() as f64)
        .sum::<f64>()
        / n as f64;
    let overall = sd(&time);
    // Lineage purity past the start, where both lineages leave steady state.
    let (mut agree, mut total) = (0usize, 0usize);
    for gr in &groups {
        let late: Vec<usize> = gr.iter().copied().filter(|&c| time[c] > 0.3).collect();
        if late.is_empty() {
            continue;
        }
        let ones = late.iter().filter(|&&c| truth.lineage[c] == 1).count();
        agree += ones.max(late.len() - ones);
        total += late.len();
    }
    let purity = agree as f64 / total as f64;
    // Nuclear cells in pseudobulks that are mostly nuclear.
    let apart = groups
        .iter()
        .filter(|gr| gr.iter().filter(|&&c| nuclear[c]).count() * 5 > gr.len() * 4)
        .map(|gr| gr.iter().filter(|&&c| nuclear[c]).count())
        .sum::<usize>() as f64
        / nuclear.iter().filter(|&&x| x).count() as f64;
    eprintln!(
        "{p_count} pseudobulks: time sd within {within:.3} vs overall {overall:.3}; \
         lineage purity past 0.3 {purity:.3}; nuclear cells apart {apart:.3}"
    );
    assert!(
        within < 0.45 * overall,
        "time sd within {within} vs {overall}"
    );
    assert!(purity > 0.9, "lineage purity {purity}");
    assert!(apart < 0.2, "nuclear cells apart {apart}");
}

/// One read at a share near 0 says no more than an even share would.
#[test]
fn a_stray_read_is_clipped_to_its_evidence() {
    let mut eta = [0.0f32; 5];
    eta[0] = -9.0;
    let entries: Vec<(u32, f32, f32)> = (0..5)
        .map(|g| (g, if g == 0 { 1.0 } else { 0.0 }, 5.0))
        .collect();
    let z = residuals(&entries, &eta);
    let n = 6f32;
    assert!(z.iter().all(|e| e.1.abs() <= n.sqrt() + 1e-6), "{z:?}");
}

/// A cell held out of the grouping joins the pseudobulk most grouped cells
/// of its collapse pseudobulk joined; with none grouped there, the largest.
#[test]
fn a_held_out_cell_follows_its_collapse_neighbours() {
    // Cells 0..4 grouped; 4 and 5 held out. Collapse: {0,1,2,4} and {3,5}…
    let collapsed = [0, 0, 0, 1, 0, 2];
    let cells = [0, 1, 2, 3];
    let grouped = [1, 1, 0, 0];
    let member = with_the_rest(&cells, &grouped, &collapsed, 2);
    assert_eq!(&member[..4], &grouped);
    assert_eq!(member[4], 1, "most of collapse 0 joined pseudobulk 1");
    // …and cell 5's collapse pseudobulk 2 has none grouped: the largest
    // (a tie of two each, the lower id).
    assert_eq!(member[5], 0);
}
