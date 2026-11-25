//! Small checks of the algorithm's parts: the gradient, the proximal step,
//! the section factor, the matching, and what the contrast recovers on a
//! planted three-donor set (no full fits, no performance thresholds).

use super::contrast_fit::*;
use super::matched_other_donor_units::*;
use super::same_section_gene_factor::*;
use senna::embed_common::*;

/// Units on a line: `units_per_donor` per donor at states 0..1, each unit's
/// state coordinates (intercept, state) and its donor.
fn units_on_a_line(
    number_of_donors: usize,
    units_per_donor: usize,
) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
    let mut state_of_unit = Vec::new();
    let mut donor_of_unit = Vec::new();
    for donor in 0..number_of_donors {
        for i in 0..units_per_donor {
            state_of_unit.push(i as f32 / (units_per_donor - 1) as f32);
            donor_of_unit.push(donor);
        }
    }
    let coordinates = state_of_unit.iter().flat_map(|&s| [1.0, s]).collect();
    (coordinates, donor_of_unit, state_of_unit)
}

/// Each unit's matched units: the units of the other donors at the same
/// position on the line (exact matches of state).
/// (distances all 0; one match per other donor).
fn matched_at_same_state(
    number_of_donors: usize,
    units_per_donor: usize,
) -> MatchedOtherDonorUnits {
    let mut matched = Vec::new();
    for donor in 0..number_of_donors {
        for i in 0..units_per_donor {
            for other in (0..number_of_donors).filter(|&o| o != donor) {
                matched.push((other * units_per_donor + i) as u32);
            }
        }
    }
    matched_units_from(matched, number_of_donors - 1)
}

/// Matched units given as a flat list, `per_unit` each, at distance 0.
fn matched_units_from(matched_unit_of_unit: Vec<u32>, per_unit: usize) -> MatchedOtherDonorUnits {
    MatchedOtherDonorUnits {
        distance_to_matched_unit: vec![0.0; matched_unit_of_unit.len()],
        matched_unit_of_unit,
        matched_per_unit: per_unit,
    }
}

/// Two sections per donor on the line: its even and its odd units.
fn two_sections_per_donor(number_of_donors: usize, units_per_donor: usize) -> Vec<usize> {
    (0..number_of_donors * units_per_donor)
        .map(|p| 2 * (p / units_per_donor) + p % 2)
        .collect()
}

#[test]
fn contrast_gradient_matches_finite_differences_with_and_without_a_section() {
    let (coordinates, donors, _) = units_on_a_line(3, 5);
    let matched = matched_at_same_state(3, 5);
    let sections = two_sections_per_donor(3, 5);
    let activity: Vec<f32> = (0..15).map(|p| 1.0 + (p % 4) as f32).collect();
    let anchors: Vec<u32> = (0..15).collect();
    let sets = contrast_sets_of_gene(&activity, &anchors, &matched);
    let states = UnitStates {
        state_coordinates: &coordinates,
        coordinates_per_unit: 2,
        donor_index_of_unit: &donors,
        number_of_donors: 3,
    };
    let deviation = vec![0.3, -0.2, 0.1, 0.5, -0.4, 0.2];
    let left_out_section_3 = LeftOutSection {
        section_index_of_unit: &sections,
        left_out_section: 3,
    };
    for left_out in [None, Some(&left_out_section_3)] {
        let mut gradient = vec![0f32; 6];
        contrast_loss_and_gradient(&deviation, &sets, &states, left_out, &mut gradient);
        let mut scratch = vec![0f32; 6];
        for k in 0..6 {
            let h = 1e-3;
            let (mut up, mut down) = (deviation.clone(), deviation.clone());
            up[k] += h;
            down[k] -= h;
            let numeric = (contrast_loss_and_gradient(&up, &sets, &states, left_out, &mut scratch)
                - contrast_loss_and_gradient(&down, &sets, &states, left_out, &mut scratch))
                / (2.0 * h);
            assert!(
                (numeric - gradient[k]).abs() < 1e-3,
                "coefficient {k}: {numeric} vs {}",
                gradient[k]
            );
        }
    }
}

#[test]
fn leaving_out_a_section_with_no_units_changes_nothing() {
    let (coordinates, donors, _) = units_on_a_line(3, 5);
    let matched = matched_at_same_state(3, 5);
    let sections = two_sections_per_donor(3, 5);
    let activity: Vec<f32> = (0..15).map(|p| 1.0 + (p % 3) as f32).collect();
    let anchors: Vec<u32> = (0..15).collect();
    let sets = contrast_sets_of_gene(&activity, &anchors, &matched);
    let states = UnitStates {
        state_coordinates: &coordinates,
        coordinates_per_unit: 2,
        donor_index_of_unit: &donors,
        number_of_donors: 3,
    };
    let deviation = vec![0.2, 0.1, -0.3, 0.4, 0.0, -0.1];
    let mut gradient = vec![0f32; 6];
    let mut gradient_without_empty = vec![0f32; 6];
    let loss = contrast_loss_and_gradient(&deviation, &sets, &states, None, &mut gradient);
    let empty_section = LeftOutSection {
        section_index_of_unit: &sections,
        left_out_section: 6,
    };
    let loss_without_empty = contrast_loss_and_gradient(
        &deviation,
        &sets,
        &states,
        Some(&empty_section),
        &mut gradient_without_empty,
    );
    assert_eq!(loss, loss_without_empty);
    assert_eq!(gradient, gradient_without_empty);
}

#[test]
fn a_refit_without_a_section_from_the_full_fit_lands_where_one_from_zero_does() {
    let (coordinates, donors, state) = units_on_a_line(3, 20);
    let matched = matched_at_same_state(3, 20);
    let sections = two_sections_per_donor(3, 20);
    // donor 2 gains the gene at the high end
    let activity: Vec<f32> = state
        .iter()
        .zip(&donors)
        .map(|(&s, &d)| if d == 2 && s > 0.5 { 40.0 } else { 5.0 })
        .collect();
    let anchors: Vec<u32> = (0..60).collect();
    let sets = contrast_sets_of_gene(&activity, &anchors, &matched);
    let states = UnitStates {
        state_coordinates: &coordinates,
        coordinates_per_unit: 2,
        donor_index_of_unit: &donors,
        number_of_donors: 3,
    };
    let settings = ContrastFitSettings {
        group_lasso_penalty: 0.01,
        max_iterations: 20000,
        gradient_mapping_tolerance: 1e-7,
    };
    let full = fit_deviation_of_gene(&sets, &states, None, None, &settings).deviation;
    let left_out = LeftOutSection {
        section_index_of_unit: &sections,
        left_out_section: 4,
    };
    let from_zero =
        fit_deviation_of_gene(&sets, &states, None, Some(&left_out), &settings).deviation;
    let from_full = super::section_jackknife::deviation_without_each_section(
        &full, &sets, &states, &sections, 6, &settings,
    );
    for (w, z) in from_full[4].deviation.iter().zip(&from_zero) {
        assert!(
            (w - z).abs() < 1e-3,
            "{:?} vs {from_zero:?}",
            from_full[4].deviation
        );
        assert!(!from_full[4].reached_iteration_cap);
    }
    // leaving out units that carry the gain changes the fit
    assert_ne!(from_zero, full);
}

#[test]
fn group_lasso_step_shrinks_each_donor_row_or_zeroes_it() {
    let mut deviation = vec![3.0, 4.0, 0.3, 0.4];
    shrink_each_donor_row_by_group_lasso(&mut deviation, 2, 1.0);
    // row 1: norm 5 -> 4; row 2: norm 0.5 < 1 -> 0
    assert!((deviation[0] - 2.4).abs() < 1e-6 && (deviation[1] - 3.2).abs() < 1e-6);
    assert_eq!(&deviation[2..], &[0.0, 0.0]);
}

#[test]
fn median_with_zeros_counts_the_units_without_signal() {
    assert_eq!(median_with_zeros(vec![5.0, 6.0], 5), 0.0);
    assert_eq!(median_with_zeros(vec![5.0, 6.0, 7.0], 5), 5.0);
    assert_eq!(median_with_zeros(vec![2.0], 2), 1.0);
}

#[test]
fn section_factor_takes_a_section_wide_shift_and_leaves_a_minority_domain() {
    // two sections of 10 units, each unit matched to the same position in the other
    let matched = matched_units_from((0..20u32).map(|p| (p + 10) % 20).collect(), 1);
    let section: Vec<usize> = (0..20).map(|p| p / 10).collect();
    // section 0: three times section 1 everywhere, plus a 3-unit domain x10
    let mut count: Vec<f32> = (0..20).map(|p| if p < 10 { 30.0 } else { 10.0 }).collect();
    for c in &mut count[0..3] {
        *c *= 10.0;
    }
    let all: Vec<u32> = (0..20).collect();
    let factor = log_factor_of_section_for_gene(&count, &all, &matched, &section, &[10, 10]);
    let expected = ((30.0 + LOG_RATIO_PSEUDO_COUNT) / (10.0 + LOG_RATIO_PSEUDO_COUNT)).ln();
    assert!(
        (factor[0] - expected).abs() < 1e-5,
        "section-wide shift: {}",
        factor[0]
    );
    assert!(factor[1] < 0.0, "the other section is the mirror image");
}

#[test]
fn an_anchor_whose_matches_come_from_one_other_donor_is_left_out() {
    // units 0..3: donor 0, 1, 2; unit 0 matched to two donors, unit 1 to one
    let donors = vec![0, 1, 2, 2];
    let matched = matched_units_from(vec![1, 2, 2, 3, 0, 1, 0, 1], 2);
    assert_eq!(
        matched.anchors_spanning_other_donors(&donors, 2),
        vec![0, 2, 3]
    );
    assert_eq!(
        matched.anchors_spanning_other_donors(&donors, 1),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn matched_units_are_never_of_the_units_own_donor() {
    let embedding = Mat::from_fn(12, 2, |unit, dim| ((unit * 7 + dim * 3) % 5) as f32);
    let donors: Vec<usize> = (0..12).map(|p| p % 3).collect();
    let matched = find_matched_other_donor_units(&embedding, &donors, 3, 3).unwrap();
    for unit in 0..12 {
        for &m in &matched.matched_unit_of_unit[unit * 3..unit * 3 + 3] {
            assert_ne!(
                donors[m as usize], donors[unit],
                "unit {unit} matched to {m}"
            );
        }
    }
}

#[test]
fn matched_distances_are_euclidean_and_nearest_first() {
    let embedding = Mat::from_fn(12, 2, |unit, dim| ((unit * 7 + dim * 3) % 5) as f32);
    let donors: Vec<usize> = (0..12).map(|p| p % 3).collect();
    let matched = find_matched_other_donor_units(&embedding, &donors, 3, 3).unwrap();
    for unit in 0..12 {
        let slots = unit * 3..unit * 3 + 3;
        let distances = &matched.distance_to_matched_unit[slots.clone()];
        assert!(distances.windows(2).all(|w| w[0] <= w[1]), "{distances:?}");
        for (&m, &distance) in matched.matched_unit_of_unit[slots].iter().zip(distances) {
            let euclidean = (embedding.row(unit) - embedding.row(m as usize)).norm();
            assert!(
                (euclidean - distance).abs() < 1e-4,
                "{euclidean} vs {distance}"
            );
        }
    }
}

#[test]
fn state_coordinates_start_with_an_intercept_and_have_unit_variance() {
    let embedding = Mat::from_fn(50, 4, |unit, dim| ((unit * (dim + 1)) % 11) as f32);
    let (coordinates, width) = state_coordinates_of_units(&embedding, 2);
    assert_eq!(width, 3);
    let axis_1: Vec<f32> = (0..50).map(|p| coordinates[p * 3 + 1]).collect();
    assert!((0..50).all(|p| coordinates[p * 3] == 1.0));
    let mean = axis_1.iter().sum::<f32>() / 50.0;
    let variance = axis_1.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 49.0;
    assert!(mean.abs() < 1e-4 && (variance - 1.0).abs() < 1e-3);
}

/// Fit one gene on the line: three donors, each unit matched to the same
/// state in the other two donors.
fn fit_on_the_line(activity: &[f32]) -> Vec<f32> {
    let (coordinates, donors, _) = units_on_a_line(3, 20);
    let matched = matched_at_same_state(3, 20);
    let anchors: Vec<u32> = (0..60).collect();
    let sets = contrast_sets_of_gene(activity, &anchors, &matched);
    let states = UnitStates {
        state_coordinates: &coordinates,
        coordinates_per_unit: 2,
        donor_index_of_unit: &donors,
        number_of_donors: 3,
    };
    fit_deviation_of_gene(
        &sets,
        &states,
        None,
        None,
        &ContrastFitSettings {
            group_lasso_penalty: 0.01,
            max_iterations: 5000,
            gradient_mapping_tolerance: 1e-7,
        },
    )
    .deviation
}

fn row_norm(deviation: &[f32], donor: usize) -> f32 {
    deviation[donor * 2..donor * 2 + 2]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
}

#[test]
fn a_gene_shared_alike_by_every_donor_has_no_deviation() {
    let (_, _, state) = units_on_a_line(3, 20);
    let activity: Vec<f32> = state.iter().map(|s| 2.0 + 10.0 * s).collect();
    let deviation = fit_on_the_line(&activity);
    assert!(deviation.iter().all(|&c| c == 0.0), "{deviation:?}");
}

#[test]
fn a_gene_gained_by_one_donor_in_one_state_deviates_in_that_donor_only() {
    let (_, donors, state) = units_on_a_line(3, 20);
    // donor 2 carries the gene 8 times more at the high end of the line
    let activity: Vec<f32> = state
        .iter()
        .zip(&donors)
        .map(|(&s, &d)| if d == 2 && s > 0.5 { 40.0 } else { 5.0 })
        .collect();
    let deviation = fit_on_the_line(&activity);
    assert!(row_norm(&deviation, 2) > 1.0, "{deviation:?}");
    assert_eq!(row_norm(&deviation, 0), 0.0, "{deviation:?}");
    assert_eq!(row_norm(&deviation, 1), 0.0, "{deviation:?}");
    // the gain is in the direction of the high states
    assert!(deviation[2 * 2 + 1] > 0.0);
}

/// The two planted genes on the line, fitted with their section jackknife
/// by the rayon path and by the batched candle path on the CPU device.
#[test]
fn batched_candle_fit_agrees_with_the_cpu_fit_and_its_jackknife() {
    use super::batched_contrast_fit::{fit_block_of_genes, SharedContrastTensors};
    use legume_numeric::candle::candle_core::Device;
    let (coordinates, donors, state) = units_on_a_line(3, 20);
    let matched = matched_at_same_state(3, 20);
    let sections = two_sections_per_donor(3, 20);
    let states = UnitStates {
        state_coordinates: &coordinates,
        coordinates_per_unit: 2,
        donor_index_of_unit: &donors,
        number_of_donors: 3,
    };
    let settings = ContrastFitSettings {
        group_lasso_penalty: 0.01,
        max_iterations: 20000,
        gradient_mapping_tolerance: 1e-7,
    };
    let shared_alike: Vec<f32> = state.iter().map(|s| 2.0 + 10.0 * s).collect();
    let gained_by_donor_2: Vec<f32> = state
        .iter()
        .zip(&donors)
        .map(|(&s, &d)| if d == 2 && s > 0.5 { 40.0 } else { 5.0 })
        .collect();
    let anchors: Vec<u32> = (0..60).collect();
    let shared =
        SharedContrastTensors::new(&anchors, &matched, &states, &sections, 6, &Device::Cpu)
            .unwrap();
    let genes = [shared_alike, gained_by_donor_2];
    let member_activity: Vec<Vec<f32>> = genes
        .iter()
        .map(|activity| {
            shared
                .member_unit
                .iter()
                .map(|&p| activity[p as usize])
                .collect()
        })
        .collect();
    let batched = fit_block_of_genes(&shared, &member_activity, &settings).unwrap();
    assert!(
        batched[0].deviation.iter().all(|&c| c == 0.0),
        "shared alike: {:?}",
        batched[0].deviation
    );
    assert!(row_norm(&batched[1].deviation, 2) > 1.0);
    assert_eq!(row_norm(&batched[1].deviation, 0), 0.0);
    for (activity, fit) in genes.iter().zip(&batched) {
        let sets = contrast_sets_of_gene(activity, &anchors, &matched);
        let deviation = fit_deviation_of_gene(&sets, &states, None, None, &settings).deviation;
        let refits = super::section_jackknife::deviation_without_each_section(
            &deviation, &sets, &states, &sections, 6, &settings,
        );
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3);
        assert!(
            close(&fit.deviation, &deviation),
            "{:?} vs {deviation:?}",
            fit.deviation
        );
        assert_eq!(fit.deviation_without_section.len(), 6);
        for (b, c) in fit.deviation_without_section.iter().zip(&refits) {
            assert!(close(b, &c.deviation), "refit {b:?} vs {:?}", c.deviation);
        }
    }
}

/// Six cells: pseudobulks 0, 0, 0, 1, 1, 1; batches A, A, B, A, A, A; three
/// donors so the struct is well formed; embedding along two axes.
fn six_cells() -> super::inputs::Stage1Cells {
    super::inputs::Stage1Cells {
        cell_names: (0..6).map(|c| format!("c{c}").into()).collect(),
        embedding: Mat::from_row_slice(
            6,
            2,
            &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ),
        unit_rows: true,
        finest_pb_of_cell: vec![0, 0, 0, 1, 1, 1],
        batch_of_cell: ["A", "A", "B", "A", "A", "A"].map(Box::from).to_vec(),
        donor_names: ["D0", "D1", "D2"].map(Box::from).to_vec(),
        donor_index_of_cell: vec![0, 0, 1, 0, 0, 0],
    }
}

#[test]
fn a_sample_is_one_pseudobulk_in_one_batch_at_its_cells_mean() {
    let cells = six_cells();
    // width 2: intercept, then one coordinate per cell (its index)
    let coordinates: Vec<f32> = (0..6).flat_map(|c| [1.0, c as f32]).collect();
    let samples = super::samples::pb_samples(&cells, &coordinates, 2);
    // pb 0 splits by batch; pb 1 lies in one batch
    assert_eq!(
        samples.sample_names,
        ["pb0@A", "pb0@B", "pb1@A"].map(Box::from)
    );
    assert_eq!(samples.sample_of_cell, vec![0, 0, 1, 2, 2, 2]);
    assert_eq!(samples.cells_in_sample, vec![2, 1, 3]);
    assert_eq!(samples.donor_index_of_sample, vec![0, 1, 0]);
    assert_eq!(
        samples.state_coordinates_of_sample,
        vec![1.0, 0.5, 1.0, 2.0, 1.0, 4.0]
    );
    for s in 0..3 {
        let norm = samples.embedding_of_sample.row(s).norm();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "sample {s} embedding norm {norm}"
        );
    }
}

#[test]
fn sample_counts_are_averaged_per_cell_and_skip_cells_not_embedded() {
    use senna::sparse_output::{write_backend, Entries};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.zarr.zip").to_string_lossy().into_owned();
    // gene 0: 2 in c0, 4 in c1 (sample pb0@A: mean 3); 6 in c2 (pb0@B: 6);
    // gene 1: 3 in c3 only (pb1@A of 3 cells: 1); a 7th column not embedded
    let triplets = vec![
        (0, 0, 2.0),
        (0, 1, 4.0),
        (0, 2, 6.0),
        (1, 3, 3.0),
        (0, 6, 100.0),
    ];
    let rows: Vec<Box<str>> = ["g0", "g1"].map(Box::from).to_vec();
    let cols: Vec<Box<str>> = (0..7).map(|c| format!("c{c}").into()).collect();
    write_backend(&path, Entries::Triplets(triplets), &rows, &cols).unwrap();
    let data = data_beans::aux::data_loading::read_data_on_shared_rows(
        data_beans::aux::data_loading::ReadSharedRowsArgs {
            data_files: vec![path.into()],
            ..Default::default()
        },
    )
    .unwrap()
    .data;
    let cells = six_cells();
    let coordinates: Vec<f32> = (0..6).flat_map(|_| [1.0, 0.0]).collect();
    let samples = super::samples::pb_samples(&cells, &coordinates, 2);
    let cell_of_column: Vec<Option<usize>> = (0..7).map(|c| (c < 6).then_some(c)).collect();
    let counts =
        super::samples::pool_sample_counts_by_gene(&data, &cell_of_column, &samples).unwrap();
    assert_eq!(counts.nonzero_samples_of_gene[0], vec![(0, 3.0), (1, 6.0)]);
    assert_eq!(counts.nonzero_samples_of_gene[1], vec![(2, 1.0)]);
}

#[test]
fn donor_of_batch_skips_a_header_and_refuses_a_batch_given_two_donors() {
    let dir = tempfile::tempdir().unwrap();
    let ok = dir.path().join("ok.tsv");
    std::fs::write(&ok, "batch\tdonor\nB0\tD0\nB1,D0\n\nB2\tD1\n").unwrap();
    let map = super::inputs::read_donor_of_batch(ok.to_str().unwrap()).unwrap();
    assert_eq!(map.len(), 3);
    assert_eq!(map.get("B1").map(AsRef::as_ref), Some("D0"));
    assert_eq!(map.get("B2").map(AsRef::as_ref), Some("D1"));
    let clash = dir.path().join("clash.tsv");
    std::fs::write(&clash, "B0\tD0\nB0\tD1\n").unwrap();
    assert!(super::inputs::read_donor_of_batch(clash.to_str().unwrap()).is_err());
}

#[test]
fn sections_index_in_order_of_first_appearance() {
    let sections = super::index_sections(&["B", "A", "B", "C"].map(Box::from));
    assert_eq!(sections.section_index_of_unit, vec![0, 1, 0, 2]);
    assert_eq!(sections.number_of_units_in_section, vec![2, 1, 1]);
    assert_eq!(sections.section_names, ["B", "A", "C"].map(Box::from));
}

#[test]
fn anchors_are_sampled_from_the_eligible_units_only() {
    let eligible: Vec<u32> = (0..100).filter(|u| u % 3 == 0).collect();
    let anchors = super::sample_anchor_units(&eligible, 10, 7);
    assert_eq!(anchors.len(), 10);
    assert!(anchors.windows(2).all(|w| w[0] < w[1]));
    assert!(anchors.iter().all(|a| eligible.contains(a)));
    assert_eq!(super::sample_anchor_units(&eligible, 1000, 7), eligible);
}

#[test]
fn each_cell_space_gets_its_euclidean_view() {
    use senna::run_manifest::CellSpace;
    let log_theta = Mat::from_row_slice(1, 2, &[0.25f32.ln(), 0.75f32.ln()]);
    let (root, unit) = super::inputs::euclidean_view(log_theta, CellSpace::LogSimplex);
    assert!(unit);
    assert!((root[(0, 0)] - 0.5).abs() < 1e-6 && (root[(0, 1)] - 0.75f32.sqrt()).abs() < 1e-6);
    let signed = Mat::from_row_slice(3, 2, &[1.0, 10.0, 2.0, 20.0, 3.0, 60.0]);
    let (z, unit) = super::inputs::euclidean_view(signed, CellSpace::Signed);
    assert!(!unit);
    for column in z.column_iter() {
        assert!(column.sum().abs() < 1e-5);
        assert!((column.norm_squared() / 3.0 - 1.0).abs() < 1e-5);
    }
    let embedding = Mat::from_row_slice(1, 2, &[3.0, 4.0]);
    let (e, unit) = super::inputs::euclidean_view(embedding, CellSpace::Embedding);
    assert!(unit && (e.row(0).norm() - 1.0).abs() < 1e-6);
}

#[test]
fn a_block_s_entries_merge_into_the_sums_already_held() {
    // sums held after one block, then the next block's entries appended
    let mut entries = vec![(0, 2.0), (3, 1.0)];
    entries.extend([(3, 4.0), (1, 1.0), (0, 1.0)]);
    super::samples::merge_by_sample(&mut entries);
    assert_eq!(entries, vec![(0, 3.0), (1, 1.0), (3, 5.0)]);
}

#[test]
fn stage_2_is_recorded_in_the_stage_1_manifest_relative_to_it() {
    use senna::run_manifest::{RunKind, RunManifest};
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("s1.senna.json");
    RunManifest::new(RunKind::Bge, "s1")
        .save(&manifest)
        .unwrap();
    let out = dir.path().join("s2").to_string_lossy().into_owned();
    super::outputs::record_in_stage1_manifest(&manifest.to_string_lossy(), &out).unwrap();
    let (m, _) = RunManifest::load(&manifest).unwrap();
    assert_eq!(
        m.stage2.donor_gene_deviation.as_deref(),
        Some("s2.donor_gene_deviation.parquet")
    );
    assert_eq!(
        m.stage2.settings.as_deref(),
        Some("s2.donor_deviation.json")
    );
}
