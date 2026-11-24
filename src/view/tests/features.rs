use super::*;

#[test]
fn features_near_a_feature_rank_by_cosine_and_leave_it_out() {
    let names = ["a", "b", "c", "d"];
    // Lengths differ on purpose: only direction counts.
    let mut rows = Mat::from_row_slice(4, 2, &[1.0, 0.0, 5.0, 0.5, 0.0, 1.0, -2.0, 0.0]);
    l2_normalize_rows_inplace(&mut rows);
    let e = FeatureEmbedding {
        axis: activity::Axis::new(names.map(Into::into).to_vec()),
        rows,
        placed: false,
    };
    let near = e.near("a", 2).unwrap();
    let got: Vec<&str> = near.iter().map(|(n, _)| n.as_ref()).collect();
    assert_eq!(got, ["b", "c"]);
    assert!((near[1].1).abs() < 1e-6);
    assert!(e.near("z", 2).is_err());
}

#[test]
fn features_near_a_place_rank_by_direction_as_the_map_is_laid_out() {
    let names = ["near", "long", "off"];
    // "long" points the query's way but sits far out, "near" is close
    // but a little off its way: the map, laid out by direction, puts
    // "long" first.
    let mut rows = Mat::from_row_slice(3, 2, &[1.0, 0.2, 10.0, 0.0, 0.0, 1.5]);
    l2_normalize_rows_inplace(&mut rows);
    let e = FeatureEmbedding {
        axis: activity::Axis::new(names.map(Into::into).to_vec()),
        rows,
        placed: true,
    };
    let z = activity::Vector::from_vec(vec![1.2, 0.0]);
    let got = e.near_place(&z, None, 3).unwrap();
    let order: Vec<&str> = got.iter().map(|(n, _)| &**n).collect();
    assert_eq!(order, ["long", "near", "off"]);
    assert!(got[0].1.abs() < 1e-6);
    // A feature's own place: itself left out.
    let (at, i) = e.place_of("near").unwrap();
    let got = e.near_place(&at, Some(i), 1).unwrap();
    assert_eq!(&*got[0].0, "long");
    // No co-embedding of this width: no ranking by place.
    let wide = activity::Vector::from_vec(vec![1.0, 0.0, 0.0]);
    assert!(e.near_place(&wide, None, 1).is_none());
}
