use super::*;

fn table() -> CloneTable {
    let b = |v: &[&str]| v.iter().map(|s| Box::from(*s)).collect::<Vec<Box<str>>>();
    CloneTable {
        donor: b(&["d1", "d1", "d1", "d2", "d2"]),
        stratum: vec![0, 2, 2, 0, 1],
        purity: Some(vec![0.0, 0.9, 0.7, 0.0, 1.0]),
        p_malig: None,
    }
}

#[test]
fn clones_and_donors_are_counted() {
    let s = of(&table());
    assert_eq!(s.cells, 5);
    assert_eq!(s.in_clones(), 3);
    assert_eq!(
        s.donors,
        [
            Donor {
                name: "d1".into(),
                cells: 3,
                in_clones: 2
            },
            Donor {
                name: "d2".into(),
                cells: 2,
                in_clones: 1
            },
        ]
    );
    let strata: Vec<usize> = s.clones.iter().map(|c| c.stratum).collect();
    assert_eq!(strata, [1, 2]);
    let two = &s.clones[1];
    assert_eq!((two.cells, two.donor.as_str(), two.share), (2, "d1", 1.0));
    assert!((two.purity.unwrap() - 0.8).abs() < 1e-6);
    assert_eq!(two.p_malig, None);
    let text = s.lines().join("\n");
    assert!(text.starts_with("5 cells: 2 mix freely, 3 (60%) in 2 donor-private clones"));
}

#[test]
fn no_clone_says_so() {
    let mut t = table();
    t.stratum = vec![0; 5];
    let s = of(&t);
    assert!(s.clones.is_empty());
    assert!(s.advice().contains("change nothing"));
}
