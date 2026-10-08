use super::*;

/// Two programmes over 40 pseudobulks (genes 0..10 high in the first half,
/// 10..20 in the second) and 10 genes at one flat rate.
#[test]
fn groups_programmes_and_leaves_flat_genes_out() {
    let p = 40;
    let totals = DMatrix::<f32>::from_fn(30, p, |g, s| {
        let first = s < p / 2;
        match g {
            0..10 => {
                if first {
                    60.0
                } else {
                    2.0
                }
            }
            10..20 => {
                if first {
                    2.0
                } else {
                    60.0
                }
            }
            _ => 30.0,
        }
    });
    let sizes = vec![1.0; p];
    let (modules, n) = gene_modules(&totals, &sizes, 2, 1, 1).unwrap();
    assert_eq!(n, 2);
    assert!(modules[20..].iter().all(Option::is_none), "{modules:?}");
    let a = modules[0].unwrap();
    let b = modules[10].unwrap();
    assert_ne!(a, b);
    assert!(modules[..10].iter().all(|&m| m == Some(a)), "{modules:?}");
    assert!(modules[10..20].iter().all(|&m| m == Some(b)), "{modules:?}");
}

/// A programme of fewer genes than the minimum is left out, the others kept.
#[test]
fn small_modules_are_left_out() {
    let p = 40;
    // 12 genes high early, 3 genes high late.
    let totals = DMatrix::<f32>::from_fn(15, p, |g, s| {
        let early = s < p / 2;
        match (g < 12, early) {
            (true, true) | (false, false) => 60.0,
            _ => 2.0,
        }
    });
    let sizes = vec![1.0; p];
    let (modules, n) = gene_modules(&totals, &sizes, 2, 5, 1).unwrap();
    assert_eq!(n, 1, "{modules:?}");
    assert!(modules[..12].iter().all(Option::is_some), "{modules:?}");
    assert!(modules[12..].iter().all(Option::is_none), "{modules:?}");
}
