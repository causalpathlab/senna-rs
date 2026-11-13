use super::*;
use senna::run_manifest;
use std::path::PathBuf;

/// A run at `from` with cells and a feature embedding.
fn target(from: &str) -> Target {
    Target {
        from: from.into(),
        out: run_manifest::derive_out_prefix(from),
        latent: None,
        has_cells: true,
        has_features: true,
        has_coembedding: false,
        can_resolve_topics: false,
    }
}

fn layout(v: Value) -> Result<Task> {
    let a: LayoutArgs = parse(v).unwrap();
    layout_task(&a, &target(&a.run))
}

#[test]
fn layout_lays_out_the_run_in_its_own_prefix_with_the_flags_given() {
    let t = layout(json!({"run": "d/r.senna.json", "method": "umap", "args": ["--knn", "30"]}))
        .unwrap();
    assert_eq!(
        t.argv,
        [
            "layout",
            "umap",
            "--from",
            "d/r.senna.json",
            "--out",
            "d/r",
            "--knn",
            "30"
        ]
    );
    assert_eq!(
        t.made,
        [("coords", PathBuf::from("d/r.umap.cell_coords.parquet"))]
    );
    let f =
        layout(json!({"run": "r.senna.json", "method": "phate", "target": "features"})).unwrap();
    assert_eq!(
        f.argv,
        [
            "layout",
            "phate",
            "--target",
            "features",
            "--from",
            "r.senna.json",
            "--out",
            "r"
        ]
    );
    assert_eq!(
        f.made,
        [("coords", PathBuf::from("r.phate.feature_coords.parquet"))]
    );
}

#[test]
fn a_umap_layout_takes_min_dist_and_spread_as_its_flags() {
    let t = layout(json!({"run": "r.senna.json", "method": "umap", "min_dist": 0.5, "spread": 1.5, "args": ["--knn", "30"]}))
        .unwrap();
    assert_eq!(
        t.argv,
        [
            "layout",
            "umap",
            "--from",
            "r.senna.json",
            "--out",
            "r",
            "--umap-min-dist",
            "0.5",
            "--umap-spread",
            "1.5",
            "--knn",
            "30"
        ]
    );
    // Either alone.
    let t = layout(json!({"run": "r.senna.json", "method": "umap", "min_dist": 0.0})).unwrap();
    assert_eq!(t.argv[6..], ["--umap-min-dist", "0"]);
}

#[test]
fn a_layout_senna_would_refuse_is_refused_before_it_runs() {
    let umap = |extra: Value| {
        let mut v = json!({"run": "r.senna.json", "method": "umap"});
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        v
    };
    for (bad, said) in [
        // Not one of phate's flags.
        (
            json!({"run": "r.senna.json", "method": "phate", "min_dist": 0.5}),
            "--umap-min-dist",
        ),
        // The kernel's range, with spread given in args.
        (
            umap(json!({"min_dist": 0.5, "args": ["--umap-spread", "0.3"]})),
            "min_dist",
        ),
        (
            umap(json!({"min_dist": 0.5, "args": ["--umap-min-dist", "0.2"]})),
            "--umap-min-dist",
        ),
        (umap(json!({"args": ["--no-such-flag"]})), "--no-such-flag"),
        // A method senna lacks for its points.
        (json!({"run": "r.senna.json", "method": "pca"}), ""),
        (
            json!({"run": "r.senna.json", "method": "tsne", "target": "features"}),
            "umap, phate",
        ),
    ] {
        let e = format!("{:#}", layout(bad.clone()).unwrap_err());
        assert!(e.contains(said), "{bad}: {e}");
    }
    // A field and the other in args are checked together.
    assert!(layout(umap(
        json!({"spread": 2.0, "args": ["--umap-min-dist=1.5"]})
    ))
    .is_ok());
}
