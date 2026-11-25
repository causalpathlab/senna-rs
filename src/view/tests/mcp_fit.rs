use super::*;

/// A cell fit leads to its clustering, then to lupin's annotation; a fit
/// with no cells to cluster, nowhere; an ODE whose root it chose and lupin
/// typed, to a root of the user's choosing, and one rooted by the user
/// with nothing open, nowhere.
#[test]
fn a_fit_s_next_call_is_the_next_step_toward_its_cell_types() {
    let run = Path::new("/nowhere/r.senna.json");
    let tool = |m: &RunManifest, said: Value| -> Vec<String> {
        next_of(m, run, &said).into_iter().map(|n| n.tool).collect()
    };
    let mut m = RunManifest::new(RunKind::Topic, "/nowhere/r");
    assert!(tool(&m, json!({})).is_empty());
    m.outputs.latent = Some("r.latent.parquet".into());
    let first = &next_of(&m, run, &json!({}))[0];
    assert_eq!(first.tool, "run_step");
    assert_eq!(first.args["out"], "/nowhere/r_clusters");
    m.cluster.clusters = Some("k.clusters.parquet".into());
    assert_eq!(tool(&m, json!({})), ["lupin annotate_run"]);

    let mut ode = RunManifest::new(RunKind::Ode, "/nowhere/r");
    ode.cluster.clusters = Some("k.clusters.parquet".into());
    assert!(tool(&ode, json!({})).is_empty());
    let untyped = json!({"root": {"n": 4, "label_cells": "…"}});
    assert_eq!(tool(&ode, untyped), ["lupin annotate_run"]);
    let chose = json!({"root": {"n": 4, "mostly": "HSC (75%)", "from": "fit"}});
    assert_eq!(tool(&ode, chose), ["view_cells"]);
    let named = json!({"root": {"n": 4, "mostly": "HSC (100%)", "from": "root-cells"}});
    assert!(tool(&ode, named.clone()).is_empty());
    let mut open = named;
    open["needs_input"] = json!({});
    assert_eq!(tool(&ode, open), ["view_cells"]);
}
