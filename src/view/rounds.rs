//! Annotation rounds: each `lupin annotate` or relabel pass writes a new
//! manifest whose `annotate.source` names the one it came from. This module
//! reads what the viewer shows about a round: how its labels differ from the
//! source round's, and the per-cluster summary and decision history lupin
//! records beside it. The viewer never applies decisions itself.
//!
//! Lupin's fields are read through the manifest's pass-through keys, so
//! senna's schema does not carry them.

use super::data::{read_pairs, LabelKind, Labels};
use super::files::{self, read_json, same_file, siblings};
use rustc_hash::FxHashMap as HashMap;
use senna::run_manifest::{self, RunManifest};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A cell type label in one canonical form for comparing: lower case, with
/// spaces, commas and underscores all read as one separator (lupin writes
/// `CT_1_a` for `CT 1, a`).
#[must_use]
pub(crate) fn label_key(s: &str) -> String {
    s.split([' ', ',', '_'])
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("_")
        .to_lowercase()
}

/// Label a round gives a cell it did not assign.
pub(super) const UNASSIGNED: &str = "unassigned";

/// `cell<TAB>label<TAB>…` with a header, as `annotate.argmax` holds it.
pub fn read_argmax(path: &Path) -> anyhow::Result<HashMap<Box<str>, Box<str>>> {
    Ok(read_pairs(path)?.into_iter().collect())
}

pub(super) fn annotate_str<'a>(m: &'a RunManifest, key: &str) -> Option<&'a str> {
    m.annotate.unknown.get(key).and_then(Value::as_str)
}

fn read_json_object(path: &Path) -> HashMap<String, Value> {
    match read_json::<Value>(path) {
        Some(Value::Object(m)) => m.into_iter().collect(),
        _ => HashMap::default(),
    }
}

/// A cluster × cell type table (`K{id}` rows, one column per type), as
/// lupin writes `cluster_celltype_nes` and `cluster_celltype_p`: per
/// cluster id, the value of each type by its `label_key`. Empty when absent
/// or unreadable, as a round from an older lupin has neither.
fn read_cluster_table(path: &Path) -> HashMap<String, HashMap<String, f64>> {
    use senna::embed_common::*;
    let Ok(MatWithNames { rows, cols, mat }) =
        Mat::from_parquet_with_row_names(&path.to_string_lossy(), Some(0))
    else {
        return HashMap::default();
    };
    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let id = r.strip_prefix('K').unwrap_or(r).to_string();
            let row = cols
                .iter()
                .enumerate()
                .map(|(j, c)| (label_key(c), f64::from(mat[(i, j)])))
                .collect();
            (id, row)
        })
        .collect()
}

/// lupin's FDR level when a round does not record one.
pub(super) const FDR_ALPHA: f64 = 0.1;

/// Whether a call's q passes `alpha`. A call without q (an older round's
/// label-only call) is kept.
fn passes(c: &Value, alpha: f64) -> bool {
    c.get("q").and_then(Value::as_f64).is_none_or(|q| q < alpha)
}

/// `{"groups": [{"name", "members": [...]}, ...]}`, as lupin's
/// `celltype_tree` holds it. Groups of one member are not coarse.
fn read_tree(v: &Value) -> Vec<(String, Vec<String>)> {
    v.get("groups")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|g| {
            let name = g.get("name")?.as_str()?.to_string();
            let members: Vec<String> = g
                .get("members")?
                .as_array()?
                .iter()
                .filter_map(|m| Some(m.as_str()?.to_string()))
                .collect();
            (members.len() > 1).then_some((name, members))
        })
        .collect()
}

/// lupin's rescored top call for a cluster, beside the label it carries.
/// `top` is `None` when no cell type passes FDR.
pub struct Evidence {
    pub top: Option<String>,
    pub q: Option<f64>,
    pub agrees: bool,
}

/// Where a round sits, and what lupin recorded about its clusters.
pub struct Round {
    /// This manifest.
    pub path: PathBuf,
    /// The round this one was made from, when it records one.
    pub source: Option<PathBuf>,
    summary: HashMap<String, Value>,
    history: HashMap<String, Value>,
    /// Groups of cell types and their members.
    tree: Vec<(String, Vec<String>)>,
    /// Curation rounds behind this round's statistics, when lupin rescored
    /// it after decisions made on the same data (post-selection).
    curated: Option<u64>,
    /// lupin's FDR level: calls at or above it are not calls.
    alpha: f64,
    /// Per cluster id and cell type: enrichment effect size (NES) and p.
    nes: HashMap<String, HashMap<String, f64>>,
    p: HashMap<String, HashMap<String, f64>>,
    /// Its q values come from the gene-set null alone (too few batches for
    /// the sample-permutation null).
    gene_set_null_only: bool,
}

impl Round {
    #[must_use]
    pub fn load(m: &RunManifest, dir: &Path, path: &Path) -> Self {
        let at = |key: &str| annotate_str(m, key).map(|rel| run_manifest::resolve(dir, rel));
        let stats = m.annotate.unknown.get("stats");
        Self {
            path: path.to_path_buf(),
            source: at("source"),
            summary: at("cluster_summary")
                .map(|p| read_json_object(&p))
                .unwrap_or_default(),
            history: at("history")
                .map(|p| read_json_object(&p))
                .unwrap_or_default(),
            tree: at("celltype_tree")
                .and_then(|p| read_json::<Value>(&p))
                .map(|v| read_tree(&v))
                .unwrap_or_default(),
            curated: stats
                .filter(|s| s.get("kind").and_then(Value::as_str) == Some("post_selection"))
                .map(|s| {
                    s.get("rounds_of_curation")
                        .and_then(Value::as_u64)
                        .unwrap_or(1)
                }),
            alpha: m
                .annotate
                .unknown
                .get("settings")
                .and_then(|s| s.pointer("/enrichment/fdr_alpha"))
                .and_then(Value::as_f64)
                .unwrap_or(FDR_ALPHA),
            nes: at("cluster_celltype_nes")
                .map(|p| read_cluster_table(&p))
                .unwrap_or_default(),
            p: at("cluster_celltype_p")
                .map(|p| read_cluster_table(&p))
                .unwrap_or_default(),
            gene_set_null_only: m
                .annotate
                .unknown
                .get("settings")
                .and_then(|s| s.pointer("/enrichment/null/sample_permutation"))
                .and_then(Value::as_u64)
                == Some(0),
        }
    }

    /// The member types of `label` when it names a coarse group: a group
    /// name that is not itself a member type (later rounds keep the tree,
    /// and a refined cluster's label may share a group's name).
    #[must_use]
    pub fn members_of(&self, label: &str) -> Option<&[String]> {
        let key = label_key(label);
        let is_member = self
            .tree
            .iter()
            .any(|(_, ms)| ms.iter().any(|m| label_key(m) == key));
        if is_member {
            return None;
        }
        self.tree
            .iter()
            .find(|(g, _)| label_key(g) == key)
            .map(|(_, m)| m.as_slice())
    }

    /// lupin's evidence for cluster `id` beside its label: the top call and
    /// its q, and whether the label agrees with it. A top call that does not
    /// pass FDR is no call, and nothing to disagree with.
    #[must_use]
    pub fn evidence(&self, id: &str) -> Option<Evidence> {
        let s = self.summary.get(id)?;
        let e = s.get("evidence")?;
        if !passes(e, self.alpha) {
            return Some(Evidence {
                top: None,
                q: e.get("q").and_then(Value::as_f64),
                agrees: true,
            });
        }
        let top = e.get("top").and_then(Value::as_str).map(String::from);
        let label = s
            .get("label")
            .and_then(Value::as_str)
            .filter(|l| *l != UNASSIGNED);
        // No label yet is not a disagreement; a coarse label and a fine top
        // call inside its group agree.
        let within_group = label.is_none()
            || match (label, &top) {
                (Some(l), Some(t)) => self
                    .members_of(l)
                    .is_some_and(|ms| ms.iter().any(|m| label_key(m) == label_key(t))),
                _ => false,
            };
        Some(Evidence {
            top,
            q: e.get("q").and_then(Value::as_f64),
            agrees: within_group || e.get("agrees").and_then(Value::as_bool).unwrap_or(true),
        })
    }

    /// How to read this round's q, for a panel heading.
    fn stats_note(&self) -> Option<String> {
        let mut notes = Vec::new();
        if let Some(n) = self.curated {
            notes.push(format!(
                "post-selection, after {n} round{} of curation",
                if n == 1 { "" } else { "s" }
            ));
        }
        if self.gene_set_null_only {
            notes.push("q from the gene-set null only".into());
        }
        (!notes.is_empty()).then(|| notes.join("; "))
    }

    /// The round made from this one, found next to it: a manifest in the same
    /// directory whose `annotate.source` resolves to this file.
    #[must_use]
    pub fn newer(&self) -> Option<PathBuf> {
        // lupin names a round's successor `{stem}.r{N}.senna.json`; prefer
        // that chain over any other round made from this one.
        let stem = run_manifest::derive_out_prefix(&files::name(&self.path));
        let mut found = siblings(&self.path, ".senna.json");
        found.sort_by_key(|p| !files::name(p).starts_with(&format!("{stem}.r")));
        found.into_iter().find(|p| {
            read_json::<Value>(p)
                .and_then(|v| v.pointer("/annotate/source")?.as_str().map(String::from))
                .is_some_and(|src| {
                    same_file(
                        &run_manifest::resolve(run_manifest::manifest_dir(p), &src),
                        &self.path,
                    )
                })
        })
    }

    /// lupin's FDR level for this round.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The label the round gives cluster `id`, from the summary.
    #[must_use]
    pub fn label(&self, id: &str) -> Option<String> {
        let s = self.summary.get(id)?;
        s.get("label").and_then(Value::as_str).map(String::from)
    }

    /// Cluster `id`'s calls that pass FDR, best first.
    fn calls(&self, id: &str) -> Vec<&Value> {
        self.summary
            .get(id)
            .and_then(|s| s.get("calls"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|c| passes(c, self.alpha))
            .collect()
    }

    /// Cell types worth considering for cluster `id`: its current label,
    /// then lupin's top calls, without repeats.
    #[must_use]
    pub fn candidates(&self, id: &str) -> Vec<String> {
        let Some(s) = self.summary.get(id) else {
            return Vec::new();
        };
        let mut out: Vec<String> = Vec::new();
        let label = s.get("label").and_then(Value::as_str);
        let calls = self.calls(id);
        for l in label.into_iter().chain(
            calls
                .iter()
                .filter_map(|c| c.get("label")?.as_str())
                .take(3),
        ) {
            if l != UNASSIGNED && !out.iter().any(|x| x == l) {
                out.push(l.to_string());
            }
        }
        out
    }

    /// Text for the panel shown when a cell of cluster `id` is clicked: the
    /// summary lupin wrote for it, then its decision history, newest first.
    #[must_use]
    pub fn cluster_lines(&self, id: &str) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(s) = self.summary.get(id) {
            let size = s.get("size").and_then(Value::as_u64).unwrap_or(0);
            let label = s.get("label").and_then(Value::as_str).unwrap_or("-");
            out.push(format!("C{id} · {size} cells · {label}"));
            if let Some(members) = self.members_of(label) {
                out.push(format!(
                    "a group of {} types (R, then L refines it):",
                    members.len()
                ));
                out.push(format!("  {}", members.join(", ")));
            }
            let num = |v: &Value, k: &str| {
                v.get(k)
                    .and_then(Value::as_f64)
                    .map_or("-".to_string(), |x| format!("{x:.3}"))
            };
            let list = |k: &str| {
                s.get(k)
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
            };
            // NES and p of one type in this cluster, from lupin's tables.
            let table = |t: &HashMap<String, HashMap<String, f64>>, l: &str| {
                t.get(id)
                    .and_then(|row| row.get(&label_key(l)))
                    .map(|x| format!("{x:.3}"))
            };
            let stats = |l: &str, q: Option<f64>| {
                let mut parts = vec![format!("q {}", q.map_or("-".into(), |x| format!("{x:.3}")))];
                parts.extend(table(&self.p, l).map(|x| format!("p {x}")));
                parts.extend(table(&self.nes, l).map(|x| format!("NES {x}")));
                parts.join("  ")
            };
            if let Some(e) = self.evidence(id).filter(|e| !e.agrees) {
                let top = e.top.as_deref().unwrap_or(UNASSIGNED);
                out.push(format!(
                    "≠ the evidence calls it {top} ({})",
                    stats(top, e.q)
                ));
            }
            let calls = self.calls(id);
            let heading = match self.stats_note() {
                Some(n) => format!("calls ({n})"),
                None => "calls".into(),
            };
            if !calls.is_empty() {
                out.push(heading);
                for c in calls.iter().take(3) {
                    let l = c.get("label").and_then(Value::as_str).unwrap_or("-");
                    let q = c.get("q").and_then(Value::as_f64);
                    out.push(format!("  {l}  {}", stats(l, q)));
                }
            } else if !list("calls").is_empty() {
                out.push(heading);
                out.push(format!("  no cell type passes FDR (q < {})", self.alpha));
            }
            let terms = list("terms");
            if !terms.is_empty() {
                out.push("terms".into());
                for t in terms.iter().take(3) {
                    let src = t.get("source").and_then(Value::as_str).unwrap_or("-");
                    let term = t.get("term").and_then(Value::as_str).unwrap_or("-");
                    out.push(format!(
                        "  [{src}] {term}  effect {}  q {}",
                        num(t, "effect"),
                        num(t, "q")
                    ));
                }
            }
            if let Some(cl) = s.get("cl") {
                let name = cl.get("name").and_then(Value::as_str).unwrap_or("-");
                let abstained = cl
                    .get("abstained")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                out.push(format!(
                    "ontology  {name}{}",
                    if abstained { " (abstained)" } else { "" }
                ));
            }
        } else {
            out.push(format!("C{id} · no summary recorded for this round"));
        }
        let history = self
            .history
            .get(id)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !history.is_empty() {
            out.push("history".into());
            for h in history.iter().take(5) {
                let field = |k: &str| h.get(k).and_then(Value::as_str).unwrap_or("-").to_string();
                let when = field("timestamp")
                    .chars()
                    .take(16)
                    .collect::<String>()
                    .replace('T', " ");
                let link = ["merged_from", "merged_into", "split_from"]
                    .iter()
                    .find_map(|k| h.get(*k).map(|v| format!("  {} {v}", k.replace('_', " "))))
                    .unwrap_or_default();
                out.push(format!(
                    "  {when}  {} {}{link}",
                    field("action"),
                    field("label")
                ));
                out.push(format!(
                    "    {} ({})",
                    field("rationale"),
                    field("decided_by")
                ));
            }
        }
        out
    }
}

/// Groupings that compare a round with its source: the source's labels, and
/// the cells whose label changed (under their new label; a cell no longer
/// assigned shows as `unassigned`).
pub fn comparisons(
    current: &HashMap<Box<str>, Box<str>>,
    previous: &HashMap<Box<str>, Box<str>>,
) -> [Labels; 2] {
    let before = Labels::new(
        LabelKind::Previous,
        previous.iter().map(|(c, l)| (c.clone(), l.clone())),
        &[UNASSIGNED],
    );
    let unassigned: Box<str> = UNASSIGNED.into();
    let mut cells: Vec<&Box<str>> = current.keys().chain(previous.keys()).collect();
    cells.sort();
    cells.dedup();
    let changed_pairs = cells.into_iter().filter_map(|c| {
        let now = current.get(c).unwrap_or(&unassigned);
        let was = previous.get(c).unwrap_or(&unassigned);
        (now != was).then(|| (c.clone(), now.clone()))
    });
    let changed = Labels::new(LabelKind::Changed, changed_pairs, &[]);
    [before, changed]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn changed_cells_carry_their_new_label_and_unchanged_ones_are_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let before = write(
            dir.path(),
            "a.argmax.tsv",
            "cell\tcell_type\tprobability\nc1\tCT1\t0.9\nc2\tCT1\t0.9\nc3\tCT2\t0.9\n",
        );
        let now: HashMap<Box<str>, Box<str>> = [("c1", "CT1"), ("c2", "CT3"), ("c3", "unassigned")]
            .map(|(a, b)| (a.into(), b.into()))
            .into_iter()
            .collect();
        let [previous, changed] = comparisons(&now, &read_argmax(&before).unwrap());
        assert_eq!(previous.by_name.len(), 3);
        assert!(!changed.by_name.contains_key("c1"));
        let level = |c: &str| changed.levels[changed.by_name[c] as usize].to_string();
        assert_eq!(level("c2"), "CT3");
        assert_eq!(level("c3"), "unassigned");
    }

    #[test]
    fn the_newer_round_is_the_one_whose_source_is_this_file() {
        let dir = tempfile::tempdir().unwrap();
        let first = write(
            dir.path(),
            "r1.senna.json",
            r#"{"version":2,"kind":"bge","prefix":"r1"}"#,
        );
        write(
            dir.path(),
            "r2.senna.json",
            r#"{"version":2,"kind":"bge","prefix":"r2","annotate":{"source":"r1.senna.json"}}"#,
        );
        write(
            dir.path(),
            "other.senna.json",
            r#"{"version":2,"kind":"bge","prefix":"o"}"#,
        );
        let (m, d) = RunManifest::load(&first).unwrap();
        let r1 = Round::load(&m, &d, &first);
        assert!(r1.source.is_none());
        let newer = r1.newer().unwrap();
        assert!(newer.ends_with("r2.senna.json"));
        let (m2, d2) = RunManifest::load(&newer).unwrap();
        let r2 = Round::load(&m2, &d2, &newer);
        assert!(r2.source.as_ref().unwrap().ends_with("r1.senna.json"));
        assert!(r2.newer().is_none());
    }

    #[test]
    fn a_coarse_label_lists_its_member_types() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s.json",
            r#"{"4":{"size":10,"label":"G1","calls":[{"label":"CT1","q":0.01}]},
                "5":{"size":3,"label":"G1","evidence":{"top":"CT2","q":0.01,"agrees":false}},
                "6":{"size":3,"label":null,"evidence":{"top":"CT2","q":0.01,"agrees":false}}}"#,
        );
        write(
            dir.path(),
            "t.json",
            r#"{"source":"marker_sharing","groups":[{"name":"G1","members":["CT1","CT2"]},{"name":"CT3","members":["CT3"]}]}"#,
        );
        let path = write(
            dir.path(),
            "r.senna.json",
            r#"{"version":2,"kind":"bge","prefix":"r","annotate":{"cluster_summary":"s.json","celltype_tree":"t.json","fine_argmax":"f.tsv"}}"#,
        );
        let (m, d) = RunManifest::load(&path).unwrap();
        let round = Round::load(&m, &d, &path);
        assert_eq!(round.members_of("g1").unwrap(), ["CT1", "CT2"]);
        assert!(round.evidence("5").unwrap().agrees);
        assert!(round.evidence("6").unwrap().agrees);
        assert!(round.members_of("CT3").is_none());
        let lines = round.cluster_lines("4");
        assert!(lines[1].starts_with("a group of 2 types"));
        assert_eq!(lines[2], "  CT1, CT2");
        assert_eq!(lines[3], "calls");

        // After curation (lupin 0.2): a group named after one of its members
        // labels that member, the calls carry the post-selection caveat with
        // NES and p beside q, and a label the evidence disagrees with is
        // flagged. A cluster where no type passes FDR has no call and no
        // dispute; q = 1 ties are not shown as calls.
        write(
            dir.path(),
            "s2.json",
            r#"{"4":{"size":10,"label":"CT1","calls":[{"label":"CT2","q":0.02},{"label":"ASDC","q":1.0}],
                "evidence":{"top":"CT2","q":0.02,"agrees":false}},
                "5":{"size":4,"label":"CT1","calls":[{"label":"ASDC","q":1.0},{"label":"CT2","q":1.0}],
                "evidence":{"top":"ASDC","q":1.0,"agrees":false}}}"#,
        );
        {
            use senna::embed_common::*;
            let rows: Vec<Box<str>> = vec!["K4".into(), "K5".into()];
            let cols: Vec<Box<str>> = vec!["CT1".into(), "CT2".into()];
            for (name, vals) in [("nes", [0.1, 2.5, 0.0, 0.3]), ("p", [0.9, 0.001, 1.0, 0.8])] {
                let m = Mat::from_row_slice(2, 2, &vals.map(|v: f64| v as f32));
                let path = dir.path().join(format!("{name}.parquet"));
                m.to_parquet_with_names(
                    &path.to_string_lossy(),
                    (Some(&rows), Some("cluster")),
                    Some(&cols),
                )
                .unwrap();
            }
        }
        write(
            dir.path(),
            "t2.json",
            r#"{"groups":[{"name":"CT1","members":["CT1","CT2"]}]}"#,
        );
        let path = write(
            dir.path(),
            "r2.senna.json",
            r#"{"version":2,"kind":"bge","prefix":"r2","annotate":{"cluster_summary":"s2.json","celltype_tree":"t2.json",
                "cluster_celltype_nes":"nes.parquet","cluster_celltype_p":"p.parquet",
                "stats":{"kind":"post_selection","rounds_of_curation":2},
                "settings":{"enrichment":{"fdr_alpha":0.1,"null":{"gene_set_randomization":10000,"sample_permutation":0,"batches":1}}}}}"#,
        );
        let (m, d) = RunManifest::load(&path).unwrap();
        let round = Round::load(&m, &d, &path);
        assert!(round.members_of("CT1").is_none());
        assert!(!round.evidence("4").unwrap().agrees);
        let lines = round.cluster_lines("4");
        assert_eq!(
            lines[1],
            "≠ the evidence calls it CT2 (q 0.020  p 0.001  NES 2.500)"
        );
        assert_eq!(
            lines[2],
            "calls (post-selection, after 2 rounds of curation; q from the gene-set null only)"
        );
        assert_eq!(lines[3], "  CT2  q 0.020  p 0.001  NES 2.500");
        assert_eq!(lines.len(), 4, "the q = 1 call is not shown: {lines:?}");
        assert_eq!(round.candidates("4"), ["CT1", "CT2"]);

        let e = round.evidence("5").unwrap();
        assert!(e.agrees && e.top.is_none());
        let lines = round.cluster_lines("5");
        assert_eq!(lines[2], "  no cell type passes FDR (q < 0.1)");
        assert_eq!(round.candidates("5"), ["CT1"]);
    }

    #[test]
    fn cluster_lines_show_summary_then_history() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s.json",
            r#"{"4":{"size":10,"label":"CT1","calls":[{"label":"CT1","q":0.01}],"terms":[],"cl":{"id":"X","name":"N","abstained":true}}}"#,
        );
        write(
            dir.path(),
            "h.json",
            r#"{"4":[{"action":"merge","label":"CT1","rationale":"same markers","decided_by":"user","timestamp":"2026-01-02T03:04:05Z","merged_from":[1,2]}]}"#,
        );
        let path = write(
            dir.path(),
            "r.senna.json",
            r#"{"version":2,"kind":"bge","prefix":"r","annotate":{"cluster_summary":"s.json","history":"h.json"}}"#,
        );
        let (m, d) = RunManifest::load(&path).unwrap();
        let lines = Round::load(&m, &d, &path).cluster_lines("4");
        assert_eq!(lines[0], "C4 · 10 cells · CT1");
        assert!(lines.contains(&"  CT1  q 0.010".to_string()));
        assert!(lines.iter().any(|l| l.contains("(abstained)")));
        assert!(lines
            .iter()
            .any(|l| l.contains("merge CT1") && l.contains("merged from [1,2]")));
        assert!(lines.iter().any(|l| l.contains("same markers (user)")));
        assert_eq!(
            Round::load(&m, &d, &path).cluster_lines("9")[0],
            "C9 · no summary recorded for this round"
        );
    }
}
