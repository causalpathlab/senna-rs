//! Annotation rounds: each `lupin annotate` or relabel pass writes a new
//! manifest whose `annotate.source` names the one it came from. This module
//! reads what the viewer shows about a round: how its labels differ from the
//! source round's, and the per-cluster summary and decision history lupin
//! records beside it. The viewer never applies decisions itself.
//!
//! Lupin's fields are read through the manifest's pass-through keys, so
//! senna's schema does not carry them.

use super::data::{read_pairs, LabelKind, Labels};
use super::files::{read_json, same_file, siblings};
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
    match read_json(path) {
        Some(Value::Object(m)) => m.into_iter().collect(),
        _ => HashMap::default(),
    }
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

/// Where a round sits, and what lupin recorded about its clusters.
pub struct Round {
    /// This manifest.
    pub path: PathBuf,
    /// The round this one was made from, when it records one.
    pub source: Option<PathBuf>,
    summary: HashMap<String, Value>,
    history: HashMap<String, Value>,
    /// Coarse groups of cell types and their members, when lupin called
    /// this round at a coarse level.
    tree: Vec<(String, Vec<String>)>,
    /// Its q values come from the gene-set null alone (too few batches for
    /// the sample-permutation null).
    gene_set_null_only: bool,
}

impl Round {
    #[must_use]
    pub fn load(m: &RunManifest, dir: &Path, path: &Path) -> Self {
        let at = |key: &str| annotate_str(m, key).map(|rel| run_manifest::resolve(dir, rel));
        Self {
            path: path.to_path_buf(),
            source: at("source"),
            summary: at("cluster_summary")
                .map(|p| read_json_object(&p))
                .unwrap_or_default(),
            history: at("history")
                .map(|p| read_json_object(&p))
                .unwrap_or_default(),
            // A round is coarse only when lupin kept the fine calls apart;
            // otherwise its tree is a record, not what its labels are.
            tree: at("celltype_tree")
                .filter(|_| annotate_str(m, "fine_argmax").is_some())
                .and_then(|p| read_json(&p))
                .map(|v| read_tree(&v))
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

    /// The member types of `label` when it names a coarse group.
    #[must_use]
    pub fn members_of(&self, label: &str) -> Option<&[String]> {
        self.tree
            .iter()
            .find(|(g, _)| label_key(g) == label_key(label))
            .map(|(_, m)| m.as_slice())
    }

    /// The round made from this one, found next to it: a manifest in the same
    /// directory whose `annotate.source` resolves to this file.
    #[must_use]
    pub fn newer(&self) -> Option<PathBuf> {
        // lupin names a round's successor `{stem}.r{N}.senna.json`; prefer
        // that chain over any other round made from this one.
        let stem = self
            .path
            .file_name()
            .map(|n| {
                n.to_string_lossy()
                    .trim_end_matches(".senna.json")
                    .to_string()
            })
            .unwrap_or_default();
        let mut found = siblings(&self.path, ".senna.json");
        found.sort_by_key(|p| {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            !name.starts_with(&format!("{stem}.r"))
        });
        found.into_iter().find(|p| {
            read_json(p)
                .and_then(|v| v.pointer("/annotate/source")?.as_str().map(String::from))
                .is_some_and(|src| {
                    same_file(
                        &run_manifest::resolve(run_manifest::manifest_dir(p), &src),
                        &self.path,
                    )
                })
        })
    }

    /// The label the round gives cluster `id`, and its top marker call with
    /// that call's bootstrap support, from the summary.
    #[must_use]
    pub fn call(&self, id: &str) -> (Option<String>, Option<(String, Option<f64>)>) {
        let Some(s) = self.summary.get(id) else {
            return (None, None);
        };
        let label = s.get("label").and_then(Value::as_str).map(String::from);
        let top = s
            .get("calls")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|c| {
                let l = c.get("label")?.as_str()?.to_string();
                Some((l, c.get("support").and_then(Value::as_f64)))
            });
        (label, top)
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
        let calls = s
            .get("calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        for l in label
            .into_iter()
            .chain(calls.filter_map(|c| c.get("label")?.as_str()).take(3))
        {
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
            let calls = list("calls");
            if !calls.is_empty() {
                out.push(if self.gene_set_null_only {
                    "calls (q from the gene-set null only)".into()
                } else {
                    "calls".into()
                });
                for c in calls.iter().take(3) {
                    let l = c.get("label").and_then(Value::as_str).unwrap_or("-");
                    out.push(format!(
                        "  {l}  q {}  support {}",
                        num(c, "q"),
                        num(c, "support")
                    ));
                }
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
            r#"{"4":{"size":10,"label":"G1","calls":[{"label":"CT1","q":0.01}]}}"#,
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
        assert!(round.members_of("CT3").is_none());
        let lines = round.cluster_lines("4");
        assert!(lines[1].starts_with("a group of 2 types"));
        assert_eq!(lines[2], "  CT1, CT2");
        assert_eq!(lines[3], "calls");

        // A re-annotated round records its tree but is not coarse, and says
        // when its q values rest on the gene-set null alone.
        let path = write(
            dir.path(),
            "r2.senna.json",
            r#"{"version":2,"kind":"bge","prefix":"r2","annotate":{"cluster_summary":"s.json","celltype_tree":"t.json","settings":{"enrichment":{"null":{"gene_set_randomization":1000,"sample_permutation":0,"batches":1}}}}}"#,
        );
        let (m, d) = RunManifest::load(&path).unwrap();
        let round = Round::load(&m, &d, &path);
        assert!(round.members_of("G1").is_none());
        let lines = round.cluster_lines("4");
        assert_eq!(lines[1], "calls (q from the gene-set null only)");
    }

    #[test]
    fn cluster_lines_show_summary_then_history() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s.json",
            r#"{"4":{"size":10,"label":"CT1","calls":[{"label":"CT1","q":0.01,"support":null}],"terms":[],"cl":{"id":"X","name":"N","abstained":true}}}"#,
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
        assert!(lines.iter().any(|l| l.contains("support -")));
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
