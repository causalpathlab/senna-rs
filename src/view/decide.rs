//! Decisions typed in the view, handed to `lupin relabel --watch` through a
//! file. The view only writes what the user decided; lupin applies it and
//! writes the next round, which the view then opens. Neither tool calls the
//! other.
//!
//! The watcher keeps `{prefix}.relabel_status.json` beside its rounds:
//! `decisions` (the file to append to), `base` and `rounds` (the rounds it
//! has written, oldest first), `latest`, and `error` when a batch was
//! refused. Paths in it are relative to the status file.

use super::files::{modified, read_json, same_file, siblings};
use senna::run_manifest;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// A running `lupin relabel --watch`, as seen through its status file.
pub struct Watcher {
    status: Option<PathBuf>,
    pub decisions: PathBuf,
    pub latest: Option<PathBuf>,
    pub error: Option<String>,
    stamp: Option<SystemTime>,
}

impl Watcher {
    /// The watcher for the round at `open`: a status file beside it whose
    /// `base` or `rounds` include that round, or, when `decisions` is given,
    /// that file (with its status file if one points at it).
    #[must_use]
    pub fn find(open: &Path, decisions: Option<&Path>) -> Option<Self> {
        for status in siblings(open, ".relabel_status.json") {
            let Some(v) = read_json(&status) else {
                continue;
            };
            let base = run_manifest::manifest_dir(&status);
            let at = |s: &str| run_manifest::resolve(base, s);
            let Some(dec) = v.get("decisions").and_then(Value::as_str).map(at) else {
                continue;
            };
            let covers = v
                .get("rounds")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .chain(v.get("base"))
                .filter_map(Value::as_str)
                .any(|r| same_file(&at(r), open));
            let named = decisions.is_some_and(|d| same_file(&dec, d));
            if covers || named {
                let mut w = Self {
                    status: Some(status),
                    decisions: dec,
                    latest: None,
                    error: None,
                    stamp: None,
                };
                w.refresh();
                return Some(w);
            }
        }
        // A decisions file named explicitly works without a status file; the
        // view then just waits for new rounds to appear.
        decisions.map(|d| Self {
            status: None,
            decisions: d.to_path_buf(),
            latest: None,
            error: None,
            stamp: None,
        })
    }

    /// The status file beside the decisions file that names it, once the
    /// watcher has written one.
    fn locate_status(&self) -> Option<PathBuf> {
        siblings(&self.decisions, ".relabel_status.json")
            .into_iter()
            .find(|p| {
                let base = run_manifest::manifest_dir(p);
                read_json(p)
                    .and_then(|v| {
                        v.get("decisions")?
                            .as_str()
                            .map(|d| run_manifest::resolve(base, d))
                    })
                    .is_some_and(|d| same_file(&d, &self.decisions))
            })
    }

    /// Re-read the status file if it changed. Returns whether it did.
    pub fn refresh(&mut self) -> bool {
        if self.status.is_none() {
            self.status = self.locate_status();
        }
        let Some(status) = &self.status else {
            return false;
        };
        let stamp = modified(status);
        if stamp == self.stamp {
            return false;
        }
        self.stamp = stamp;
        let Some(v) = read_json(status) else {
            return false;
        };
        let base = run_manifest::manifest_dir(status);
        self.latest = v
            .get("latest")
            .and_then(Value::as_str)
            .map(|s| run_manifest::resolve(base, s));
        self.error = v
            .pointer("/error/message")
            .and_then(Value::as_str)
            .map(String::from);
        true
    }

    /// `path` as a decision's `round` field: relative to the decisions file
    /// when it sits under the same directory, absolute otherwise.
    #[must_use]
    pub fn round_ref(&self, path: &Path) -> String {
        let dir = run_manifest::manifest_dir(&self.decisions)
            .canonicalize()
            .ok();
        match (dir, path.canonicalize().ok()) {
            (Some(d), Some(p)) => p.strip_prefix(&d).map_or_else(
                |_| p.to_string_lossy().into_owned(),
                |r| r.to_string_lossy().into_owned(),
            ),
            _ => path.to_string_lossy().into_owned(),
        }
    }

    /// Append one decision as a single newline-terminated line, in one write,
    /// so the watcher never reads half of it.
    pub fn append(&self, decision: &Value) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(decision)?;
        line.push('\n');
        anyhow::ensure!(
            line.len() < 4096,
            "decision is {} bytes; keep it under 4 KB (shorter rationale or fewer evidence items)",
            line.len()
        );
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.decisions)?;
        f.write_all(line.as_bytes())?;
        Ok(())
    }
}

/// What a decision does. Label, merge and keep act on clusters; the marker
/// actions edit the marker table the next round starts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Label,
    Merge,
    Keep,
    MarkersAdd,
    MarkersDrop,
}

impl Action {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Action::Label => "label",
            Action::Merge => "merge",
            Action::Keep => "keep",
            Action::MarkersAdd => "markers_add",
            Action::MarkersDrop => "markers_drop",
        }
    }

    /// Whether the user types a label for it (keep reuses the current one).
    #[must_use]
    pub fn needs_label(self) -> bool {
        self != Action::Keep
    }
}

/// One decision, ready to be written.
pub struct Decision {
    pub action: Action,
    pub clusters: Vec<i64>,
    pub features: Vec<Box<str>>,
    pub label: String,
    pub rationale: String,
    pub evidence: Vec<Value>,
}

impl Decision {
    /// The JSON line lupin reads. `round` names the round whose cluster ids
    /// the decision uses.
    #[must_use]
    pub fn to_json(&self, round: &str) -> Value {
        let mut v = serde_json::json!({
            "action": self.action.name(),
            "label": self.label,
            "rationale": self.rationale,
            "decided_by": "user",
            "round": round,
            "evidence": self.evidence,
        });
        match self.action {
            Action::Merge => v["clusters"] = self.clusters.clone().into(),
            Action::Label | Action::Keep => {
                if let [one] = self.clusters.as_slice() {
                    v["cluster"] = (*one).into();
                } else {
                    v["clusters"] = self.clusters.clone().into();
                }
            }
            Action::MarkersAdd | Action::MarkersDrop => {
                v["features"] = self
                    .features
                    .iter()
                    .map(AsRef::as_ref)
                    .collect::<Vec<&str>>()
                    .into();
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_watcher_is_found_by_its_rounds_and_lines_are_appended_whole() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::write(d.join("r.senna.json"), "{}").unwrap();
        std::fs::write(d.join("r.r1.senna.json"), "{}").unwrap();
        std::fs::write(
            d.join("r.relabel_status.json"),
            r#"{"decisions":"d.jsonl","base":"r.senna.json","rounds":["r.senna.json","r.r1.senna.json"],
                "latest":"r.r1.senna.json","error":{"lines":[1,1],"message":"no rationale"}}"#,
        )
        .unwrap();
        let w = Watcher::find(&d.join("r.r1.senna.json"), None).unwrap();
        assert!(w.decisions.ends_with("d.jsonl"));
        assert!(w.latest.as_ref().unwrap().ends_with("r.r1.senna.json"));
        assert_eq!(w.error.as_deref(), Some("no rationale"));
        assert_eq!(w.round_ref(&d.join("r.r1.senna.json")), "r.r1.senna.json");

        let dec = Decision {
            action: Action::Merge,
            clusters: vec![3, 5],
            features: Vec::new(),
            label: "CT1".into(),
            rationale: "same markers".into(),
            evidence: Vec::new(),
        };
        w.append(&dec.to_json("r.r1.senna.json")).unwrap();
        let add = Decision {
            action: Action::MarkersAdd,
            clusters: Vec::new(),
            features: vec!["GENE1".into()],
            label: "CT1".into(),
            rationale: "high in CT1".into(),
            evidence: Vec::new(),
        };
        w.append(&add.to_json("r.r1.senna.json")).unwrap();
        let text = std::fs::read_to_string(d.join("d.jsonl")).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert!(text.ends_with('\n'));
        assert_eq!(lines[0]["clusters"], serde_json::json!([3, 5]));
        assert_eq!(lines[0]["round"], "r.r1.senna.json");
        assert_eq!(lines[1]["action"], "markers_add");
        assert_eq!(lines[1]["features"], serde_json::json!(["GENE1"]));
    }

    #[test]
    fn a_round_no_watcher_covers_finds_none_unless_named() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::write(d.join("x.senna.json"), "{}").unwrap();
        assert!(Watcher::find(&d.join("x.senna.json"), None).is_none());
        let w = Watcher::find(&d.join("x.senna.json"), Some(&d.join("mine.jsonl"))).unwrap();
        assert!(w.decisions.ends_with("mine.jsonl"));
        assert!(w.latest.is_none());
    }
}
