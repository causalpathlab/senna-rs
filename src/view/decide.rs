//! Decisions typed in the view, applied by lupin. For each one the view runs
//! `lupin relabel -f <round on screen> -d - --next` with the decision as one
//! JSON line on stdin; lupin writes the next round and prints its path, and
//! the view opens it. The view applies nothing itself.
//!
//! A `lupin relabel --watch` may also be running on the same chain (for
//! decisions made elsewhere); its `{prefix}.relabel_status.json` names the
//! latest round, which the view follows. Paths in it are relative to it.

use super::files::{modified, read_json, same_file, siblings};
use senna::run_manifest;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

/// A running `lupin relabel --watch`, as seen through its status file.
pub struct Watcher {
    status: Option<PathBuf>,
    pub decisions: PathBuf,
    pub latest: Option<PathBuf>,
    pub error: Option<String>,
    stamp: Option<SystemTime>,
    /// The decisions directory's modification time at the last search for a
    /// status file, so the search reruns only when files come or go.
    dir_stamp: Option<SystemTime>,
}

impl Watcher {
    /// The watcher for the round at `open`: a status file beside it whose
    /// `base` or `rounds` include that round.
    #[must_use]
    pub fn find(open: &Path) -> Option<Self> {
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
            if covers {
                let mut w = Self {
                    status: Some(status),
                    decisions: dec,
                    latest: None,
                    error: None,
                    stamp: None,
                    dir_stamp: None,
                };
                w.refresh();
                return Some(w);
            }
        }
        None
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
            let dir = modified(run_manifest::manifest_dir(&self.decisions));
            if dir != self.dir_stamp {
                self.dir_stamp = dir;
                self.status = self.locate_status();
            }
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
}

/// Apply the decisions as the next round, or only preview their effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Next,
    Preview,
}

/// What `lupin relabel` answered.
pub enum Reply {
    /// The round it wrote (`--next`).
    Round(PathBuf),
    /// What the decisions would change, without writing (`--preview`).
    Preview(Value),
    /// Why it refused, and the latest round when the one on screen was stale.
    Refused {
        reason: String,
        latest: Option<PathBuf>,
    },
}

/// Run `lupin relabel -f <round> -d - --next|--preview` with `decisions` on
/// stdin, one line each; lupin applies them together as one round (or
/// previews them). `Err` means lupin could not be run at all (or cannot
/// relabel).
pub fn relabel(
    lupin: &str,
    round: &Path,
    decisions: &[Value],
    mode: Mode,
) -> Result<Reply, String> {
    let flag = match mode {
        Mode::Next => "--next",
        Mode::Preview => "--preview",
    };
    let mut child = Command::new(lupin)
        .args(["relabel", "-f"])
        .arg(round)
        .args(["-d", "-", flag])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            format!("cannot run `{lupin}` ({e}); pass --lupin <path> or set SENNA_LUPIN")
        })?;
    let mut lines = String::new();
    for d in decisions {
        lines.push_str(&serde_json::to_string(d).map_err(|e| e.to_string())?);
        lines.push('\n');
    }
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(lines.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if out.status.success() {
        return match mode {
            Mode::Preview => serde_json::from_str(&stdout)
                .map(Reply::Preview)
                .map_err(|e| format!("lupin's preview is not JSON: {e}")),
            Mode::Next => stdout
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .map(|p| Reply::Round(PathBuf::from(p.trim())))
                .ok_or_else(|| "lupin wrote no round path".to_string()),
        };
    }
    if stderr.contains("unrecognized subcommand") || stderr.contains("unexpected argument") {
        return Err(format!(
            "`{lupin}` does not support `relabel {flag}` (an older build?); \
             pass --lupin <path> or set SENNA_LUPIN"
        ));
    }
    let last = stderr
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("lupin failed without a reason")
        .trim();
    let reason = last.strip_prefix("Error: ").unwrap_or(last).to_string();
    // "<round> is not the latest round (that is <path>); reload and decide again"
    let latest = reason
        .split_once("(that is ")
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(p, _)| PathBuf::from(p.trim()));
    Ok(Reply::Refused { reason, latest })
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
    fn a_watcher_is_found_by_its_rounds_and_its_status_read() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::write(d.join("r.senna.json"), "{}").unwrap();
        std::fs::write(d.join("r.r1.senna.json"), "{}").unwrap();
        std::fs::write(d.join("x.senna.json"), "{}").unwrap();
        std::fs::write(
            d.join("r.relabel_status.json"),
            r#"{"decisions":"d.jsonl","base":"r.senna.json","rounds":["r.senna.json","r.r1.senna.json"],
                "latest":"r.r1.senna.json","error":{"lines":[1,1],"message":"no rationale"}}"#,
        )
        .unwrap();
        let w = Watcher::find(&d.join("r.r1.senna.json")).unwrap();
        assert!(w.latest.as_ref().unwrap().ends_with("r.r1.senna.json"));
        assert_eq!(w.error.as_deref(), Some("no rationale"));
        assert!(Watcher::find(&d.join("x.senna.json")).is_none());
    }

    /// A stand-in `lupin` that behaves as `relabel --next` does.
    #[cfg(unix)]
    fn fake_lupin(dir: &Path, body: &str) -> String {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join("lupin");
        std::fs::write(
            &p,
            format!("#!/bin/sh\ncat > {}/stdin.txt\n{body}\n", dir.display()),
        )
        .unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p.to_string_lossy().into_owned()
    }

    #[cfg(unix)]
    #[test]
    fn relabel_next_reads_the_new_round_or_the_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let dec = Decision {
            action: Action::Merge,
            clusters: vec![3, 5],
            features: Vec::new(),
            label: "CT1".into(),
            rationale: "same markers".into(),
            evidence: Vec::new(),
        };
        let json = dec.to_json("r.senna.json");

        let ok = fake_lupin(d, "echo r.r1.senna.json");
        let Ok(Reply::Round(p)) =
            relabel(&ok, Path::new("r.senna.json"), &[json.clone()], Mode::Next)
        else {
            panic!("expected a round");
        };
        assert_eq!(p, PathBuf::from("r.r1.senna.json"));
        let sent: Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("stdin.txt")).unwrap()).unwrap();
        assert_eq!(sent["clusters"], serde_json::json!([3, 5]));
        assert_eq!(sent["round"], "r.senna.json");

        let stale = fake_lupin(
            d,
            "echo 'Error: r.senna.json is not the latest round (that is r.r2.senna.json); reload and decide again' >&2; exit 1",
        );
        let Ok(Reply::Refused { reason, latest }) = relabel(
            &stale,
            Path::new("r.senna.json"),
            &[json.clone()],
            Mode::Next,
        ) else {
            panic!("expected a refusal");
        };
        assert!(reason.starts_with("r.senna.json is not the latest round"));
        assert_eq!(latest, Some(PathBuf::from("r.r2.senna.json")));

        let old = fake_lupin(
            d,
            "echo \"error: unrecognized subcommand 'relabel'\" >&2; exit 2",
        );
        assert!(
            relabel(&old, Path::new("r.senna.json"), &[json.clone()], Mode::Next)
                .err()
                .unwrap()
                .contains("does not support")
        );
        assert!(relabel(
            "/nonexistent/lupin",
            Path::new("r.senna.json"),
            &[json.clone()],
            Mode::Next
        )
        .is_err());

        let preview = fake_lupin(d, "echo '{\"cells_changed\": 7, \"clusters\": {}}'");
        let Ok(Reply::Preview(v)) = relabel(
            &preview,
            Path::new("r.senna.json"),
            &[json.clone(), json],
            Mode::Preview,
        ) else {
            panic!("expected a preview");
        };
        assert_eq!(v["cells_changed"], 7);
        let sent = std::fs::read_to_string(d.join("stdin.txt")).unwrap();
        assert_eq!(sent.lines().count(), 2);
    }
}
