//! Relabel mode's draft: decisions staged while visiting clusters one by one,
//! turned into lupin decision lines only when the user submits them all as
//! one round. The draft is saved beside the round, so leaving the mode or the
//! view loses nothing.

use super::decide::{Action, Decision};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What was decided for one cluster.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "lowercase")]
pub enum Verdict {
    Label { label: String, rationale: String },
    Keep { label: String, rationale: String },
}

/// A marker-panel edit staged for one feature: add it to a type's markers,
/// or drop it from one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mark {
    Include { cell_type: String },
    Exclude { cell_type: String },
}

/// Everything staged for one cluster.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClusterDraft {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    /// Marker edits, by feature; the score is the feature's expected log
    /// fold change in this cluster, kept as evidence.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub marks: BTreeMap<String, (Mark, f32)>,
}

/// A merge of clusters under one label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Merge {
    pub clusters: Vec<i64>,
    pub label: String,
    pub rationale: String,
}

/// A staged marker edit, grouped under (cell type, include?): the feature,
/// its score, and the cluster that motivated it.
type StagedEdit = (String, f32, i64);

/// The staged decisions for one round.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    /// The round whose cluster ids these decisions use.
    pub round: PathBuf,
    #[serde(default)]
    pub clusters: BTreeMap<i64, ClusterDraft>,
    #[serde(default)]
    pub merges: Vec<Merge>,
}

impl Draft {
    /// `{round}` with `.senna.json` replaced by `.relabel_draft.json`.
    #[must_use]
    pub fn path_for(round: &Path) -> PathBuf {
        let s = round.to_string_lossy();
        let stem = s.strip_suffix(".senna.json").unwrap_or(&s);
        PathBuf::from(format!("{stem}.relabel_draft.json"))
    }

    /// The saved draft for `round`, or an empty one.
    #[must_use]
    pub fn load(round: &Path) -> Self {
        std::fs::read_to_string(Self::path_for(round))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| Self {
                round: round.to_path_buf(),
                ..Default::default()
            })
    }

    pub fn save(&self) -> anyhow::Result<()> {
        std::fs::write(
            Self::path_for(&self.round),
            serde_json::to_string_pretty(self)?,
        )?;
        Ok(())
    }

    /// Remove the saved draft (after it was applied).
    pub fn discard(&self) {
        let _ = std::fs::remove_file(Self::path_for(&self.round));
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
            && self
                .clusters
                .values()
                .all(|c| c.verdict.is_none() && c.marks.is_empty())
    }

    /// Clusters with a verdict or in a merge.
    #[must_use]
    pub fn decided(&self) -> usize {
        let merged: std::collections::BTreeSet<i64> = self
            .merges
            .iter()
            .flat_map(|m| m.clusters.iter().copied())
            .collect();
        self.clusters
            .iter()
            .filter(|(id, c)| c.verdict.is_some() || merged.contains(id))
            .count()
            + merged
                .iter()
                .filter(|id| !self.clusters.contains_key(id))
                .count()
    }

    pub fn cluster(&mut self, id: i64) -> &mut ClusterDraft {
        self.clusters.entry(id).or_default()
    }

    /// The decision lines for lupin: marker edits first (grouped by cell type
    /// and direction), then merges, labels and keeps.
    #[must_use]
    pub fn decisions(&self) -> Vec<Decision> {
        let mut out = Vec::new();
        let mut edits: BTreeMap<(String, bool), Vec<StagedEdit>> = BTreeMap::new();
        for (&id, c) in &self.clusters {
            for (f, (mark, score)) in &c.marks {
                let (t, add) = match mark {
                    Mark::Include { cell_type } => (cell_type.clone(), true),
                    Mark::Exclude { cell_type } => (cell_type.clone(), false),
                };
                edits
                    .entry((t, add))
                    .or_default()
                    .push((f.clone(), *score, id));
            }
        }
        for ((cell_type, add), feats) in edits {
            let mut clusters: Vec<i64> = feats.iter().map(|x| x.2).collect();
            clusters.dedup();
            let ids: Vec<String> = clusters.iter().map(|c| format!("C{c}")).collect();
            let rationale = if add {
                format!("high in {} (expected log fold change)", ids.join(", "))
            } else {
                format!("low in {} although listed as a marker", ids.join(", "))
            };
            out.push(Decision {
                action: if add {
                    Action::MarkersAdd
                } else {
                    Action::MarkersDrop
                },
                clusters: Vec::new(),
                features: feats.iter().map(|x| x.0.as_str().into()).collect(),
                label: cell_type,
                rationale,
                evidence: feats
                    .iter()
                    .take(10)
                    .map(|(f, v, _)| {
                        serde_json::json!({"kind": "marker", "term": f, "stat": "expected_lfc", "value": v})
                    })
                    .collect(),
            });
        }
        for m in &self.merges {
            out.push(Decision {
                action: Action::Merge,
                clusters: m.clusters.clone(),
                features: Vec::new(),
                label: m.label.clone(),
                rationale: m.rationale.clone(),
                evidence: Vec::new(),
            });
        }
        for (&id, c) in &self.clusters {
            let (action, label, rationale) = match &c.verdict {
                Some(Verdict::Label { label, rationale }) => (Action::Label, label, rationale),
                Some(Verdict::Keep { label, rationale }) => (Action::Keep, label, rationale),
                None => continue,
            };
            let evidence = c
                .marks
                .iter()
                .filter(|(_, (m, _))| matches!(m, Mark::Include { .. }))
                .take(5)
                .map(|(f, (_, v))| {
                    serde_json::json!({"kind": "marker", "term": f, "stat": "expected_lfc", "value": v})
                })
                .collect();
            out.push(Decision {
                action,
                clusters: vec![id],
                features: Vec::new(),
                label: label.clone(),
                rationale: rationale.clone(),
                evidence,
            });
        }
        out
    }

    /// The decisions as JSON lines naming `round`.
    #[must_use]
    pub fn lines(&self, round: &str) -> Vec<Value> {
        self.decisions().iter().map(|d| d.to_json(round)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_decisions_become_marker_edits_then_merges_then_labels() {
        let mut d = Draft {
            round: "r.senna.json".into(),
            ..Default::default()
        };
        let c = d.cluster(3);
        c.marks.insert(
            "GENE1".into(),
            (
                Mark::Include {
                    cell_type: "CT1".into(),
                },
                2.0,
            ),
        );
        c.marks.insert(
            "GENE2".into(),
            (
                Mark::Exclude {
                    cell_type: "CT2".into(),
                },
                0.1,
            ),
        );
        c.verdict = Some(Verdict::Label {
            label: "CT1".into(),
            rationale: "GENE1 high".into(),
        });
        d.cluster(4).marks.insert(
            "GENE3".into(),
            (
                Mark::Include {
                    cell_type: "CT1".into(),
                },
                1.5,
            ),
        );
        d.merges.push(Merge {
            clusters: vec![5, 6],
            label: "CT3".into(),
            rationale: "same program".into(),
        });
        assert_eq!(d.decided(), 3);
        let lines = d.lines("r.senna.json");
        let actions: Vec<&str> = lines
            .iter()
            .map(|l| l["action"].as_str().unwrap())
            .collect();
        assert_eq!(actions, ["markers_add", "markers_drop", "merge", "label"]);
        assert_eq!(lines[0]["label"], "CT1");
        assert_eq!(lines[0]["features"], serde_json::json!(["GENE1", "GENE3"]));
        assert_eq!(lines[1]["features"], serde_json::json!(["GENE2"]));
        assert_eq!(lines[2]["clusters"], serde_json::json!([5, 6]));
        assert_eq!(lines[3]["cluster"], 3);
        assert_eq!(lines[3]["evidence"][0]["term"], "GENE1");
        assert!(lines
            .iter()
            .all(|l| !l["rationale"].as_str().unwrap().is_empty()));
    }

    #[test]
    fn a_draft_is_saved_beside_its_round_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let round = dir.path().join("r.senna.json");
        let mut d = Draft::load(&round);
        assert!(d.is_empty());
        d.cluster(1).verdict = Some(Verdict::Keep {
            label: "CT1".into(),
            rationale: "clear".into(),
        });
        d.save().unwrap();
        assert!(Draft::path_for(&round).ends_with("r.relabel_draft.json"));
        assert_eq!(Draft::load(&round), d);
        d.discard();
        assert!(Draft::load(&round).is_empty());
    }
}
