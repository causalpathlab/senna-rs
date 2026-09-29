//! Scores kept current with the staged marker edits: lupin rescores the
//! round against them in the background (`relabel --preview`), so each
//! cluster's calls follow the edits instead of waiting for the next round.
//! lupin chooses how: when the set of cell types stays the same, only the
//! types the edits touch are scored again (within seconds); an edit that adds
//! or removes a whole type moves every gene's weight, and all are rescored.

use super::*;
use crate::view::decide::Action;
use crate::view::rounds::{fdr_alpha, label_key, UNASSIGNED};

/// Calls shown for the visited cluster.
const LIVE_CALLS: usize = 3;

/// Staged marker edits: per cell type and direction, the features.
pub(crate) type Edits = Vec<(Action, String, Vec<Box<str>>)>;

/// One cell type's rescored call in one cluster.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Call {
    pub label: String,
    /// Its share of the cluster (lupin's Q).
    pub share: f64,
    pub nes: Option<f64>,
    pub p: Option<f64>,
    pub q: Option<f64>,
}

/// Per cluster id, its calls, smallest q first.
pub(crate) type Scores = BTreeMap<i64, Vec<Call>>;

/// lupin's scores against the staged marker edits.
pub(crate) struct Live {
    /// The edits they are for.
    pub edits: Edits,
    /// `None` while lupin works; else the scores, or why there are none.
    pub scores: Option<Result<Scores, String>>,
}

/// A `lupin relabel --preview`'s `scores` (per cluster id, each type's
/// share, NES, p and q), smallest q first as the round's calls are.
pub(crate) fn parse_scores(v: &serde_json::Value) -> Result<Scores, String> {
    let Some(per) = v["scores"].as_object() else {
        // lupin 0.2 always answers `scores`, null when it cannot rescore.
        return Err(if v.get("scores").is_some() {
            "this round has no cached statistics to rescore; S rescores it fully".into()
        } else {
            "this lupin does not rescore live (an older build?); pass --lupin <path>".into()
        });
    };
    let mut out = Scores::new();
    for (id, calls) in per {
        let id: i64 = id
            .parse()
            .map_err(|_| format!("lupin's scores name cluster {id:?}"))?;
        let mut calls: Vec<Call> = calls
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| {
                Some(Call {
                    label: c["label"].as_str()?.to_string(),
                    share: c["share"].as_f64().unwrap_or(0.0),
                    nes: c["nes"].as_f64(),
                    p: c["p"].as_f64(),
                    q: c["q"].as_f64(),
                })
            })
            .filter(|c| c.label != UNASSIGNED)
            .collect();
        calls.sort_by(|a, b| {
            let q = |c: &Call| c.q.unwrap_or(f64::INFINITY);
            q(a).total_cmp(&q(b)).then(b.share.total_cmp(&a.share))
        });
        out.insert(id, calls);
    }
    Ok(out)
}

impl Scene {
    /// The rescored calls of cluster `id`, when there are live scores.
    fn live_calls(&self, id: i64) -> Option<&[Call]> {
        let live = self.review.as_ref()?.live.as_ref()?;
        match &live.scores {
            Some(Ok(s)) => s.get(&id).map(Vec::as_slice),
            _ => None,
        }
    }

    /// The type the evidence calls cluster `o` when it differs from its
    /// label: from the live scores when there are any, else as recorded.
    pub(crate) fn disputed(&self, o: &Overview) -> Option<String> {
        let Some(calls) = self.live_calls(o.id) else {
            return o.disputed.clone();
        };
        let round = self.data.round.as_ref();
        let alpha = fdr_alpha(round);
        let label = o.label.as_deref().filter(|l| *l != UNASSIGNED)?;
        let top = &calls.iter().find(|c| c.q.is_some_and(|q| q < alpha))?.label;
        // A coarse label and a fine top call inside its group agree.
        let within = round
            .and_then(|r| r.members_of(label))
            .is_some_and(|ms| ms.iter().any(|m| label_key(m) == label_key(top)));
        (!same_type(label, top) && !within).then(|| top.clone())
    }

    /// Sidebar lines for the visited cluster's live scores: lupin's calls
    /// with the staged marker edits, beside the q the round recorded.
    pub(crate) fn live_lines(&self) -> Vec<String> {
        let Some(r) = self.review.as_ref() else {
            return Vec::new();
        };
        let Some(live) = &r.live else {
            return Vec::new();
        };
        let n = live.edits.len();
        let edits = format!("{n} marker edit{}", if n == 1 { "" } else { "s" });
        let mut out = vec![String::new()];
        match &live.scores {
            None => out.push(format!("lupin: rescoring with {edits}…")),
            Some(Err(e)) => out.push(format!("lupin could not rescore: {e}")),
            Some(Ok(scores)) => {
                let id = r.cluster();
                let round = self.data.round.as_ref();
                let alpha = fdr_alpha(round);
                out.push(format!("lupin, rescored with {edits}:"));
                let calls = scores.get(&id).map_or(&[][..], Vec::as_slice);
                let passing: Vec<&Call> = calls
                    .iter()
                    .filter(|c| c.q.is_some_and(|q| q < alpha))
                    .take(LIVE_CALLS)
                    .collect();
                if passing.is_empty() {
                    out.push(format!("  no cell type passes FDR (q < {alpha})"));
                }
                let fmt = |x: Option<f64>| x.map_or("-".to_string(), |x| format!("{x:.3}"));
                for c in passing {
                    let was = round
                        .and_then(|r| r.recorded_q(&id.to_string(), &c.label))
                        .map_or_else(|| "new".to_string(), |q| format!("was {q:.3}"));
                    out.push(format!(
                        "  {:<18} q {} ({was})  NES {}  share {:.2}",
                        c.label,
                        fmt(c.q),
                        fmt(c.nes),
                        c.share
                    ));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_read_smallest_q_first_without_unassigned() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"scores":{"3":[
                {"label":"CT1","share":0.6,"nes":2.0,"p":0.001,"q":0.02},
                {"label":"CT2","share":0.3,"nes":1.5,"p":0.0001,"q":0.004},
                {"label":"unassigned","share":0.1,"nes":null,"p":null,"q":null},
                {"label":"CT3","share":0.0,"nes":null,"p":null,"q":null}]}}"#,
        )
        .unwrap();
        let s = parse_scores(&v).unwrap();
        let labels: Vec<&str> = s[&3].iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, ["CT2", "CT1", "CT3"]);
        assert_eq!(s[&3][1].nes, Some(2.0));

        let none: serde_json::Value = serde_json::from_str(r#"{"scores":null}"#).unwrap();
        assert!(parse_scores(&none)
            .unwrap_err()
            .contains("no cached statistics"));
        let old: serde_json::Value = serde_json::from_str(r#"{"clusters":{}}"#).unwrap();
        assert!(parse_scores(&old).unwrap_err().contains("older build"));
    }
}
