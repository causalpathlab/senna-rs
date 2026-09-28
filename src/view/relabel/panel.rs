//! Relabel mode's sidebar and overview text.

use super::*;

impl Scene {
    /// Sidebar text for merge mode.
    pub fn merge_lines(&self) -> Option<Vec<String>> {
        let r = self.review.as_ref()?;
        let m = r.merge.as_ref()?;
        let chosen: Vec<&Overview> = r
            .overview
            .iter()
            .filter(|o| m.chosen.contains(&o.id))
            .collect();
        let total: usize = chosen.iter().map(|o| o.size).sum();
        let mut out = vec![
            format!("merge · {} clusters · {total} cells", chosen.len()),
            match &m.best {
                Some((t, fit)) => format!("together, markers fit {t} best ({fit:+.2})"),
                None => "together: no marker fit".into(),
            },
            if chosen.len() < 2 {
                "next: choose at least one more cluster (↑↓, space)".into()
            } else {
                "next: enter to name the merged cluster".into()
            },
            String::new(),
        ];
        for o in chosen {
            out.push(format!(
                "  C{:<4} {:>6} cells  now {}",
                o.id,
                o.size,
                o.label.as_deref().unwrap_or("unassigned")
            ));
        }
        Some(out)
    }

    /// Sidebar text for the cluster overview (the left panel).
    pub fn overview_lines(&self) -> Option<Vec<String>> {
        let r = self.review.as_ref()?;
        let cursor = r.merge.as_ref().map_or(r.at, |m| m.cursor);
        let cursor_best = r
            .overview
            .get(cursor)
            .and_then(|o| o.best.as_ref())
            .map(|b| b.0.clone());
        let merged: std::collections::BTreeSet<i64> = r
            .draft
            .merges
            .iter()
            .flat_map(|m| m.clusters.iter().copied())
            .collect();
        let mut out = vec![
            format!("clusters · {} decided", r.draft.decided()),
            "? unassigned  → markers suggest".into(),
            "↓ a group to refine  ✓ decided".into(),
            String::new(),
        ];
        for (k, o) in r.overview.iter().enumerate() {
            let here = if k == cursor { "▸" } else { " " };
            let pick = match &r.merge {
                Some(m) if m.chosen.contains(&o.id) => "[x]",
                Some(_) => "[ ]",
                None => "",
            };
            let staged = r.draft.clusters.get(&o.id).and_then(|c| c.verdict.as_ref());
            let status = if let Some(v) = staged {
                match v {
                    Verdict::Label { label, .. } | Verdict::Keep { label, .. } => {
                        format!("✓ {label}")
                    }
                }
            } else if merged.contains(&o.id) {
                "✓ merged".into()
            } else if o.label.is_none() {
                match &o.best {
                    Some((b, _)) => format!("? → {b}"),
                    None => "?".into(),
                }
            } else if o.coarse {
                // The group's name is on the map; the panel says where to go.
                match &o.best {
                    Some((b, _)) => format!("↓ {b}"),
                    None => format!("↓ {}", o.label.as_deref().unwrap_or_default()),
                }
            } else if o.suggests_change() {
                format!("→ {}", o.best.as_ref().map_or("", |b| b.0.as_str()))
            } else {
                o.label.clone().unwrap_or_default()
            };
            let similar = r.merge.is_some()
                && k != cursor
                && cursor_best.is_some()
                && o.best.as_ref().map(|b| &b.0) == cursor_best.as_ref();
            out.push(format!(
                "{here}{pick}{}C{:<4}{:>6} {status}",
                if similar { "≈" } else { " " },
                o.id,
                o.size
            ));
        }
        Some(out)
    }

    /// Sidebar text for relabel mode.
    pub fn review_lines(&self) -> Option<Vec<String>> {
        let r = self.review.as_ref()?;
        let id = r.cluster();
        let (label, _) = self.cluster_call(id);
        let staged = r.draft.clusters.get(&id);
        let size = self.focus.and_then(|f| {
            let g = self.groups()?;
            Some(g.iter().filter(|&&x| x == f && x != NONE).count())
        });
        let marks = staged.map_or(0, |c| c.marks.len());
        let verdict = staged.and_then(|c| c.verdict.as_ref());
        let merged = r.draft.merges.iter().find(|m| m.clusters.contains(&id));
        let next = if verdict.is_some() || merged.is_some() {
            "next: ] for the next cluster · S when done (p previews)"
        } else if marks == 0 {
            "next: check the fit below, then + / - features (a accepts ?), then L"
        } else {
            "next: L to label it (or K to keep it, M to merge it with others)"
        };
        let mut out = vec![
            format!(
                "relabel · C{id} ({} of {}) · {} decided",
                r.at + 1,
                r.overview.len(),
                r.draft.decided()
            ),
            format!(
                "{} cells · now {}",
                size.unwrap_or(0),
                label.as_deref().unwrap_or("unassigned")
            ),
            next.to_string(),
        ];
        if let Some(members) = label
            .as_deref()
            .and_then(|l| self.data.round.as_ref()?.members_of(l))
        {
            out.push(format!(
                "a coarse group of {} types: label it one of them (tab)",
                members.len()
            ));
        }
        if let Some(v) = staged.and_then(|c| c.verdict.as_ref()) {
            out.push(match v {
                Verdict::Label { label, .. } => format!("staged: label {label}"),
                Verdict::Keep { label, .. } => format!("staged: keep {label}"),
            });
        }
        if let Some(m) = merged {
            let others: Vec<String> = m
                .clusters
                .iter()
                .filter(|&&c| c != id)
                .map(|c| format!("C{c}"))
                .collect();
            out.push(format!(
                "staged: merge with {} as {}",
                others.join(" "),
                m.label
            ));
        }
        out.push(format!("target {}", r.target.as_deref().unwrap_or("-")));
        out.push("markers fit here (tab picks the target):".into());
        for c in &r.candidates {
            let fit = r.fits.iter().find(|f| &f.0 == c);
            let mark = if r.target.as_ref() == Some(c) {
                "▸"
            } else {
                " "
            };
            out.push(match fit {
                Some((_, v, n)) => format!(" {mark} {c:<22} {v:+.2}  ({n} markers)"),
                None => format!(" {mark} {c:<22}     ·"),
            });
        }
        out.push(String::new());
        out.push("   mark feature       here  marker of".into());
        for (k, row) in r.rows.iter().enumerate() {
            let staged = staged.and_then(|c| c.marks.get(row.feature.as_ref()));
            let sign = match (staged, &row.proposal) {
                (Some((Mark::Include { .. }, _)), _) => "+",
                (Some((Mark::Exclude { .. }, _)), _) => "−",
                (None, Some(Mark::Include { .. })) => "?+",
                (None, Some(Mark::Exclude { .. })) => "?−",
                (None, None) => "",
            };
            let cursor = if k == r.row { "▸" } else { " " };
            let score = if row.score.is_finite() {
                format!("{:+5.1}", row.score)
            } else {
                "    ·".into()
            };
            out.push(format!(
                "{cursor}{sign:<2} {:<12} {score}  {}",
                row.feature,
                row.marker_of.join(",")
            ));
        }
        if let Some(near) = &self.near {
            out.push(String::new());
            out.push(format!("near {} (k pins their names)", near.cell));
            let names: Vec<&str> = near.features.iter().map(|(f, _)| f.as_ref()).collect();
            out.push(format!("  {}", names.join(" ")));
        }
        if let Some(p) = &r.preview {
            out.push(String::new());
            out.extend(p.iter().cloned());
        }
        Some(out)
    }
}

/// Lupin's preview, as sidebar lines.
#[must_use]
pub fn preview_lines(v: &serde_json::Value) -> Vec<String> {
    let mut out = vec![format!(
        "preview · {} cells would change{}",
        v["cells_changed"].as_u64().unwrap_or(0),
        if v["rescored"].as_bool() == Some(false) {
            " (calls not rescored)"
        } else {
            ""
        }
    )];
    if let Some(cs) = v["clusters"].as_object() {
        for (id, c) in cs {
            let s = |k: &str| c[k].as_str().unwrap_or("unassigned").to_string();
            let top = c["calls"][0]["label"].as_str().unwrap_or("-");
            out.push(format!(
                "  C{id}: {} → {}   top call {top}",
                s("label_before"),
                s("label_after")
            ));
        }
    }
    if let Some(ms) = v["markers"].as_object() {
        for (t, m) in ms {
            let n = |k: &str| m[k].as_array().map_or(0, Vec::len);
            out.push(format!("  markers {t}: +{} −{}", n("added"), n("dropped")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preview_reads_as_before_and_after_per_cluster() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"rescored":false,"clusters":{"7":{"label_before":"CT1","label_after":"CT2",
                "top_before":"CT1","calls":[{"label":"CT1","score":null,"q":null,"support":null}]}},
                "cells_changed":388,"markers":{"CT2":{"added":["GENE1","GENE2"],"dropped":[]}}}"#,
        )
        .unwrap();
        let lines = preview_lines(&v);
        assert_eq!(
            lines[0],
            "preview · 388 cells would change (calls not rescored)"
        );
        assert!(lines[1].contains("C7: CT1 → CT2"));
        assert!(lines[2].contains("markers CT2: +2 −0"));
    }
}
