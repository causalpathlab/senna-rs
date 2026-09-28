//! Merge mode: choosing clusters to stage as one.

use super::*;

impl Scene {
    /// Enter merge mode with the current cluster chosen.
    pub fn begin_merge(&mut self) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let id = r.cluster();
        r.merge = Some(MergeSel {
            cursor: r.at,
            chosen: std::iter::once(id).collect(),
            levels: Vec::new(),
            best: None,
        });
        self.refresh_merge();
    }

    /// Choose or unchoose the cluster under the merge cursor.
    pub fn toggle_merge_cursor(&mut self) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let Some(m) = r.merge.as_mut() else { return };
        let id = r.overview[m.cursor].id;
        if !m.chosen.remove(&id) {
            m.chosen.insert(id);
        }
        self.refresh_merge();
    }

    /// Recompute the chosen clusters' joint fit, and which to draw.
    fn refresh_merge(&mut self) {
        let Some(li) = self.colour else { return };
        let Some(chosen) = self
            .review
            .as_ref()
            .and_then(|r| r.merge.as_ref())
            .map(|m| m.chosen.clone())
        else {
            return;
        };
        let ids = &self.data.labels[li].ids;
        let levels: Vec<bool> = ids.iter().map(|id| chosen.contains(id)).collect();
        let scores = match self.review_sums() {
            Some((activity, Ok(sums))) => activity
                .union_contrast(sums, |g| levels.get(g).copied().unwrap_or(false))
                .ok()
                .map(|(v, _)| v),
            _ => None,
        };
        let best = scores.and_then(|v| self.evidence().best_fit(&v));
        if let Some(m) = self.review.as_mut().and_then(|r| r.merge.as_mut()) {
            m.levels = levels;
            m.best = best;
        }
    }

    /// Stage a merge of the clusters chosen in merge mode, and leave it.
    pub fn stage_merge(&mut self, label: String, rationale: String) {
        let Some(r) = self.review.as_mut() else {
            return;
        };
        let Some(m) = r.merge.take() else { return };
        r.draft.merges.push(Merge {
            clusters: m.chosen.into_iter().collect(),
            label,
            rationale,
        });
    }
}
