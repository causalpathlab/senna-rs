//! How well types fit a cluster, and the review list: from score vectors in
//! the model's feature order, looked up through columns resolved once.

use super::*;
use rustc_hash::FxHashMap as HashMap;

/// What a review scores against, fixed for its whole visit: the marker
/// table and the model's features.
#[derive(Default)]
pub(crate) struct Evidence {
    /// The marker table, as the features listed under each type.
    pub markers: BTreeMap<String, Vec<Box<str>>>,
    /// Each type (in `markers` order) and its markers' feature columns.
    cols: Vec<(String, Vec<usize>)>,
    /// The model's features, in score order.
    features: Vec<Box<str>>,
    /// Column of each feature name (the last, for a repeated name).
    index: HashMap<Box<str>, usize>,
}

impl Evidence {
    pub(crate) fn new(markers: BTreeMap<String, Vec<Box<str>>>, features: Vec<Box<str>>) -> Self {
        let index: HashMap<Box<str>, usize> = features
            .iter()
            .enumerate()
            .map(|(i, f)| (f.clone(), i))
            .collect();
        let cols = markers
            .iter()
            .map(|(t, ms)| {
                let c = ms.iter().filter_map(|m| index.get(m).copied()).collect();
                (t.clone(), c)
            })
            .collect();
        Self {
            markers,
            cols,
            features,
            index,
        }
    }

    fn score_of(&self, scores: &[f32], f: &str) -> Option<f32> {
        scores.get(*self.index.get(f)?).copied()
    }

    /// How well each type's markers fit, best first: the mean of its best
    /// `FIT_TOP` marker scores (missing ones as zero), with how many it has.
    pub(crate) fn type_fits(&self, scores: &[f32]) -> Vec<(String, f32, usize)> {
        let mut fits: Vec<(String, f32, usize)> = self
            .cols
            .iter()
            .filter_map(|(t, cols)| {
                let mut v: Vec<f32> = cols
                    .iter()
                    .filter_map(|&c| scores.get(c).copied())
                    .filter(|v| v.is_finite())
                    .collect();
                if v.is_empty() {
                    return None;
                }
                v.sort_by(|a, b| b.total_cmp(a));
                let fit = v.iter().take(FIT_TOP).sum::<f32>() / FIT_TOP as f32;
                Some((t.clone(), fit, v.len()))
            })
            .collect();
        fits.sort_by(|a, b| b.1.total_cmp(&a.1));
        fits
    }

    /// The type whose markers fit best.
    pub(crate) fn best_fit(&self, scores: &[f32]) -> Option<(String, f32)> {
        self.type_fits(scores)
            .into_iter()
            .next()
            .map(|(t, fit, _)| (t, fit))
    }

    /// The `top` highest finite scores, best first, ties by name; one per
    /// feature name.
    fn ranked(&self, scores: &[f32], top: usize) -> Vec<(&str, f32)> {
        let mut ranked: Vec<(&str, f32)> = self
            .features
            .iter()
            .zip(scores)
            .enumerate()
            .filter(|&(i, (f, v))| v.is_finite() && self.index.get(f) == Some(&i))
            .map(|(_, (f, &v))| (f.as_ref(), v))
            .collect();
        let order = |a: &(&str, f32), b: &(&str, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(b.0));
        if ranked.len() > top && top > 0 {
            ranked.select_nth_unstable_by(top - 1, order);
            ranked.truncate(top);
        }
        ranked.sort_by(order);
        ranked.truncate(top);
        ranked
    }

    /// Top DE features, then each candidate's best markers, with proposals.
    /// `−` only for the labelled type: a low marker of another type faults
    /// the type.
    pub(crate) fn review_rows(
        &self,
        scores: &[f32],
        candidates: &[String],
        target: Option<&str>,
        labelled: Option<&str>,
    ) -> Vec<Row> {
        let markers = &self.markers;
        let lists = |t: &str, f: &str| {
            markers
                .get(t)
                .is_some_and(|ms| ms.iter().any(|m| m.as_ref() == f))
        };
        let marker_of = |f: &str| -> Vec<String> {
            candidates.iter().filter(|t| lists(t, f)).cloned().collect()
        };
        let mut rows: Vec<Row> = self
            .ranked(scores, TOP_DE)
            .into_iter()
            .map(|(f, v)| Row {
                feature: f.into(),
                score: v,
                marker_of: marker_of(f),
                proposal: (v >= PROPOSE_ADD && !target.is_some_and(|t| lists(t, f)))
                    .then(|| {
                        target.map(|t| Mark::Include {
                            cell_type: t.to_string(),
                        })
                    })
                    .flatten(),
            })
            .collect();
        for t in candidates {
            let Some(ms) = markers.get(t) else { continue };
            let mut listed: Vec<(&Box<str>, f32)> = ms
                .iter()
                .map(|m| (m, self.score_of(scores, m).unwrap_or(f32::NAN)))
                .collect();
            listed.sort_by(|a, b| {
                let key = |v: f32| if v.is_finite() { v } else { f32::NEG_INFINITY };
                key(b.1).total_cmp(&key(a.1))
            });
            for (m, v) in listed.into_iter().take(MARKERS_PER_TYPE) {
                if rows.iter().any(|r| r.feature == *m) {
                    continue;
                }
                let proposal =
                    (labelled == Some(t.as_str()) && v < PROPOSE_DROP).then(|| Mark::Exclude {
                        cell_type: t.clone(),
                    });
                rows.push(Row {
                    feature: m.clone(),
                    score: v,
                    marker_of: marker_of(m),
                    proposal,
                });
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> Evidence {
        let mut markers = BTreeMap::new();
        markers.insert("CT1".to_string(), vec!["GENE1".into(), "GENE2".into()]);
        markers.insert("CT2".to_string(), vec!["GENE3".into(), "GENE9".into()]);
        let features = ["GENE1", "GENE2", "GENE3", "GENE4"]
            .map(Into::into)
            .to_vec();
        Evidence::new(markers, features)
    }

    #[test]
    fn fits_average_the_top_markers_with_missing_as_zero() {
        let e = evidence();
        let fits = e.type_fits(&[1.0, 2.0, 0.5, f32::NEG_INFINITY]);
        assert_eq!(fits[0].0, "CT1");
        assert!((fits[0].1 - 3.0 / FIT_TOP as f32).abs() < 1e-6);
        assert_eq!(fits[0].2, 2);
        assert_eq!((fits[1].0.as_str(), fits[1].2), ("CT2", 1));
        assert!(e.type_fits(&[]).is_empty());
        assert_eq!(e.best_fit(&[1.0, 2.0, 0.5, 0.0]).unwrap().0, "CT1");
    }

    #[test]
    fn rows_rank_by_score_then_name_and_list_candidate_markers() {
        let e = evidence();
        let rows = e.review_rows(
            &[0.2, 0.9, 0.9, f32::NEG_INFINITY],
            &["CT2".to_string()],
            Some("CT2"),
            Some("CT2"),
        );
        let names: Vec<&str> = rows.iter().map(|r| r.feature.as_ref()).collect();
        assert_eq!(names, ["GENE2", "GENE3", "GENE1", "GENE9"]);
        assert_eq!(rows[1].marker_of, ["CT2"]);
        assert!(matches!(rows[0].proposal, Some(Mark::Include { .. })));
        assert!(rows[1].proposal.is_none());
        assert!(rows[3].score.is_nan());
        assert!(rows[3].proposal.is_none());
    }
}
