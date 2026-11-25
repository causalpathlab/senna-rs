//! The critique's output tables: per-model labels, per-pair judgements and
//! the summary.

use super::{
    names_of, pair_keys, ranks::far_share, ranks::known_ranks, ranks::median, ranks::merge_rate,
    ranks::Bounds, Level,
};
use legume_numeric::matrix::parquet::{write_named_table, Column};

/// One model's labels, as `{out}.critique.labels.{model}.parquet`: the drawn
/// questions the answer made merges, the pairs that model is to push apart.
/// `channel` is `A` throughout (pseudobulk pairs, plan §7); channel B, gene
/// pairs, is not built.
#[derive(Default)]
pub(super) struct LabelTable {
    level: Vec<i32>,
    pb_a: Vec<i32>,
    pb_b: Vec<i32>,
    weight: Vec<f32>,
    answer_rank: Vec<f32>,
    own_rank: Vec<i32>,
    answered_by: Vec<Box<str>>,
}

impl LabelTable {
    /// Add model `mi`'s drawn questions that are merges; returns how many.
    pub(super) fn push(
        &mut self,
        l: usize,
        lv: &Level,
        j: &JudgedLevel,
        mi: usize,
        asked: &[usize],
        names: &[String],
    ) -> usize {
        let committee = &j.committees[mi];
        let by = names_of(names, committee.iter().copied());
        let before = self.level.len();
        for &c in asked.iter().filter(|&&c| j.merged[mi][c]) {
            let (a, b) = (j.pairs[c].0 as usize, j.pairs[c].1 as usize);
            self.level.push(l as i32);
            self.pb_a.push(lv.pb_id[a] as i32);
            self.pb_b.push(lv.pb_id[b] as i32);
            self.weight.push(far_share(j.ranks, committee, c, j.bounds));
            self.answer_rank
                .push(j.answered[mi][c].map_or(f32::NAN, |x| x as f32));
            self.own_rank.push(j.ranks[mi][c] as i32);
            self.answered_by.push(by.clone());
        }
        self.level.len() - before
    }

    pub(super) fn write(&self, path: &str) -> anyhow::Result<()> {
        let key = pair_keys(&self.level, &self.pb_a, &self.pb_b);
        let channel: Vec<Box<str>> = vec![Box::from("A"); self.level.len()];
        let cols: Vec<(Box<str>, Column)> = vec![
            ("channel".into(), Column::Str(&channel)),
            ("level".into(), Column::I32(&self.level)),
            ("pb_a".into(), Column::I32(&self.pb_a)),
            ("pb_b".into(), Column::I32(&self.pb_b)),
            ("weight".into(), Column::F32(&self.weight)),
            ("answer_rank".into(), Column::F32(&self.answer_rank)),
            ("own_rank".into(), Column::I32(&self.own_rank)),
            ("answered_by".into(), Column::Str(&self.answered_by)),
        ];
        write_named_table(path, "pair", &key, &cols)
    }
}

/// One level's candidate pairs and what the models said about them.
pub(super) struct JudgedLevel<'a> {
    pub(super) bounds: Bounds,
    pub(super) pairs: &'a [(u32, u32)],
    /// `ranks[model][pair]`, `u32::MAX` where unknown.
    pub(super) ranks: &'a [Vec<u32>],
    /// Per model, the models that answered it.
    pub(super) committees: &'a [Vec<usize>],
    /// `answered[model][pair]`: the committee's median rank.
    pub(super) answered: &'a [Vec<Option<f64>>],
    /// `merged[model][pair]`.
    pub(super) merged: &'a [Vec<bool>],
}

/// One row per candidate pair, as `{out}.critique.pairs.parquet`.
#[derive(Default)]
pub(super) struct PairTable {
    level: Vec<i32>,
    pb_a: Vec<i32>,
    pb_b: Vec<i32>,
    n_cells_a: Vec<i32>,
    n_cells_b: Vec<i32>,
    /// Per model; `NaN` where the model has no cells in either pseudobulk.
    rank: Vec<Vec<f32>>,
    median_rank: Vec<f32>,
    /// Largest minus smallest known rank. Not `rank_spread`: per-model columns
    /// are `rank_{model}`, and a run named `spread` would collide.
    spread: Vec<f32>,
    /// The models that merged the pair, comma-separated; empty when none did.
    merged_by: Vec<Box<str>>,
    /// With `--cell-labels`: the overlap of the two label compositions.
    pub(super) label_overlap: Vec<f32>,
}

impl PairTable {
    /// Add one level's pairs.
    pub(super) fn push_level(&mut self, l: usize, lv: &Level, j: &JudgedLevel, names: &[String]) {
        let JudgedLevel {
            pairs,
            ranks,
            merged,
            ..
        } = *j;
        let all: Vec<usize> = (0..ranks.len()).collect();
        self.rank.resize_with(ranks.len(), Vec::new);
        for (c, &(a, b)) in pairs.iter().enumerate() {
            let sorted = known_ranks(ranks, &all, c);
            let (a, b) = (a as usize, b as usize);
            self.level.push(l as i32);
            self.pb_a.push(lv.pb_id[a] as i32);
            self.pb_b.push(lv.pb_id[b] as i32);
            self.n_cells_a.push(lv.n_cells[a] as i32);
            self.n_cells_b.push(lv.n_cells[b] as i32);
            for (col, rk) in self.rank.iter_mut().zip(ranks) {
                col.push(if rk[c] == u32::MAX {
                    f32::NAN
                } else {
                    rk[c] as f32
                });
            }
            self.median_rank
                .push(median(&sorted).map_or(f32::NAN, |m| m as f32));
            self.spread.push(match (sorted.first(), sorted.last()) {
                (Some(lo), Some(hi)) => (hi - lo) as f32,
                _ => f32::NAN,
            });
            self.merged_by
                .push(names_of(names, (0..merged.len()).filter(|&m| merged[m][c])));
        }
    }

    pub(super) fn write(&self, path: &str, names: &[String]) -> anyhow::Result<()> {
        let key = pair_keys(&self.level, &self.pb_a, &self.pb_b);
        let rank_names: Vec<Box<str>> = names.iter().map(|n| format!("rank_{n}").into()).collect();
        let mut cols: Vec<(Box<str>, Column)> = vec![
            ("level".into(), Column::I32(&self.level)),
            ("pb_a".into(), Column::I32(&self.pb_a)),
            ("pb_b".into(), Column::I32(&self.pb_b)),
            ("n_cells_a".into(), Column::I32(&self.n_cells_a)),
            ("n_cells_b".into(), Column::I32(&self.n_cells_b)),
        ];
        for (name, col) in rank_names.iter().zip(&self.rank) {
            cols.push((name.clone(), Column::F32(col)));
        }
        cols.push(("median_rank".into(), Column::F32(&self.median_rank)));
        cols.push(("spread".into(), Column::F32(&self.spread)));
        cols.push(("merged_by".into(), Column::Str(&self.merged_by)));
        if !self.label_overlap.is_empty() {
            cols.push(("label_overlap".into(), Column::F32(&self.label_overlap)));
        }
        write_named_table(path, "pair", &key, &cols)
    }
}

/// One row per model and level, as `{out}.critique.summary.parquet`.
#[derive(Default)]
pub(super) struct SummaryTable {
    model: Vec<Box<str>>,
    level: Vec<i32>,
    n_pb: Vec<i32>,
    far_rank: Vec<i32>,
    near: Vec<i32>,
    merges: Vec<i32>,
    /// Merges over near pairs: the report card.
    merge_rate: Vec<f32>,
    /// With `--cell-labels`: mean label overlap of the merged pairs, and of
    /// all the model's near pairs (the base a merge is judged against).
    pub(super) merge_overlap: Vec<f32>,
    pub(super) near_overlap: Vec<f32>,
}

impl SummaryTable {
    pub(super) fn push(
        &mut self,
        level: usize,
        model: &str,
        n_pb: usize,
        b: Bounds,
        near: usize,
        merges: usize,
    ) {
        self.model.push(model.into());
        self.level.push(level as i32);
        self.n_pb.push(n_pb as i32);
        self.far_rank.push(b.far as i32);
        self.near.push(near as i32);
        self.merges.push(merges as i32);
        self.merge_rate.push(merge_rate(merges, near));
    }

    pub(super) fn write(&self, path: &str) -> anyhow::Result<()> {
        let mut cols: Vec<(Box<str>, Column)> = vec![
            ("level".into(), Column::I32(&self.level)),
            ("n_pb".into(), Column::I32(&self.n_pb)),
            ("far_rank".into(), Column::I32(&self.far_rank)),
            ("near_pairs".into(), Column::I32(&self.near)),
            ("merges".into(), Column::I32(&self.merges)),
            ("merge_rate".into(), Column::F32(&self.merge_rate)),
        ];
        if !self.merge_overlap.is_empty() {
            cols.push(("merge_overlap".into(), Column::F32(&self.merge_overlap)));
            cols.push(("near_overlap".into(), Column::F32(&self.near_overlap)));
        }
        write_named_table(path, "model", &self.model, &cols)
    }
}
