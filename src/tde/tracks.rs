//! Row-grammar track assignment for `senna tde`.
//!
//! A feature axis is one gene axis carrying several **tracks**: the base
//! gene count (`{gene}/count/spliced`, optionally `{gene}/count/unspliced`)
//! plus, for every co-measured modality passed with `--modality`, that
//! modality's two channel rows (`{gene}/m6a/{methylated,unmethylated}`,
//! `{gene}/atoi/{edited,unedited}`, `{gene}/apa/{proximal,distal}`). Every
//! row is `{gene}/{modality}/{channel}`, split by
//! [`data_beans::aux::feature_rows::parse_feature_row`] — the modality and
//! channel are read from the row itself, never from a file name or load
//! order.
//!
//! [`assign_tracks`] is the single place this grammar is enforced: it turns
//! a feature-name axis into a [`TrackPlan`]: per-row track and gene, read
//! for per-gene HVG pooling ([`super::hvg::gem_hvg_row_weights`]) and, via
//! [`TrackPlan::pair_rows`], for cutting the axis to its base rows.
//!
//! No heuristics: a row that does not fit the grammar, carries a subunit
//! (tracks are gene-level only), names a modality outside
//! `{count, m6a, atoi, apa}`, or is a `count` row with a channel other than
//! `spliced`/`unspliced` (so the pooled `{gene}/count/total` track some
//! producers also write is an ERROR here, not a silent third gene) is
//! rejected, naming up to ten offending rows and the total.

use std::collections::BTreeSet;

use data_beans::aux::feature_rows::{parse_feature_row, APA, ATOI, COUNT, M6A, SPLICED, UNSPLICED};
use rustc_hash::FxHashMap;

/// One track: a `(modality, channel)` pair on the feature axis.
#[derive(Clone, Debug)]
pub(crate) struct Track {
    pub id: u32,
    pub modality: Box<str>,
    pub channel: Box<str>,
}

/// The row-grammar-derived plan for one feature axis.
#[derive(Clone, Debug)]
pub(crate) struct TrackPlan {
    /// `tracks[0]` is always `(count, spliced)` — the base track every gene
    /// must have a row on. `tracks[1]` is `(count, unspliced)` when the axis
    /// has one; the rest are sorted by `(modality, channel)`.
    pub tracks: Vec<Track>,
    /// Track id per feature row, in `feature_names` order.
    pub row_track: Vec<u32>,
    /// Dense gene id per row, over ALL tracks. Ids are assigned in first-seen
    /// row order (scanning every track together, not track 0 alone).
    pub row_gene: Vec<u32>,
    /// Gene keys in id order.
    pub gene_names: Vec<Box<str>>,
    /// `base_rows[r]` iff row `r` is on track 0 (`count/spliced`).
    pub base_rows: Vec<bool>,
}

/// Assign every row of a feature axis to a track and a gene, from the row
/// grammar alone. See the module docs for the rejected shapes.
pub(crate) fn assign_tracks(feature_names: &[Box<str>]) -> anyhow::Result<TrackPlan> {
    let mut bad_parse: Vec<usize> = Vec::new();
    let mut bad_subunit: Vec<usize> = Vec::new();
    let mut bad_modality: Vec<usize> = Vec::new();
    let mut bad_count_channel: Vec<usize> = Vec::new();
    let mut parsed: Vec<Option<data_beans::aux::feature_rows::FeatureRow<'_>>> =
        Vec::with_capacity(feature_names.len());

    for (r, name) in feature_names.iter().enumerate() {
        let Some(row) = parse_feature_row(name) else {
            bad_parse.push(r);
            parsed.push(None);
            continue;
        };
        if row.subunit.is_some() {
            bad_subunit.push(r);
            parsed.push(None);
            continue;
        }
        if !matches!(row.modality, COUNT | M6A | ATOI | APA) {
            bad_modality.push(r);
            parsed.push(None);
            continue;
        }
        if row.modality == COUNT && !matches!(row.channel, SPLICED | UNSPLICED) {
            bad_count_channel.push(r);
            parsed.push(None);
            continue;
        }
        parsed.push(Some(row));
    }

    if !bad_parse.is_empty() {
        return Err(rows_error(
            "do not parse as `{gene}/{modality}/{channel}` feature rows",
            &bad_parse,
            feature_names,
        ));
    }
    if !bad_subunit.is_empty() {
        return Err(rows_error(
            "carry a subunit; tracks are gene-level only",
            &bad_subunit,
            feature_names,
        ));
    }
    if !bad_modality.is_empty() {
        return Err(rows_error(
            "name a modality outside {count, m6a, atoi, apa}",
            &bad_modality,
            feature_names,
        ));
    }
    if !bad_count_channel.is_empty() {
        return Err(rows_error(
            "are `count` rows with a channel outside {spliced, unspliced}",
            &bad_count_channel,
            feature_names,
        ));
    }

    let has_base = parsed
        .iter()
        .flatten()
        .any(|row| row.modality == COUNT && row.channel == SPLICED);
    anyhow::ensure!(
        has_base,
        "no base rows: at least one `{{gene}}/count/spliced` row is required"
    );

    // Track ids: 0 = count/spliced, 1 = count/unspliced when present, the
    // rest sorted ascending by (modality, channel).
    let mut pairs: BTreeSet<(Box<str>, Box<str>)> = BTreeSet::new();
    for row in parsed.iter().flatten() {
        pairs.insert((row.modality.into(), row.channel.into()));
    }
    let base_key: (Box<str>, Box<str>) = (COUNT.into(), SPLICED.into());
    let unspliced_key: (Box<str>, Box<str>) = (COUNT.into(), UNSPLICED.into());

    let mut tracks: Vec<Track> = vec![Track {
        id: 0,
        modality: COUNT.into(),
        channel: SPLICED.into(),
    }];
    let mut track_id_of: FxHashMap<(Box<str>, Box<str>), u32> = FxHashMap::default();
    track_id_of.insert(base_key.clone(), 0);

    if pairs.contains(&unspliced_key) {
        let id = tracks.len() as u32;
        tracks.push(Track {
            id,
            modality: COUNT.into(),
            channel: UNSPLICED.into(),
        });
        track_id_of.insert(unspliced_key.clone(), id);
    }

    for (modality, channel) in pairs
        .iter()
        .filter(|k| **k != base_key && **k != unspliced_key)
    {
        let id = tracks.len() as u32;
        tracks.push(Track {
            id,
            modality: modality.clone(),
            channel: channel.clone(),
        });
        track_id_of.insert((modality.clone(), channel.clone()), id);
    }

    // Genes: first-seen row order, scanning every track together.
    let mut gene_ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut gene_names: Vec<Box<str>> = Vec::new();
    let mut row_track: Vec<u32> = Vec::with_capacity(feature_names.len());
    let mut row_gene: Vec<u32> = Vec::with_capacity(feature_names.len());
    let mut base_rows: Vec<bool> = Vec::with_capacity(feature_names.len());

    for row in parsed.iter().flatten() {
        let gid = *gene_ids.entry(row.gene.into()).or_insert_with(|| {
            let g = gene_names.len() as u32;
            gene_names.push(row.gene.into());
            g
        });
        let tid = track_id_of[&(
            Box::<str>::from(row.modality),
            Box::<str>::from(row.channel),
        )];
        row_gene.push(gid);
        row_track.push(tid);
        base_rows.push(tid == 0);
    }

    Ok(TrackPlan {
        tracks,
        row_track,
        row_gene,
        gene_names,
        base_rows,
    })
}

/// Build an error naming up to ten offending rows and the total count.
fn rows_error(what: &str, rows: &[usize], feature_names: &[Box<str>]) -> anyhow::Error {
    let total = rows.len();
    let preview: Vec<&str> = rows
        .iter()
        .take(10)
        .map(|&r| feature_names[r].as_ref())
        .collect();
    let more = total.saturating_sub(preview.len());
    if more > 0 {
        anyhow::anyhow!("{total} feature row(s) {what}: {preview:?} (+{more} more)")
    } else {
        anyhow::anyhow!("{total} feature row(s) {what}: {preview:?}")
    }
}

impl TrackPlan {
    /// For every gene with a base (`count/spliced`) row, ascending by that
    /// row: the base row, and the gene's row on `track`, if it has one. What
    /// `graph_embedding_util::split_displaced` cuts the axis by.
    pub(crate) fn pair_rows(&self, track: u32) -> anyhow::Result<(Vec<usize>, Vec<Option<usize>>)> {
        let mut on_track: FxHashMap<u32, usize> = FxHashMap::default();
        for (r, (&t, &g)) in self.row_track.iter().zip(&self.row_gene).enumerate() {
            if t == track {
                anyhow::ensure!(
                    on_track.insert(g, r).is_none(),
                    "gene {} has more than one row on track {track}",
                    self.gene_names[g as usize]
                );
            }
        }
        let base = self.rows_of(0);
        let paired = base
            .iter()
            .map(|&r| on_track.get(&self.row_gene[r]).copied())
            .collect();
        Ok((base, paired))
    }

    /// Row indices on `track`, ascending.
    pub(crate) fn rows_of(&self, track: u32) -> Vec<usize> {
        self.row_track
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == track)
            .map(|(r, _)| r)
            .collect()
    }
}

#[cfg(test)]
#[path = "tracks/tests.rs"]
mod tests;
