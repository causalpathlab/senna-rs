//! Group-level reads of a feature: contrasts of the focused group against
//! the rest, per-group levels and means, and the suggestions built on them.

use super::*;

impl Activity {
    /// Every feature's score for a view, in the model's feature order, with
    /// the features' names. `-inf` marks a feature left out.
    ///
    /// `universe` names the cells in view (the view `key`); `group` marks, in
    /// the same order, the focused group's cells. With a group, a feature
    /// scores by its expected log fold change, group over the rest of the
    /// view (for an embedding run, `ρ_g · (mean z_group − mean z_rest)`).
    /// Without one, by how much it varies across the view relative to how
    /// much it varies over all cells, among features at least as variable as
    /// the median, which points at what shapes a zoomed-in layout. Where the
    /// run records a per-feature baseline, features below its median are left
    /// out, so rarely expressed features with noisy loadings do not lead.
    ///
    /// A nudge from the model, not a test: `o` shows the observed counts.
    pub fn contrast(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        group: Option<&[bool]>,
    ) -> Result<(Vec<f32>, &[Box<str>]), String> {
        let rows = self.rows(Source::Expected, key, universe)?;
        let e = self.expected()?;
        let pick = |want: Option<bool>| -> Vec<usize> {
            rows.iter()
                .enumerate()
                .filter(|&(k, &r)| {
                    r != NO_ROW && want.is_none_or(|w| group.is_some_and(|g| g[k] == w))
                })
                .map(|(_, &r)| r as usize)
                .collect()
        };
        let scores: Vec<f32> = match (&e.model, group) {
            (_, Some(_)) => {
                let (inside, rest) = (pick(Some(true)), pick(Some(false)));
                if inside.is_empty() || rest.is_empty() {
                    return Err(NO_CONTRAST.into());
                }
                let table = e.table();
                e.score(
                    &row_sum(table, &inside),
                    inside.len(),
                    &row_sum(table, &rest),
                    rest.len(),
                )
            }
            (Model::Embedding { z, rho, .. }, None) => {
                let cells = pick(None);
                // A view of every cell has nothing to compare against, so
                // its variance ranks alone; a zoomed view ranks by how much
                // more a feature varies there than everywhere.
                let whole = cells.len() == z.nrows();
                let here = quadratic_forms(rho, &row_cov(z, &cells));
                let everywhere = e.variance_everywhere.get_or_init(|| {
                    let all: Vec<usize> = (0..z.nrows()).collect();
                    quadratic_forms(rho, &row_cov(z, &all))
                });
                let mut sorted = everywhere.clone();
                sorted.sort_unstable_by(f32::total_cmp);
                let floor = sorted[sorted.len() / 2];
                let mut ratio: Vec<f32> = here
                    .iter()
                    .zip(everywhere)
                    .map(|(&h, &a)| {
                        if a >= floor && a > 0.0 {
                            if whole {
                                h
                            } else {
                                h / a
                            }
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                drop_below_median(&mut ratio, e.baseline());
                ratio
            }
            (Model::Topic { .. }, None) => {
                return Err("on a topic run, focus a group first to get suggestions".into());
            }
        };
        Ok((scores, &e.features.names))
    }

    /// Per-group sums for `contrast` of groups: `groups` gives each point
    /// of `universe` (the view `key`) its group (`>= n_groups` for none).
    /// The one pass over the cells; the contrasts below are cheap after it.
    pub fn group_sums(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        groups: &[u32],
        n_groups: usize,
    ) -> Result<GroupSums, String> {
        let rows = self.rows(Source::Expected, key, universe)?;
        let table = self.expected()?.table();
        let zero = Vector::zeros(table.ncols());
        let mut out = GroupSums {
            sums: vec![zero.clone(); n_groups],
            counts: vec![0; n_groups],
            none: zero,
            none_count: 0,
        };
        for (&r, &g) in rows.iter().zip(groups) {
            if r == NO_ROW {
                continue;
            }
            let (sum, count) = match out.sums.get_mut(g as usize) {
                Some(s) => (s, &mut out.counts[g as usize]),
                None => (&mut out.none, &mut out.none_count),
            };
            for (a, &b) in sum.iter_mut().zip(table.row(r as usize).iter()) {
                *a += b;
            }
            *count += 1;
        }
        Ok(out)
    }

    /// `contrast` for every group at once, each over the other groups'
    /// cells (cells with no group left out). One score vector per group
    /// (empty for an empty group), and the feature names.
    pub fn cluster_contrasts(&mut self, sums: &GroupSums) -> Result<GroupContrasts<'_>, String> {
        let e = self.expected()?;
        let total = sums
            .sums
            .iter()
            .fold(Vector::zeros(sums.none.len()), |a, b| a + b);
        let all: usize = sums.counts.iter().sum();
        let out = sums
            .sums
            .iter()
            .zip(&sums.counts)
            .map(|(s, &n)| {
                if n == 0 {
                    return Vec::new();
                }
                e.score(s, n, &(&total - s), all - n)
            })
            .collect();
        Ok((out, &e.features.names))
    }

    /// Every feature's expected level in each group's mean cell, from
    /// `sums`: feature-major (`features × groups`), `NaN` for an empty group
    /// and, on an embedding run with a baseline, for features below its
    /// median (rarely expressed, with noisy loadings). Embedding runs:
    /// `ρ_g · z̄ + bias_g`; topic runs: `ln(β_g · θ̄)`. And the features.
    pub fn group_levels(&mut self, sums: &GroupSums) -> Result<(Vec<f32>, &[Box<str>]), String> {
        let e = self.expected()?;
        let n = sums.sums.len();
        let d = e.features.names.len();
        let mut out = vec![f32::NAN; d * n];
        for (g, (s, &count)) in sums.sums.iter().zip(&sums.counts).enumerate() {
            if count == 0 {
                continue;
            }
            let mean = s / count as f32;
            let level: Vec<f32> = match (&e.model, e.linear()) {
                (Model::Embedding { rho, bias, .. }, _) => (rho * &mean)
                    .iter()
                    .enumerate()
                    .map(|(f, &v)| v + bias.as_ref().map_or(0.0, |b| b[f]))
                    .collect(),
                (_, Some((_, beta))) => (beta * &mean).iter().map(|&v| v.max(1e-12).ln()).collect(),
                _ => unreachable!("a topic run has linear tables"),
            };
            for (f, v) in level.into_iter().enumerate() {
                out[f * n + g] = v;
            }
        }
        if let Some((b, median)) = e.baseline().filter(|(b, _)| b.len() == d) {
            for (f, &v) in b.iter().enumerate() {
                if v.is_nan() || v < median {
                    out[f * n..(f + 1) * n].fill(f32::NAN);
                }
            }
        }
        Ok((out, &e.features.names))
    }

    /// Each point's topic mixture (θ, summing to one), `k` values a point,
    /// `NaN` for a point the latent has no row for; and `k`. Runs with topics only.
    pub fn mixtures(
        &mut self,
        _key: usize,
        universe: &[Box<str>],
    ) -> Result<(Vec<f32>, usize), String> {
        let m = self.mixture()?;
        let k = m.theta.ncols();
        let mut out = vec![f32::NAN; universe.len() * k];
        for (p, name) in universe.iter().enumerate() {
            let Some(&r) = m.index.get(name) else {
                continue;
            };
            let row = m.theta.row(r as usize);
            let total: f32 = row.iter().sum::<f32>().max(1e-12);
            for (o, &v) in out[p * k..(p + 1) * k].iter_mut().zip(row.iter()) {
                *o = v / total;
            }
        }
        Ok((out, k))
    }

    /// Each feature's mean level in each group of the points in view
    /// (`groups`, `>= n_groups` for none), feature-major (`features.len() ×
    /// n_groups`): the mean `ln(1 + count)` over the group's cells from the
    /// data files, or, when they cannot be read, the model's expected level
    /// of the group's mean cell. Also says which source it used. A feature
    /// neither has is `NaN`.
    pub fn group_means(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        groups: &[u32],
        n_groups: usize,
        features: &[Box<str>],
    ) -> Result<(Vec<f32>, Source), String> {
        if let Ok(v) = self.observed_group_means(key, universe, groups, n_groups, features) {
            return Ok((v, Source::Observed));
        }
        let sums = self.group_sums(key, universe, groups, n_groups)?;
        let (levels, _) = self.group_levels(&sums)?;
        let index = &self.expected()?.features.index;
        let mut out = vec![f32::NAN; features.len() * n_groups];
        for (i, f) in features.iter().enumerate() {
            if let Some(g) = index.match_gene(f) {
                out[i * n_groups..(i + 1) * n_groups]
                    .copy_from_slice(&levels[g * n_groups..(g + 1) * n_groups]);
            }
        }
        Ok((out, Source::Expected))
    }

    pub(super) fn observed_group_means(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        groups: &[u32],
        n_groups: usize,
        features: &[Box<str>],
    ) -> Result<Vec<f32>, String> {
        let rows = self.rows(Source::Observed, key, universe)?;
        let o = self.observed()?;
        // Each data column's group, and each group's size in columns.
        let mut group_of = vec![u32::MAX; o.n_cells];
        let mut size = vec![0usize; n_groups];
        for (&r, &g) in rows.iter().zip(groups) {
            if r != NO_ROW && (g as usize) < n_groups {
                group_of[r as usize] = g;
                size[g as usize] += 1;
            }
        }
        let matched: Vec<(usize, usize)> = features
            .iter()
            .enumerate()
            .filter_map(|(i, f)| Some((i, o.features.index.match_gene(f)?)))
            .collect();
        let mut out = vec![f32::NAN; features.len() * n_groups];
        if matched.is_empty() {
            return Ok(out);
        }
        // One row a read: data-beans' multi-row reads from a .zarr.zip can
        // return wrong values or panic (seen in 0.6.12); single-row reads,
        // as `raw` makes, are stable.
        for &(i, g) in &matched {
            let csr = o
                .data
                .read_rows_csr(std::iter::once(g))
                .map_err(|e| e.to_string())?;
            let mut sum = vec![0f32; n_groups];
            let row = csr.row(0);
            for (&c, &v) in row.col_indices().iter().zip(row.values()) {
                if let Some(s) = sum.get_mut(group_of[c] as usize) {
                    *s += v.ln_1p();
                }
            }
            for (j, (s, &n)) in sum.iter().zip(&size).enumerate() {
                if n > 0 {
                    out[i * n_groups + j] = s / n as f32;
                }
            }
        }
        Ok(out)
    }

    /// `contrast` of the union of the groups `chosen` picks, over every
    /// other cell in view (cells with no group included), from `sums`.
    pub fn union_contrast(
        &mut self,
        sums: &GroupSums,
        chosen: impl Fn(usize) -> bool,
    ) -> Result<(Vec<f32>, &[Box<str>]), String> {
        let e = self.expected()?;
        let mut inside = Vector::zeros(sums.none.len());
        let mut rest = sums.none.clone();
        let (mut n_in, mut n_rest) = (0, sums.none_count);
        for (g, (s, &n)) in sums.sums.iter().zip(&sums.counts).enumerate() {
            if chosen(g) {
                inside += s;
                n_in += n;
            } else {
                rest += s;
                n_rest += n;
            }
        }
        if n_in == 0 || n_rest == 0 {
            return Err(NO_CONTRAST.into());
        }
        Ok((e.score(&inside, n_in, &rest, n_rest), &e.features.names))
    }

    /// The `top` best-scoring features of `contrast`, best first.
    pub fn suggest(
        &mut self,
        key: usize,
        universe: &[Box<str>],
        group: Option<&[bool]>,
        top: usize,
    ) -> Result<Vec<(Box<str>, f32)>, String> {
        let (scores, names) = self.contrast(key, universe, group)?;
        Ok(best(&scores, names, top))
    }
}
