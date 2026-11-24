//! Features and cells near a place in the run's embedding: near a set of
//! cells, along a direction, or near a feature.

use super::*;

impl Activity {
    /// Features whose expected level is highest in `cells` (one clicked
    /// cell, or a cluster) relative to the average cell: for an embedding
    /// run, `ρ_g · (mean z of the cells − mean z)`, the genes nearest them in
    /// the embedding. Cells not in the model are skipped; rarely expressed
    /// features are left out as in `contrast`.
    pub fn near_cells<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a str>,
        top: usize,
    ) -> Result<Vec<(Box<str>, f32)>, String> {
        let at = self.cells_z(cells)?;
        let e = self.expected()?;
        let Model::Embedding { z, rho, .. } = &e.model else {
            return Err("neighbouring features need an embedding run".into());
        };
        let d = at - e.z_mean(z);
        let mut scores: Vec<f32> = (rho * d).iter().copied().collect();
        drop_below_median(&mut scores, e.baseline());
        Ok(best(&scores, &e.features.names, top))
    }

    /// The mean cell embedding `z` of `cells` (those in the model), for
    /// scores against the average cell.
    pub fn cells_z<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a str>,
    ) -> Result<Vector, String> {
        let e = self.expected()?;
        let Model::Embedding { z, .. } = &e.model else {
            return Err("placing cells needs an embedding run".into());
        };
        let index = e.cell_index();
        let rows: Vec<usize> = cells
            .into_iter()
            .filter_map(|c| index.get(c).map(|&n| n as usize))
            .collect();
        if rows.is_empty() {
            return Err("none of these cells is in the model".into());
        }
        Ok(row_mean(z, &rows))
    }

    /// The way `cells` (those in the model) point together, the mean of
    /// their unit embeddings: where they sit on the cell map, whose cells
    /// are laid out by direction.
    pub fn cells_direction<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a str>,
    ) -> Result<Vector, String> {
        let e = self.expected()?;
        let Model::Embedding { z, .. } = &e.model else {
            return Err("placing cells needs an embedding run".into());
        };
        let index = e.cell_index();
        let rows: Vec<usize> = cells
            .into_iter()
            .filter_map(|c| index.get(c).map(|&n| n as usize))
            .collect();
        if rows.is_empty() {
            return Err("none of these cells is in the model".into());
        }
        let sum = rows.iter().fold(Vector::zeros(z.ncols()), |s, &r| {
            s + unit(z.row(r).transpose())
        });
        Ok(unit(sum))
    }

    /// The `top` cells whose embedding points nearest `at`'s way (see
    /// [`nearest_directions`]).
    pub fn cells_near(&mut self, at: &Vector, top: usize) -> Result<Vec<(Box<str>, f32)>, String> {
        let e = self.expected()?;
        let Model::Embedding { z, .. } = &e.model else {
            return Err("placing cells needs an embedding run".into());
        };
        nearest_directions(z, at, &e.cells, None, top)
            .ok_or_else(|| "the co-embedding and the cell embedding differ in width".into())
    }

    /// Cells whose expected level of `feature` is highest relative to the
    /// average cell (`ρ_g · (z_cell − mean z)`): the cells nearest that
    /// feature in the embedding, the mirror of `near_cell`.
    pub fn near_feature(
        &mut self,
        feature: &str,
        top: usize,
    ) -> Result<Vec<(Box<str>, f32)>, String> {
        let e = self.expected()?;
        let Model::Embedding { z, rho, .. } = &e.model else {
            return Err("neighbouring cells need an embedding run".into());
        };
        let g = e
            .features
            .index
            .match_gene(feature)
            .ok_or_else(|| format!("no feature {feature} in the model"))?;
        let r = rho.row(g).transpose();
        let mean = e.z_mean(z);
        let mut scores = z * &r;
        scores.add_scalar_mut(-mean.dot(&r));
        Ok(best(scores.as_slice(), &e.cells, top))
    }
}
