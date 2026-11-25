//! `view_map`'s `chart`: a chart drawn in place of the map, as `H` draws
//! them in the view.

use crate::view::chart::Kind;
use crate::view::Scene;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// A chart drawn in place of the map, or the map.
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum ChartArg {
    Structure,
    Heatmap,
    Gains,
    Volcano,
    Off,
}

impl ChartArg {
    fn kind(self) -> Option<Kind> {
        match self {
            Self::Structure => Some(Kind::Structure),
            Self::Heatmap => Some(Kind::Heatmap),
            Self::Gains => Some(Kind::Gains),
            Self::Volcano => Some(Kind::Volcano),
            Self::Off => None,
        }
    }

    /// The chart `scene` draws.
    pub(super) fn of(scene: &Scene) -> Self {
        match scene.chart.as_ref().map(|c| c.kind) {
            Some(Kind::Structure) => Self::Structure,
            Some(Kind::Heatmap) => Self::Heatmap,
            Some(Kind::Gains) => Self::Gains,
            Some(Kind::Volcano) => Self::Volcano,
            None => Self::Off,
        }
    }

    /// Whether the run can draw it: the stage-2 charts need a stage 2.
    pub(super) fn check(self, has_stage2: bool) -> Result<()> {
        if matches!(self, Self::Gains | Self::Volcano) {
            ensure!(has_stage2, crate::view::stage2::NO_STAGE2);
        }
        Ok(())
    }
}

/// Draw `chart` (made at once), or the map again; a chart is made again
/// for what it depends on each time the view is drawn.
pub(super) fn apply_chart(scene: &mut Scene, chart: Option<ChartArg>) {
    match chart.map(ChartArg::kind) {
        Some(Some(kind)) if scene.chart.as_ref().map(|c| c.kind) != Some(kind) => {
            scene.start_chart(kind);
        }
        Some(None) => scene.chart = None,
        _ => {}
    }
    scene.refresh_chart();
}
