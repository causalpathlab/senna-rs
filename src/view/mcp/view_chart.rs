//! `view_map`'s `chart`: a chart drawn in place of the map, as `H` draws
//! them in the view.

use crate::view::chart::Kind;
use crate::view::Scene;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// A chart drawn in place of the map ([`Kind`], by its name), or `off`:
/// the map.
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(try_from = "String", into = "String")]
pub(super) struct ChartArg(Option<Kind>);

impl TryFrom<String> for ChartArg {
    type Error = String;

    fn try_from(name: String) -> Result<Self, String> {
        use clap::ValueEnum;
        if name == "off" {
            return Ok(Self(None));
        }
        Kind::from_str(&name, false)
            .map(|k| Self(Some(k)))
            .map_err(|_| {
                let all: Vec<&str> = Kind::value_variants().iter().map(|k| k.slug()).collect();
                format!("no chart {name}: {}, or off", all.join(", "))
            })
    }
}

impl From<ChartArg> for String {
    fn from(c: ChartArg) -> Self {
        c.0.map_or("off", Kind::slug).into()
    }
}

impl ChartArg {
    /// The chart `scene` draws.
    pub(super) fn of(scene: &Scene) -> Self {
        Self(scene.chart.as_ref().map(|c| c.kind))
    }

    /// Whether the run can draw it: the stage-2 charts need a stage 2.
    pub(super) fn check(self, has_stage2: bool) -> Result<()> {
        ensure!(
            has_stage2 || !self.0.is_some_and(Kind::needs_stage2),
            crate::view::stage2::NO_STAGE2
        );
        Ok(())
    }
}

/// Draw `chart` (made at once), or the map again; a chart on is made again
/// for what it depends on each time the view is drawn.
pub(super) fn apply_chart(scene: &mut Scene, chart: Option<ChartArg>) {
    match chart {
        Some(ChartArg(Some(kind))) if scene.chart.as_ref().map(|c| c.kind) != Some(kind) => {
            scene.start_chart(kind);
        }
        Some(ChartArg(None)) => scene.chart = None,
        _ => scene.refresh_chart(),
    }
}
