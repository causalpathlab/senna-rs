//! Zooming into one group's own layout, and back out.

use super::*;

impl Scene {
    /// The focused group's zoom request; the geometry table is read once.
    pub fn zoom_request(&mut self) -> Result<ZoomRequest, String> {
        if self.current().axis() != Axis::Cells {
            return Err("zoom into a group from a cell view".into());
        }
        let (Some(f), Some(groups)) = (self.focus, self.groups()) else {
            return Err("focus a group first ([ ] or click), then press z".into());
        };
        let points = &self.current().points;
        let names: Vec<Box<str>> = groups
            .iter()
            .zip(&points.names)
            .filter(|&(&g, _)| g == f)
            .map(|(_, n)| n.clone())
            .collect();
        let label = self.levels()[f as usize].to_string();
        if self.geometry.is_none() {
            let (m, dir) = self
                .data
                .run
                .as_ref()
                .ok_or("no manifest to read the embedding from")?;
            let g = sublayout::Geometry::load(m, dir).map_err(|e| e.to_string())?;
            self.geometry = Some(std::sync::Arc::new(g));
        }
        let geometry = self.geometry.clone().expect("just loaded");
        Ok(ZoomRequest {
            label,
            names,
            geometry,
        })
    }

    /// Add a layout of one group, zoomed from `parent`, and switch to it.
    pub fn add_zoomed(
        &mut self,
        parent: usize,
        label: &str,
        names: Vec<Box<str>>,
        xy: Vec<[f32; 2]>,
    ) {
        let method = format!("{} › {label}", self.data.spaces[parent].method);
        self.data.spaces.push(data::Space {
            method,
            kind: SpaceKind::Cells,
            points: std::sync::Arc::new(data::Points::new(names, xy)),
            backdrop: None,
            parent: Some(parent),
        });
        self.focus = None;
        self.set_space(self.data.spaces.len() - 1);
    }

    /// Back to the layout this one was zoomed from. Returns whether there was one.
    pub fn zoom_out(&mut self) -> bool {
        match self.current().parent {
            Some(p) => {
                self.set_space(p);
                true
            }
            None => false,
        }
    }

    /// The top of the current zoom chain.
    pub(super) fn root(&self) -> usize {
        let mut at = self.space;
        while let Some(p) = self.data.spaces[at].parent {
            at = p;
        }
        at
    }

    /// Back to the top of the zoom chain. Returns whether it moved.
    pub fn zoom_to_root(&mut self) -> bool {
        let at = self.root();
        if at == self.space {
            return false;
        }
        self.set_space(at);
        true
    }
}
