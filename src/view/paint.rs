//! Turning the scene into layers, labels and a finished frame.

use super::*;

impl Scene {
    /// Layers for the renderer: the backdrop (if any) under the points.
    pub fn layers(&self) -> Vec<Paint<'_>> {
        let space = self.current();
        let mut layers = Vec::new();
        let mut keys = Vec::new();
        if let Some(b) = space.backdrop {
            layers.push(Paint {
                points: &self.data.spaces[b].points,
                groups: None,
                styles: &[],
                focus: None,
                selected: None,
                muted: true,
                size: 0.8 * self.scale,
                levels: None,
                order: None,
            });
            keys.push((b, None, None, 0, 0, true));
        }
        let shown = self.current_shown();
        let merge = self.review.as_ref().and_then(|r| r.merge.as_ref());
        let selected = merge.map(|m| m.levels.as_slice());
        layers.push(Paint {
            points: &space.points,
            groups: self.groups(),
            styles: &self.styles,
            focus: self.focus,
            selected,
            muted: false,
            size: self.scale * if space.backdrop.is_some() { 1.5 } else { 1.0 },
            levels: shown.map(|s| (&s.levels, self.ramp.as_slice())),
            order: None,
        });
        keys.push((
            self.space,
            self.colour,
            self.focus,
            shown.map_or(0, |s| s.id),
            merge.map_or(0, |m| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                m.chosen.hash(&mut h);
                h.finish() | 1
            }),
            false,
        ));
        // The order only changes with these keys, not with the camera: reuse
        // it across pans and zooms instead of re-ranking every point.
        let mut cache = self.orders.borrow_mut();
        for (paint, key) in layers.iter_mut().zip(keys) {
            let order = match cache.iter().find(|(k, _)| *k == key) {
                Some((_, o)) => o.clone(),
                None => {
                    let o = std::sync::Arc::new(paint.draw_order());
                    cache.insert(0, (key, o.clone()));
                    cache.truncate(4);
                    o
                }
            };
            paint.order = Some(order);
        }
        layers
    }

    /// Group label anchors for the space and grouping on screen, computed
    /// once per (space, grouping).
    fn medians_of(
        &self,
        groups: &[u32],
        n: usize,
    ) -> std::cell::Ref<'_, Option<((usize, usize), render::Medians)>> {
        let key = (self.space, self.colour.unwrap_or(usize::MAX));
        if self
            .medians
            .borrow()
            .as_ref()
            .is_none_or(|(k, _)| *k != key)
        {
            *self.medians.borrow_mut() =
                Some((key, group_medians(&self.current().points, groups, n)));
        }
        self.medians.borrow()
    }

    /// Where group `g`'s label sits on the cell map on screen, in data
    /// coordinates.
    pub fn group_centre(&self, g: u32) -> Option<[f32; 2]> {
        if self.current().axis() != Axis::Cells {
            return None;
        }
        let groups = self.groups()?;
        let medians = self.medians_of(groups, self.levels().len());
        let (_, m) = medians.as_ref()?;
        m.get(g as usize).copied().flatten().map(|(xy, _)| xy)
    }

    /// The group whose label is under pixel `(px, py)` in the frame last
    /// drawn with camera `vp`: only labels that frame showed count.
    pub fn group_at(&self, vp: &Viewport, px: f32, py: f32) -> Option<u32> {
        let hits = self.label_hits.borrow();
        let (at, drawn) = hits.as_ref()?;
        if at != vp {
            return None;
        }
        drawn
            .iter()
            .filter(|(_, [x0, y0, x1, y1])| (*x0..=*x1).contains(&px) && (*y0..=*y1).contains(&py))
            .min_by(|a, b| {
                let d = |r: &[f32; 4]| (0.5 * (r[0] + r[2]) - px).hypot(0.5 * (r[1] + r[3]) - py);
                d(&a.1).total_cmp(&d(&b.1))
            })
            .map(|(g, _)| *g)
    }

    /// Where group `g`'s label was drawn in the last frame, if it was.
    #[cfg(test)]
    pub fn label_rect(&self, g: u32) -> Option<[f32; 4]> {
        let hits = self.label_hits.borrow();
        let (_, drawn) = hits.as_ref()?;
        drawn.iter().find(|(k, _)| *k == g).map(|(_, r)| *r)
    }

    /// Text height on the map for a terminal cell `cell_px` tall: text grows
    /// and shrinks with the dots.
    fn text_px(&self, cell_px: f32) -> f32 {
        cell_px * self.scale
    }

    /// The bold font of names on a cell map (group labels, the picked
    /// feature), for text `cell_px` from `text_px`.
    fn label_font(&self, cell_px: f32) -> Font {
        Font::for_cell_height(cell_px * self.text_scale, true)
    }

    /// Labels for a finished frame. Cells: each group's name at its median.
    /// Features: the features' own names (a group of features, such as one
    /// type's markers, is often scattered, so a median would point at
    /// nothing); labelled ones first, and every visible one when few remain.
    pub fn labels(&self, vp: &Viewport, cell_px: f32) -> Vec<Label> {
        let mut out = Vec::new();
        if !self.show_labels {
            return out;
        }
        let space = self.current();
        let groups = self.groups();
        let visible = |i: usize| {
            let (x, y) = vp.to_px(space.points.xy[i]);
            x >= 0.0 && y >= 0.0 && x < vp.w as f32 && y < vp.h as f32
        };

        if space.axis() == Axis::Cells {
            let Some(groups) = groups else { return out };
            let font = self.label_font(cell_px);
            let n = self.levels().len();
            let cached = self.medians_of(groups, n);
            let medians = cached.as_ref().map_or(&[][..], |(_, m)| &m[..]);
            for (g, m) in medians.iter().enumerate() {
                let &Some((xy, size)) = m else { continue };
                if self.focus.is_some_and(|f| f as usize != g) || self.styles[g].hidden {
                    continue;
                }
                let (x, y) = vp.to_px(xy);
                out.push(Label {
                    text: self.levels()[g].to_string(),
                    x,
                    y,
                    ink: self.styles[g].ink,
                    tier: Tier::Map,
                    priority: size as f32,
                    font,
                    group: Some(g as u32),
                });
            }
            return out;
        }

        const MAX_NAMED: usize = 400;
        const NAME_ALL_BELOW: usize = 120;
        let font = Font::for_cell_height(cell_px * 0.8 * self.text_scale, false);
        let group_of = |i: usize| groups.map_or(NONE, |g| g[i]);
        let wanted = |g: u32| match self.focus {
            Some(f) => g == f,
            None => g != NONE,
        };
        let order = space.points.order.iter().map(|&i| i as usize);
        let mut named: Vec<usize> = order
            .clone()
            .filter(|&i| wanted(group_of(i)) && visible(i))
            .take(MAX_NAMED)
            .collect();
        let n_visible = order
            .clone()
            .filter(|&i| visible(i))
            .take(NAME_ALL_BELOW + 1)
            .count();
        if n_visible <= NAME_ALL_BELOW {
            named.extend(order.filter(|&i| !wanted(group_of(i)) && visible(i)));
        }
        let n = named.len();
        for (rank, i) in named.into_iter().enumerate() {
            let g = group_of(i);
            let (x, y) = vp.to_px(space.points.xy[i]);
            if g != NONE && self.styles[g as usize].hidden {
                continue;
            }
            let ink = if g == NONE {
                color::linear_rgb(color::INK)
            } else {
                self.styles[g as usize].ink
            };
            out.push(Label {
                text: space.points.names[i].to_string(),
                x,
                y: y - font.line_height() as f32 * 0.7,
                ink,
                tier: Tier::Map,
                priority: (n - rank) as f32,
                font,
                group: None,
            });
        }
        out
    }

    /// A colour key, for feature views only: their labels are feature names,
    /// so the group a colour stands for has to be spelled out once. Groups
    /// with the most points first.
    pub fn legend(&self) -> Vec<(String, Rgb, Rgb)> {
        const MAX_ENTRIES: usize = 24;
        let (Some(count), Axis::Features) = (self.group_sizes(), self.current().axis()) else {
            return Vec::new();
        };
        let mut ids: Vec<usize> = (0..count.len())
            .filter(|&g| {
                count[g] > 0 && !self.styles[g].hidden && self.focus.is_none_or(|f| f as usize == g)
            })
            .collect();
        ids.sort_by_key(|&g| std::cmp::Reverse(count[g]));
        ids.truncate(MAX_ENTRIES);
        ids.into_iter()
            .map(|g| {
                (
                    self.levels()[g].to_string(),
                    self.styles[g].colour,
                    self.styles[g].ink,
                )
            })
            .collect()
    }

    /// Where the picked feature sits in a feature space, if it is one there.
    pub(super) fn picked_point(&self) -> Option<usize> {
        let Some(Pick::One(name)) = &self.pick else {
            return None;
        };
        let space = self.current();
        if space.axis() != Axis::Features {
            return None;
        }
        let mut cached = self.feature_index.borrow_mut();
        if cached.as_ref().is_none_or(|(s, _)| *s != self.space) {
            *cached = Some((self.space, GeneIndex::build(&space.points.names)));
        }
        cached
            .as_ref()
            .and_then(|(_, index)| index.match_gene(name))
    }

    /// Labels, key and marks over a composited frame.
    pub fn decorate(&self, frame: &mut render::Frame, vp: &Viewport, cell_px: f32) {
        let cell_px = self.text_px(cell_px);
        // The picked feature is marked even with labels off: it is the answer
        // to what was just asked for.
        let mut reserved = Vec::new();
        let mut labels = Vec::new();
        let ink = color::highlight_ink();
        let picked = self.picked_point();
        if let Some(i) = picked {
            let font = self.label_font(cell_px);
            let at = vp.to_px(self.current().points.xy[i]);
            let text = self.current().points.names[i].to_string();
            let r = 0.45 * cell_px;
            mark(
                frame,
                &mut labels,
                at,
                r,
                ink,
                font,
                (Tier::Picked, 0.0),
                text,
            );
        }
        // A clicked cell or feature (and pinned sets): the centre, named,
        // with an edge to each neighbour at its place.
        let (cells, features) = self.near_spaces();
        let font = Font::for_cell_height(cell_px * 0.85 * self.text_scale, true);
        let (centre_r, near_r) = (0.4 * cell_px, 0.25 * cell_px);
        for near in self.locked.iter().chain(&self.near) {
            // Where the centre is drawn, and the point it is, if one.
            let point = |space: Option<usize>| {
                let k = space?;
                let i = self.point_of(k, &near.name)?;
                Some((vp.to_px(self.data.spaces[k].points.xy[i]), Some((k, i))))
            };
            let centre = match near.centre {
                Centre::Cell => point(cells),
                Centre::Feature => point(features),
                Centre::Group { space, xy } => (cells == Some(space)).then(|| (vp.to_px(xy), None)),
                Centre::None => None,
            };
            let place = |space: Option<usize>, list: &'_ [(Box<str>, f32)], tier: Tier| {
                let Some(k) = space else {
                    return Vec::new();
                };
                list.iter()
                    .enumerate()
                    .filter_map(|(rank, (n, _))| {
                        let i = self.point_of(k, n)?;
                        let xy = vp.to_px(self.data.spaces[k].points.xy[i]);
                        Some(((tier, -(rank as f32)), n.to_string(), xy))
                    })
                    .collect::<Vec<_>>()
            };
            let mut placed = place(features, &near.features, Tier::NearFeature);
            placed.extend(place(cells, &near.cells, Tier::NearCell));
            if let Some((at, point)) = centre {
                for &(_, _, to) in &placed {
                    render::draw_edge(frame, at, to, centre_r, near_r, ink, 0.7);
                }
                // A centre that is the picked point is marked already.
                if !point.is_some_and(|(k, i)| k == self.space && picked == Some(i)) {
                    let text = near.name.to_string();
                    mark(
                        frame,
                        &mut labels,
                        at,
                        centre_r,
                        ink,
                        font,
                        (Tier::Centre, 0.0),
                        text,
                    );
                }
            }
            for (rank, text, at) in placed {
                mark(frame, &mut labels, at, near_r, ink, font, rank, text);
            }
        }
        let font = Font::for_cell_height(cell_px * 0.8 * self.text_scale, false);
        if let Some(s) = self.current_shown() {
            reserved.push(render::draw_ramp_key(frame, &s.title, &self.ramp, font));
        } else if self.show_labels {
            reserved.extend(render::draw_legend(frame, &self.legend(), font));
        }
        if self.show_labels {
            labels.extend(self.labels(vp, cell_px));
        }
        let drawn = draw_labels(frame, labels, &reserved);
        // The label under the pointer is framed: a click there picks it.
        let hovered = self.hover.and_then(|h| drawn.iter().find(|(g, _)| *g == h));
        if let Some(&(_, [x0, y0, x1, y1])) = hovered {
            let ink = color::highlight_ink();
            for (a, b) in [
                ((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0)),
            ] {
                render::draw_edge(frame, a, b, 0.0, 0.0, ink, 1.0);
            }
        }
        *self.label_hits.borrow_mut() = Some((*vp, drawn));
    }
}

/// Ring the point at `at` (radius `r`) and name it just above the ring.
#[allow(clippy::too_many_arguments)]
fn mark(
    frame: &mut render::Frame,
    labels: &mut Vec<Label>,
    (x, y): (f32, f32),
    r: f32,
    ink: Rgb,
    font: Font,
    (tier, priority): (Tier, f32),
    text: String,
) {
    render::draw_ring(frame, x, y, r, ink);
    labels.push(Label {
        text,
        x,
        y: y - r - font.line_height() as f32 * 0.6,
        ink,
        tier,
        priority,
        font,
        group: None,
    });
}
