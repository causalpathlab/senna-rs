//! The settings panel at the top of the sidebar: what the map shows, each
//! setting a click away from its next (`›`, or the value) or previous (`‹`)
//! value. The keys beside them do the same.

use super::*;
use crate::view::recompute::Step;

/// One setting the panel shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Setting {
    Layout,
    Map,
    Colour,
    Values,
    Labels,
    Dots,
    Near,
    Sidebar,
}

impl Setting {
    pub const ALL: [Setting; 8] = [
        Setting::Layout,
        Setting::Map,
        Setting::Colour,
        Setting::Values,
        Setting::Labels,
        Setting::Dots,
        Setting::Near,
        Setting::Sidebar,
    ];

    fn label(self) -> &'static str {
        match self {
            Setting::Layout => "layout",
            Setting::Map => "map",
            Setting::Colour => "colour",
            Setting::Values => "values",
            Setting::Labels => "labels",
            Setting::Dots => "dots",
            Setting::Near => "near",
            Setting::Sidebar => "sidebar",
        }
    }

    /// The key that steps it, said beside it.
    fn key(self) -> &'static str {
        match self {
            Setting::Layout => "m",
            Setting::Map => "tab",
            Setting::Colour => "c",
            Setting::Values => "o",
            Setting::Labels => "t",
            Setting::Dots => "< >",
            Setting::Near => "",
            Setting::Sidebar => "b",
        }
    }
}

/// Rows the panel takes: a title, a row per setting, a blank line.
pub(super) const PANEL_ROWS: u16 = Setting::ALL.len() as u16 + 2;

/// Columns before a value: the frame, a space and the label.
const LABEL_W: u16 = 10;

impl App {
    /// Whether the sidebar starts with the panel: not in relabel mode, whose
    /// panels need the room.
    pub(super) fn panel_shown(&self) -> bool {
        self.scene.review.is_none() && self.menu.is_none()
    }

    /// A setting's value as shown.
    fn setting_value(&self, s: Setting) -> String {
        let scene = &self.scene;
        match s {
            Setting::Layout => scene.current().method.clone(),
            Setting::Map => scene.current().title().into(),
            Setting::Colour => scene.colour_said().into(),
            Setting::Values => match scene.source {
                crate::view::activity::Source::Expected => "model".into(),
                crate::view::activity::Source::Observed => "counts".into(),
            },
            Setting::Labels => scene.labels_said().into(),
            Setting::Dots => format!("{:.2}×", scene.scale),
            Setting::Near => format!("{} features", scene.near_count),
            Setting::Sidebar => "on".into(),
        }
    }

    /// Step setting `s` to its next (`d` = 1) or previous (-1) value.
    pub(super) fn step_setting(&mut self, s: Setting, d: isize) {
        match s {
            Setting::Layout => self.step_method(d, true),
            Setting::Map => self.step_view(d),
            Setting::Colour => self.change(|sc| sc.step_colour(d)),
            Setting::Values => self.change(Scene::toggle_source),
            Setting::Labels => {
                self.change_text(|sc| sc.step_labels(d));
                self.redecorate();
            }
            Setting::Dots => self.change(|sc| sc.resize(d)),
            Setting::Near => self.change(|sc| sc.step_near_count(d)),
            Setting::Sidebar => {
                self.sidebar = false;
                self.message = Some("sidebar hidden · b to show".into());
            }
        }
    }

    /// The layout method `d` along those of this map's kind in the run; with
    /// `offer`, then those the run lacks, which open the recompute menu set
    /// to make one.
    pub(super) fn step_method(&mut self, d: isize, offer: bool) {
        let spaces = &self.scene.data.spaces;
        let cur = self.scene.current();
        let mut methods: Vec<String> = Vec::new();
        for s in spaces.iter().filter(|s| s.parent.is_none()) {
            if !methods.contains(&s.method) {
                methods.push(s.method.clone());
            }
        }
        let have = methods.len();
        if offer {
            let step = match cur.kind {
                crate::view::SpaceKind::Features => Step::FeatureLayout,
                _ => Step::CellLayout,
            };
            for m in step.settings() {
                if !methods.iter().any(|x| x == m) {
                    methods.push((*m).to_string());
                }
            }
        }
        if methods.len() < 2 {
            self.message = Some("only one layout method in this run".into());
            return;
        }
        let at = methods.iter().position(|m| *m == cur.method).unwrap_or(0) as isize;
        let next = (at + d).rem_euclid(methods.len() as isize) as usize;
        if next >= have {
            let method = methods[next].clone();
            self.open_recompute_with(Some(&method));
            if self.modal.is_some() {
                self.message = Some(format!(
                    "this run has no {method} layout yet · ctrl+r computes it"
                ));
            }
            return;
        }
        let m = &methods[next];
        let same = spaces
            .iter()
            .position(|s| s.method == *m && s.kind == cur.kind);
        let any = spaces.iter().position(|s| s.method == *m);
        if let Some(i) = same.or(any) {
            self.switch_space(i);
        }
    }

    /// The panel's lines for a sidebar `width` columns wide.
    pub(super) fn panel_lines(&self) -> Vec<Line<'static>> {
        let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
        let mut lines = vec![Line::from(vec![
            Span::styled(" Settings", bold),
            Span::styled("  click ‹ › to change", hint()),
        ])];
        for s in Setting::ALL {
            let label = format!(" {:<w$}", s.label(), w = usize::from(LABEL_W) - 2);
            lines.push(Line::from(vec![
                Span::raw(label),
                Span::styled("‹ ", hint()),
                Span::styled(self.setting_value(s), bold),
                Span::styled(" ›", hint()),
                Span::styled(format!("  {}", s.key()), hint()),
            ]));
        }
        lines.push(Line::from(""));
        lines
    }

    /// A click at terminal cell (`col`, `row`). Returns whether it was on
    /// a setting: left of its value steps back, on or right of it forward.
    pub(super) fn panel_click(&mut self, col: u16, row: u16) -> bool {
        let p = self.panel;
        let inside = col >= p.x && col < p.x + p.width && row >= p.y && row < p.y + p.height;
        if !inside {
            return false;
        }
        let Some(&s) = (row - p.y)
            .checked_sub(1)
            .and_then(|i| Setting::ALL.get(usize::from(i)))
        else {
            return true;
        };
        // `‹ ` is the two columns after the label.
        let d = if col < p.x + LABEL_W + 1 { -1 } else { 1 };
        self.step_setting(s, d);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        App::new(
            crate::view::tests::scene(),
            Picker::halfblocks(),
            "r.senna.json".into(),
            "lupin".into(),
        )
    }

    #[test]
    fn every_setting_steps_both_ways_and_comes_back() {
        for s in [Setting::Colour, Setting::Labels, Setting::Dots] {
            let mut a = app();
            let before = a.setting_value(s);
            a.step_setting(s, 1);
            assert_ne!(a.setting_value(s), before, "{s:?}");
            a.step_setting(s, -1);
            assert_eq!(a.setting_value(s), before, "{s:?}");
        }
    }

    #[test]
    fn the_layout_steps_through_the_runs_methods() {
        let mut a = app();
        assert_eq!(a.setting_value(Setting::Layout), "umap");
        a.step_setting(Setting::Layout, 1);
        assert_eq!(a.setting_value(Setting::Layout), "phate");
        a.step_method(-1, false);
        assert_eq!(a.setting_value(Setting::Layout), "umap");
    }

    #[test]
    fn a_click_left_of_a_value_steps_back_and_on_it_forward() {
        let mut a = app();
        a.panel = Rect::new(50, 0, 40, PANEL_ROWS);
        let labels_row = 1 + Setting::ALL
            .iter()
            .position(|s| *s == Setting::Labels)
            .unwrap() as u16;
        assert_eq!(a.setting_value(Setting::Labels), "medium");
        assert!(a.panel_click(50 + LABEL_W, labels_row));
        assert_eq!(a.setting_value(Setting::Labels), "small");
        assert!(a.panel_click(50 + LABEL_W + 4, labels_row));
        assert_eq!(a.setting_value(Setting::Labels), "medium");
        // Off the panel: not taken.
        assert!(!a.panel_click(10, labels_row));
    }
}
