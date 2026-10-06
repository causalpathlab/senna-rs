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
}

impl Setting {
    pub const ALL: [Setting; 7] = [
        Setting::Layout,
        Setting::Map,
        Setting::Colour,
        Setting::Values,
        Setting::Labels,
        Setting::Dots,
        Setting::Near,
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
        }
    }
}

/// Rows the panel takes: a title, a row per setting, a blank line.
pub(super) const PANEL_ROWS: u16 = Setting::ALL.len() as u16 + 2;

/// Columns before a value: the frame, a space and the label.
const LABEL_W: u16 = 10;

/// The title row: its words, then the control that hides the sidebar.
const TITLE: &str = " Settings";
const TITLE_HINT: &str = "  click ‹ › to change";
const HIDE: &str = "   hide ×";

impl App {
    /// Hide the sidebar (space or `b` shows it again).
    pub(super) fn hide_sidebar(&mut self) {
        self.sidebar = false;
        self.message = Some("sidebar hidden · space shows it".into());
    }

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
        }
    }

    /// The layout method `d` along those of this map's kind in the run; with
    /// `offer`, then those the run lacks, which open the recompute menu set
    /// to make one.
    pub(super) fn step_method(&mut self, d: isize, offer: bool) {
        let cur = self.scene.current();
        let (kind, now) = (cur.kind, cur.method.clone());
        // Each method, and whether the run has it.
        let mut methods: Vec<(String, bool)> = Vec::new();
        for s in self.scene.data.spaces.iter().filter(|s| s.parent.is_none()) {
            if !methods.iter().any(|(m, _)| *m == s.method) {
                methods.push((s.method.clone(), true));
            }
        }
        if offer {
            for m in Step::on_screen(kind, &now).settings() {
                if !methods.iter().any(|(x, _)| x == m) {
                    methods.push(((*m).to_string(), false));
                }
            }
        }
        if methods.len() < 2 {
            self.message = Some("only one layout method in this run".into());
            return;
        }
        let at = methods.iter().position(|(m, _)| *m == now).unwrap_or(0) as isize;
        let (method, have) = methods[(at + d).rem_euclid(methods.len() as isize) as usize].clone();
        if have {
            self.show_layout(&method, kind);
            return;
        }
        self.open_recompute_with(Some(&method));
        if self.modal.is_some() {
            self.message = Some(format!(
                "this run has no {method} layout yet · {} computes it",
                crate::tui::RUN_KEY
            ));
        }
    }

    /// The panel's lines for a sidebar `width` columns wide.
    pub(super) fn panel_lines(&self) -> Vec<Line<'static>> {
        let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
        let mut lines = vec![Line::from(vec![
            Span::styled(TITLE, bold),
            Span::styled(TITLE_HINT, hint()),
            Span::raw(HIDE),
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
        if !p.contains(ratatui::layout::Position::new(col, row)) {
            return false;
        }
        // The title row: its end hides the sidebar.
        let hide_at = p.x + 1 + (TITLE.chars().count() + TITLE_HINT.chars().count()) as u16;
        if row == p.y {
            if col >= hide_at {
                self.hide_sidebar();
            }
            return true;
        }
        let Some(&s) = Setting::ALL.get(usize::from(row - p.y - 1)) else {
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
        // The title row's end hides the sidebar.
        assert!(a.panel_click(50 + 2, 0));
        assert!(a.sidebar);
        assert!(a.panel_click(50 + 38, 0));
        assert!(!a.sidebar);
    }
}
