//! Post-fit steps on the screens: under each queued fit on the Methods
//! screen, its steps, queued with space as a fit is; each queued step's
//! flags a page of the parameters screen, after its fit's.

use super::steps::Step;
use super::{App, Screen};
use ratatui::crossterm::event::{KeyCode, KeyEvent};

/// A row of the Methods screen: a method, and one of its steps.
pub(super) type Entry = (usize, Option<usize>);

impl App {
    /// The Methods screen's rows: every method, each queued one with its
    /// steps under it.
    pub(super) fn method_entries(&self) -> Vec<Entry> {
        self.rows
            .iter()
            .enumerate()
            .flat_map(|(i, r)| {
                let steps = if r.on { r.steps.len() } else { 0 };
                std::iter::once((i, None)).chain((0..steps).map(move |s| (i, Some(s))))
            })
            .collect()
    }

    /// The parameters screen's pages in `[ ]`'s order: the shared one
    /// (`None`) when two fits or more are queued, then each method and its
    /// queued steps.
    pub(super) fn pages(&self) -> Vec<Option<Entry>> {
        let shared = (self.shared_rows().len() > 1).then_some(None);
        let each = self.param_methods().into_iter().flat_map(|m| {
            let r = &self.rows[m];
            let on = r.steps.iter().enumerate().filter(|(_, s)| s.on && r.on);
            std::iter::once(Some((m, None))).chain(on.map(move |(s, _)| Some((m, Some(s)))))
        });
        shared.into_iter().chain(each).collect()
    }

    /// The step whose page the parameters screen shows, while it is queued.
    pub(super) fn page_step(&self) -> Option<&Step> {
        let r = self.rows.get(self.param_method)?;
        let step = r.steps.get(self.param_step?)?;
        (r.on && step.on && !self.shared_page).then_some(step)
    }

    /// Show step `s` of method `m` on the parameters screen.
    pub(super) fn show_step(&mut self, m: usize, s: usize) {
        if (self.param_method, self.param_step) != (m, Some(s)) || self.shared_page {
            self.field_row = 0;
        }
        (self.param_method, self.param_step) = (m, Some(s));
        self.shared_page = false;
        self.screen = Screen::Params;
    }

    /// Keep both screens off a step that is no longer listed: its fit or
    /// itself unqueued.
    pub(super) fn settle_steps(&mut self) {
        if self.param_step.is_some() && self.page_step().is_none() {
            self.param_step = None;
            self.field_row = 0;
        }
        let r = &self.rows[self.methods_cursor.0];
        if self
            .methods_cursor
            .1
            .is_some_and(|s| !r.on || s >= r.steps.len())
        {
            self.methods_cursor.1 = None;
        }
    }

    /// A key on a step's row of the Methods screen, as on a method's: space
    /// queues it, enter queues it and shows its flags. Whether it was one.
    pub(super) fn step_key(&mut self, k: KeyEvent) -> bool {
        let (m, Some(s)) = self.methods_cursor else {
            return false;
        };
        let step = &mut self.rows[m].steps[s];
        match k.code {
            KeyCode::Char(' ') => step.on = !step.on,
            KeyCode::Enter | KeyCode::Right => {
                step.on = true;
                self.show_step(m, s);
            }
            KeyCode::Char('o') => {
                self.message = Some("a step writes beside its fit: o on the fit's row".into());
            }
            _ => return false,
        }
        true
    }
}
