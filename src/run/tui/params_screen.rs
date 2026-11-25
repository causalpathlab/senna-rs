//! The parameters screen: the shared page, each fit's form, and editing its fields.

use super::form::{Field, Kind, Method};
use super::jobs::Tool;
use super::{data::Pick, form, App, Screen, Target};
use ratatui::crossterm::event::{KeyCode, KeyEvent};

impl App {
    pub(super) fn show_shared(&mut self) {
        self.shared_page = true;
        self.field_row = 0;
    }

    pub(super) fn show_params(&mut self, m: usize) {
        if self.param_method != m || self.shared_page || self.param_step.is_some() {
            self.field_row = 0;
        }
        self.param_method = m;
        self.param_step = None;
        self.shared_page = false;
        self.screen = Screen::Params;
    }

    /// Queued senna fits: what the shared page sets together.
    pub(super) fn shared_rows(&self) -> Vec<usize> {
        (0..self.rows.len())
            .filter(|&i| self.rows[i].on && self.rows[i].tool == Tool::Senna)
            .collect()
    }

    /// The flags at least two queued fits have with the same default, in the
    /// order first met, as a form: each with the value they all hold, or
    /// none (and listed even when advanced) where they differ; a choice
    /// offers only the values every fit takes. A flag whose default differs
    /// between fits means something different in each, so it stays on each
    /// fit's page.
    pub(super) fn shared_form(&self) -> Method {
        let rows = self.shared_rows();
        // Each flag's fields across the fits, in the order first met.
        let mut seen: std::collections::HashMap<&str, usize> = Default::default();
        let mut found: Vec<Vec<&Field>> = Vec::new();
        for &r in &rows {
            for f in &self.rows[r].form.fields {
                match seen.get(f.long.as_str()) {
                    Some(&at) => found[at].push(f),
                    None => {
                        seen.insert(&f.long, found.len());
                        found.push(vec![f]);
                    }
                }
            }
        }
        let fields = found
            .into_iter()
            .filter(|each| each.len() > 1 && each.iter().all(|f| f.default == each[0].default))
            .map(|each| {
                let mut x = each[0].clone();
                // A choice the page sets in every fit: only values all take.
                if let Kind::Choice(values) = &mut x.kind {
                    values.retain(|v| {
                        each.iter()
                            .all(|f| matches!(&f.kind, Kind::Choice(vs) if vs.contains(v)))
                    });
                }
                if each.iter().any(|f| f.value != x.value) {
                    x.value = String::new();
                    // Where the fits differ the page says so, advanced or not.
                    x.advanced = false;
                }
                x
            })
            .collect();
        let mut form = self.rows[rows[0]].form.clone();
        form.name = "shared".into();
        form.fields = fields;
        form
    }

    /// Flag `long` in each queued fit that has it: the fit and its value.
    pub(super) fn shared_values(&self, long: &str) -> Vec<(String, String)> {
        self.shared_rows()
            .into_iter()
            .filter_map(|r| {
                let f = self.rows[r].form.fields.iter().find(|f| f.long == long)?;
                Some((self.rows[r].label.clone(), f.value.clone()))
            })
            .collect()
    }

    /// Whether the queued fits hold flag `long` at different values.
    pub(super) fn differs(&self, long: &str) -> bool {
        let values = self.shared_values(long);
        values.iter().any(|(_, v)| *v != values[0].1)
    }

    /// The form the parameters screen shows: the method's, a step's after
    /// it, or the shared page's.
    pub(super) fn page_form(&self) -> std::borrow::Cow<'_, Method> {
        if self.shared_page {
            std::borrow::Cow::Owned(self.shared_form())
        } else if let Some(step) = self.page_step() {
            std::borrow::Cow::Borrowed(&step.form)
        } else {
            std::borrow::Cow::Borrowed(&self.rows[self.param_method].form)
        }
    }

    /// The field `target` names, as the page shows it: on the shared page
    /// with the value every fit holds, or none where they differ.
    fn field_of(&self, target: &Target) -> Option<Field> {
        match target {
            Target::Field(m, i) => self.rows.get(*m)?.form.fields.get(*i).cloned(),
            Target::Step(m, s, i) => self
                .rows
                .get(*m)?
                .steps
                .get(*s)?
                .form
                .fields
                .get(*i)
                .cloned(),
            Target::Shared(long) => self
                .shared_form()
                .fields
                .into_iter()
                .find(|f| f.long == *long),
            _ => None,
        }
    }

    /// Change the flag `target` names with `change`: a shared flag in every
    /// queued fit that has it, each taking the changed value.
    pub(super) fn change_field(&mut self, target: &Target, change: impl Fn(&mut Field)) {
        match target {
            Target::Field(m, i) => change(&mut self.rows[*m].form.fields[*i]),
            Target::Step(m, s, i) => change(&mut self.rows[*m].steps[*s].form.fields[*i]),
            Target::Shared(long) => {
                let Some(mut x) = self.field_of(target) else {
                    return;
                };
                change(&mut x);
                self.each_shared(long, |f| f.value.clone_from(&x.value));
            }
            _ => {}
        }
    }

    /// The flag `target` names back to its default: a shared flag to each
    /// fit's own, as defaults differ between methods.
    fn reset_field(&mut self, target: &Target) {
        match target {
            Target::Field(m, i) => self.rows[*m].form.fields[*i].reset(),
            Target::Step(m, s, i) => self.rows[*m].steps[*s].form.fields[*i].reset(),
            Target::Shared(long) => self.each_shared(long, Field::reset),
            _ => {}
        }
    }

    /// Flag `long` in every queued fit that has it.
    fn each_shared(&mut self, long: &str, mut change: impl FnMut(&mut Field)) {
        for r in self.shared_rows() {
            if let Some(f) = self.rows[r].form.fields.iter_mut().find(|f| f.long == long) {
                change(f);
            }
        }
    }

    /// Take `text` as the value of the flag `target` names. Left empty, a
    /// flag with a default goes back to it, since leaving it out means that.
    pub(super) fn set_text(&mut self, target: &Target, text: &str) {
        let Some(field) = self.field_of(target) else {
            return;
        };
        if !text.trim().is_empty() {
            self.change_field(target, |f| f.value = text.to_string());
            return;
        }
        match target {
            // Nothing typed over values that differ: each fit keeps its own.
            Target::Shared(long) if self.differs(long) => {
                self.message = Some(format!("--{long}: nothing typed, each fit keeps its value"));
            }
            // Each fit back to its own default, which may differ.
            Target::Shared(long) => {
                self.reset_field(target);
                self.message = Some(format!("--{long} back to each fit's default"));
            }
            _ if !field.default.is_empty() => {
                self.reset_field(target);
                self.message = Some(format!(
                    "--{} cannot be left unset: back to its default {}",
                    field.long, field.default
                ));
            }
            _ => self.change_field(target, |f| f.value.clear()),
        }
    }

    /// Methods the parameters screen steps through: the queued ones, or
    /// all when none is.
    pub(super) fn param_methods(&self) -> Vec<usize> {
        let on: Vec<usize> = (0..self.rows.len()).filter(|&i| self.rows[i].on).collect();
        if on.is_empty() {
            (0..self.rows.len()).collect()
        } else {
            on
        }
    }

    // ───────────── parameters ─────────────

    /// Field indices of the method shown, as listed: advanced ones only
    /// when asked for or changed, narrowed by the filter.
    pub(super) fn visible(&self) -> Vec<usize> {
        let f = self.filter.to_lowercase();
        self.page_form()
            .fields
            .iter()
            .enumerate()
            .filter(|(_, x)| self.advanced || !x.advanced || !x.is_default())
            .filter(|(_, x)| f.is_empty() || x.long.contains(&f))
            .map(|(i, _)| i)
            .collect()
    }

    /// The field under the cursor, by index on the page shown.
    pub(super) fn field(&self) -> Option<usize> {
        self.visible().get(self.field_row).copied()
    }

    /// What a key on field `i` of the page shown changes: that flag of the
    /// method shown, or on the shared page, by name, that flag of every
    /// queued fit.
    fn target(&self, i: usize) -> Target {
        if self.shared_page {
            Target::Shared(self.page_form().fields[i].long.clone())
        } else if let (Some(s), Some(_)) = (self.param_step, self.page_step()) {
            Target::Step(self.param_method, s, i)
        } else {
            Target::Field(self.param_method, i)
        }
    }

    pub(super) fn params_key(&mut self, k: KeyEvent) {
        if let Some(row) =
            legume_tui::keys::list_move(&k, self.field_row, self.visible().len(), true)
        {
            self.field_row = row;
            return;
        }
        match k.code {
            KeyCode::Char('[' | ']') => {
                let pages = self.pages();
                let here = (!self.shared_page).then_some((self.param_method, self.param_step));
                let at = pages.iter().position(|&p| p == here).unwrap_or(0);
                let d = if k.code == KeyCode::Char(']') {
                    1
                } else {
                    pages.len() - 1
                };
                match pages[(at + d) % pages.len()] {
                    Some((m, None)) => self.show_params(m),
                    Some((m, Some(s))) => self.show_step(m, s),
                    None => self.show_shared(),
                }
            }
            KeyCode::Char('a') => {
                let on = self.field();
                self.advanced = !self.advanced;
                // Stay on the same flag when it is still listed.
                self.field_row = on.and_then(|i| self.row_of(i)).unwrap_or(0);
            }
            KeyCode::Char('/') => {
                let text = self.filter.clone();
                self.edit(Target::Filter, text);
            }
            KeyCode::Esc if !self.filter.is_empty() => {
                self.filter.clear();
                self.field_row = 0;
            }
            KeyCode::Char('r') => {
                if let Some(i) = self.field() {
                    self.reset_field(&self.target(i));
                }
            }
            KeyCode::Char('R') => {
                let targets: Vec<Target> = if self.shared_page {
                    let form = self.shared_form();
                    form.fields
                        .into_iter()
                        .map(|f| Target::Shared(f.long))
                        .collect()
                } else {
                    (0..self.page_form().fields.len())
                        .map(|i| self.target(i))
                        .collect()
                };
                for t in &targets {
                    self.reset_field(t);
                }
                self.message = Some(if self.shared_page {
                    "every shared flag back to each fit's default".into()
                } else {
                    "every flag back to its default".into()
                });
            }
            KeyCode::Char('e' | 'f') => {
                let Some(i) = self.field() else { return };
                if self.page_form().fields[i].kind != Kind::Text {
                    return;
                }
                if k.code == KeyCode::Char('f') {
                    self.browse_for(self.target(i), Pick::File);
                } else {
                    let text = self.page_form().fields[i].value.clone();
                    self.edit(self.target(i), text);
                }
            }
            KeyCode::Char(' ') | KeyCode::Left | KeyCode::Right | KeyCode::Enter => {
                let Some(i) = self.field() else { return };
                let x = self.page_form().fields[i].clone();
                let t = self.target(i);
                match (&x.kind, k.code) {
                    (Kind::Flag { .. }, KeyCode::Char(' ') | KeyCode::Enter) => {
                        self.change_field(&t, Field::toggle);
                    }
                    (Kind::Choice(_), KeyCode::Left) => self.change_field(&t, |f| f.cycle(-1)),
                    (Kind::Choice(_), _) => self.change_field(&t, |f| f.cycle(1)),
                    (Kind::Text, KeyCode::Enter) if form::takes_table(&x) => {
                        self.browse_for(self.target(i), Pick::Table);
                    }
                    (Kind::Text, KeyCode::Enter) => self.edit(self.target(i), x.value),
                    _ => {}
                }
            }
            _ => {}
        }
    }
}
