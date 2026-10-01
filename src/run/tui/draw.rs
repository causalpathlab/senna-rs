//! Drawing `senna run`: a tab line, the screen, a message and the keys.
//! Same page and popups as `senna view`.

use super::jobs::State;
use super::{App, Kind, Screen, Target};
use crate::view::color;
use crate::view::tui::draw::{first_row, hint, page, popup, selected, At};
use crate::view::tui::rgb;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn rule() -> Style {
    Style::default().fg(rgb(color::MUTED))
}

/// A line too long for `w` columns, cut with an ellipsis.
fn fit(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(w.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

impl App {
    pub(super) fn draw(&self, f: &mut ratatui::Frame) {
        let area = f.area();
        f.render_widget(Block::default().style(page()), area);
        let [top, body, status] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(3),
        ])
        .areas(area);
        f.render_widget(Paragraph::new(self.tabs()), top);
        let inner = Rect {
            x: body.x + 1,
            width: body.width.saturating_sub(2),
            ..body
        };
        match self.screen {
            Screen::Data => self.draw_data(f, inner),
            Screen::Methods => self.draw_methods(f, inner),
            Screen::Params => self.draw_params(f, inner),
            Screen::Run => self.draw_run(f, inner),
        }
        f.render_widget(Paragraph::new(self.status()), status);
        if let Some(b) = &self.browser {
            let rows = usize::from(area.height).saturating_sub(12).max(3);
            popup(f, area, b.lines(rows), 110, At::Middle, color::TEXT);
        } else if self.confirm.is_some() {
            let rows = usize::from(area.height).saturating_sub(4);
            popup(
                f,
                area,
                self.confirm_lines(rows),
                120,
                At::Middle,
                color::TEXT,
            );
        }
        if let Some(e) = &self.editor {
            let what = match e.target {
                Target::Out(i) => format!(" --out for {}", self.rows[i].form.name),
                Target::Field(m, i) => format!(" --{}", self.rows[m].form.fields[i].long),
                Target::Filter => " flags containing".to_string(),
            };
            let lines = vec![
                Line::from(Span::styled(what, bold())),
                Line::from(format!(" {}▏", e.text)),
                Line::from(Span::styled(" enter keep   esc cancel", hint())),
            ];
            popup(f, area, lines, 90, At::Middle, color::TEXT);
        }
    }

    fn tabs(&self) -> Vec<Line<'static>> {
        let mut spans = vec![Span::styled(" senna run   ", bold())];
        for (i, s) in self.screens().into_iter().enumerate() {
            let label = format!(" {} {} ", i + 1, s.title());
            spans.push(if s == self.screen {
                Span::styled(label, selected())
            } else {
                Span::styled(label, hint())
            });
            spans.push(Span::raw(" "));
        }
        let queued: Vec<&str> = self
            .rows
            .iter()
            .filter(|r| r.on)
            .map(|r| r.form.name.as_str())
            .collect();
        let summary = format!(
            "  {} data · {}",
            self.pairs.len(),
            if queued.is_empty() {
                "no method".to_string()
            } else {
                queued.join(", ")
            }
        );
        spans.push(Span::styled(summary, hint()));
        vec![Line::from(spans), Line::from("")]
    }

    fn status(&self) -> Vec<Line<'static>> {
        let keys = match self.screen {
            Screen::Data => "a add data   b batch files (several: paired by name)   x clear batch   X clear all   d remove   J K reorder   enter methods",
            Screen::Methods => "space queue   enter flags   o change --out",
            Screen::Params => "space / enter change   ← → choices   r reset   R reset all   a advanced   / filter   [ ] method",
            Screen::Run => "s stop   v open the results in senna view",
        };
        vec![
            Line::from(Span::styled(
                format!(" {}", self.message.clone().unwrap_or_default()),
                bold(),
            )),
            Line::from(Span::styled(format!(" {keys}"), hint())),
            Line::from(Span::styled(
                " tab / 1-4 screens   g review and run   q quit",
                hint(),
            )),
        ]
    }

    fn draw_data(&self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines = vec![Line::from(Span::styled(
            "Data files, with the batch labels of each",
            bold(),
        ))];
        lines.push(Line::from(""));
        if self.pairs.is_empty() {
            lines.push(Line::from(Span::styled(
                "  none yet: a opens the file browser",
                hint(),
            )));
        }
        let w = usize::from(area.width);
        let name_w = self
            .pairs
            .iter()
            .map(|p| self.shown(&p.data).chars().count())
            .max()
            .unwrap_or(0)
            .min(w / 2);
        for (i, p) in self.pairs.iter().enumerate() {
            let batch = p
                .batch
                .as_deref()
                .map_or_else(|| "no batch labels".to_string(), |b| self.shown(b));
            let text = format!(
                " {:<name_w$}  {}  ·  {}",
                fit(&self.shown(&p.data), name_w),
                p.info,
                batch
            );
            let style = if i == self.pair_row {
                selected()
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(fit(&text, w), style)));
        }
        if let Some(why) = super::data::batch_problem(&self.pairs) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(why, bold())));
        }
        f.render_widget(Paragraph::new(lines), area);
    }

    fn draw_methods(&self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines = vec![
            Line::from(Span::styled(
                "Methods to fit on these data, each to its own --out",
                bold(),
            )),
            Line::from(""),
        ];
        let w = usize::from(area.width);
        let out_w = self
            .rows
            .iter()
            .map(|r| r.out.chars().count())
            .max()
            .unwrap_or(0)
            .min(32);
        for (i, r) in self.rows.iter().enumerate() {
            let changed = r.form.changed();
            let text = format!(
                " [{}] {:<13} --out {:<out_w$}  {:<11} {}",
                if r.on { "x" } else { " " },
                r.form.name,
                fit(&r.out, out_w),
                if changed == 0 {
                    "defaults".to_string()
                } else {
                    format!("{changed} changed")
                },
                r.form.about.lines().next().unwrap_or_default()
            );
            let style = if i == self.method_row {
                selected()
            } else if r.on {
                bold()
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(fit(&text, w), style)));
            if r.on {
                if let Some(warn) = self.shape_warning(&r.form.name) {
                    lines.push(Line::from(Span::styled(format!("      {warn}"), hint())));
                }
            }
        }
        f.render_widget(Paragraph::new(lines), area);
    }

    fn draw_params(&self, f: &mut ratatui::Frame, area: Rect) {
        let [head, list, help] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(10),
        ])
        .areas(area);
        let mut tabs = Vec::new();
        for m in self.param_methods() {
            let name = format!(" {} ", self.rows[m].form.name);
            tabs.push(if m == self.param_method {
                Span::styled(name, selected())
            } else {
                Span::styled(name, hint())
            });
        }
        let mut sub = format!(
            "{}{}",
            if self.advanced {
                "all flags"
            } else {
                "flags (a shows advanced ones)"
            },
            if self.filter.is_empty() {
                String::new()
            } else {
                format!(" containing “{}”", self.filter)
            }
        );
        sub.insert_str(0, "  ");
        tabs.push(Span::styled(sub, hint()));
        f.render_widget(Paragraph::new(vec![Line::from(tabs), Line::from("")]), head);

        let form = &self.rows[self.param_method].form;
        let blamed = self.blamed_here();
        let visible = self.visible();
        let rows = usize::from(list.height);
        let start = first_row(self.field_row, rows, visible.len());
        let long_w = form
            .fields
            .iter()
            .map(|x| x.long.chars().count() + 2)
            .max()
            .unwrap_or(0)
            .min(36);
        let w = usize::from(list.width);
        let mut lines = Vec::new();
        for (row, &i) in visible.iter().enumerate().skip(start).take(rows) {
            let x = &form.fields[i];
            let mark = if blamed.as_deref() == Some(x.long.as_str()) {
                "!"
            } else if x.required {
                "*"
            } else if x.advanced {
                "·"
            } else {
                " "
            };
            let flag = format!("{mark}--{:<width$}", x.long, width = long_w);
            let value = fit(&x.shown(), 28);
            let help = x.help.lines().next().unwrap_or_default().to_string();
            if row == self.field_row {
                let text = format!("{flag} {value:<28}  {help}");
                lines.push(Line::from(Span::styled(fit(&text, w), selected())));
            } else {
                let vstyle = if x.is_default() { hint() } else { bold() };
                let rest = w.saturating_sub(flag.chars().count() + 31);
                lines.push(Line::from(vec![
                    Span::raw(flag),
                    Span::raw(" "),
                    Span::styled(format!("{value:<28}"), vstyle),
                    Span::raw("  "),
                    Span::styled(fit(&help, rest), hint()),
                ]));
            }
        }
        if visible.is_empty() {
            lines.push(Line::from(Span::styled("  no flag matches", hint())));
        }
        f.render_widget(Paragraph::new(lines), list);

        let mut text = Vec::new();
        if let Some(&i) = visible.get(self.field_row) {
            let x = &form.fields[i];
            let kind = match &x.kind {
                Kind::Flag { .. } => "switch".to_string(),
                Kind::Choice(v) => v
                    .iter()
                    .map(|s| if s.is_empty() { "(unset)" } else { s.as_str() })
                    .collect::<Vec<_>>()
                    .join(" | "),
                Kind::Text => "value".to_string(),
            };
            let default = if x.default.is_empty() {
                "(unset)".to_string()
            } else {
                x.default.clone()
            };
            text.push(Line::from(vec![
                Span::styled(format!("--{}", x.long), bold()),
                Span::styled(format!("   {kind}   default {default}"), hint()),
            ]));
            text.extend(x.long_help.lines().map(|l| Line::from(l.to_string())));
        }
        f.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .block(Block::new().borders(Borders::TOP).border_style(rule())),
            help,
        );
    }

    /// The flag clap complained about on the method shown, if it did.
    /// Checked again only when the command line changes.
    fn blamed_here(&self) -> Option<String> {
        let m = self.param_method;
        let form = &self.rows[m].form;
        let argv = form.argv(&["x".to_string()], &[], "x");
        if let Some((cm, ca, blamed)) = self.blame.borrow().as_ref() {
            if *cm == m && *ca == argv {
                return blamed.clone();
            }
        }
        let blamed = super::form::check(&self.cli, &argv)
            .err()
            .and_then(|why| super::form::blamed(&why, &form.fields).map(str::to_string));
        *self.blame.borrow_mut() = Some((m, argv, blamed.clone()));
        blamed
    }

    fn draw_run(&self, f: &mut ratatui::Frame, area: Rect) {
        let Some(q) = &self.queue else { return };
        let rows = usize::from(area.height).saturating_sub(q.jobs.len() + 3);
        // Copied out, so the worker writing the log is not held up.
        let (states, last, finished, log) = {
            let Ok(s) = q.shared.lock() else { return };
            let skip = s.log.len().saturating_sub(rows);
            let log: Vec<String> = s.log.iter().skip(skip).cloned().collect();
            (s.states.clone(), s.last.clone(), s.finished, log)
        };
        let w = usize::from(area.width);
        let mut lines = vec![
            Line::from(Span::styled(
                if finished {
                    "Finished"
                } else {
                    "Running, one after another"
                },
                bold(),
            )),
            Line::from(""),
        ];
        for ((j, state), last) in q.jobs.iter().zip(&states).zip(&last) {
            let (said, style) = match state {
                State::Waiting => ("waiting".to_string(), hint()),
                State::Running => (format!("running  {last}"), bold()),
                State::Done => ("done".to_string(), Style::default()),
                State::Failed(why) => (format!("failed: {why}"), bold()),
                State::Stopped => ("stopped".to_string(), hint()),
            };
            let text = format!(
                " {:<13} {:<24} {said}",
                j.method,
                fit(&self.shown(&j.manifest()), 24)
            );
            lines.push(Line::from(Span::styled(fit(&text, w), style)));
        }
        lines.push(Line::from(""));
        lines.extend(
            log.iter()
                .map(|l| Line::from(Span::styled(fit(l, w), hint()))),
        );
        f.render_widget(Paragraph::new(lines), area);
    }

    fn confirm_lines(&self, rows: usize) -> Vec<Line<'static>> {
        let Some(planned) = &self.confirm else {
            return Vec::new();
        };
        let mut body: Vec<Line<'static>> = Vec::new();
        for p in planned {
            body.push(Line::from(vec![
                Span::styled(format!(" {}", p.job.method), bold()),
                Span::styled(
                    format!("   recorded in {}", self.shown(&p.job.script())),
                    hint(),
                ),
            ]));
            let lines = super::script::command_lines(&p.job.argv);
            let n = lines.len();
            body.extend(lines.into_iter().enumerate().map(|(k, l)| {
                let indent = if k == 0 { "   " } else { "     " };
                let more = if k + 1 < n { " \\" } else { "" };
                Line::from(format!("{indent}{l}{more}"))
            }));
            if let Some(why) = &p.problem {
                body.push(Line::from(Span::styled(format!("   ✗ {why}"), bold())));
            }
            if let Some(w) = &p.warning {
                body.push(Line::from(Span::styled(format!("   note: {w}"), hint())));
            }
            body.push(Line::from(""));
        }
        let blocked = planned.iter().any(|p| p.problem.is_some());
        let mut out = vec![
            Line::from(Span::styled(
                format!(
                    " Run {} fit{} in turn",
                    planned.len(),
                    if planned.len() == 1 { "" } else { "s" }
                ),
                bold(),
            )),
            Line::from(Span::styled(
                " each command is saved as its {out}.cmd.sh, which will not run over a result",
                hint(),
            )),
            Line::from(""),
        ];
        let room = rows.saturating_sub(out.len() + 3);
        let max = body.len().saturating_sub(room);
        self.confirm_max.set(max);
        let skip = self.confirm_scroll.min(max);
        out.extend(body.into_iter().skip(skip).take(room));
        out.push(Line::from(Span::styled(
            if blocked {
                " enter go to the problem   c copy   ↑ ↓ scroll   esc back"
            } else {
                " enter run   c copy the commands   ↑ ↓ scroll   esc back"
            },
            hint(),
        )));
        if let Some(m) = &self.message {
            out.push(Line::from(Span::styled(format!(" {m}"), bold())));
        }
        out
    }
}
