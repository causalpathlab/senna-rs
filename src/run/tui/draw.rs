//! Drawing `senna run`: a tab line, the screen, a message and the keys.
//! Same page and popups as `senna view`.

use super::jobs::{State, Tool, CLONES_FLAG};
use super::{batches, App, Batch, Kind, Screen, Target};
use crate::tui::child::Progress;
use crate::tui::style::{bold, first_row, hint, page, popup, rgb, selected, At, MUTED, TEXT};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

/// A progress bar `width` cells wide with its count and what it counts;
/// a spinner, with no count, as what it is doing.
fn gauge(p: &Progress, width: usize) -> String {
    if p.len == 0 {
        return format!("… {}", p.what);
    }
    let filled = (width as u64 * p.pos.min(p.len) / p.len) as usize;
    format!(
        "{}{} {}/{} {}",
        "█".repeat(filled),
        "░".repeat(width - filled),
        p.pos,
        p.len,
        p.what
    )
}

fn rule() -> Style {
    Style::default().fg(rgb(MUTED))
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
            let lines = b.lines(usize::from(area.height).saturating_sub(2));
            popup(f, area, lines, 110, At::Middle, TEXT);
        } else if self.confirm.is_some() {
            let rows = usize::from(area.height).saturating_sub(4);
            popup(f, area, self.confirm_lines(rows), 120, At::Middle, TEXT);
        } else if self.asking() {
            let rows = usize::from(area.height).saturating_sub(4);
            popup(f, area, self.clones_lines(rows), 100, At::Middle, TEXT);
        } else if let Some((row, at)) = self.relabel {
            let rows = usize::from(area.height).saturating_sub(10).max(3);
            popup(
                f,
                area,
                self.relabel_lines(row, at, rows),
                90,
                At::Middle,
                TEXT,
            );
        }
        if let Some(e) = &self.editor {
            let what = match e.target {
                Target::Out(i) => format!(
                    " --out for {} (empty: under the output header)",
                    self.rows[i].label
                ),
                Target::Header => {
                    " Output header: what every result of this run is named after".to_string()
                }
                Target::Field(m, i) => format!(" --{}", self.rows[m].form.fields[i].long),
                Target::Filter => " flags containing".to_string(),
                Target::BatchName(i) => format!(
                    " batch of {} (empty: its own name)",
                    crate::tui::name(&self.pairs[i].data)
                ),
                Target::Label(..) => " new name for this label (empty: as it was)".to_string(),
            };
            let mut lines = vec![
                Line::from(Span::styled(what, bold())),
                Line::from(format!(" {}▏", e.text)),
            ];
            if e.target == Target::Header {
                lines.extend([
                    Line::from(""),
                    Line::from(Span::styled(
                        " e.g. exp1 → exp1_svd, exp1_topic, …   results/ → results/svd, …",
                        hint(),
                    )),
                    Line::from(Span::styled(
                        " O on Methods changes it later; o names one --out by hand",
                        hint(),
                    )),
                    Line::from(Span::styled(" enter keep   esc no header", hint())),
                ]);
            } else {
                lines.push(Line::from(Span::styled(" enter keep   esc cancel", hint())));
            }
            popup(f, area, lines, 90, At::Middle, TEXT);
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
            .map(|r| r.label.as_str())
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
            Screen::Data => "a add data   n name the batch   b label files (several: paired by name)   e rename labels   x own name   X all own   d remove   J K reorder",
            Screen::Methods => "space queue   enter flags   o change --out   O output header",
            Screen::Params => "space / enter change   ← → choices   r reset   R reset all   a advanced   / filter   [ ] method",
            Screen::Run => "↑ ↓ PgUp PgDn scroll the log   End follow it   s stop   v open the results in senna view",
        };
        vec![
            Line::from(Span::styled(
                format!(" {}", self.message.clone().unwrap_or_default()),
                bold(),
            )),
            Line::from(Span::styled(format!(" {keys}"), hint())),
            Line::from(Span::styled(
                format!(
                    " tab / 1-4 screens   {} review and run   q quit",
                    self.go_key()
                ),
                hint(),
            )),
        ]
    }

    fn draw_data(&self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines = vec![Line::from(Span::styled(
            "Data files, and the batch of their cells",
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
            let batch = match &p.batch {
                Batch::Own if p.tags.is_some() => "its barcodes' @batch tags".to_string(),
                Batch::Own => format!("{} (its name)", batches::own_name(&p.data)),
                Batch::Named(name) => format!("{name} (named)"),
                Batch::Labels {
                    file,
                    renamed,
                    counts,
                } => format!(
                    "{}: {} label{}{}",
                    self.shown(file),
                    counts.len(),
                    if counts.len() == 1 { "" } else { "s" },
                    if renamed.is_empty() {
                        String::new()
                    } else {
                        format!(", {} renamed", renamed.len())
                    }
                ),
            };
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
        let summary = batches::summary(&self.pairs);
        if !summary.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!(
                    "{} batch{}",
                    summary.len(),
                    if summary.len() == 1 { "" } else { "es" }
                ),
                bold(),
            )));
            for (name, files, cells) in summary {
                let cells = cells.map_or_else(|| "…".to_string(), |c| c.to_string());
                let text = format!(
                    "  {name:<24} {cells:>8} cells  {files} file{}",
                    if files == 1 { "" } else { "s" }
                );
                lines.push(Line::from(Span::styled(fit(&text, w), hint())));
            }
        }
        f.render_widget(Paragraph::new(lines), area);
    }

    /// The labels of data row `row`'s label file, `at` under the cursor.
    fn relabel_lines(&self, row: usize, at: usize, rows: usize) -> Vec<Line<'static>> {
        let Some(Batch::Labels {
            file,
            counts,
            renamed,
        }) = self.pairs.get(row).map(|p| &p.batch)
        else {
            return Vec::new();
        };
        let mut out = vec![
            Line::from(Span::styled(
                format!(" Labels of {}", self.shown(file)),
                bold(),
            )),
            Line::from(""),
        ];
        let width = counts
            .iter()
            .map(|(l, _)| l.chars().count())
            .max()
            .unwrap_or(0)
            .min(30);
        for (i, (label, n)) in counts
            .iter()
            .enumerate()
            .skip(first_row(at, rows, counts.len()))
            .take(rows)
        {
            let to = renamed
                .get(label)
                .map_or_else(String::new, |r| format!("→ {r}"));
            let text = format!(" {label:<width$}  {n:>8} cells  {to}");
            let style = if i == at {
                selected()
            } else {
                Style::default()
            };
            out.push(Line::from(Span::styled(text, style)));
        }
        out.push(Line::from(""));
        out.push(Line::from(Span::styled(
            " ↑ ↓ choose   enter rename (empty: as it was)   esc back",
            hint(),
        )));
        out
    }

    fn draw_methods(&self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines = vec![
            Line::from(Span::styled(
                "Methods to fit on these data, each to its own --out",
                bold(),
            )),
            Line::from(vec![
                Span::styled(" output header ", hint()),
                if self.header.is_empty() {
                    Span::styled("(none: O sets one)", hint())
                } else {
                    Span::styled(self.header.clone(), bold())
                },
            ]),
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
        if let Err(why) = &self.mung {
            lines.push(Line::from(Span::styled(
                fit(&format!(" mung clones (CNV clones first): {why}"), w),
                hint(),
            )));
        }
        for (i, r) in self.rows.iter().enumerate() {
            let changed = r.form.changed();
            let text = format!(
                " [{}] {:<13} --out {:<out_w$}{} {:<11} {}",
                if r.on { "x" } else { " " },
                r.label,
                fit(&r.out, out_w),
                // Typed by hand: the header leaves it be.
                if r.typed { " ✎" } else { "  " },
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
            if r.tool == Tool::Mung {
                lines.push(Line::from(Span::styled(
                    "      runs first; once it is done you see its clones and choose whether the fits get --cnv-clones",
                    hint(),
                )));
                lines.push(Line::from(""));
            }
            if r.on {
                if let Some(warn) = self.shape_warning(r) {
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
            let name = format!(" {} ", self.rows[m].label);
            tabs.push(if m == self.param_method {
                Span::styled(name, selected())
            } else {
                Span::styled(name, hint())
            });
        }
        let mut sub = format!(
            "{}{}",
            if self.advanced {
                "all flags (a hides advanced ones, marked ·)"
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
            let value = if x.is_default() && self.filled_by_clones(x) {
                "← mung clones".to_string()
            } else {
                fit(&x.shown(), 28)
            };
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
        let blamed = super::form::check(self.command_of(self.rows[m].tool), &argv)
            .err()
            .and_then(|why| super::form::blamed(&why, &form.fields).map(str::to_string));
        *self.blame.borrow_mut() = Some((m, argv, blamed.clone()));
        blamed
    }

    fn draw_run(&self, f: &mut ratatui::Frame, area: Rect) {
        let Some(q) = &self.queue else { return };
        let rows = usize::from(area.height).saturating_sub(q.jobs.len() + 3);
        self.log_rows.set(rows);
        // Copied out, so the worker writing the log is not held up.
        let (states, last, progress, finished, log, top) = {
            let Ok(s) = q.shared.lock() else { return };
            let end = s.dropped + s.log.len();
            // Followed to its end, or from the line scrolled to.
            let top = self.log_top.map_or(end.saturating_sub(rows), |t| {
                t.clamp(s.dropped, end.saturating_sub(rows).max(s.dropped))
            });
            let log: Vec<String> = s
                .log
                .iter()
                .skip(top - s.dropped)
                .take(rows)
                .cloned()
                .collect();
            let at_end = top + rows >= end;
            (
                s.states.clone(),
                s.last.clone(),
                s.progress.clone(),
                s.finished,
                log,
                (top, end, at_end),
            )
        };
        let w = usize::from(area.width);
        let done = states
            .iter()
            .filter(|s| !matches!(s, State::Waiting | State::Running))
            .count();
        let head = if finished {
            "Finished".to_string()
        } else {
            let at = states
                .iter()
                .position(|s| *s == State::Running)
                .map_or(done, |i| i + 1);
            format!("Running {at} of {}, one after another", states.len())
        };
        let mut lines = vec![Line::from(Span::styled(head, bold())), Line::from("")];
        for (((j, state), last), progress) in q.jobs.iter().zip(&states).zip(&last).zip(&progress) {
            let name = format!(
                " {:<13} {:<24} ",
                j.method,
                fit(&self.shown(&j.result()), 24)
            );
            let (said, style) = match state {
                State::Waiting => ("waiting".to_string(), hint()),
                State::Running => match progress {
                    Some(p) => (gauge(p, 24), bold()),
                    None => (format!("running  {last}"), bold()),
                },
                State::Done => ("done".to_string(), Style::default()),
                State::Failed(why) => (format!("failed: {why}"), bold()),
                State::Stopped => ("stopped".to_string(), hint()),
            };
            lines.push(Line::from(Span::styled(
                fit(&format!("{name}{said}"), w),
                style,
            )));
        }
        let (top, end, at_end) = top;
        lines.push(Line::from(Span::styled(
            if at_end {
                String::new()
            } else {
                format!(
                    " ── lines {}-{} of {end}; End follows the log",
                    top + 1,
                    top + log.len()
                )
            },
            hint(),
        )));
        lines.extend(
            log.iter()
                .map(|l| Line::from(Span::styled(fit(l, w), hint()))),
        );
        f.render_widget(Paragraph::new(lines), area);
    }

    /// Whether row `x` of the method shown is filled by the queued
    /// `mung clones` step.
    fn filled_by_clones(&self, x: &super::Field) -> bool {
        self.rows[self.param_method].tool == Tool::Senna
            && x.long == CLONES_FLAG
            && self.clones_row().is_some()
    }

    /// The popup the queue waits on after `mung clones`.
    fn clones_lines(&self, rows: usize) -> Vec<Line<'static>> {
        let mut out = vec![
            Line::from(Span::styled(" mung clones is done: its clones", bold())),
            Line::from(""),
        ];
        let Some(q) = &self.queue else { return out };
        let Ok(shared) = q.shared.lock() else {
            return out;
        };
        let Some(told) = &shared.asking else {
            return out;
        };
        let advice = match told {
            Ok((lines, advice)) => {
                let room = rows.saturating_sub(7).max(1);
                let first = self.clones_scroll.min(lines.len().saturating_sub(room));
                out.extend(
                    lines
                        .iter()
                        .skip(first)
                        .take(room)
                        .map(|l| Line::from(format!(" {l}"))),
                );
                if lines.len() > room {
                    out.push(Line::from(Span::styled(" ↑ ↓ scroll", hint())));
                }
                (*advice).to_string()
            }
            Err(why) => format!("cannot read the clones: {why}"),
        };
        out.push(Line::from(""));
        out.push(Line::from(Span::styled(format!(" {advice}"), bold())));
        out.push(Line::from(""));
        out.push(Line::from(Span::styled(
            " y keep them (--cnv-clones)   n run without   s stop",
            hint(),
        )));
        out
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
            let lines = super::script::command_lines(&p.job.command(), p.job.tool);
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
