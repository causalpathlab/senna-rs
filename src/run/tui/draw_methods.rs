//! Drawing the Methods screen: every fit with its `--out`, under the
//! output header, and under each queued fit the steps offered after it.

use super::form::Method;
use super::jobs::Tool;
use super::steps::Step;
use super::App;
use crate::tui::style::{bold, hint, selected, CHROME};
use legume_tui::chrome::short;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

impl App {
    pub(super) fn draw_methods(&self, f: &mut ratatui::Frame, area: Rect) {
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
        const HEAD: usize = 3;
        let mut at = 0;
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
                short(&format!(" mung clones (CNV clones first): {why}"), w),
                hint(),
            )));
        }
        for (i, r) in self.rows.iter().enumerate() {
            let (on, changed, about) = told(&r.form, r.on);
            let text = format!(
                " {on} {:<13} --out {:<out_w$}{} {changed:<11} {about}",
                r.label,
                short(&r.out, out_w),
                // Typed by hand: the header leaves it be.
                if r.typed { " ✎" } else { "  " },
            );
            let here = self.methods_cursor == (i, None);
            let style = if here {
                selected()
            } else if r.on {
                bold()
            } else {
                Style::default()
            };
            if here {
                at = lines.len();
            }
            lines.push(Line::from(Span::styled(short(&text, w), style)));
            if r.on {
                for (s, step) in r.steps.iter().enumerate() {
                    let here = self.methods_cursor == (i, Some(s));
                    if here {
                        at = lines.len();
                    }
                    lines.push(step_line(step, here, w));
                }
            }
            if r.tool == Tool::Mung {
                // A note under the row, wrapped by word under its name.
                let note = "runs first; once it is done you see its clones and choose whether the fits get --cnv-clones";
                for l in legume_tui::text::wrap_words(note, w.saturating_sub(6)) {
                    lines.push(Line::from(Span::styled(format!("      {l}"), hint())));
                }
                lines.push(Line::from(""));
            }
        }
        // The heading stays; the rows scroll to keep the cursor in sight.
        let room = usize::from(area.height).saturating_sub(HEAD);
        let body = lines.split_off(HEAD.min(lines.len()));
        lines.extend(CHROME.scrolled(body, at.saturating_sub(HEAD), room));
        f.render_widget(Paragraph::new(lines), area);
    }
}

/// A step's row under its fit: queued or not, its flags, what it does.
fn step_line(step: &Step, here: bool, w: usize) -> Line<'static> {
    let (on, changed, about) = told(&step.form, step.on);
    let text = format!(
        "     {on} then {:<16} {changed:<11} {about}",
        step.form.name
    );
    let style = if here {
        selected()
    } else if step.on {
        bold()
    } else {
        hint()
    };
    Line::from(Span::styled(short(&text, w), style))
}

/// What a fit's or a step's row tells of `form`: queued (`on`) or not, its
/// flags left at their defaults or how many changed, and the first line of
/// what it does.
fn told(form: &Method, on: bool) -> (&'static str, String, &str) {
    let changed = match form.changed() {
        0 => "defaults".to_string(),
        n => format!("{n} changed"),
    };
    let about = form.about.lines().next().unwrap_or_default();
    (if on { "[x]" } else { "[ ]" }, changed, about)
}
