//! Drawing: the map image, the status line, the sidebar and help.

use super::*;

impl App {
    pub(super) fn draw(&self, f: &mut ratatui::Frame) {
        let page = page();
        f.render_widget(Block::default().style(page), f.area());
        let [_, status] = Layout::vertical([Constraint::Min(1), Constraint::Length(STATUS_LINES)])
            .areas(f.area());
        let map = self.map;
        if let Some(p) = &self.proto {
            f.render_widget(Image::new(p), map);
        }

        // Status area: what is on screen (or the latest message), then the
        // keys that act here.
        let mut first = match self.modal.as_ref().and_then(Modal::line) {
            Some(line) => line,
            None => match &self.relabeling {
                Some(r) if matches!(r.job, RelabelJob::Annotate) => format!(
                    "lupin is annotating this run… {:.0} s · {}",
                    r.started.elapsed().as_secs_f32(),
                    r.progress.lock().map(|p| p.clone()).unwrap_or_default()
                ),
                Some(r) => format!(
                    "lupin is {} {} decision(s)… {:.1} s · editing is locked until it answers",
                    if matches!(r.job, RelabelJob::Draft(Mode::Preview)) {
                        "previewing"
                    } else {
                        "applying"
                    },
                    r.sent.len(),
                    r.started.elapsed().as_secs_f32()
                ),
                // What was just said first; senna's progress when nothing was.
                None => self
                    .message
                    .clone()
                    .or_else(|| self.recompute_line())
                    .unwrap_or_else(|| self.scene.caption()),
            },
        };
        if self.job.is_some() {
            first.push_str("   · drawing…");
        }
        let [main, more] = self.status_keys();
        let lines = vec![
            Line::from(format!(" {first}")),
            Line::from(format!(" {main}")),
            Line::from(Span::styled(format!(" {more}"), hint())),
        ];
        f.render_widget(Paragraph::new(lines).style(page), status);

        let side = self.side;
        if side.width > 0 {
            if let Some(menu) = &self.menu {
                self.draw_menu(f, side, menu, page);
            } else if let Some(lines) = self.side_lines() {
                let text: Vec<Line> = lines.iter().map(|l| Line::from(format!(" {l}"))).collect();
                f.render_widget(
                    Paragraph::new(text)
                        .wrap(ratatui::widgets::Wrap { trim: false })
                        .block(side_block())
                        .style(page),
                    side,
                );
            }
        }

        if self.left.width > 0 {
            if let Some(lines) = self.scene.overview_lines() {
                // Keep the cursor row in sight.
                let rows = usize::from(self.left.height);
                let at = lines.iter().position(|l| l.starts_with('▸')).unwrap_or(0);
                let skip = (at + rows / 2)
                    .saturating_sub(rows)
                    .min(lines.len().saturating_sub(rows));
                let text: Vec<Line> = lines
                    .iter()
                    .skip(skip)
                    .map(|l| Line::from(format!("{l} ")))
                    .collect();
                f.render_widget(
                    Paragraph::new(text)
                        .block(
                            Block::new()
                                .borders(ratatui::widgets::Borders::RIGHT)
                                .border_style(Style::default().fg(rgb(color::MUTED))),
                        )
                        .style(page),
                    self.left,
                );
            }
        }

        if let Some((text, _)) = &self.toast {
            toast(f, map, text);
        }

        if let Some(Modal::Submit(decisions)) = &self.modal {
            let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
            let mut lines = vec![
                Line::from(Span::styled(" Submit this round to lupin?", bold)),
                Line::from(""),
            ];
            lines.extend(decisions.iter().map(|d| Line::from(format!("  {d}"))));
            lines.push(Line::from(""));
            lines.push(Line::from(
                " lupin writes a new round from these; this one stays as it is",
            ));
            lines.push(Line::from(Span::styled(
                " S or enter submits   any other key cancels",
                hint(),
            )));
            popup(f, map, lines, 76, At::Middle, color::TEXT);
        }

        if let Some(Modal::Recompute(_, menu)) = &self.modal {
            let lines = App::recompute_lines(menu, &files::name(&self.from));
            popup(f, map, lines, 72, At::Middle, color::TEXT);
        }

        if self.help {
            popup(f, map, self.help_lines(), 96, At::Middle, color::TEXT);
        }
    }

    /// The style menu, docked on the right of the map.
    pub(super) fn draw_menu(&self, f: &mut ratatui::Frame, map: Rect, menu: &Menu, page: Style) {
        let enc = color::encoder();
        let to_color = |c: color::Rgb| {
            let [r, g, b] = c.map(|v| enc.encode(v));
            Color::Rgb(r, g, b)
        };
        let levels = self.scene.levels();
        let chrome = FIELDS.len() as u16 + 6;
        let r = map;
        let h = r.height;
        let list_rows = h.saturating_sub(chrome).max(1) as usize;
        let first = menu
            .row
            .saturating_sub(list_rows / 2)
            .min(levels.len().saturating_sub(list_rows));

        let mut lines: Vec<Line> = Vec::new();
        for (g, level) in levels.iter().enumerate().skip(first).take(list_rows) {
            let st = self.scene.style_of(g);
            let res = self.scene.resolved(g);
            let mark = Span::styled(
                format!(" {} ", st.shape.glyph()),
                Style::default().fg(res.map_or(Color::Reset, |r| to_color(r.colour))),
            );
            let name = match (g == menu.row, st.hidden) {
                (true, _) => selected(),
                (false, true) => hint().add_modifier(ratatui::style::Modifier::CROSSED_OUT),
                (false, false) => Style::default(),
            };
            lines.push(Line::from(vec![
                mark,
                Span::styled(level.to_string(), name),
            ]));
        }
        let st = self.scene.style_of(menu.row);
        let res = self.scene.resolved(menu.row);
        let values = [
            Line::from(vec![
                Span::raw("■■■■ "),
                Span::styled(
                    if st.colour.is_some() {
                        "custom"
                    } else {
                        "default"
                    },
                    hint(),
                ),
            ])
            .style(Style::default().fg(res.map_or(Color::Reset, |r| to_color(r.colour)))),
            Line::from(format!("{} {}", st.shape.glyph(), st.shape.name())),
            Line::from(format!("{:.1}", st.alpha)),
            Line::from(format!("{:.2}×", st.size)),
            Line::from(if st.hidden { "hidden" } else { "shown" }),
        ];
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!(" {}", levels[menu.row]),
            Style::default().add_modifier(ratatui::style::Modifier::BOLD),
        )));
        for (k, (field, value)) in FIELDS.iter().zip(values).enumerate() {
            let here = k == menu.field;
            let cursor = if here { " ▸ " } else { "   " };
            let label = if here { selected() } else { Style::default() };
            let mut spans = vec![
                Span::styled(format!("{cursor}{field:<8}"), label),
                Span::raw(" "),
            ];
            spans.extend(value.spans.into_iter().map(|s| s.patch_style(value.style)));
            lines.push(Line::from(spans));
        }
        lines.push(Line::from(Span::styled(format!(" {MENU_HINT}"), hint())));

        f.render_widget(
            Paragraph::new(lines)
                .wrap(ratatui::widgets::Wrap { trim: false })
                .block(side_block())
                .style(page),
            r,
        );
    }
}

/// The page's colours: ink on the map's background.
pub(super) fn page() -> Style {
    Style::default()
        .bg(rgb(color::BACKGROUND))
        .fg(rgb(color::TEXT))
}

/// Key hints and other secondary lines: a step lighter than the text.
pub(super) fn hint() -> Style {
    Style::default().fg(rgb(color::HINT))
}

/// The line under a menu's cursor: a dark bar, so where you are is plain.
pub(super) fn selected() -> Style {
    Style::default()
        .bg(rgb(color::TEXT))
        .fg(rgb(color::BACKGROUND))
        .add_modifier(ratatui::style::Modifier::BOLD)
}

/// Where a popup sits over its area.
#[derive(Clone, Copy)]
pub(super) enum At {
    Top,
    Middle,
}

/// A bordered popup of `lines` over `area`, at most `max_w` columns wide
/// and as tall as its lines.
pub(super) fn popup(
    f: &mut ratatui::Frame,
    area: Rect,
    lines: Vec<Line<'_>>,
    max_w: u16,
    at: At,
    border: [u8; 3],
) {
    let w = max_w.min(area.width);
    let h = (lines.len() as u16 + 2).min(area.height);
    let y = match at {
        At::Top => area.y + 1.min(area.height - h),
        At::Middle => area.y + (area.height - h) / 2,
    };
    let r = Rect::new(area.x + (area.width - w) / 2, y, w, h);
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(lines)
            .wrap(ratatui::widgets::Wrap { trim: false })
            .block(Block::bordered().border_style(Style::default().fg(rgb(border))))
            .style(page()),
        r,
    );
}

/// A one-line notice at the top of `area`: lupin answered, a file saved.
pub(super) fn toast(f: &mut ratatui::Frame, area: Rect, text: &str) {
    let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
    let line = Line::from(Span::styled(text.to_string(), bold)).centered();
    let w = text.chars().count() as u16 + 4;
    popup(f, area, vec![line], w, At::Top, color::TEXT);
}

/// The sidebar's frame: one thin rule on the map side, nothing else.
fn side_block() -> Block<'static> {
    Block::new()
        .borders(ratatui::widgets::Borders::LEFT)
        .border_style(Style::default().fg(rgb(color::MUTED)))
}
