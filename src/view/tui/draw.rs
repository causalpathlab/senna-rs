//! Drawing: the map image, the status line, the sidebar and help.

use super::*;

impl App {
    pub(super) fn draw(&self, f: &mut ratatui::Frame) {
        let page = Style::default()
            .bg(rgb(color::BACKGROUND))
            .fg(rgb(color::INK));
        f.render_widget(Block::default().style(page), f.area());
        let [_, status] = Layout::vertical([Constraint::Min(1), Constraint::Length(STATUS_LINES)])
            .areas(f.area());
        let map = self.map;
        if let Some(p) = &self.proto {
            f.render_widget(Image::new(p), map);
        }

        // Status area: what is on screen (or the latest message), then the
        // keys that act here.
        let mut first = match (&self.search, &self.prompt) {
            (Some((q, hits)), _) => {
                let shown: Vec<&str> = hits.iter().take(6).map(AsRef::as_ref).collect();
                format!("/{q}   {}", shown.join("  "))
            }
            (None, Some(p)) => p.line(),
            (None, None) => match &self.relabeling {
                Some(r) => format!(
                    "lupin is {} {} decision(s)… {:.1} s · editing is locked until it answers",
                    if matches!(r.job, RelabelJob::Preview) {
                        "previewing"
                    } else {
                        "applying"
                    },
                    r.sent.len(),
                    r.started.elapsed().as_secs_f32()
                ),
                None => self.message.clone().unwrap_or_else(|| self.scene.caption()),
            },
        };
        if self.job.is_some() {
            first.push_str("   · drawing…");
        }
        let [main, more] = self.status_keys();
        let lines = vec![
            Line::from(format!(" {first}")),
            Line::from(Span::styled(
                format!(" {main}"),
                Style::default().fg(rgb(color::INK)),
            )),
            Line::from(Span::styled(
                format!(" {more}"),
                Style::default().fg(rgb(color::MUTED)),
            )),
        ];
        f.render_widget(Paragraph::new(lines).style(page), status);

        let side = self.side;
        if side.width > 0 {
            if let Some(menu) = &self.menu {
                self.draw_menu(f, side, menu, page);
            } else if let Some(lines) = self
                .scene
                .merge_lines()
                .or_else(|| self.scene.review_lines())
                .or_else(|| self.info.clone())
                .or_else(|| self.scene.suggestion_lines())
                .or_else(|| self.scene.near_lines())
            {
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

        if self.help {
            let lines = self.help_lines();
            let w = 96.min(map.width);
            let h = (lines.len() as u16 + 2).min(map.height);
            let r = Rect::new(
                map.x + (map.width - w) / 2,
                map.y + (map.height - h) / 2,
                w,
                h,
            );
            f.render_widget(Clear, r);
            f.render_widget(
                Paragraph::new(lines)
                    .wrap(ratatui::widgets::Wrap { trim: false })
                    .block(Block::bordered().border_style(Style::default().fg(rgb(color::MUTED))))
                    .style(page),
                r,
            );
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

        let dim = Style::default().fg(rgb(color::MUTED));
        let mut lines: Vec<Line> = Vec::new();
        for (g, level) in levels.iter().enumerate().skip(first).take(list_rows) {
            let st = self.scene.style_of(g);
            let res = self.scene.resolved(g);
            let mark = Span::styled(
                format!(" {} ", st.shape.glyph()),
                Style::default().fg(res.map_or(Color::Reset, |r| to_color(r.colour))),
            );
            let mut name = Style::default();
            if st.hidden {
                name = dim;
            }
            if g == menu.row {
                name = name.add_modifier(ratatui::style::Modifier::REVERSED);
            }
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
                    dim,
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
            let cursor = if k == menu.field { " ▸ " } else { "   " };
            let mut spans = vec![Span::raw(format!("{cursor}{field:<8} "))];
            spans.extend(value.spans.into_iter().map(|s| s.patch_style(value.style)));
            lines.push(Line::from(spans));
        }
        lines.push(Line::from(Span::styled(format!(" {MENU_HINT}"), dim)));

        f.render_widget(
            Paragraph::new(lines)
                .wrap(ratatui::widgets::Wrap { trim: false })
                .block(side_block())
                .style(page),
            r,
        );
    }
}

/// The sidebar's frame: one thin rule on the map side, nothing else.
fn side_block() -> Block<'static> {
    Block::new()
        .borders(ratatui::widgets::Borders::LEFT)
        .border_style(Style::default().fg(rgb(color::MUTED)))
}
