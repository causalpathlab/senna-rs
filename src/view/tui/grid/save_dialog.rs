//! Saving: what can be saved (one view, the grid, every view), under which
//! name, and writing the PDF.

use super::*;
use crate::view::deck;
use legume_tui::save::{SaveDialog, SaveKey, SaveRequest, SaveTarget};

/// What a PDF takes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scope {
    /// The session on screen.
    View,
    /// Every session, one grid on one page.
    Grid,
    /// Every session, one page each.
    Pages,
}

impl Scope {
    pub(super) const ALL: [Scope; 3] = [Scope::View, Scope::Grid, Scope::Pages];

    pub(super) fn name(self, n: usize) -> String {
        match self {
            Scope::View => "this view".into(),
            Scope::Grid => format!("all {n} runs, one grid"),
            Scope::Pages => format!("all {n} runs, one page each"),
        }
    }
}

/// What the save dialog offers, as the deck was when it opened.
pub(super) struct Target {
    /// Sessions open.
    pub(super) n: usize,
    /// The name the view on screen offers, and the first run's.
    pub(super) view: String,
    pub(super) first: String,
    /// Screen size in pixels of what each scope pictures, for its aspect
    /// and to scale text: the view on screen, and the grid's area.
    pub(super) view_px: (f32, f32),
    pub(super) grid_px: (f32, f32),
}

impl SaveTarget for Target {
    type Scope = Scope;

    fn scopes(&self) -> Vec<Scope> {
        if self.n > 1 {
            Scope::ALL.to_vec()
        } else {
            vec![Scope::View]
        }
    }

    fn scope_name(&self, s: Scope) -> String {
        s.name(self.n)
    }

    fn default_stem(&self, s: Scope) -> String {
        match s {
            Scope::View => self.view.clone(),
            Scope::Grid => format!("{}.view.grid", self.first),
            Scope::Pages => format!("{}.view.pages", self.first),
        }
    }

    fn exts(&self, _: Scope) -> Vec<String> {
        vec!["pdf".into()]
    }

    /// One page per run is as wide as the view on screen, its first page
    /// named in the size.
    fn aspect(&self, s: Scope) -> f32 {
        let (w, h) = match s {
            Scope::Grid => self.grid_px,
            _ => self.view_px,
        };
        w / h.max(1.0)
    }

    fn size_note(&self, s: Scope) -> String {
        match s {
            Scope::Pages => format!("(first of {} pages)", self.n),
            _ => String::new(),
        }
    }
}

impl Deck {
    /// What the save dialog offers now.
    pub(super) fn save_target(&self) -> Target {
        let app = &self.apps[self.at];
        let view_px = app.view_px();
        let grid_px = if self.area.width > 0 {
            (
                f32::from(self.area.width) * self.cell.0,
                f32::from(self.area.height) * self.cell.1,
            )
        } else {
            view_px
        };
        Target {
            n: self.apps.len(),
            view: app.pdf_name(),
            first: here(&self.apps[0].scene.data.prefix),
            view_px,
            grid_px,
        }
    }

    /// Open the save dialog on `scope`, or on this view when one is open.
    pub(super) fn open_save(&mut self, scope: Scope) {
        let t = self.save_target();
        let scope = if t.scopes().contains(&scope) {
            scope
        } else {
            Scope::View
        };
        self.save = Some((SaveDialog::new(&t, scope, NUMBERING), t));
    }

    /// A key in the save dialog: enter on a name that may be written
    /// renders and writes the PDF, the rest edit the dialog.
    pub(super) fn save_event(
        &mut self,
        terminal: &mut DefaultTerminal,
        k: KeyEvent,
    ) -> anyhow::Result<()> {
        let Some((d, t)) = self.save.as_mut() else {
            return Ok(());
        };
        match d.key(k, t) {
            SaveKey::Open => {}
            SaveKey::Cancelled => self.save = None,
            SaveKey::Save(req) => {
                d.say("saving…");
                self.draw(terminal)?;
                let (mut d, t) = self.save.take().expect("the dialog is open");
                match self.write_pdf(&req, &t) {
                    Ok(msg) => self.say(msg),
                    Err(e) => {
                        d.say(format!("not saved: {e}"));
                        self.save = Some((d, t));
                    }
                }
            }
        }
        if self.save.is_none() {
            // The dialog covered part of the picture, and a first save opens
            // the saved-figures strip: draw it all again.
            self.drawn = None;
        }
        Ok(())
    }

    /// Render and write the PDF `req` asks for. Returns what to say.
    pub(super) fn write_pdf(
        &mut self,
        req: &SaveRequest<Scope>,
        t: &Target,
    ) -> Result<String, String> {
        let (w, h) = req.px;
        let dpi = req.dpi;
        let view = |app| view_at(app, w, self.cell.1);
        let pages = match req.scope {
            Scope::View => deck::view_pages(&[view(&self.apps[self.at])]),
            Scope::Pages => deck::view_pages(&self.apps.iter().map(view).collect::<Vec<_>>()),
            Scope::Grid => {
                let scenes: Vec<&Scene> = self.apps.iter().map(|a| &a.scene).collect();
                let cams: Vec<Option<Viewport>> = self.apps.iter().map(|a| a.vp).collect();
                let titles: Vec<String> = (0..self.apps.len()).map(|k| self.title(k)).collect();
                let cell = self.cell.1 * w as f32 / t.grid_px.0.max(1.0);
                vec![deck::render_grid(&scenes, &cams, &titles, (w, h), cell)]
            }
        };
        legume_tui::export::write_pdf(&pages, dpi as f32, &req.path).map_err(|e| e.to_string())?;
        // Listed on the left with its first page as the thumbnail; failing
        // to list it never fails the save.
        let what = format!(
            "{} · {} in · {dpi} dpi",
            t.scope_name(req.scope),
            req.width_in
        );
        let listed = match self
            .gallery
            .add(&[&req.path], &what, Some(&pages[0].img), ())
        {
            Ok(()) => String::new(),
            Err(e) => format!(" · not listed: {e}"),
        };
        let pages = if pages.len() > 1 {
            format!("{} pages · ", pages.len())
        } else {
            String::new()
        };
        Ok(format!(
            "saved {} · {pages}{w} × {h} px at {dpi} dpi · text kept as text{listed}",
            req.path.display()
        ))
    }
}
