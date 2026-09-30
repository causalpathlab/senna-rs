//! Help that follows the context. The status line lists the keys that act
//! where you are; `?` explains them, grouped by task.

use super::*;

/// Where the user is, as far as keys are concerned.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Context {
    Browse,
    /// A feature's activity is on the map.
    Feature,
    Relabel,
    /// Choosing clusters to merge.
    Merge,
    /// Typing a label or rationale.
    Prompt,
    /// Browsing for the marker panel `lupin annotate` reads.
    MarkersFile,
    /// A structure plot or heatmap in place of the map.
    Chart,
    /// Confirming a relabel round before it goes to lupin.
    Submit,
    /// Choosing what senna recomputes for the run.
    Recompute,
    Search,
    StyleMenu,
}

/// `?` help for browsing, by task.
const BROWSE_HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Look around",
        &[
            (
                "click a cell",
                "its cluster's summary, and the features nearest it, with edges to them",
            ),
            (
                "click a feature",
                "on a feature map: the features nearest it (cosine), and the cells nearest it",
            ),
            (
                "click a label",
                "a cluster's label on the map: the features most up in that cluster",
            ),
            ("[  ]", "focus the previous / next group (drawn on top)"),
            (
                "c",
                "colour by the next grouping (annotation, cluster, topic, …)",
            ),
            ("tab  shift-tab", "cells, features on cells, features, of this layout"),
            ("m", "next layout method (umap, phate, …)"),
            (
                "z  Z  0",
                "lay out the focused group on its own / back up / back to the top",
            ),
            ("scroll  + -", "zoom the camera in / out"),
            ("arrows  drag", "pan"),
        ],
    ),
    (
        "Features",
        &[
            (
                "n",
                "suggest features: what sets the focused group apart, or what varies here",
            ),
            (
                "g  G",
                "step through the suggestions (else the focused group's markers)",
            ),
            ("/", "search a feature by name"),
            ("a", "activity of the focused group's whole marker set"),
            ("o", "expected (model) or observed (counts)"),
            ("k", "pin names on the map: those nearest the clicked cell or feature, or the feature on screen (again clears)"),
            ("x  esc", "back to group colours"),
        ],
    ),
    (
        "Charts",
        &[
            (
                "H",
                "structure plot (runs with topics, bge included: each cell's topic mixture, a panel per group), heatmap, back to the map",
            ),
            (
                "+  -",
                "heatmap: more or fewer top features per group (z-scored mean ln(1 + count))",
            ),
        ],
    ),
    (
        "Annotation",
        &[
            (
                "R",
                "relabel mode: go cluster by cluster, then hand lupin one round",
            ),
            (",  .", "previous (source) / next annotation round"),
            (
                "A",
                "annotate this run with lupin: browse to a marker panel (★ fits the run best)",
            ),
        ],
    ),
    (
        "Several views (d, or several -f manifests)",
        &[
            (
                "d",
                "copy this view beside it (1×2, then 2×2, …) to compare colourings or features",
            ),
            (
                "w",
                "every open view in a grid: point or arrows choose, click or enter opens, 1-9 by number",
            ),
            ("{  }", "previous / next view"),
            (
                "f",
                "saved figures on the left, newest first, kept in .senna-view/ here",
            ),
            ("X  (grid)", "close the view under the pointer (shift: it cannot be undone)"),
        ],
    ),
    (
        "Other",
        &[
            (
                "e",
                "style of a group: colour, shape, opacity, size, hidden",
            ),
            ("t", "labels on the map: small, medium, large, largest, off, in turn"),
            ("b", "sidebar on or off"),
            (
                "<  >",
                "every dot and all text smaller / bigger (each group's own style stays)",
            ),
            (
                "r",
                "recompute this run's layouts or clusters: choose in a menu, then enter",
            ),
            (
                "s",
                "save as PDF: this view, or every run (one grid, or a page each); width, dpi, file name",
            ),
            ("ctrl-r  ctrl-l", "reload the run from disk and redraw the screen"),
            ("q", "quit"),
        ],
    ),
];

/// `?` help in relabel mode: the steps, in order.
const RELABEL_HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "1. Check the fit",
        &[
            (
                "left panel",
                "every cluster: ? unassigned, → markers suggest another type, ✓ decided",
            ),
            (
                "right panel",
                "candidate types and how well their markers fit this cluster",
            ),
            (
                "tab",
                "choose the working type: + adds markers to it, L offers it as the label (the best fit is chosen for you)",
            ),
        ],
    ),
    (
        "2. Look at the features",
        &[
            (
                "↑  ↓  enter",
                "choose a feature; show its activity on the map",
            ),
            (
                "click a cell",
                "the features nearest it; k pins their names on the map",
            ),
        ],
    ),
    (
        "3. Mark features",
        &[
            (
                "+  -  space",
                "include in / exclude from the markers; clear (then the next feature)",
            ),
            ("a", "accept every proposal (?+ add, ?- drop)"),
            (
                "(live)",
                "lupin rescores the edited types as you mark; its calls show under the fit",
            ),
        ],
    ),
    (
        "4. Decide",
        &[
            (
                "L",
                "label the cluster (the working type and a rationale filled in; enter twice), then on to the next undecided one",
            ),
            ("K", "keep its current call, then on to the next undecided one"),
            (
                "M",
                "merge mode: ↑ ↓ move, space choose, enter name the merged cluster, esc cancel",
            ),
        ],
    ),
    (
        "5. Move on, then finish",
        &[
            ("→  ←  ]  [", "next / previous cluster (or click one)"),
            ("p", "ask lupin what everything staged would change"),
            ("S", "hand lupin everything as one round"),
            ("R  esc", "leave; the draft is kept for later"),
        ],
    ),
    (
        "How the keys read",
        &[
            ("L K M S R", "uppercase decides or writes: label, keep, merge, submit, leave"),
            ("+ - a p k", "lowercase looks or edits the draft: marks, proposals, preview, names"),
        ],
    ),
];

impl App {
    pub(super) fn context(&self) -> Context {
        if let Some(m) = &self.modal {
            m.context()
        } else if self.menu.is_some() {
            Context::StyleMenu
        } else if let Some(r) = &self.scene.review {
            if r.merge.is_some() {
                Context::Merge
            } else {
                Context::Relabel
            }
        } else if self.scene.chart.is_some() {
            Context::Chart
        } else if self.scene.pick.is_some() {
            Context::Feature
        } else {
            Context::Browse
        }
    }

    /// The keys that act here, for the two key lines of the status area:
    /// the main ones, then the rest.
    pub(super) fn status_keys(&self) -> [&'static str; 2] {
        match self.context() {
            Context::Browse => [
                "click a cell: its cluster and the features nearest it (k pins their names)   [ ] focus a group   c change the colouring   n suggest features",
                "R relabel clusters   A annotate with lupin   , . rounds   tab / m other layouts   z lay out a group   d copy view   w all views   s save PDF   ? all keys   q quit",
            ],
            Context::Feature => [
                "g / G next or previous feature   o switch between counts and model   a the group's whole marker set   x back to group colours",
                "/ search a feature   n new suggestions   k pin its name   [ ] focus a group   ? all keys",
            ],
            Context::Merge => [
                "↑ ↓ move   space choose or unchoose the cluster   enter name the merged cluster",
                "≈ marks clusters whose markers fit the same type   esc or M cancel   ? the steps",
            ],
            Context::Relabel => [
                "→ ← next or previous cluster   ↑ ↓ choose a feature   enter show it   + / - include or exclude   a accept all ? proposals",
                "tab working type   L label   K keep   M merge clusters   k pin names   p preview   S submit all   R leave   ? the steps",
            ],
            Context::Prompt => [
                "type, or keep what is filled in   enter accepts   esc cancels",
                "tab completes a known cell type (while typing the label)",
            ],
            Context::Chart => [
                "H next chart (structure plot, heatmap, map)   c group by another grouping   + / - features per group (heatmap)",
                "s save PDF   ctrl-r redraw   ? all keys   q quit",
            ],
            // The popup says what the keys do.
            Context::Submit | Context::Recompute | Context::MarkersFile => ["", ""],
            Context::Search => [
                "type part of a feature name   enter shows the first match   esc cancels",
                "",
            ],
            Context::StyleMenu => [
                "↑ ↓ choose a group   ← → change the value   tab next property",
                "space show / hide   r reset   enter done",
            ],
        }
    }

    /// The `?` overlay for the current context.
    pub(super) fn help_lines(&self) -> Vec<Line<'static>> {
        let sections = if matches!(
            self.context(),
            Context::Relabel | Context::Merge | Context::Submit
        ) {
            RELABEL_HELP
        } else {
            BROWSE_HELP
        };
        let bold = Style::default().add_modifier(ratatui::style::Modifier::BOLD);
        let mut out = Vec::new();
        for (title, keys) in sections {
            out.push(Line::from(Span::styled(format!(" {title}"), bold)));
            for (k, v) in *keys {
                out.push(Line::from(format!("   {k:<15} {v}")));
            }
            out.push(Line::from(""));
        }
        out.push(Line::from(" any key closes this"));
        out
    }
}
