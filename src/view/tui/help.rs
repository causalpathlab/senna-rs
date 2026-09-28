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
    /// Typing the markers file for `lupin annotate`.
    MarkersFile,
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
                "its cluster's summary, and the features nearest it",
            ),
            ("[  ]", "focus the previous / next group (drawn on top)"),
            (
                "c",
                "colour by the next grouping (annotation, cluster, topic, …)",
            ),
            ("tab  shift-tab", "cells, features on cells, features"),
            ("m", "next layout method (umap, phate, …)"),
            (
                "z  Z  0",
                "lay out the focused group on its own / back up / back to the top",
            ),
            ("scroll  + -", "zoom"),
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
            ("k  l", "pin names on the map: those nearest the clicked cell, or the feature on screen (again clears)"),
            ("x  esc", "back to group colours"),
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
                "annotate this run with lupin (asks for a markers file; tab completes)",
            ),
        ],
    ),
    (
        "Other",
        &[
            (
                "e",
                "style of a group: colour, shape, opacity, size, hidden",
            ),
            ("b  t", "sidebar / text labels on or off"),
            ("s", "save this view as PNG"),
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
                "choose the target type (the best fit is chosen for you)",
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
                "the features nearest it; k or l pins their names on the map",
            ),
        ],
    ),
    (
        "3. Mark features",
        &[
            (
                "+  -  space",
                "include in / exclude from the markers; clear",
            ),
            ("a", "accept every proposal (?+ add, ?- drop)"),
        ],
    ),
    (
        "4. Decide",
        &[
            (
                "L",
                "label the cluster (target and rationale filled in; enter twice)",
            ),
            ("K", "keep its current call"),
            (
                "M",
                "merge mode: ↑ ↓ move, space choose, enter name the merged cluster, esc cancel",
            ),
        ],
    ),
    (
        "5. Move on, then finish",
        &[
            ("]  [", "next / previous cluster (or click one)"),
            ("p", "ask lupin what everything staged would change"),
            ("S", "hand lupin everything as one round"),
            ("R  esc", "leave; the draft is kept for later"),
        ],
    ),
];

impl App {
    pub(super) fn context(&self) -> Context {
        if self.search.is_some() {
            Context::Search
        } else if self.prompt.is_some() {
            Context::Prompt
        } else if self.markers_input.is_some() {
            Context::MarkersFile
        } else if self.menu.is_some() {
            Context::StyleMenu
        } else if let Some(r) = &self.scene.review {
            if r.merge.is_some() {
                Context::Merge
            } else {
                Context::Relabel
            }
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
                "R relabel clusters   A annotate with lupin   , . rounds   tab / m other layouts   z zoom into a group   e style   ? all keys   q quit",
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
                "] / [ next or previous cluster   ↑ ↓ choose a feature   enter show it   + / - include or exclude   a accept all ? proposals",
                "tab target type   L label   K keep   M merge clusters   k pin names   p preview with lupin   S submit all   R leave   ? the steps",
            ],
            Context::Prompt => [
                "type, or keep what is filled in   enter accepts   esc cancels",
                "tab completes a known cell type (while typing the label)",
            ],
            Context::MarkersFile => [
                "type the path of a markers file (type, then its markers; TSV, or one comma per line)",
                "tab completes the path   enter runs lupin annotate   esc cancels",
            ],
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
        let sections = if matches!(self.context(), Context::Relabel | Context::Merge) {
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
