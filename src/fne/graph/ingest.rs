//! Reading sources into the [`TypedGraphBuilder`]: gene pair files and their
//! derived relations, typed edge files, node text, memberships and gene sets,
//! ontologies, and region files tiled onto windows.

use super::{
    canonical_term_id, outside_locus, parse_weight, GenePairMode, NodeText, PpiOpts,
    TypedGraphBuilder, GENE_TYPE, PPR_EPS, REGION_TYPE, TERM_TYPE,
};
use data_beans::aux::gene_sets::{read_membership_pairs, GeneSets};
use data_beans::aux::ontology::{Ontology, Rel};
use genomic_data::coordinates::PeakCoord;
use legume_numeric::matrix::common_io::{file_stem, read_lines_of_words_delim};
use legume_numeric::matrix::membership::detect_delimiter;
use legume_numeric::matrix::pair_graph::FeaturePairGraph;
use log::{info, warn};
use rustc_hash::FxHashMap;

impl TypedGraphBuilder {
    /// A gene-gene pair file: `gene1 gene2 [weight]`, its own relation
    /// `gene:gene/<stem>`, then the QC and derived relations of `opts`.
    pub(crate) fn add_pair_file(&mut self, path: &str, opts: &PpiOpts) -> anyhow::Result<()> {
        let stem = file_stem(path);
        let rel_name = format!("{GENE_TYPE}:{GENE_TYPE}/{stem}");
        let r = self.relation(&rel_name, GENE_TYPE, GENE_TYPE);
        let (n_rows, _) = self.ingest_gene_pairs(path, GenePairMode::Stem { rel: r })?;
        let rel = &self.relations[r];
        info!(
            "fne: {path}: {n_rows} pair rows → relation `{}` with {} unique edges ({} self-loops, {} repeats dropped)",
            rel.name,
            rel.edges.len(),
            rel.n_self_loops,
            rel.n_repeats
        );
        if opts.any() {
            self.refine_pair_relation(r, &stem, opts);
        }
        Ok(())
    }

    /// Mixed gene–gene file: `gene1 gene2 relation [weight]`. Each distinct
    /// relation token becomes `gene:gene/<relation>`; no PPI QC / SNN / PPR.
    pub(crate) fn add_named_pair_file(&mut self, path: &str) -> anyhow::Result<()> {
        let (n_rows, touched) = self.ingest_gene_pairs(path, GenePairMode::Named)?;
        anyhow::ensure!(
            n_rows > 0,
            "{path}: no named-pair rows (need `gene gene relation [weight]`)"
        );
        let names: Vec<String> = touched
            .iter()
            .map(|&r| {
                format!(
                    "`{}`:{}",
                    self.relations[r].name,
                    self.relations[r].edges.len()
                )
            })
            .collect();
        info!(
            "fne: {path}: {n_rows} named-pair rows → {} relation(s) ({})",
            touched.len(),
            names.join(", ")
        );
        Ok(())
    }

    /// Shared gene–gene TSV ingest: columns 0/1 are endpoints; weight and
    /// relation naming follow [`GenePairMode`].
    fn ingest_gene_pairs(
        &mut self,
        path: &str,
        mode: GenePairMode,
    ) -> anyhow::Result<(usize, Vec<usize>)> {
        let (min_cols, weight_col) = match mode {
            GenePairMode::Stem { .. } => (2, 2),
            GenePairMode::Named => (3, 3),
        };
        let read = read_lines_of_words_delim(path, detect_delimiter(path), -1)?;
        let mut n_rows = 0usize;
        let mut tok_to_r: FxHashMap<Box<str>, usize> = FxHashMap::default();
        let mut touched: Vec<usize> = Vec::new();
        for line in &read.lines {
            if line.len() < min_cols || line[0].starts_with('#') {
                continue;
            }
            let r = match mode {
                GenePairMode::Stem { rel } => rel,
                GenePairMode::Named => {
                    let rel_tok = line[2].trim();
                    anyhow::ensure!(
                        !rel_tok.is_empty(),
                        "{path}: empty relation name on a named-pair row"
                    );
                    match tok_to_r.get(rel_tok) {
                        Some(&r) => r,
                        None => {
                            let name = format!("{GENE_TYPE}:{GENE_TYPE}/{rel_tok}");
                            let r = self.relation(&name, GENE_TYPE, GENE_TYPE);
                            tok_to_r.insert(rel_tok.into(), r);
                            r
                        }
                    }
                }
            };
            let weight = parse_weight(line.get(weight_col).map(AsRef::as_ref), path)?;
            self.link(r, GENE_TYPE, &line[0], GENE_TYPE, &line[1], weight);
            if !touched.contains(&r) {
                touched.push(r);
            }
            n_rows += 1;
        }
        Ok((n_rows, touched))
    }

    /// The QC pipeline of `legume_numeric::matrix::pair_graph` on one pair relation
    /// (in place), then its second-order and diffusion relations.
    fn refine_pair_relation(&mut self, r: usize, stem: &str, opts: &PpiOpts) {
        let t = self.type_index[GENE_TYPE];
        let mut graph = FeaturePairGraph {
            feature_names: self.types[t].names.clone(),
            n_features: self.types[t].names.len(),
            feature_edges: {
                let mut e: Vec<(usize, usize)> = self.relations[r]
                    .edges
                    .keys()
                    .map(|&(u, v)| (u as usize, v as usize))
                    .collect();
                e.sort_unstable();
                e
            },
        };
        let before = graph.feature_edges.len();
        graph.prune_by_shared_neighbors(opts.min_shared_neighbors);
        graph.cap_per_node_degree(opts.max_degree);
        graph.prune_by_min_degree(opts.min_degree);
        if graph.feature_edges.len() != before {
            let keep: FxHashMap<(u32, u32), ()> = graph
                .feature_edges
                .iter()
                .map(|&(u, v)| ((u as u32, v as u32), ()))
                .collect();
            self.relations[r].edges.retain(|k, _| keep.contains_key(k));
            info!(
                "fne: relation `{}` after PPI QC: {} of {before} edges kept",
                self.relations[r].name,
                self.relations[r].edges.len()
            );
        }
        if opts.snn_k > 0 {
            let snn = graph.shared_neighbor_edges(opts.snn_min_shared.max(1), opts.snn_k);
            let rs = self.relation(
                &format!("{GENE_TYPE}:{GENE_TYPE}/{stem}/snn"),
                GENE_TYPE,
                GENE_TYPE,
            );
            for &(u, v, c, un) in &snn {
                self.add_edge(rs, u as u32, v as u32, c as f32 / un.max(1) as f32);
            }
            info!(
                "fne: relation `{}`: {} second-order pairs (top {} per gene by Jaccard, ≥ {} shared neighbours)",
                self.relations[rs].name,
                snn.len(),
                opts.snn_k,
                opts.snn_min_shared.max(1)
            );
        }
        if opts.ppr_k > 0 {
            let ppr = graph.personalized_pagerank_top_k(opts.ppr_restart, PPR_EPS, opts.ppr_k);
            let rp = self.relation(
                &format!("{GENE_TYPE}:{GENE_TYPE}/{stem}/ppr"),
                GENE_TYPE,
                GENE_TYPE,
            );
            let mut n = 0usize;
            for (u, targets) in ppr.iter().enumerate() {
                let top = targets.first().map_or(0.0, |&(_, m)| m).max(1e-12);
                for &(v, m) in targets {
                    self.add_edge(rp, u as u32, v as u32, m / top);
                    n += 1;
                }
            }
            info!(
                "fne: relation `{}`: {n} directed top-{} diffusion targets ({} unique pairs; restart {})",
                self.relations[rp].name,
                opts.ppr_k,
                self.relations[rp].edges.len(),
                opts.ppr_restart
            );
        }
    }

    /// A typed edge file: `lhs_type lhs rhs_type rhs [weight]`; rows join
    /// the relation `lhs_type:rhs_type`.
    pub(crate) fn add_typed_file(&mut self, path: &str) -> anyhow::Result<()> {
        let read = read_lines_of_words_delim(path, detect_delimiter(path), -1)?;
        let mut n_rows = 0usize;
        let mut n_short = 0usize;
        let mut touched: Vec<usize> = Vec::new();
        for line in &read.lines {
            if line.is_empty() || line[0].starts_with('#') {
                continue;
            }
            if line.len() < 4 {
                n_short += 1;
                continue;
            }
            let (lt, rt) = (line[0].as_ref(), line[2].as_ref());
            let rel_name = format!("{lt}:{rt}");
            let r = self.relation(&rel_name, lt, rt);
            let weight = parse_weight(line.get(4).map(AsRef::as_ref), path)?;
            self.link(r, lt, &line[1], rt, &line[3], weight);
            if !touched.contains(&r) {
                touched.push(r);
            }
            n_rows += 1;
        }
        if n_short > 0 {
            warn!("fne: {path}: {n_short} rows had fewer than four columns and were skipped");
        }
        let names: Vec<String> = touched
            .iter()
            .map(|&r| format!("`{}`", self.relations[r].name))
            .collect();
        info!(
            "fne: {path}: {n_rows} typed rows → relation(s) {}",
            names.join(", ")
        );
        Ok(())
    }

    /// Attach text to a node (inserting it if new); a later call fills only
    /// the parts still missing.
    pub(crate) fn set_text(&mut self, ty: &str, name: &str, text: NodeText) {
        let (t, i) = self.node(ty, name);
        let slot = self.types[t].texts.entry(i).or_default();
        if slot.name.is_none() {
            slot.name = text.name;
        }
        if slot.text.is_none() {
            slot.text = text.text;
        }
    }

    /// A membership file `gene <TAB> label` → relation `gene:<ty>/<stem>`
    /// with the labels as nodes of type `ty`.
    pub(crate) fn add_membership_file(&mut self, ty: &str, path: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !ty.is_empty() && ty != GENE_TYPE,
            "--membership: the label type must be a non-empty name other than `{GENE_TYPE}` (got `{ty}`)"
        );
        let pairs = read_membership_pairs(path)?;
        let rel_name = format!("{GENE_TYPE}:{ty}/{}", file_stem(path));
        let r = self.relation(&rel_name, GENE_TYPE, ty);
        for (gene, label) in &pairs {
            self.link(r, GENE_TYPE, gene, ty, label, 1.0);
        }
        let rel = &self.relations[r];
        info!(
            "fne: {path}: {} membership rows → relation `{}` with {} unique edges over {} `{ty}` labels",
            pairs.len(),
            rel.name,
            rel.edges.len(),
            self.types[self.type_index[ty]].names.len()
        );
        Ok(())
    }

    /// Gene sets (a GAF after propagation, or a GMT) → relation
    /// `gene:term/<stem>`; sets outside `[min, max]` members are dropped
    /// (`max` 0 = no cap). Names and descriptions become term text.
    pub(crate) fn add_gene_sets(
        &mut self,
        sets: &GeneSets,
        stem: &str,
        min_members: usize,
        max_members: usize,
    ) -> usize {
        let rel_name = format!("{GENE_TYPE}:{TERM_TYPE}/{stem}");
        let r = self.relation(&rel_name, GENE_TYPE, TERM_TYPE);
        let mut terms: Vec<&Box<str>> = sets.term_genes.keys().collect();
        terms.sort();
        let mut n_kept = 0usize;
        for term in terms {
            let genes = &sets.term_genes[term];
            let n = genes.len();
            if n < min_members || (max_members > 0 && n > max_members) {
                continue;
            }
            n_kept += 1;
            let mut members: Vec<&Box<str>> = genes.iter().collect();
            members.sort();
            for g in members {
                self.link(r, GENE_TYPE, g, TERM_TYPE, term, 1.0);
            }
            if let Some(desc) = sets.names.get(term) {
                self.set_text(
                    TERM_TYPE,
                    term,
                    NodeText {
                        name: Some(desc.clone()),
                        text: None,
                    },
                );
            }
        }
        let rel = &self.relations[r];
        info!(
            "fne: gene sets `{stem}`: {n_kept} of {} sets within [{min_members}, {}] members → relation `{}` with {} edges",
            sets.term_genes.len(),
            if max_members == 0 {
                "∞".to_string()
            } else {
                max_members.to_string()
            },
            rel.name,
            rel.edges.len()
        );
        n_kept
    }

    /// The ontology's hierarchy over the term nodes already in the graph:
    /// relations `term:term/is_a` and `term:term/part_of` (child → parent),
    /// plus the terms' names and definitions as text.
    pub(crate) fn add_ontology(&mut self, onto: &Ontology) {
        let Some(&t) = self.type_index.get(TERM_TYPE) else {
            warn!("fne: --obo given but no term nodes are in the graph; its hierarchy is skipped");
            return;
        };
        // The ontology's own spelling of every id whose canonical CURIE
        // differs (EFO's OBO writes `efo:EFO_0004340` for `EFO:0004340`).
        let original: FxHashMap<Box<str>, &str> = onto
            .ids()
            .filter_map(|id| {
                let c = canonical_term_id(id);
                (c.as_ref() != id).then(|| (c.into_owned().into_boxed_str(), id))
            })
            .collect();
        let n_present = self.types[t].names.len();
        for i in 0..n_present {
            let id = &self.types[t].names[i];
            let id = original.get(id).copied().unwrap_or(id);
            let text = NodeText {
                name: onto.name(id).map(Box::from),
                text: onto.def(id).map(Box::from),
            };
            if text != NodeText::default() {
                let slot = self.types[t].texts.entry(i as u32).or_default();
                if slot.name.is_none() {
                    slot.name = text.name;
                }
                if slot.text.is_none() {
                    slot.text = text.text;
                }
            }
        }
        let r_is_a = self.relation(
            &format!("{TERM_TYPE}:{TERM_TYPE}/is_a"),
            TERM_TYPE,
            TERM_TYPE,
        );
        let r_part_of = self.relation(
            &format!("{TERM_TYPE}:{TERM_TYPE}/part_of"),
            TERM_TYPE,
            TERM_TYPE,
        );
        let index = &self.types[t].index;
        let mut edges: Vec<(u32, u32, Rel)> = onto
            .edges()
            .filter_map(|(child, parent, rel)| {
                Some((
                    *index.get(canonical_term_id(child).as_ref())?,
                    *index.get(canonical_term_id(parent).as_ref())?,
                    rel,
                ))
            })
            .collect();
        edges.sort_unstable_by_key(|(c, p, rel)| (*c, *p, matches!(rel, Rel::PartOf)));
        let n = edges.len();
        for (c, p, rel) in edges {
            let r = match rel {
                Rel::IsA => r_is_a,
                Rel::PartOf => r_part_of,
            };
            // Hierarchy edges are directed child → parent; the undirected
            // dedup of a same-type relation only folds the two directions.
            self.add_edge(r, c, p, 1.0);
        }
        info!(
            "fne: ontology hierarchy over {n_present} present terms: {n} edges ({} is_a, {} part_of)",
            self.relations[r_is_a].edges.len(),
            self.relations[r_part_of].edges.len()
        );
    }

    /// A region→gene link file `region gene [score]`; every region is tiled
    /// onto `window`-bp windows and each window links the gene in the
    /// relation `region:gene/<stem>` with the score as the edge weight.
    pub(crate) fn add_region_file(&mut self, path: &str, window: i64) -> anyhow::Result<()> {
        let rel_name = format!("{REGION_TYPE}:{GENE_TYPE}/{}", file_stem(path));
        let r = self.relation(&rel_name, REGION_TYPE, GENE_TYPE);
        let read = read_lines_of_words_delim(path, detect_delimiter(path), -1)?;
        let mut n_rows = 0usize;
        let mut n_bad = 0usize;
        for line in &read.lines {
            if line.len() < 2 || line[0].starts_with('#') {
                continue;
            }
            let Some(region) = outside_locus(&line[0]) else {
                n_bad += 1;
                continue;
            };
            let weight = parse_weight(line.get(2).map(AsRef::as_ref), path)?;
            self.link_region(r, &region, window, GENE_TYPE, &line[1], weight);
            n_rows += 1;
        }
        if n_bad > 0 {
            warn!(
                "fne: {path}: {n_bad} rows had a region that is not a coordinate and were skipped"
            );
        }
        let rel = &self.relations[r];
        info!(
            "fne: {path}: {n_rows} region rows on {}-bp windows → relation `{}` with {} edges over {} windows",
            window,
            rel.name,
            rel.edges.len(),
            self.types[self.type_index[REGION_TYPE]].names.len()
        );
        Ok(())
    }

    /// Link a region to a node of type `rt` from every `window`-bp window
    /// the region overlaps; the region nodes are shared by every input.
    pub(crate) fn link_region(
        &mut self,
        r: usize,
        region: &PeakCoord,
        window: i64,
        rt: &str,
        rhs: &str,
        weight: f32,
    ) {
        let mut wins = Vec::new();
        self.windows(region, window, &mut wins);
        let j = self.node_id(rt, rhs);
        self.link_windows(r, &wins, j, weight);
    }

    /// Local ids of the `window`-bp windows `region` overlaps (see
    /// [`genomic_data::coordinates::tile_windows`]), inserting new ones, into
    /// `out`. A window is named by its locus key, `chr:start-end` with the
    /// `chr` prefix dropped (`region` comes from [`outside_locus`]), written
    /// into a reused buffer, so a window already seen costs a lookup and no
    /// allocation.
    pub(crate) fn windows(&mut self, region: &PeakCoord, window: i64, out: &mut Vec<u32>) {
        use std::fmt::Write;
        out.clear();
        let t = self.type_id(REGION_TYPE);
        let (first, last) = if window <= 0 {
            (0, 0)
        } else {
            (
                region.start.div_euclid(window),
                (region.end - 1).max(region.start).div_euclid(window),
            )
        };
        let mut name = std::mem::take(&mut self.name_buf);
        for i in first..=last {
            name.clear();
            let (start, end) = if window <= 0 {
                (region.start, region.end)
            } else {
                (i * window, (i + 1) * window)
            };
            let _ = write!(name, "{}:{start}-{end}", region.chr);
            let nodes = &mut self.types[t];
            let id = match nodes.index.get(name.as_str()) {
                Some(&id) => id,
                None => {
                    let id = nodes.names.len() as u32;
                    let key: Box<str> = name.as_str().into();
                    nodes.names.push(key.clone());
                    nodes.index.insert(key, id);
                    id
                }
            };
            out.push(id);
        }
        self.name_buf = name;
    }
}
