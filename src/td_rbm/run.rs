//! `senna td-rbm`: read a two-track count backend, collapse its cells into
//! multilevel pseudobulks, pick the genes, start the genes' embedding and the
//! curve from the finest pseudobulks in their expression order (as `senna
//! time` starts), fit the machine on samples of every level
//! ([`super::train`]) and read every cell.

use super::args::TdRbmArgs;
use super::counts::{for_cell_blocks, pb_tracks, select_genes};
use super::curve::{grid_times, Spline};
use super::levels::pb_levels;
use super::model::Init0;
use super::start::{base_start, rho_start, row_states, start_along, STATE_DIMS};
use super::train::{fit, Level, TrainConfig, BETA0, GAMMA0};
use super::tree::Forest;
use data_beans::sparse_io::open_sparse_matrix_by_path;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};

/// Columns `0..n` of `rows` (`[N × n]` row-major) as named columns.
fn columns(prefix: &str, rows: &[Vec<f32>], n: usize) -> Vec<(Box<str>, Vec<f32>)> {
    (0..n)
        .map(|j| {
            (
                format!("{prefix}{}", j + 1).into(),
                rows.iter().map(|r| r[j]).collect(),
            )
        })
        .collect()
}

fn write(
    path: &str,
    key: &str,
    names: &[Box<str>],
    cols: &[(Box<str>, Vec<f32>)],
) -> anyhow::Result<()> {
    let cols: Vec<(Box<str>, Column)> = cols
        .iter()
        .map(|(n, v)| (n.clone(), Column::F32(v)))
        .collect();
    write_named_table(path, key, names, &cols)
}

pub fn run_td_rbm(args: &TdRbmArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.batch > 0 && args.chains > 0,
        "--batch and --chains must be positive"
    );
    anyhow::ensure!(
        args.depth > 0 && args.state_depth > 0 && args.assign_depth > 0.0,
        "--depth, --state-depth and --assign-depth must be positive"
    );
    anyhow::ensure!(
        args.states > 0 && args.epochs > args.time_epochs,
        "--states must be positive and --epochs exceed --time-epochs, so the states are fitted"
    );
    let dev = args.device.to_device(args.device_no)?;
    let cells = open_sparse_matrix_by_path(&args.tracks)?.column_names()?;
    let levels = pb_levels(&args.tracks, args.levels as usize)?;
    let member = levels.last().expect("at least one level").clone();
    let pbs = pb_tracks(&args.tracks, &levels)?;
    let finest = pbs.last().expect("at least one level");
    let genes = select_genes(finest, args.min_reads, args.max_genes);
    anyhow::ensure!(!genes.is_empty(), "no gene left to fit");
    let (g, n_pb) = (genes.len(), finest.cells.len());
    let f = 2 * g;
    info!(
        "td-rbm: {} cells; {} levels, {n_pb} finest pseudobulks; {} genes with both tracks, {g} fitted",
        cells.len(),
        levels.len(),
        finest.genes.len(),
    );
    // Every cell's reads on the fitted genes, `[C × F]` (unspliced genes
    // first); with `--steady-unspliced`, every cell's reads of a gene split
    // anew between the tracks at the gene's overall unspliced share (the
    // negative control: no splicing dynamics, the same either way in time).
    let names: Vec<Box<str>> = genes.iter().map(|&j| finest.genes[j].clone()).collect();
    let mut x = Vec::with_capacity(cells.len() * f);
    for_cell_blocks(&args.tracks, &names, |_, u, s| {
        for i in 0..u.nrows() {
            x.extend((0..g).map(|j| u[(i, j)]));
            x.extend((0..g).map(|j| s[(i, j)]));
        }
        Ok(())
    })?;
    let c = cells.len();
    if let Some(seed) = args.steady_unspliced {
        let mut rng = StdRng::seed_from_u64(seed);
        for j in 0..g {
            let (u, t) = (0..c).fold((0f64, 0f64), |(u, t), i| {
                let (a, b) = (f64::from(x[i * f + j]), f64::from(x[i * f + g + j]));
                (u + a, t + a + b)
            });
            let share = u / t.max(1.0);
            for i in 0..c {
                let n = (x[i * f + j] + x[i * f + g + j]).round() as u64;
                let a = Binomial::new(n, share).map_or(0, |b| b.sample(&mut rng));
                x[i * f + j] = a as f32;
                x[i * f + g + j] = (n - a) as f32;
            }
        }
        info!("td-rbm: every gene's reads split at its overall unspliced share (seed {seed})");
    }
    // Every level's pseudobulks, `[P × F]`, coarse to fine: sums of their
    // cells' reads.
    let rows: Vec<Vec<f32>> = levels
        .iter()
        .zip(&pbs)
        .map(|(member, t)| {
            let mut r = vec![0f32; t.cells.len() * f];
            for (i, p) in member.iter().enumerate() {
                let p = p.expect("the collapse places every cell");
                r[p * f..(p + 1) * f]
                    .iter_mut()
                    .zip(&x[i * f..(i + 1) * f])
                    .for_each(|(a, b)| *a += b);
            }
            r
        })
        .collect();

    // Where the fit starts: a forest over clusters of the finest
    // pseudobulks, every tree provisionally rooted at one end of its longest
    // path (or where `--root` says), the lineages its root-to-leaf paths; the
    // genes' embedding and baselines from every finest pseudobulk, every
    // lineage's curve through its pseudobulks in their order from the root.
    let fin = rows.last().expect("a level");
    let states = row_states(fin, f, STATE_DIMS);
    let forest = Forest::new(&states, args.clusters, args.knn, args.seed);
    let of_cell: Vec<usize> = member
        .iter()
        .map(|p| p.expect("the collapse places every cell"))
        .collect();
    let mut roots: Vec<usize> = (0..forest.trees.len()).map(|t| forest.far_end(t)).collect();
    let mut given = vec![false; forest.trees.len()];
    if let Some(path) = &args.root {
        let wanted: std::collections::HashSet<String> = std::fs::read_to_string(path)?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        let mut votes = vec![0usize; forest.clusters];
        for (i, c) in cells.iter().enumerate() {
            if wanted.contains(c.as_ref()) {
                votes[forest.cluster[of_cell[i]]] += 1;
            }
        }
        let top = (0..forest.clusters)
            .max_by_key(|&c| votes[c])
            .expect("a cluster");
        anyhow::ensure!(votes[top] > 0, "none of the --root cells is in the data");
        roots[forest.tree[top]] = top;
        given[forest.tree[top]] = true;
    }
    info!(
        "td-rbm: {} clusters in {} trees ({} with one cluster)",
        forest.clusters,
        forest.trees.len(),
        forest.trees.iter().filter(|t| t.len() == 1).count()
    );
    let (pu, ps): (Vec<f32>, Vec<f32>) = (
        fin.chunks(f).flat_map(|r| r[..g].to_vec()).collect(),
        fin.chunks(f).flat_map(|r| r[g..].to_vec()).collect(),
    );
    let h = args.embedding_dim.min(n_pb.saturating_sub(1)).max(1);
    let rho = rho_start(&pu, &ps, n_pb, g, h);
    let base = base_start(&ps, n_pb, g);
    let (k, m) = (args.grid as usize, args.knots as usize);
    let values = Spline::new(m, &grid_times(k), &Device::Cpu)?
        .values
        .flatten_all()?
        .to_vec1::<f32>()?;
    let begin = start_along(
        fin,
        f,
        &states,
        &forest,
        &roots,
        &rho,
        &values,
        (g, h, k, m),
        args.knn,
    );
    let l = begin.paths.len();

    let cfg = TrainConfig {
        grid: k,
        knots: m,
        states: args.states,
        embedding: h,
        lineages: l,
        concentration: args.lineage_concentration,
        epochs: args.epochs,
        time_epochs: args.time_epochs,
        steady_epochs: args.steady_epochs,
        direction_draws: args.direction_draws,
        knn: args.knn,
        smooth: args.smooth,
        assign_depth: args.assign_depth,
        depth: args.depth,
        state_depth: args.state_depth,
        batch: args.batch,
        chains: args.chains,
        sweeps: args.sweeps,
        lr: args.lr,
        seed: args.seed,
    };
    let levels_in: Vec<Level> = rows
        .into_iter()
        .zip(&pbs)
        .map(|(x, t)| Level {
            x,
            cells: t.cells.clone(),
        })
        .collect();
    let mut begin = begin;
    let curve = std::mem::take(&mut begin.curve);
    let model = fit(
        &levels_in,
        f,
        &Init0 {
            rho: &rho,
            base: &base,
            curve: &curve,
            beta: BETA0,
            gamma: GAMMA0,
        },
        begin,
        &forest,
        &states,
        &roots,
        &given,
        &cfg,
        &dev,
    )?;

    let out = model.summarise(&x, f, args.depth as f32, &dev)?;
    let rho_rows = model.energy.rho().to_vec2::<f32>()?;
    let (map, map_velocity) = map_space(&x, f, &rho_rows, &out.velocity)?;
    drop(x);

    // The start as the chosen roots have it, so its lineages are the
    // fitted ones' paths.
    let start: Vec<f32> = of_cell.iter().map(|&p| model.start_order[p]).collect();
    let pb_names: Vec<Box<str>> = of_cell.iter().map(|p| format!("pb{p}").into()).collect();
    let lineage_names: Vec<Box<str>> = out
        .lineage
        .iter()
        .map(|&i| format!("L{}", i + 1).into())
        .collect();
    let start_names: Vec<Box<str>> = of_cell
        .iter()
        .map(|&p| format!("L{}", model.start_lineage[p] + 1).into())
        .collect();
    let cluster_names: Vec<Box<str>> = of_cell
        .iter()
        .map(|&p| format!("C{}", forest.cluster[p]).into())
        .collect();
    let p_lineage = columns("p_L", &out.p_lineage, l);
    let mut cols: Vec<(Box<str>, Column)> = vec![
        ("lineage".into(), Column::Str(&lineage_names)),
        ("tau".into(), Column::F32(&out.tau)),
        ("tau_sd".into(), Column::F32(&out.tau_sd)),
        ("tau_start".into(), Column::F32(&start)),
        ("lineage_start".into(), Column::Str(&start_names)),
        ("pb".into(), Column::Str(&pb_names)),
        ("cluster".into(), Column::Str(&cluster_names)),
    ];
    cols.extend(p_lineage.iter().map(|(n, v)| (n.clone(), Column::F32(v))));
    write_named_table(
        &format!("{}.cell_time.parquet", args.out),
        "cell",
        &cells,
        &cols,
    )?;
    write(
        &format!("{}.cell_state.parquet", args.out),
        "cell",
        &cells,
        &columns("h", &out.state, args.states),
    )?;
    write(
        &format!("{}.cell_velocity_rho.parquet", args.out),
        "cell",
        &cells,
        &columns("v", &out.velocity, h),
    )?;
    // The map's space: every cell's expression on the leading principal
    // components (layouts, clusters), and its velocity there.
    let pcs = map.first().map_or(0, Vec::len);
    write(
        &format!("{}.cell_embedding.parquet", args.out),
        "cell",
        &cells,
        &columns("pc", &map, pcs),
    )?;
    write(
        &format!("{}.cell_velocity.parquet", args.out),
        "cell",
        &cells,
        &columns("pc", &map_velocity, pcs),
    )?;

    // Every used lineage's direction and what it rests on.
    {
        use std::io::Write;
        let mut file = std::fs::File::create(format!("{}.direction.tsv", args.out))?;
        let speed = model
            .energy
            .speed(Some(&model.log_weight))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        writeln!(file, "lineage\ttree\tclusters\tweight\tspeed\tdetermined\tp_forward\tlog_bf\tlog_bf_sd\tlog_abf\tgenes_for\tgenes_against")?;
        for d in &model.directions {
            let path = &model.paths[d.lineage];
            let along: Vec<String> = path.clusters.iter().map(|c| format!("C{c}")).collect();
            writeln!(
                file,
                "L{}\tT{}\t{}\t{:.4}\t{:.4}\t{}\t{:.6}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
                d.lineage + 1,
                path.tree + 1,
                along.join(">"),
                model.log_weight[d.lineage].exp(),
                speed[d.lineage],
                model.roots[path.tree].determined,
                d.p_forward,
                d.log_bf,
                d.log_bf_sd,
                d.log_abf,
                d.gene_votes.iter().filter(|&&v| v > 0.0).count(),
                d.gene_votes.iter().filter(|&&v| v < 0.0).count(),
            )?;
        }
    }
    // Every tree's root: the candidates' posterior.
    {
        use std::io::Write;
        let mut file = std::fs::File::create(format!("{}.roots.tsv", args.out))?;
        writeln!(
            file,
            "tree\tcandidate\tp\tchosen\tprovisional\tdetermined\tgiven\tmirrored"
        )?;
        for c in &model.roots {
            for (&a, &p) in c.candidates.iter().zip(&c.p) {
                writeln!(
                    file,
                    "T{}\tC{}\t{:.6}\t{}\tC{}\t{}\t{}\t{}",
                    c.tree + 1,
                    a,
                    p,
                    a == c.chosen,
                    c.provisional,
                    c.determined,
                    c.given,
                    c.mirrored
                )?;
            }
        }
    }
    // Every tree's edges and the splicing along them.
    {
        use std::io::Write;
        let mut file = std::fs::File::create(format!("{}.edges.tsv", args.out))?;
        writeln!(
            file,
            "tree\tfrom\tto\tlog_bf\tgenes_for\tgenes_against\tunder_chosen_root"
        )?;
        for c in &model.roots {
            let (parent, _) = forest.rooted(c.chosen);
            for &(a, b, v, up, down) in &c.edges {
                let kept = parent[b] == Some(a);
                writeln!(
                    file,
                    "T{}\tC{}\tC{}\t{:.3}\t{}\t{}\t{}",
                    c.tree + 1,
                    a,
                    b,
                    v,
                    up,
                    down,
                    if kept { "same" } else { "reversed" }
                )?;
            }
        }
    }
    let vote_cols: Vec<(Box<str>, Vec<f32>)> = model
        .directions
        .iter()
        .flat_map(|d| {
            [
                (
                    format!("log_bf_L{}", d.lineage + 1).into(),
                    d.gene_votes.clone(),
                ),
                (format!("z_L{}", d.lineage + 1).into(), d.gene_z.clone()),
            ]
        })
        .collect();
    if !vote_cols.is_empty() {
        write(
            &format!("{}.gene_direction.parquet", args.out),
            "gene",
            &names,
            &vote_cols,
        )?;
    }

    let e = &model.energy;
    let row = |t: &Tensor| -> anyhow::Result<Vec<f32>> { Ok(t.flatten_all()?.to_vec1::<f32>()?) };
    write(
        &format!("{}.gene_rates.parquet", args.out),
        "gene",
        &names,
        &[
            ("beta".into(), row(&e.beta()?)?),
            ("gamma".into(), row(&e.gamma()?)?),
            ("base".into(), row(e.base())?),
        ],
    )?;
    write(
        &format!("{}.feature_embedding.parquet", args.out),
        "gene",
        &names,
        &columns("rho", &e.rho().to_vec2::<f32>()?, h),
    )?;
    let t = model.tables(true)?;
    write(
        &format!("{}.gene_states.parquet", args.out),
        "gene",
        &names,
        &columns("w", &t.w.narrow(0, 0, g)?.to_vec2::<f32>()?, args.states),
    )?;
    let times: Vec<Box<str>> = (0..l)
        .flat_map(|i| {
            grid_times(k)
                .into_iter()
                .map(move |t| format!("L{}:{t:.4}", i + 1).into())
        })
        .collect();
    let mut curve = columns("z", &t.curve.to_vec2::<f32>()?, h);
    curve.extend(columns("dz", &t.velocity.to_vec2::<f32>()?, h));
    curve.push(("nu".into(), row(&t.size)?));
    write(
        &format!("{}.curve.parquet", args.out),
        "tau",
        &times,
        &curve,
    )?;
    write_manifest(args)?;
    info!(
        "Wrote {0}.{{cell_time,cell_state,cell_velocity,cell_velocity_rho,cell_embedding,gene_direction,gene_rates,feature_embedding,gene_states,curve}}.parquet, {0}.{{direction,roots,edges}}.tsv and {0}.senna.json",
        args.out
    );
    Ok(())
}

/// `{out}.senna.json`: the run as `senna view` and the other readers open
/// it. The cells' positions on their curves are the cell embedding and the
/// genes' ρ the feature embedding; the time, velocity and state tables are
/// recorded beside them. The two-track input is not recorded as data: its
/// rows are tracks, not the genes the embedding names.
fn write_manifest(args: &TdRbmArgs) -> anyhow::Result<()> {
    use senna::run_manifest::{
        record_train_args, write_run_manifest, RunDescription, RunKind, RunManifest,
    };
    let prefix: &str = &args.out;
    write_run_manifest(&RunDescription {
        train_args: Some(record_train_args(args)?),
        kind: RunKind::TdRbm,
        prefix,
        data_input: &[],
        data_multiome: None,
        data_batch: &[],
        data_input_null: &[],
        dictionary_suffix: None,
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        pb_reference_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_coembedding_suffix: None,
        carried: None,
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: None,
        cell_embedding_suffix: Some("cell_embedding.parquet"),
        cell_encoder_suffix: None,
        default_colour_by: "cluster",
        has_latent: false,
        has_cell_to_pb: false,
        has_pb_tree: false,
    })?;
    let path = std::path::PathBuf::from(format!("{prefix}.senna.json"));
    let (mut m, _) = RunManifest::load(&path)?;
    let base = std::path::Path::new(prefix)
        .file_name()
        .map_or_else(|| prefix.to_string(), |n| n.to_string_lossy().into_owned());
    m.outputs.cell_time = Some(format!("{base}.cell_time.parquet"));
    m.outputs.cell_velocity = Some(format!("{base}.cell_velocity.parquet"));
    m.outputs.cell_state = Some(format!("{base}.cell_state.parquet"));
    m.save(&path)
}

/// Principal components of the map space.
const MAP_PCS: usize = 30;

/// Every cell's place in the map space, and its velocity there.
type MapSpace = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// The cells' map space and their velocity in it, as scVelo has them:
/// every cell's `log1p` reads per 10⁴ (both tracks summed) on the fitted
/// genes, centred, on its leading principal components; and every cell's
/// velocity read out per gene (`ρ_g · v`, a `d log s_g/dt`, centred over
/// the genes) taken through the same components. `x` is `[C × F]`, `rho`
/// `[G × H]`, `velocity` `[C × H]`.
fn map_space(
    x: &[f32],
    f: usize,
    rho: &[Vec<f32>],
    velocity: &[Vec<f32>],
) -> anyhow::Result<MapSpace> {
    use legume_numeric::matrix::traits::RandomizedAlgs;
    let g = f / 2;
    let c = x.len() / f;
    let mut m = nalgebra::DMatrix::<f32>::zeros(c, g);
    for (i, row) in x.chunks(f).enumerate() {
        let total: f32 = row.iter().sum::<f32>().max(1.0);
        for j in 0..g {
            m[(i, j)] = ((row[j] + row[g + j]) / total * 1e4).ln_1p();
        }
    }
    for j in 0..g {
        let mean = m.column(j).mean();
        m.column_mut(j).add_scalar_mut(-mean);
    }
    let k = MAP_PCS.min(c.saturating_sub(1)).min(g).max(1);
    let (u, s, v) = m.rsvd(k)?; // [C × k], [k], [G × k]
    let k = s.len();
    let scores: Vec<Vec<f32>> = (0..c)
        .map(|i| (0..k).map(|j| u[(i, j)] * s[j]).collect())
        .collect();
    let moved: Vec<Vec<f32>> = velocity
        .iter()
        .map(|vc| {
            let per_gene: Vec<f32> = rho
                .iter()
                .map(|r| r.iter().zip(vc).map(|(a, b)| a * b).sum())
                .collect();
            let mean = per_gene.iter().sum::<f32>() / g.max(1) as f32;
            (0..k)
                .map(|j| (0..g).map(|q| (per_gene[q] - mean) * v[(q, j)]).sum())
                .collect()
        })
        .collect();
    Ok((scores, moved))
}
