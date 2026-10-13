//! `senna td-rbm`: read a two-track count backend, collapse its cells into
//! multilevel pseudobulks, pick the genes, start the genes' embedding and the
//! curve from the finest pseudobulks in their expression order (as `senna
//! time` starts), fit the machine on samples of every level
//! ([`super::train`]) and read every cell.

use super::args::TdRbmArgs;
use super::counts::{for_cell_blocks, pb_tracks, select_genes};
use super::curve::{grid_times, Spline};
use super::init::{diffusion_order, expression_states};
use super::levels::pb_levels;
use super::model::Init0;
use super::start::{base_start, curve_start, kmeans, rho_start, STATE_DIMS};
use super::train::{fit, Level, TrainConfig, BETA0, GAMMA0};
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
        args.depth > 0 && args.assign_depth > 0.0,
        "--depth and --assign-depth must be positive"
    );
    anyhow::ensure!(
        args.grid >= 2 && args.knots >= 4,
        "--grid needs 2 times or more, --knots 4 or more"
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

    // Where the fit starts: the finest pseudobulks split into lineages by
    // k-means on their expression states, each lineage's pseudobulks in their
    // diffusion order, the genes' embedding and
    // baselines from every finest pseudobulk, and every lineage's curve
    // through its pseudobulks in that order.
    let states = expression_states(finest, &genes, STATE_DIMS);
    let l = args.max_lineages.max(1);
    let start_lineage = kmeans(&states, l, args.seed);
    let mut pb_order = vec![0f32; n_pb];
    let fin = rows.last().expect("a level");
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
    let mut curve = Vec::with_capacity(l * m * h);
    for lin in 0..l {
        let members: Vec<usize> = (0..n_pb).filter(|&p| start_lineage[p] == lin).collect();
        let order: Vec<f32> = if members.len() > 2 {
            let sub = nalgebra::DMatrix::from_fn(members.len(), states.ncols(), |i, j| {
                states[(members[i], j)]
            });
            diffusion_order(&sub, args.knn.min(members.len() - 1))
        } else {
            (0..members.len())
                .map(|i| (i as f32 + 0.5) / members.len().max(1) as f32)
                .collect()
        };
        for (&p, &o) in members.iter().zip(&order) {
            pb_order[p] = o;
        }
        let (pu, ps) = (&pu, &ps);
        let tot: Vec<f32> = members
            .iter()
            .flat_map(|&p| (0..g).map(move |j| pu[p * g + j] + ps[p * g + j]))
            .collect();
        curve.extend(curve_start(&tot, &order, &rho, &values, (g, h, k, m)));
    }

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
        orient_epochs: args.orient_epochs,
        orient_draws: args.orient_draws,
        knn: args.knn,
        assign_depth: args.assign_depth,
        depth: args.depth,
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
        &start_lineage,
        &cfg,
        &dev,
    )?;

    let out = model.summarise(&x, f, args.depth as f32, &dev)?;
    drop(x);

    let of_cell: Vec<usize> = member
        .iter()
        .map(|p| p.expect("the collapse places every cell"))
        .collect();
    let start: Vec<f32> = of_cell.iter().map(|&p| pb_order[p]).collect();
    let pb_names: Vec<Box<str>> = of_cell.iter().map(|p| format!("pb{p}").into()).collect();
    let lineage_names: Vec<Box<str>> = out
        .lineage
        .iter()
        .map(|&i| format!("L{}", i + 1).into())
        .collect();
    let start_names: Vec<Box<str>> = of_cell
        .iter()
        .map(|&p| format!("L{}", start_lineage[p] + 1).into())
        .collect();
    let p_lineage = columns("p_L", &out.p_lineage, l);
    let mut cols: Vec<(Box<str>, Column)> = vec![
        ("lineage".into(), Column::Str(&lineage_names)),
        ("tau".into(), Column::F32(&out.tau)),
        ("tau_sd".into(), Column::F32(&out.tau_sd)),
        ("tau_start".into(), Column::F32(&start)),
        ("lineage_start".into(), Column::Str(&start_names)),
        ("pb".into(), Column::Str(&pb_names)),
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
        &format!("{}.cell_velocity.parquet", args.out),
        "cell",
        &cells,
        &columns("v", &out.velocity, h),
    )?;

    // Every used lineage's direction and what it rests on.
    {
        use std::io::Write;
        let mut file = std::fs::File::create(format!("{}.direction.tsv", args.out))?;
        let speed = model.energy.speed()?.flatten_all()?.to_vec1::<f32>()?;
        writeln!(file, "lineage\tweight\tspeed\tdetermined\treversed\tz\tlambda\tlambda_null\tlambda_sd\tlambda_per_read\tgenes_for\tgenes_against")?;
        for d in &model.directions {
            writeln!(
                file,
                "L{}\t{:.4}\t{:.4}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{}\t{}",
                d.lineage + 1,
                model.log_weight[d.lineage].exp(),
                speed[d.lineage],
                d.determined,
                d.reversed,
                d.z,
                d.lambda,
                d.lambda_null,
                d.lambda_sd,
                d.lambda_per_read,
                d.gene_votes.iter().filter(|&&v| v > 0.0).count(),
                d.gene_votes.iter().filter(|&&v| v < 0.0).count(),
            )?;
        }
    }
    let vote_cols: Vec<(Box<str>, Vec<f32>)> = model
        .directions
        .iter()
        .map(|d| {
            (
                format!("vote_L{}", d.lineage + 1).into(),
                d.gene_votes.clone(),
            )
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
    info!(
        "Wrote {0}.{{cell_time,cell_state,cell_velocity,gene_direction,gene_rates,feature_embedding,gene_states,curve}}.parquet and {0}.direction.tsv",
        args.out
    );
    Ok(())
}
