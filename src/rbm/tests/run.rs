use super::*;
use crate::planted::planted_zarr;
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: RbmArgs,
}

fn parse(argv: &[&str]) -> RbmArgs {
    let mut full = vec!["rbm"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).expect("valid argv").args
}

#[test]
fn a_fit_writes_every_table_and_a_manifest_and_freezes_onto_its_own_rho() {
    let dir = tempfile::tempdir().unwrap();
    let data = planted_zarr(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let common = [
        "-t",
        "3",
        "-i",
        "3",
        "--embedding-dim",
        "4",
        "--minibatch-size",
        "20",
        "--chains",
        "20",
        "--device",
        "cpu",
    ];
    let mut argv = vec![data.as_str(), "-o", &first];
    argv.extend_from_slice(&common);
    fit_rbm(&parse(&argv)).unwrap();

    for suffix in [
        "latent",
        "pb_latent",
        "cell_score",
        "feature_embedding",
        "hidden_embedding",
        "dictionary",
        "feature_bias",
        "log_likelihood",
    ] {
        let path = format!("{first}.{suffix}.parquet");
        assert!(std::path::Path::new(&path).exists(), "missing {path}");
    }
    let latent = Mat::from_parquet(&format!("{first}.latent.parquet")).unwrap();
    assert_eq!(latent.mat.ncols(), 3);
    assert!(latent.mat.iter().all(|x| x.is_finite()));
    let saved = legume_numeric::candle::candle_core::safetensors::load(
        format!("{first}.safetensors"),
        &legume_numeric::candle::candle_core::Device::Cpu,
    )
    .unwrap();
    for key in ["rho", "alpha", "b", "c", "mu", "lambda"] {
        assert!(saved.contains_key(key), "the saved machine lacks {key}");
    }
    let rho = Mat::from_parquet(&format!("{first}.feature_embedding.parquet")).unwrap();
    assert_eq!(rho.mat.ncols(), 4);

    let (m, _) = senna::run_manifest::RunManifest::load(std::path::Path::new(&format!(
        "{first}.senna.json"
    )))
    .unwrap();
    assert_eq!(m.kind, senna::run_manifest::RunKind::Rbm);
    let recorded: RbmArgs = serde_json::from_value(m.train_args.expect("train args").args).unwrap();
    assert_eq!(recorded.n_latent, 3);
    assert_eq!(recorded.embedding_dim, 4);

    // A second fit pinned to the first's ρ keeps it.
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let mut argv = vec![
        data.as_str(),
        "-o",
        &second,
        "--freeze-feature-embedding",
        &first,
    ];
    argv.extend_from_slice(&common);
    fit_rbm(&parse(&argv)).unwrap();
    let rho2 = Mat::from_parquet(&format!("{second}.feature_embedding.parquet")).unwrap();
    assert_eq!(rho2.rows, rho.rows);
    let diff = (&rho2.mat - &rho.mat).abs().max();
    assert!(diff < 1e-5, "frozen ρ moved by {diff}");
}
