use super::*;
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Shape {
    Round,
    Flat,
}

#[derive(Args, Debug)]
struct Fit {
    #[arg(required = true, value_delimiter = ',')]
    data_files: Vec<Box<str>>,
    #[arg(long, short)]
    out: Box<str>,
    #[arg(long, short, value_delimiter = ',')]
    batch_files: Option<Vec<Box<str>>>,
    #[arg(long, default_value_t = 10)]
    steps: usize,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, value_enum, default_value = "round")]
    shape: Shape,
    #[arg(long, value_enum)]
    maybe_shape: Option<Shape>,
    #[arg(long)]
    fast: bool,
    #[arg(long, value_delimiter = ',', default_values_t = [1, 2])]
    levels: Vec<usize>,
    #[arg(long, num_args = 1..)]
    names: Vec<String>,
    #[arg(long, hide = true)]
    secret: Option<f32>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    Fit(Fit),
}

#[derive(Parser, Debug)]
struct Cli {
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

fn cli() -> Command {
    use clap::CommandFactory;
    let mut c = Cli::command();
    c.build();
    c
}

fn field<'a>(m: &'a mut Method, long: &str) -> &'a mut Field {
    m.fields.iter_mut().find(|f| f.long == long).unwrap()
}

#[test]
fn rows_come_from_the_clap_definition() {
    let m = Method::new(&cli(), "fit").unwrap();
    let longs: Vec<&str> = m.fields.iter().map(|f| f.long.as_str()).collect();
    assert_eq!(
        longs,
        [
            "steps",
            "seed",
            "shape",
            "maybe-shape",
            "fast",
            "levels",
            "names",
            "secret"
        ]
    );
    assert!(m.takes_batches());
    let f = &m.fields;
    assert_eq!(f[0].default, "10");
    assert_eq!(f[2].kind, Kind::Choice(vec!["round".into(), "flat".into()]));
    assert_eq!(
        f[3].kind,
        Kind::Choice(vec![String::new(), "round".into(), "flat".into()])
    );
    assert_eq!(f[4].kind, Kind::Flag { on: true });
    assert_eq!(f[5].default, "1,2");
    assert!(f[7].advanced && !f[0].advanced);
}

#[test]
fn an_untouched_form_passes_only_what_senna_run_fills() {
    let cli = cli();
    let m = Method::new(&cli, "fit").unwrap();
    let argv = m.argv(&["d1.zarr".into(), "d2.zarr".into()], &[], "o");
    assert_eq!(argv, ["fit", "d1.zarr", "d2.zarr", "--out", "o"]);
    assert_eq!(check(&cli, &argv), Ok(()));
    assert_eq!(m.changed(), 0);
}

#[test]
fn changed_rows_reach_the_command_line_as_clap_takes_them() {
    let cli = cli();
    let mut m = Method::new(&cli, "fit").unwrap();
    field(&mut m, "steps").value = "20".into();
    field(&mut m, "fast").toggle();
    field(&mut m, "shape").cycle(1);
    field(&mut m, "levels").value = "3, 4".into();
    field(&mut m, "names").value = "a,b".into();
    field(&mut m, "seed").value = "  ".into();
    let argv = m.argv(&["d.zarr".into()], &["b1.tsv".into()], "o");
    assert_eq!(
        argv,
        [
            "fit",
            "d.zarr",
            "--batch-files",
            "b1.tsv",
            "--out",
            "o",
            "--steps",
            "20",
            "--shape",
            "flat",
            "--fast",
            "--levels",
            "3,4",
            "--names",
            "a",
            "b"
        ]
    );
    assert_eq!(check(&cli, &argv), Ok(()));
    assert_eq!(m.changed(), 5);
    field(&mut m, "steps").reset();
    assert_eq!(m.changed(), 4);
}

#[test]
fn an_unset_choice_cycles_back_to_unset() {
    let mut m = Method::new(&cli(), "fit").unwrap();
    let f = field(&mut m, "maybe-shape");
    assert!(f.is_default() && f.shown() == "(unset)");
    f.cycle(-1);
    assert_eq!(f.value, "flat");
    f.cycle(1);
    assert!(f.is_default());
}

#[test]
fn clap_complaints_name_the_row() {
    let cli = cli();
    let mut m = Method::new(&cli, "fit").unwrap();
    field(&mut m, "steps").value = "many".into();
    let why = check(&cli, &m.argv(&["d.zarr".into()], &[], "o")).unwrap_err();
    assert!(why.contains("--steps"), "{why}");
    assert_eq!(blamed(&why, &m.fields), Some("steps"));
    field(&mut m, "steps").reset();
    let why = check(&cli, &m.argv(&[], &[], "o")).unwrap_err();
    assert_eq!(blamed(&why, &m.fields), None);
}

#[test]
fn batch_paths_pass_whole_even_with_spaces() {
    let m = Method::new(&cli(), "fit").unwrap();
    let argv = m.argv(
        &["d.zarr".into()],
        &["my labels/b 1.tsv".into(), "b2.tsv".into()],
        "o",
    );
    assert_eq!(argv[2..4], ["--batch-files", "my labels/b 1.tsv,b2.tsv"]);
}
