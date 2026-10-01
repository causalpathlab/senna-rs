use super::*;
use crate::run::tui::form::{check, Kind, Method};

const CLONES: &str = r#"{"describe":1,"program":"tool","version":"9.9.9","command":"split",
 "about":"Split cells","args":[
 {"id":"query","long":null,"short":null,"positional":true,"help":"Inputs","long_help":null,
  "action":"append","value_type":"text","values":[],"default":[],"delimiter":null,
  "num_args_min":1,"num_args_max":null,"hidden":false,"required":false,"global":false},
 {"id":"out","long":"out","short":"o","positional":false,"help":"Prefix","long_help":null,
  "action":"set","value_type":"text","values":[],"default":[],"delimiter":null,
  "num_args_min":1,"num_args_max":1,"hidden":false,"required":true,"global":false},
 {"id":"k_max","long":"k-max","short":null,"positional":false,"help":"K","long_help":"Largest K",
  "action":"set","value_type":"unsigned","values":[],"default":["8"],"delimiter":null,
  "num_args_min":1,"num_args_max":1,"hidden":false,"required":false,"global":false},
 {"id":"engine","long":"engine","short":null,"positional":false,"help":"How","long_help":null,
  "action":"set","value_type":"text","values":["a","b"],"default":["a"],"delimiter":null,
  "num_args_min":1,"num_args_max":1,"hidden":false,"required":false,"global":false},
 {"id":"skip","long":"skip","short":null,"positional":false,"help":null,"long_help":null,
  "action":"append","value_type":"text","values":[],"default":["x","y"],"delimiter":",",
  "num_args_min":1,"num_args_max":1,"hidden":true,"required":false,"global":false},
 {"id":"refs","long":"refs","short":null,"positional":false,"help":null,"long_help":null,
  "action":"append","value_type":"text","values":[],"default":[],"delimiter":null,
  "num_args_min":1,"num_args_max":null,"hidden":false,"required":false,"global":false},
 {"id":"fast","long":"fast","short":null,"positional":false,"help":null,"long_help":null,
  "action":"set_true","value_type":"text","values":[],"default":["false"],"delimiter":null,
  "num_args_min":0,"num_args_max":0,"hidden":false,"required":false,"global":false},
 {"id":"help","long":"help","short":"h","positional":false,"help":"Print help","long_help":null,
  "action":"help","value_type":"text","values":[],"default":[],"delimiter":null,
  "num_args_min":0,"num_args_max":0,"hidden":false,"required":false,"global":false}
]}"#;

fn argv(words: &[&str]) -> Vec<String> {
    words.iter().map(ToString::to_string).collect()
}

#[test]
fn a_description_becomes_a_form_like_senna_s_own() {
    let d = Description::parse(CLONES).unwrap();
    let cmd = d.command();
    let m = Method::new(&cmd, "split").unwrap();
    assert_eq!(m.about, "Split cells");
    let field = |l: &str| m.fields.iter().find(|f| f.long == l).unwrap();
    assert_eq!(field("k-max").default, "8");
    assert_eq!(field("k-max").long_help, "Largest K");
    assert_eq!(
        field("engine").kind,
        Kind::Choice(vec!["a".into(), "b".into()])
    );
    assert!(field("skip").advanced);
    assert_eq!(field("skip").default, "x,y");
    assert_eq!(field("fast").kind, Kind::Flag { on: true });
    assert!(m.fields.iter().all(|f| f.long != "out" && f.long != "help"));
    let line = m.argv(&argv(&["a.zarr", "b.zarr"]), &[], "cnv");
    assert_eq!(line, argv(&["split", "a.zarr", "b.zarr", "--out", "cnv"]));
    assert_eq!(check(&cmd, &line), Ok(()));
}

#[test]
fn the_rebuilt_command_checks_values_and_names() {
    let cmd = Description::parse(CLONES).unwrap().command();
    let ok = |w: &[&str]| check(&cmd, &argv(w));
    assert!(ok(&["split", "q", "--out", "o", "--k-max", "3", "--refs", "r1", "r2"]).is_ok());
    assert!(ok(&["split", "q", "--out", "o", "--k-max", "nope"])
        .unwrap_err()
        .contains("--k-max"));
    assert!(ok(&["split", "q", "--out", "o", "--engine", "c"]).is_err());
    assert!(ok(&["split", "q", "--out", "o", "--nope"]).is_err());
    assert!(ok(&["split", "q"]).is_err());
}

#[test]
fn another_format_is_refused() {
    let other = CLONES.replacen(r#""describe":1"#, r#""describe":2"#, 1);
    assert!(Description::parse(&other).is_err());
    assert!(Description::parse("not json").is_err());
}
