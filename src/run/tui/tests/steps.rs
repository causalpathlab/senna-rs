use super::*;

#[test]
fn a_step_is_offered_only_to_the_families_it_runs_on_and_parses_untouched() {
    use clap::CommandFactory;
    let mut cli = crate::Cli::command();
    cli.build();
    let names =
        |m: &str| -> Vec<&'static str> { offered(&cli, m).iter().map(|s| s.spec.name).collect() };
    for m in ["topic", "bge", "svd", "vae"] {
        assert!(names(m).contains(&"donor-deviation"), "{m}");
    }
    for m in ["simba", "ode"] {
        assert!(!names(m).contains(&"donor-deviation"), "{m}");
        assert_eq!(names(m)[0], "layout tsne", "{m}");
    }
    for m in super::super::METHODS {
        for step in offered(&cli, m) {
            let argv = step.argv("o");
            assert_eq!(
                super::super::form::check(&cli, &argv),
                Ok(()),
                "{m}: {argv:?}"
            );
        }
    }
}
