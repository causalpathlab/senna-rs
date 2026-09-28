//! `senna docs` — the method write-ups, compiled into the binary.
//!
//! `include_str!`, not paths read at runtime. The binary is often the only thing on the machine
//! that ran the analysis (installed with `cargo install`, or copied to a cluster with no checkout
//! beside it), and a doc you cannot reach from there is a doc nobody reads. It also means the
//! build breaks if one of these files is moved or deleted — which enforces that they *exist*,
//! though not that they are *current*.

use anyhow::Result;
use clap::builder::PossibleValue;
use clap::{Args, ValueEnum};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Topic {
    /// Reference-based deconvolution of pseudobulk profiles.
    Deconvolve,
    // Moved to lupin; kept so the old names say where to look.
    #[value(hide = true)]
    Annotation,
    #[value(hide = true)]
    Grouping,
    #[value(hide = true)]
    OntologyPlan,
    #[value(hide = true)]
    RootingPlan,
}

/// Every write-up, in one place: the topic, a one-line blurb, and the text.
///
/// The listing `senna docs` prints and the text `senna docs <TOPIC>` prints are both read from
/// here, so the index can never advertise a topic the command cannot serve — which is exactly
/// what happens when the two are maintained separately.
const DOCS: &[(Topic, &str, &str)] = &[(
    Topic::Deconvolve,
    "METHOD  reference-based deconvolution of pseudobulk profiles",
    include_str!("../docs/deconvolve.md"),
)];

#[derive(Args, Debug)]
pub struct DocsArgs {
    #[arg(
        value_enum,
        help = "Which write-up to print (omit to list what there is)"
    )]
    pub topic: Option<Topic>,
}

pub fn run_docs(args: &DocsArgs) -> Result<()> {
    let Some(want) = args.topic else {
        println!("senna method write-ups (`senna docs <TOPIC>` to read one):\n");
        for (topic, blurb, _) in DOCS {
            // The slug clap will actually ACCEPT. Deriving it with `format!("{topic:?}")` prints
            // `ontologyplan` while the parser wants `ontology-plan`, so the listing would
            // advertise names the command then refuses — the one failure this table prevents.
            let slug = topic
                .to_possible_value()
                .as_ref()
                .map(PossibleValue::get_name)
                .unwrap_or_default()
                .to_string();
            println!("  {slug:<14} {blurb}");
        }
        println!(
            "\nAnnotation and lineage write-ups moved to lupin: `lupin docs annotation`, \
             `grouping`, `ontology-plan`, `rooting-plan`."
        );
        println!(
            "\nThe per-cell feature matrices these commands read are built by `faba`; \
             see `faba docs profiling`.\n"
        );
        return Ok(());
    };
    let Some(text) = DOCS
        .iter()
        .find(|(t, _, _)| *t == want)
        .map(|(_, _, text)| *text)
    else {
        let slug = want
            .to_possible_value()
            .as_ref()
            .map(PossibleValue::get_name)
            .unwrap_or_default()
            .to_string();
        println!("This write-up moved to lupin: run `lupin docs {slug}`.");
        return Ok(());
    };
    println!("{text}");
    Ok(())
}
