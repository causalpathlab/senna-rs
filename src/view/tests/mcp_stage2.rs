use super::*;
use crate::view::mcp::jobs::senna_parses;
use std::path::PathBuf;

#[test]
fn donor_deviation_runs_on_the_run_and_names_its_gene_table_first() {
    let a: DonorDeviationCall = parse(json!({
        "from": "r.senna.json",
        "donor_of_batch": "donors.tsv",
        "args": ["--device", "cuda"],
    }))
    .unwrap();
    let t = donor_deviation_task(&a);
    assert_eq!(
        t.argv,
        [
            "donor-deviation",
            "--from",
            "r.senna.json",
            "-o",
            "r.stage2",
            "--donor-of-batch",
            "donors.tsv",
            "--device",
            "cuda"
        ]
    );
    senna_parses(&t.argv).unwrap();
    assert_eq!(
        t.made[0],
        (
            "donor_gene_deviation",
            PathBuf::from("r.stage2.donor_gene_deviation.parquet")
        )
    );
    assert_eq!(t.made.len(), 4);
    let bare: DonorDeviationCall = parse(json!({"from": "r", "out": "s2"})).unwrap();
    assert_eq!(
        donor_deviation_task(&bare).argv,
        ["donor-deviation", "--from", "r", "-o", "s2"]
    );
}
