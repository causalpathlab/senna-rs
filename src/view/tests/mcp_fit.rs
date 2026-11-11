use super::*;

#[test]
fn help_covers_predict_update_and_each_layout() {
    let help = |c: &str| match &run_help(parse(json!({"method": c})).unwrap()).unwrap()[0] {
        Content::Text(t) => t.clone(),
        _ => panic!("help is text"),
    };
    assert!(help("predict").contains("--model"));
    assert!(help("update").contains("--model"));
    assert!(help("layout umap").contains("--knn"));
    assert!(help("layout phate").contains("--knn"));
    assert!(run_help(parse(json!({"method": "layout pca"})).unwrap()).is_err());
    assert!(run_help(parse(json!({"method": "view"})).unwrap()).is_err());
}
