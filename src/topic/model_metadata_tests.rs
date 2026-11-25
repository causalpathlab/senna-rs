//! Reading a saved model's metadata back.

use super::TopicModelMetadata;

const FIELDS: &str = r#""model_type":"indexed_topic_masked","decoder_types":["nb"],
    "decoder_weights":[1.0],"n_features_encoder":4,"n_features_full":4,"n_topics":2,
    "encoder_hidden":[8],"num_levels":1,"level_decoder_dims":[4],"adj_method":"batch",
    "has_coarsening":false"#;

fn write(dir: &std::path::Path, extra: &str) -> String {
    let prefix = dir.join("m").to_string_lossy().into_owned();
    std::fs::write(
        format!("{prefix}.model.json"),
        format!("{{{FIELDS}{extra}}}"),
    )
    .unwrap();
    prefix
}

/// A model that recorded an encoder context window (any value, as the old
/// read treated any recorded one as windowed) was scored on a top-K
/// shortlist; scoring it on every gene would give a different latent without
/// a word, so scoring is refused. It still loads, so it can be continued.
#[test]
fn a_model_with_a_context_window_loads_but_is_not_scored() {
    let dir = tempfile::tempdir().unwrap();
    for k in ["64", "0"] {
        let prefix = write(dir.path(), &format!(r#","enc_context_size":{k}"#));
        let m = TopicModelMetadata::load(&prefix).unwrap();
        let err = m.ensure_window_free(&prefix).unwrap_err().to_string();
        assert!(err.contains("context window"), "{err}");
    }
    for extra in ["", r#","enc_context_size":null"#] {
        let prefix = write(dir.path(), extra);
        let m = TopicModelMetadata::load(&prefix).unwrap();
        assert!(m.ensure_window_free(&prefix).is_ok());
    }
}
