//! Reading a saved model's metadata back.

use super::TopicModelMetadata;

const FIELDS: &str = r#""model_type":"indexed_topic_masked","decoder_types":["nb"],
    "decoder_weights":[1.0],"n_features_encoder":4,"n_features_full":4,"n_topics":2,
    "encoder_hidden":[8],"num_levels":1,"level_decoder_dims":[4],"adj_method":"batch",
    "has_coarsening":false"#;

fn write_and_load(dir: &std::path::Path, extra: &str) -> anyhow::Result<TopicModelMetadata> {
    let prefix = dir.join("m").to_string_lossy().into_owned();
    std::fs::write(
        format!("{prefix}.model.json"),
        format!("{{{FIELDS}{extra}}}"),
    )?;
    TopicModelMetadata::load(&prefix)
}

/// A model that recorded an encoder context window was scored on a top-K
/// shortlist; scoring it on every gene would give a different latent without
/// a word, so loading it is refused and the message says why.
#[test]
fn a_model_with_a_context_window_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let Err(err) = write_and_load(dir.path(), r#","enc_context_size":64"#) else {
        panic!("a windowed model loaded");
    };
    let err = err.to_string();
    assert!(err.contains("context window"), "{err}");

    assert!(write_and_load(dir.path(), r#","enc_context_size":0"#).is_ok());
    assert!(write_and_load(dir.path(), "").is_ok());
}
