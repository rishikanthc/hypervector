// tests/integration_test.rs

use hypervector::encoder::ObjectEncoder;
use hypervector::mbat::MBAT;
use hypervector::{set_global_seed, Hypervector, TieBreaker};
use serde_json::json;

#[test]
fn test_crate_import_and_basic_usage() {
    // Set the global RNG seed for reproducibility.
    set_global_seed(42);

    // Test generating a hypervector and its self cosine similarity.
    let hv = Hypervector::<MBAT>::generate(1000);
    // A hypervector should be identical to itself, yielding cosine similarity of 1.0.
    assert!(
        (hv.cosine_similarity(&hv) - 1.0).abs() < 1e-6,
        "Self similarity must be 1.0"
    );

    // Test the object encoder:
    let mut encoder = ObjectEncoder::<MBAT>::new(1000, TieBreaker::AlwaysPositive);
    let json_obj = json!({
        "name": "Alice",
        "age": 30,
        "interests": ["rust", "coding", "hiking"]
    });

    // Encode the JSON object into a hypervector.
    let encoded = encoder.encode_object(&json_obj);
    // Ensure that the encoded hypervector has the expected dimensionality.
    assert_eq!(encoded.to_vec().len(), 1000);

    // Verify that the encoded object can be fetched from the encoder’s codebook.
    let fetched = encoder
        .get_encoded_object(&json_obj)
        .expect("The encoded object should be retrievable from the codebook.");
    assert_eq!(
        &encoded, fetched,
        "The fetched encoding should match the computed encoding."
    );
}
