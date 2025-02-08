//! Object encoder module.
//!
//! This module implements an object encoder for hyperdimensional computing using vector symbolic architectures (VSAs).
//!
//! The encoder takes as input a JSON object (using [serde_json::Value]) and produces an encoded hypervector.
//! It maintains two codebooks:
//! 1. **Token Codebook:** Maps each unique JSON token (from keys or values) to a randomly generated hypervector.
//! 2. **Object (Document) Codebook:** Maps a stringified JSON object to its encoded hypervector.
//!
//! Each key–value pair in the JSON object is encoded by binding the hypervector for the key with that for the value.
//! The final object representation is computed by bundling all these bound hypervectors together.
//!
//! This encoder is generic over any VSA type (i.e. any type implementing the [`VSA`] trait).
//! The methods [`token_codebook`], [`object_codebook`], and [`get_encoded_object`] allow you to inspect or retrieve the generated codebooks.

use serde_json::Value;
use std::collections::HashMap;

use crate::{Hypervector, TieBreaker, VSA};

/// An object encoder that produces hypervector encodings for JSON objects.
///
/// # Type Parameters
///
/// * `V` - A type that implements the [`VSA`] trait.
pub struct ObjectEncoder<V: VSA> {
    dim: usize,
    tie_breaker: TieBreaker,
    /// Codebook mapping unique tokens (from keys or values) to hypervectors.
    token_codebook: HashMap<String, Hypervector<V>>,
    /// Codebook mapping stringified JSON objects to their encoded hypervector.
    object_codebook: HashMap<String, Hypervector<V>>,
}

impl<V: VSA> ObjectEncoder<V> {
    /// Creates a new `ObjectEncoder`.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the hypervectors to generate.
    /// * `tie_breaker` - The tie-breaking rule used during bundling.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hypervector::encoder::ObjectEncoder;
    /// use hypervector::TieBreaker;
    /// use hypervector::mbat::MBAT;
    ///
    /// // For deterministic behavior in tests, use a fixed tie-breaker.
    /// let encoder = ObjectEncoder::<MBAT>::new(10000, TieBreaker::AlwaysPositive);
    /// ```
    pub fn new(dim: usize, tie_breaker: TieBreaker) -> Self {
        Self {
            dim,
            tie_breaker,
            token_codebook: HashMap::new(),
            object_codebook: HashMap::new(),
        }
    }

    /// Retrieves (or generates) the hypervector associated with a given token.
    ///
    /// Tokens are represented as strings. If the token does not exist in the codebook,
    /// a new random hypervector is generated (using [`Hypervector::generate`]) and stored.
    pub fn get_token_vector(&mut self, token: &str) -> Hypervector<V> {
        self.token_codebook
            .entry(token.to_string())
            .or_insert_with(|| Hypervector::<V>::generate(self.dim))
            .clone()
    }

    /// Encodes a JSON object into a hypervector.
    ///
    /// The JSON object is expected to be a JSON object (i.e. a map). For each key–value pair:
    ///
    /// 1. The key (a string) is used as a token.
    /// 2. If the value is an array, each element is encoded (if an element is an object, it is
    ///    encoded recursively; otherwise its token is used) and then bundled together.
    ///    Otherwise, the value is converted to a token string (via [`Self::value_to_token`])
    ///    and a hypervector is generated.
    /// 3. The key hypervector is bound with the (possibly bundled) value hypervector.
    ///
    /// All resulting bound hypervectors are then bundled (using the VSA’s bundling operator)
    /// to yield a final hypervector representing the object.
    ///
    /// The computed hypervector is stored in the object codebook. On subsequent calls
    /// with the same JSON object, the stored encoding is returned.
    pub fn encode_object(&mut self, json_obj: &Value) -> Hypervector<V> {
        // Use the string representation as a key.
        let obj_str = json_obj.to_string();
        if let Some(existing) = self.object_codebook.get(&obj_str) {
            return existing.clone();
        }

        let obj = json_obj
            .as_object()
            .expect("Expected a JSON object for encoding");
        let mut bound_vectors = Vec::with_capacity(obj.len());

        for (key, value) in obj {
            let key_vector = self.get_token_vector(key);
            let value_vector = match value {
                // For an array value, encode each element then bundle them.
                Value::Array(arr) => {
                    let mut elem_vectors = Vec::with_capacity(arr.len());
                    for elem in arr {
                        // If an element is an object, encode it recursively.
                        if elem.is_object() {
                            elem_vectors.push(self.encode_object(elem));
                        } else {
                            elem_vectors.push(self.get_token_vector(&Self::value_to_token(elem)));
                        }
                    }
                    // If the array is empty, use a default token.
                    if elem_vectors.is_empty() {
                        self.get_token_vector("empty_array")
                    } else if elem_vectors.len() == 1 {
                        elem_vectors.remove(0)
                    } else {
                        Hypervector::<V>::bundle_many(&elem_vectors, self.tie_breaker)
                    }
                }
                // (Optionally, you could also handle nested objects recursively here.)
                // Value::Object(_) => self.encode_object(value),
                _ => {
                    let token = Self::value_to_token(value);
                    self.get_token_vector(&token)
                }
            };

            // Bind the key vector with the (possibly bundled) value vector.
            let bound = key_vector.bind(&value_vector);
            bound_vectors.push(bound);
        }

        // Bundle all bound vectors into a single hypervector.
        let object_vector = Hypervector::<V>::bundle_many(&bound_vectors, self.tie_breaker);
        self.object_codebook.insert(obj_str, object_vector.clone());
        object_vector
    }

    /// Retrieves an encoded hypervector for a given JSON object, if it was encoded previously.
    ///
    /// Returns `None` if the JSON object has not been encoded.
    pub fn get_encoded_object(&self, json_obj: &Value) -> Option<&Hypervector<V>> {
        self.object_codebook.get(&json_obj.to_string())
    }

    /// Converts a JSON value into a string token.
    ///
    /// For simple types (strings, numbers, booleans) the conversion is straightforward.
    /// For other types (arrays or objects), the entire JSON value is stringified.
    fn value_to_token(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => value.to_string(),
        }
    }

    /// Returns a reference to the token codebook.
    ///
    /// This codebook maps each token (from keys or primitive values) to its generated hypervector.
    pub fn token_codebook(&self) -> &HashMap<String, Hypervector<V>> {
        &self.token_codebook
    }

    /// Returns a reference to the object (document) codebook.
    ///
    /// This codebook maps the string representation of JSON objects to their encoded hypervectors.
    pub fn object_codebook(&self) -> &HashMap<String, Hypervector<V>> {
        &self.object_codebook
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbat::MBAT;
    use crate::{Hypervector, TieBreaker};
    use serde_json::json;

    /// Test that calling `get_token_vector` with the same token yields identical hypervectors.
    #[test]
    fn test_token_vector_consistency() {
        let mut encoder = ObjectEncoder::<MBAT>::new(1000, TieBreaker::AlwaysPositive);
        let token = "exampleToken";
        let vec1 = encoder.get_token_vector(token);
        let vec2 = encoder.get_token_vector(token);
        assert_eq!(
            vec1, vec2,
            "Token vectors should be consistent across multiple calls"
        );
    }

    /// Test encoding a JSON object.
    #[test]
    fn test_encode_object() {
        let mut encoder = ObjectEncoder::<MBAT>::new(1000, TieBreaker::AlwaysPositive);
        let json_obj = json!({
            "firstName": "John",
            "lastName": "Doe",
            "isActive": true
        });

        let encoded1 = encoder.encode_object(&json_obj);
        let retrieved = encoder
            .get_encoded_object(&json_obj)
            .expect("Encoded object should be stored in the codebook");
        assert_eq!(
            encoded1, *retrieved,
            "Stored encoding should match computed encoding"
        );

        let encoded2 = encoder.encode_object(&json_obj);
        assert_eq!(
            encoded1, encoded2,
            "Re-encoding the same object should produce the same hypervector"
        );
    }

    /// Test that different JSON objects produce different encodings.
    #[test]
    fn test_different_objects_have_different_encodings() {
        let mut encoder = ObjectEncoder::<MBAT>::new(1000, TieBreaker::AlwaysPositive);
        let json_obj1 = json!({
            "firstName": "John",
            "lastName": "Doe",
            "isActive": true
        });
        let json_obj2 = json!({
            "firstName": "Jane",
            "lastName": "Doe",
            "isActive": false
        });

        let encoded1 = encoder.encode_object(&json_obj1);
        let encoded2 = encoder.encode_object(&json_obj2);

        assert_ne!(
            encoded1, encoded2,
            "Different JSON objects should have different encodings"
        );
    }

    /// Additional test: Generate 10 random entity hypervectors, verify that they are nearly orthogonal,
    /// then bind all possible combinations. For a particular pair, the re-computed binding should have a cosine
    /// similarity of ~1 with the stored binding for that pair, while bindings for other pairs remain nearly orthogonal.
    #[test]
    fn test_entity_binding_retrieval() {
        let mut encoder = ObjectEncoder::<MBAT>::new(1000, TieBreaker::AlwaysPositive);
        let num_entities = 10;
        let mut entity_vectors = Vec::new();

        // Generate hypervectors for tokens "entity0", "entity1", ... "entity9"
        for i in 0..num_entities {
            let token = format!("entity{}", i);
            let hv = encoder.get_token_vector(&token);
            entity_vectors.push(hv);
        }

        // Check that all entity hypervectors are nearly orthogonal.
        for i in 0..num_entities {
            for j in (i + 1)..num_entities {
                let sim = entity_vectors[i].cosine_similarity(&entity_vectors[j]);
                assert!(
                    sim < 0.2,
                    "Entity hypervectors {} and {} are not orthogonal enough: sim = {}",
                    i,
                    j,
                    sim
                );
            }
        }

        // Compute and store bindings for all unique pairs.
        let mut pair_bindings = Vec::new();
        for i in 0..num_entities {
            for j in (i + 1)..num_entities {
                let binding = entity_vectors[i].bind(&entity_vectors[j]);
                pair_bindings.push(((i, j), binding));
            }
        }

        // Choose a particular pair, e.g., (3, 7), for which to "retrieve" the binding.
        let query_pair = (3, 7);
        let query_binding = entity_vectors[query_pair.0].bind(&entity_vectors[query_pair.1]);

        // Compare the query binding to each stored binding.
        for &((i, j), ref binding) in &pair_bindings {
            let sim = query_binding.cosine_similarity(binding);
            if (i, j) == query_pair {
                assert!(
                    (sim - 1.0).abs() < 1e-6,
                    "Correct pair ({},{}) similarity not close to 1: sim = {}",
                    i,
                    j,
                    sim
                );
            } else {
                assert!(
                    sim < 0.1,
                    "Binding for pair ({},{}) has unexpected similarity with query pair: sim = {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    /// Test retrieval of the codebooks.
    #[test]
    fn test_fetch_codebooks() {
        let mut encoder = ObjectEncoder::<MBAT>::new(1000, TieBreaker::AlwaysPositive);

        // Encode a couple of objects.
        let json_obj1 = json!({
            "firstName": "John",
            "lastName": "Doe",
            "isActive": true
        });
        let json_obj2 = json!({
            "firstName": "Jane",
            "lastName": "Smith",
            "isActive": false
        });

        encoder.encode_object(&json_obj1);
        encoder.encode_object(&json_obj2);

        // Fetch token codebook.
        let token_book = encoder.token_codebook();
        assert!(
            !token_book.is_empty(),
            "Token codebook should not be empty after encoding objects"
        );
        println!("Token codebook has {} entries.", token_book.len());

        // Fetch object (document) codebook.
        let object_book = encoder.object_codebook();
        assert!(
            !object_book.is_empty(),
            "Object codebook should not be empty after encoding objects"
        );
        println!("Object codebook has {} entries.", object_book.len());
    }
}

#[cfg(test)]
mod ssp_tests {
    use super::*;
    use crate::ssp::SSP;
    use crate::{Hypervector, TieBreaker};
    use serde_json::json;

    /// Test that calling `get_token_vector` with the same token yields identical hypervectors for SSP.
    #[test]
    fn test_ssp_token_vector_consistency() {
        let mut encoder = ObjectEncoder::<SSP>::new(1000, TieBreaker::AlwaysPositive);
        let token = "exampleToken";
        let vec1 = encoder.get_token_vector(token);
        let vec2 = encoder.get_token_vector(token);
        assert_eq!(
            vec1, vec2,
            "SSP token vectors should be consistent across multiple calls"
        );
    }

    /// Test encoding a JSON object using SSP.
    #[test]
    fn test_ssp_encode_object() {
        let mut encoder = ObjectEncoder::<SSP>::new(1000, TieBreaker::AlwaysPositive);
        let json_obj = json!({
            "firstName": "John",
            "lastName": "Doe",
            "isActive": true
        });

        let encoded1 = encoder.encode_object(&json_obj);
        let retrieved = encoder
            .get_encoded_object(&json_obj)
            .expect("SSP: Encoded object should be stored in the codebook");
        assert_eq!(
            encoded1, *retrieved,
            "SSP: Stored encoding should match computed encoding"
        );

        let encoded2 = encoder.encode_object(&json_obj);
        assert_eq!(
            encoded1, encoded2,
            "SSP: Re-encoding the same object should produce the same hypervector"
        );
    }

    /// Test that different JSON objects produce different encodings using SSP.
    #[test]
    fn test_ssp_different_objects_have_different_encodings() {
        let mut encoder = ObjectEncoder::<SSP>::new(1000, TieBreaker::AlwaysPositive);
        let json_obj1 = json!({
            "firstName": "John",
            "lastName": "Doe",
            "isActive": true
        });
        let json_obj2 = json!({
            "firstName": "Jane",
            "lastName": "Doe",
            "isActive": false
        });

        let encoded1 = encoder.encode_object(&json_obj1);
        let encoded2 = encoder.encode_object(&json_obj2);
        assert_ne!(
            encoded1, encoded2,
            "SSP: Different JSON objects should have different encodings"
        );
    }

    /// Additional test: Generate 10 random entity hypervectors using SSP, verify that they are nearly orthogonal,
    /// then bind all possible combinations. For a particular pair, the re-computed binding should have a cosine
    /// similarity of ~1 with the stored binding for that pair, while bindings for other pairs remain nearly orthogonal.
    #[test]
    fn test_ssp_entity_binding_retrieval() {
        let mut encoder = ObjectEncoder::<SSP>::new(1000, TieBreaker::AlwaysPositive);
        let num_entities = 10;
        let mut entity_vectors = Vec::new();

        // Generate hypervectors for tokens "entity0", "entity1", ... "entity9"
        for i in 0..num_entities {
            let token = format!("entity{}", i);
            let hv = encoder.get_token_vector(&token);
            entity_vectors.push(hv);
        }

        // Check that all entity hypervectors are nearly orthogonal.
        for i in 0..num_entities {
            for j in (i + 1)..num_entities {
                let sim = entity_vectors[i].cosine_similarity(&entity_vectors[j]);
                assert!(
                    sim < 0.2,
                    "SSP: Entity hypervectors {} and {} are not orthogonal enough: sim = {}",
                    i,
                    j,
                    sim
                );
            }
        }

        // Compute and store bindings for all unique pairs.
        let mut pair_bindings = Vec::new();
        for i in 0..num_entities {
            for j in (i + 1)..num_entities {
                let binding = entity_vectors[i].bind(&entity_vectors[j]);
                pair_bindings.push(((i, j), binding));
            }
        }

        // Choose a particular pair, e.g., (3, 7), for which to "retrieve" the binding.
        let query_pair = (3, 7);
        let query_binding = entity_vectors[query_pair.0].bind(&entity_vectors[query_pair.1]);

        // Compare the query binding to each stored binding.
        for &((i, j), ref binding) in &pair_bindings {
            let sim = query_binding.cosine_similarity(binding);
            if (i, j) == query_pair {
                assert!(
                    (sim - 1.0).abs() < 1e-6,
                    "SSP: Correct pair ({},{}) similarity not close to 1: sim = {}",
                    i,
                    j,
                    sim
                );
            } else {
                assert!(
                    sim < 0.1,
                    "SSP: Binding for pair ({},{}) has unexpected similarity with query pair: sim = {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    /// Test retrieval of the codebooks using SSP.
    #[test]
    fn test_ssp_fetch_codebooks() {
        let mut encoder = ObjectEncoder::<SSP>::new(1000, TieBreaker::AlwaysPositive);

        // Encode a couple of objects.
        let json_obj1 = json!({
            "firstName": "John",
            "lastName": "Doe",
            "isActive": true
        });
        let json_obj2 = json!({
            "firstName": "Jane",
            "lastName": "Smith",
            "isActive": false
        });

        encoder.encode_object(&json_obj1);
        encoder.encode_object(&json_obj2);

        // Fetch token codebook.
        let token_book = encoder.token_codebook();
        assert!(
            !token_book.is_empty(),
            "SSP: Token codebook should not be empty after encoding objects"
        );
        println!("SSP: Token codebook has {} entries.", token_book.len());

        // Fetch object (document) codebook.
        let object_book = encoder.object_codebook();
        assert!(
            !object_book.is_empty(),
            "SSP: Object codebook should not be empty after encoding objects"
        );
        println!("SSP: Object codebook has {} entries.", object_book.len());
    }
}
