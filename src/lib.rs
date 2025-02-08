//! # Hypervector Crate
//!
//! This crate implements high-dimensional vectors for hyperdimensional computing and
//! Vector Symbolic Architectures (VSAs). It currently provides two implementations:
//!
//! - **MBAT**: Bipolar vectors (elements in {-1, +1}).
//! - **SSP**: Semantic Spatial Parameters (implemented in a separate module).
//!
//! Core components such as a global RNG, the `VSA` trait, and a generic `Hypervector` type
//! are defined here.

pub mod hypervector {
    pub mod encoder;
    pub mod mbat; // if you have this module

    // Re-export core items
    pub use crate::{Hypervector, TieBreaker, VSA};
}

use once_cell::sync::Lazy;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};
use std::sync::Mutex;

// Now you can safely import MBAT using the hypervector module:
use crate::hypervector::mbat::MBAT;

/// A lazily-initialized, thread-safe global RNG used for hypervector generation and operations.
///
/// This global RNG is used by all methods that do not receive their own RNG.
static GLOBAL_RNG: Lazy<Mutex<StdRng>> = Lazy::new(|| Mutex::new(StdRng::from_entropy()));

/// Sets the seed for the global RNG. This affects all hypervector generation and operations that
/// use the global RNG.
///
/// # Example
///
/// ```rust
/// use hypervector::set_global_seed;
///
/// // Set the global RNG seed to 42 for reproducibility.
/// set_global_seed(42);
/// ```
pub fn set_global_seed(seed: u64) {
    let mut rng = GLOBAL_RNG.lock().unwrap();
    *rng = StdRng::seed_from_u64(seed);
}

/// Options for tie-breaking during the bundling operation.
///
/// When bundling two hypervectors, an element-wise sum may result in a tie (i.e. a zero).
/// This enum specifies how such ties are resolved.
#[derive(Debug, Clone, Copy)]
pub enum TieBreaker {
    /// Always choose +1 when a tie occurs.
    AlwaysPositive,
    /// Always choose -1 when a tie occurs.
    AlwaysNegative,
    /// Randomly choose between +1 and -1 when a tie occurs.
    Random,
}

/// The `VSA` trait defines the interface for a Vector Symbolic Architecture.
/// New VSA implementations (such as SSP, MBAT, or FHRR) can be added by implementing this trait.
///
/// # Associated Types
///
/// * `Elem` - The type used to represent each element in the hypervector.
pub trait VSA: Sized + Clone {
    /// The type used to represent each element in the hypervector.
    type Elem: Copy + std::fmt::Debug + PartialEq + Into<f32>;

    /// Generates a random hypervector of the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensionality of the hypervector.
    /// * `rng` - A mutable reference to a random number generator.
    fn generate(dim: usize, rng: &mut impl Rng) -> Self;

    /// Bundles (superposes) two hypervectors.
    ///
    /// For example, for MBAT this is typically implemented as the element-wise sum followed by a
    /// sign function with tie-breaking.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to bundle with.
    /// * `tie_breaker` - The rule to resolve ties.
    /// * `rng` - A mutable reference to a random number generator.
    fn bundle(&self, other: &Self, tie_breaker: crate::TieBreaker, rng: &mut impl Rng) -> Self;

    /// Binds two hypervectors.
    ///
    /// For MBAT, this is implemented as the element-wise product.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to bind with.
    fn bind(&self, other: &Self) -> Self;

    /// Computes the cosine similarity between two hypervectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to compare with.
    fn cosine_similarity(&self, other: &Self) -> f32;

    /// Computes the normalized Hamming distance between two hypervectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to compare with.
    fn hamming_distance(&self, other: &Self) -> f32;

    /// Converts the hypervector into a plain `Vec<f32>`.
    fn to_vec(&self) -> Vec<f32>;

    /// Bundles many hypervectors (folding a slice using the bundling operation).
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty.
    fn bundle_many(vectors: &[Self], tie_breaker: crate::TieBreaker, rng: &mut impl Rng) -> Self {
        assert!(
            !vectors.is_empty(),
            "Cannot bundle an empty slice of hypervectors"
        );
        let mut result = vectors[0].clone();
        for vec in &vectors[1..] {
            result = result.bundle(vec, tie_breaker, rng);
        }
        result
    }

    /// Binds many hypervectors (folding a slice using the binding operation).
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty.
    fn bind_many(vectors: &[Self]) -> Self {
        assert!(
            !vectors.is_empty(),
            "Cannot bind an empty slice of hypervectors"
        );
        let mut result = vectors[0].clone();
        for vec in &vectors[1..] {
            result = result.bind(vec);
        }
        result
    }
}

/// A generic hypervector type parameterized over a VSA implementation.
///
/// This type wraps an inner hypervector and provides high-level operations for generation,
/// bundling, binding, similarity comparisons, and conversion to a plain vector.
///
/// # Type Parameters
///
/// * `V` - A type that implements the `VSA` trait.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hypervector<V: VSA> {
    /// The underlying hypervector.
    pub inner: V,
}

impl<V: VSA> Hypervector<V> {
    /// Generates a random hypervector of the given dimension using the global RNG.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensionality of the hypervector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hypervector::Hypervector;
    /// use hypervector::mbat::MBAT;
    ///
    /// let hv = Hypervector::<MBAT>::generate(1000);
    /// ```
    pub fn generate(dim: usize) -> Self {
        let mut rng = GLOBAL_RNG.lock().unwrap();
        Self {
            inner: V::generate(dim, &mut *rng),
        }
    }

    /// Generates many random hypervectors.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensionality of each hypervector.
    /// * `count` - The number of hypervectors to generate.
    pub fn generate_many(dim: usize, count: usize) -> Vec<Self> {
        let mut rng = GLOBAL_RNG.lock().unwrap();
        (0..count)
            .map(|_| Self {
                inner: V::generate(dim, &mut *rng),
            })
            .collect()
    }

    /// Bundles (superposes) this hypervector with another using the specified tie-breaking rule.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to bundle with.
    /// * `tie_breaker` - The tie-breaking rule to use.
    pub fn bundle(&self, other: &Self, tie_breaker: crate::TieBreaker) -> Self {
        let mut rng = GLOBAL_RNG.lock().unwrap();
        Self {
            inner: self.inner.bundle(&other.inner, tie_breaker, &mut *rng),
        }
    }

    /// Binds this hypervector with another.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to bind with.
    pub fn bind(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.bind(&other.inner),
        }
    }

    /// Computes the cosine similarity between this hypervector and another.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to compare with.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        self.inner.cosine_similarity(&other.inner)
    }

    /// Computes the normalized Hamming distance between this hypervector and another.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to compare with.
    pub fn hamming_distance(&self, other: &Self) -> f32 {
        self.inner.hamming_distance(&other.inner)
    }

    /// Converts this hypervector into a plain vector of `f32`.
    pub fn to_vec(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    /// Bundles many hypervectors by extracting their inner representations and then wrapping
    /// the bundled result back into a `Hypervector`.
    ///
    /// # Arguments
    ///
    /// * `vectors` - A slice of hypervectors to bundle.
    /// * `tie_breaker` - The tie-breaking rule to use.
    pub fn bundle_many(vectors: &[Self], tie_breaker: crate::TieBreaker) -> Self {
        let mut rng = GLOBAL_RNG.lock().unwrap();
        let inners: Vec<V> = vectors.iter().map(|hv| hv.inner.clone()).collect();
        Self {
            inner: V::bundle_many(&inners, tie_breaker, &mut *rng),
        }
    }

    /// Binds many hypervectors by extracting their inner representations and then wrapping
    /// the bound result back into a `Hypervector`.
    ///
    /// # Arguments
    ///
    /// * `vectors` - A slice of hypervectors to bind.
    pub fn bind_many(vectors: &[Self]) -> Self {
        let inners: Vec<V> = vectors.iter().map(|hv| hv.inner.clone()).collect();
        Self {
            inner: V::bind_many(&inners),
        }
    }
}

/// Overloads the `+` operator for bundling hypervectors.
///
/// This implementation uses a default tie-breaker of `Random`.
impl<V: VSA> Add for Hypervector<V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let tie_breaker = crate::TieBreaker::Random;
        let mut rng = GLOBAL_RNG.lock().unwrap();
        Self {
            inner: self.inner.bundle(&rhs.inner, tie_breaker, &mut *rng),
        }
    }
}

/// Overloads the `*` operator for binding hypervectors.
impl<V: VSA> Mul for Hypervector<V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner.bind(&rhs.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Bring the mbat module into scope.
    use crate::hypervector::mbat;
    // Now you can alias HV as follows:
    type HV = Hypervector<mbat::MBAT>;

    #[test]
    fn test_codebook_pair_bindings() {
        let dim = 10000;
        let codebook: Vec<HV> = HV::generate_many(dim, 10);

        // Compute bindings for all unique pairs (i, j) with i < j.
        let mut pair_bindings = Vec::new();
        for i in 0..codebook.len() {
            for j in (i + 1)..codebook.len() {
                let binding = codebook[i].bind(&codebook[j]);
                pair_bindings.push(((i, j), binding));
            }
        }

        // For each binding, recompute it and check that the cosine similarity is 1.
        for &((i, j), ref binding) in &pair_bindings {
            let recomputed = codebook[i].bind(&codebook[j]);
            let sim: f32 = binding.cosine_similarity(&recomputed);
            assert!(
                (sim - 1.0).abs() < 1e-6,
                "Recomputed binding differs for pair ({}, {})",
                i,
                j
            );
        }

        // Compare bindings from different pairs; they should be nearly orthogonal.
        for (idx1, &((i, j), ref binding1)) in pair_bindings.iter().enumerate() {
            for (idx2, &((k, l), ref binding2)) in pair_bindings.iter().enumerate() {
                if idx1 == idx2 {
                    continue;
                }
                let sim: f32 = binding1.cosine_similarity(&binding2);
                assert!(
                sim.abs() < 0.1,
                "Binding for pair ({}, {}) has cosine similarity {} with binding for pair ({}, {})",
                i,
                j,
                sim,
                k,
                l
            );
            }
        }
    }

    /// Helper function to generate two random hypervectors.
    fn generate_two(dim: usize) -> (HV, HV) {
        (HV::generate(dim), HV::generate(dim))
    }

    /// Verifies that two random high-dimensional bipolar vectors are nearly orthogonal.
    #[test]
    fn test_random_vectors_orthogonal() {
        let dim = 10000;
        let (a, b) = generate_two(dim);
        let cos_sim = a.cosine_similarity(&b);
        assert!(
            cos_sim.abs() < 0.1,
            "Expected near-orthogonality but got cosine similarity {}",
            cos_sim
        );
    }

    /// Verifies that bundling two hypervectors yields a vector that is similar to each constituent.
    #[test]
    fn test_bundling_similarity() {
        let dim = 10000;
        let a = HV::generate(dim);
        let b = HV::generate(dim);
        let bundled = a.bundle(&b, TieBreaker::Random);
        let sim_a = bundled.cosine_similarity(&a);
        let sim_b = bundled.cosine_similarity(&b);
        assert!(
            (sim_a - 0.5).abs() < 0.1,
            "Bundled vector similarity with first constituent was {}",
            sim_a
        );
        assert!(
            (sim_b - 0.5).abs() < 0.1,
            "Bundled vector similarity with second constituent was {}",
            sim_b
        );
    }

    /// Verifies that binding two hypervectors produces a result nearly orthogonal to each constituent.
    #[test]
    fn test_binding_orthogonality() {
        let dim = 10000;
        let a = HV::generate(dim);
        let b = HV::generate(dim);
        let bound = a.bind(&b);
        let sim_a = a.cosine_similarity(&bound);
        let sim_b = b.cosine_similarity(&bound);
        assert!(
            sim_a.abs() < 0.1,
            "Binding similarity with first constituent was {}",
            sim_a
        );
        assert!(
            sim_b.abs() < 0.1,
            "Binding similarity with second constituent was {}",
            sim_b
        );
    }

    /// Verifies that converting a hypervector to a plain vector yields the expected values.
    #[test]
    fn test_to_vec_conversion() {
        let dim = 100;
        let hv = HV::generate(dim);
        let vec_f32 = hv.to_vec();
        assert_eq!(vec_f32.len(), dim);
        // For MBAT hypervectors, each element should be either 1.0 or -1.0.
        for &x in &vec_f32 {
            assert!(x == 1.0 || x == -1.0, "Element {} is not 1.0 or -1.0", x);
        }
    }
}
