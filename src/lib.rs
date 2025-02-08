//! # Hypervector Crate
//!
//! This crate implements high-dimensional vectors for hyperdimensional computing / VSA.
//! It currently provides two implementations:
//! - **MBAT**: Bipolar vectors (elements in {-1, +1}).
//! - **SPP**: Semantic Spatial Parameters (in a separate module).
//!
//! The core components (global RNG, VSA trait, Hypervector type, etc.) are defined here.

use once_cell::sync::Lazy;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};
use std::sync::Mutex;

/// A global RNG used for hypervector generation and any randomness in operations.
/// Set the global seed via [`set_global_seed`].
static GLOBAL_RNG: Lazy<Mutex<StdRng>> = Lazy::new(|| Mutex::new(StdRng::from_entropy()));

/// Set the seed for the global RNG. This affects all hypervector generation and operations.
///
/// # Example
///
/// ```
/// use hypervector::set_global_seed;
///
/// set_global_seed(42);
/// ```
pub fn set_global_seed(seed: u64) {
    let mut rng = GLOBAL_RNG.lock().unwrap();
    *rng = StdRng::seed_from_u64(seed);
}

/// Tie-breaking options for the bundling operation.
///
/// When bundling, an element-wise sum may result in a tie (zero). This enum lets you choose how to resolve it.
#[derive(Debug, Clone, Copy)]
pub enum TieBreaker {
    /// Always choose +1 when a tie occurs.
    AlwaysPositive,
    /// Always choose –1 when a tie occurs.
    AlwaysNegative,
    /// Randomly choose between +1 and –1 when a tie occurs.
    Random,
}

/// The trait defining the interface for a VSA algorithm.
/// New VSA implementations (like SPP) can be added by implementing this trait.
pub trait VSA: Sized + Clone {
    /// The type used to represent each element in the hypervector.
    type Elem: Copy + std::fmt::Debug + PartialEq + Into<f32>;

    /// Generate a random hypervector of a given dimension.
    fn generate(dim: usize, rng: &mut impl Rng) -> Self;

    /// Bundle (superpose) two hypervectors.
    ///
    /// For MBAT, this is the element-wise sum followed by a sign function with tie-breaking.
    fn bundle(&self, other: &Self, tie_breaker: TieBreaker, rng: &mut impl Rng) -> Self;

    /// Bind two hypervectors.
    ///
    /// For MBAT, this is the element-wise product.
    fn bind(&self, other: &Self) -> Self;

    /// Compute the cosine similarity between two hypervectors.
    fn cosine_similarity(&self, other: &Self) -> f32;

    /// Compute the normalized Hamming distance between two hypervectors.
    fn hamming_distance(&self, other: &Self) -> f32;

    /// Convert the hypervector into a plain `Vec<f32>`.
    fn to_vec(&self) -> Vec<f32>;

    /// Bundle many hypervectors (folding a slice using the bundling operation).
    fn bundle_many(vectors: &[Self], tie_breaker: TieBreaker, rng: &mut impl Rng) -> Self {
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
}

/// A generic Hypervector type parametrized over a VSA implementation.
///
/// This type wraps an inner hypervector and provides high-level operations (generation, bundling, binding, similarity checks).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hypervector<V: VSA> {
    pub inner: V,
}

impl<V: VSA> Hypervector<V> {
    /// Generate a random hypervector of a given dimension using the global RNG.
    pub fn generate(dim: usize) -> Self {
        let mut rng = GLOBAL_RNG.lock().unwrap();
        Self {
            inner: V::generate(dim, &mut *rng),
        }
    }

    /// Generate many random hypervectors.
    pub fn generate_many(dim: usize, count: usize) -> Vec<Self> {
        let mut rng = GLOBAL_RNG.lock().unwrap();
        (0..count)
            .map(|_| Self {
                inner: V::generate(dim, &mut *rng),
            })
            .collect()
    }

    /// Bundle (superpose) two hypervectors with the specified tie-breaking rule.
    pub fn bundle(&self, other: &Self, tie_breaker: TieBreaker) -> Self {
        let mut rng = GLOBAL_RNG.lock().unwrap();
        Self {
            inner: self.inner.bundle(&other.inner, tie_breaker, &mut *rng),
        }
    }

    /// Bind two hypervectors.
    pub fn bind(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.bind(&other.inner),
        }
    }

    /// Compute the cosine similarity between two hypervectors.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        self.inner.cosine_similarity(&other.inner)
    }

    /// Compute the normalized Hamming distance between two hypervectors.
    pub fn hamming_distance(&self, other: &Self) -> f32 {
        self.inner.hamming_distance(&other.inner)
    }

    /// Convert the hypervector into a plain vector of `f32` (for example, for Lance DB compatibility).
    pub fn to_vec(&self) -> Vec<f32> {
        self.inner.to_vec()
    }
}

// Overload the `+` operator for bundling.
// Uses a default tie-breaker of `Random`.
impl<V: VSA> Add for Hypervector<V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let tie_breaker = TieBreaker::Random;
        let mut rng = GLOBAL_RNG.lock().unwrap();
        Self {
            inner: self.inner.bundle(&rhs.inner, tie_breaker, &mut *rng),
        }
    }
}

// Overload the `*` operator for binding.
impl<V: VSA> Mul for Hypervector<V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner.bind(&rhs.inner),
        }
    }
}

// Declare submodules for different VSA implementations.
pub mod mbat;
pub mod spp; // SPP is implemented in src/spp.rs

#[cfg(test)]
mod tests {
    use super::*;
    // For testing MBAT, we use the alias HV.
    type HV = Hypervector<mbat::MBAT>;

    /// Helper: generate two random hypervectors.
    fn generate_two(dim: usize) -> (HV, HV) {
        (HV::generate(dim), HV::generate(dim))
    }

    /// With high-dimensional bipolar vectors, two random vectors should be nearly orthogonal.
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

    /// Test that bundling two hypervectors yields a vector similar to its constituents.
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

    /// Test that binding two hypervectors produces a result nearly orthogonal to each constituent.
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

    /// Test creating a codebook of 10 hypervectors, bundling pairs, and verifying retrieval.
    ///
    /// This test generates 10 random hypervectors, computes bindings for every unique pair,
    /// verifies that recomputed bindings for the same pair yield a cosine similarity of 1,
    /// and ensures that bindings from different pairs are nearly orthogonal.
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
            let sim = binding.cosine_similarity(&recomputed);
            assert!(
                (sim - 1.0).abs() < 1e-6,
                "Recomputed binding differs for pair ({}, {})",
                i,
                j
            );
        }

        // Compare bindings from different pairs.
        // They should be nearly orthogonal.
        for (idx1, &((i, j), ref binding1)) in pair_bindings.iter().enumerate() {
            for (idx2, &((k, l), ref binding2)) in pair_bindings.iter().enumerate() {
                if idx1 == idx2 {
                    continue;
                }
                let sim = binding1.cosine_similarity(binding2);
                // Allow a small chance of nonzero similarity due to randomness.
                assert!(
                    sim.abs() < 0.1,
                    "Binding for pair ({}, {}) has cosine similarity {} with binding for pair ({}, {})",
                    i, j, sim, k, l
                );
            }
        }
    }

    /// Test conversion to `Vec<f32>`.
    #[test]
    fn test_to_vec_conversion() {
        let dim = 100;
        let hv = HV::generate(dim);
        let vec_f32 = hv.to_vec();
        assert_eq!(vec_f32.len(), dim);
        for &x in &vec_f32 {
            assert!(x == 1.0 || x == -1.0, "Element {} is not 1.0 or -1.0", x);
        }
    }
}
