//! # MBAT Module
//!
//! This module provides an implementation of MBAT (Matrix Binding of Additive Terms)
//! using bipolar vectors. An MBAT hypervector is internally stored as a `Vec<i8>` where each
//! element is either -1 or +1. This module implements the [`VSA`] trait for MBAT,
//! providing methods for hypervector generation, bundling, binding, similarity measurement,
//! and conversion to a plain vector.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{TieBreaker, VSA};

/// MBAT (Matrix Binding of Additive Terms) implementation using bipolar vectors.
///
/// Internally, an MBAT hypervector is stored as a `Vec<i8>` where each element is either -1 or +1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MBAT {
    /// The underlying bipolar vector.
    pub data: Vec<i8>,
}

impl VSA for MBAT {
    type Elem = i8;

    /// Generate a random MBAT hypervector of the specified dimension.
    ///
    /// Each element is randomly set to either +1 or -1 with equal probability.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensionality of the hypervector.
    /// * `rng` - A mutable reference to a random number generator.
    fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        let data = (0..dim)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();
        MBAT { data }
    }

    /// Bundle (superpose) two MBAT hypervectors.
    ///
    /// Bundling is performed element-wise by computing the sum of corresponding elements and then
    /// applying the sign function. In the event of a tie (i.e. when the sum is zero), the provided
    /// tie-breaker is used to determine the output.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to bundle with.
    /// * `tie_breaker` - The rule to resolve ties.
    /// * `rng` - A mutable reference to a random number generator.
    fn bundle(&self, other: &Self, tie_breaker: TieBreaker, rng: &mut impl Rng) -> Self {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Dimension mismatch in bundling"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| {
                let sum = a + b;
                if sum > 0 {
                    1
                } else if sum < 0 {
                    -1
                } else {
                    // Tie: apply the specified tie-breaking rule.
                    match tie_breaker {
                        TieBreaker::AlwaysPositive => 1,
                        TieBreaker::AlwaysNegative => -1,
                        TieBreaker::Random => {
                            if rng.gen_bool(0.5) {
                                1
                            } else {
                                -1
                            }
                        }
                    }
                }
            })
            .collect();
        MBAT { data }
    }

    /// Bind two MBAT hypervectors.
    ///
    /// Binding is implemented as the element-wise product.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to bind with.
    fn bind(&self, other: &Self) -> Self {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Dimension mismatch in binding"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        MBAT { data }
    }

    /// Compute the cosine similarity between two MBAT hypervectors.
    ///
    /// For bipolar vectors, cosine similarity is computed as the dot product of the vectors divided
    /// by the dimension.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to compare with.
    fn cosine_similarity(&self, other: &Self) -> f32 {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Dimension mismatch in cosine similarity"
        );
        let dot: i32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a as i32) * (b as i32))
            .sum();
        let dim = self.data.len() as f32;
        dot as f32 / dim
    }

    /// Compute the normalized Hamming distance between two MBAT hypervectors.
    ///
    /// The normalized Hamming distance is defined as the fraction of positions where the two
    /// vectors differ.
    ///
    /// # Arguments
    ///
    /// * `other` - The hypervector to compare with.
    fn hamming_distance(&self, other: &Self) -> f32 {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Dimension mismatch in hamming distance"
        );
        let mismatches = self
            .data
            .iter()
            .zip(other.data.iter())
            .filter(|(&a, &b)| a != b)
            .count();
        mismatches as f32 / self.data.len() as f32
    }

    /// Convert the MBAT hypervector into a plain vector of `f32` values.
    ///
    /// Each element is converted from `i8` to `f32`.
    fn to_vec(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x as f32).collect()
    }

    /// Create an MBAT hypervector from a plain `Vec<f32>`.
    ///
    /// This function converts each element of the vector: nonnegative values become `1` and
    /// negative values become `-1`.
    ///
    /// # Arguments
    ///
    /// * `v` - A vector of `f32` values, typically containing only `1.0` and `-1.0`.
    fn from_vec(v: Vec<f32>) -> Self {
        let data = v
            .into_iter()
            .map(|x| if x >= 0.0 { 1 } else { -1 })
            .collect();
        MBAT { data }
    }
}
