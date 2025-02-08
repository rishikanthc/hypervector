use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{TieBreaker, VSA};

/// MBAT (Matrix Binding of Additive Terms) implementation using bipolar vectors.
///
/// Internally, an MBAT hypervector is stored as a `Vec<i8>` where each element is either -1 or +1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MBAT {
    pub data: Vec<i8>,
}

impl VSA for MBAT {
    type Elem = i8;

    /// Generate a random MBAT hypervector of the specified dimension.
    ///
    /// Each element is randomly set to either +1 or -1.
    fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        let data = (0..dim)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();
        MBAT { data }
    }

    /// Bundle (superpose) two MBAT hypervectors.
    ///
    /// Bundling is performed element-wise: the sum is computed and then the sign function is applied.
    /// In the event of a tie (i.e. when the sum is zero), the provided tie-breaker is used.
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
    /// For bipolar vectors, the dot product divided by the dimension is used.
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
    /// This is defined as the fraction of positions where the two vectors differ.
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

    /// Convert the MBAT hypervector into a plain `Vec<f32>`.
    fn to_vec(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x as f32).collect()
    }
}
