//! # Spatial Semantic Pointer (SSP) Module
//!
//! This module implements the Spatial Semantic Pointer (SSP) VSA. An SSP is simply a vector
//! of `f32` values. SSP operations (bundling, binding, similarity, etc.) are defined here and
//! the type implements the [`VSA`] trait.
//!
//! Additionally, this module provides an [`SSPEncoder`] which converts an input sample into an
//! SSP vector based on a set of frequency means.

use crate::{TieBreaker, VSA};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;

/// A spatial semantic pointer (SSP) is represented as a vector of `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct SSP {
    /// The underlying data representing the SSP.
    pub data: Vec<f32>,
}

impl VSA for SSP {
    /// The element type for SSP is `f32`.
    type Elem = f32;

    /// Generate a random SSP of the given dimension using the provided RNG.
    fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        SSP::generate(dim, rng)
    }

    /// Bundle two SSPs.
    ///
    /// For SSP, bundling is implemented as element‑wise addition. The tie-breaker and RNG are ignored.
    fn bundle(&self, other: &Self, _tie_breaker: TieBreaker, _rng: &mut impl Rng) -> Self {
        self.bundle(other)
    }

    /// Bind two SSPs.
    ///
    /// Binding is implemented using a naive circular convolution (see [`SSP::bind`]).
    fn bind(&self, other: &Self) -> Self {
        self.bind(other)
    }

    /// Compute the cosine similarity between two SSPs.
    fn cosine_similarity(&self, other: &Self) -> f32 {
        self.cosine_similarity(other)
    }

    /// Compute the normalized Hamming distance between two SSPs.
    fn hamming_distance(&self, other: &Self) -> f32 {
        self.hamming_distance(other)
    }

    /// Converts the SSP into a plain vector of `f32` values.
    fn to_vec(&self) -> Vec<f32> {
        self.to_vec()
    }
}

impl SSP {
    /// Generates a random SSP of the given dimension.
    ///
    /// Each element is sampled from a normal distribution with mean 0 and standard deviation 1.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensionality of the SSP.
    /// * `rng` - A mutable reference to a random number generator.
    pub fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data = (0..dim).map(|_| normal.sample(rng)).collect();
        SSP { data }
    }

    /// Bundles (superposes) two SSPs using element‑wise addition.
    ///
    /// # Arguments
    ///
    /// * `other` - The other SSP to bundle with.
    ///
    /// # Panics
    ///
    /// Panics if the two SSPs do not have the same dimensionality.
    pub fn bundle(&self, other: &SSP) -> SSP {
        assert_eq!(self.data.len(), other.data.len());
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        SSP { data }
    }

    /// Binds two SSPs using a naive circular convolution.
    ///
    /// This O(n²) algorithm is provided for illustration purposes. After computing the convolution,
    /// the result is normalized to have unit length.
    ///
    /// # Arguments
    ///
    /// * `other` - The other SSP to bind with.
    ///
    /// # Panics
    ///
    /// Panics if the two SSPs do not have the same dimensionality.
    pub fn bind(&self, other: &SSP) -> SSP {
        let n = self.data.len();
        assert_eq!(n, other.data.len());
        let mut result = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let k = (i + n - j) % n;
                sum += self.data[j] * other.data[k];
            }
            result[i] = sum;
        }
        // Normalize the result so that bound vectors are unit-length.
        let norm = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in result.iter_mut() {
                *x /= norm;
            }
        }
        SSP { data: result }
    }

    /// Computes the cosine similarity between two SSPs.
    ///
    /// Cosine similarity is defined as the dot product of the two vectors divided by the product
    /// of their magnitudes.
    ///
    /// # Arguments
    ///
    /// * `other` - The other SSP to compare with.
    pub fn cosine_similarity(&self, other: &SSP) -> f32 {
        let dot: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_self = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other = other.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_self * norm_other)
    }

    /// Computes the fraction of elements with different signs between two SSPs.
    ///
    /// This is interpreted as a normalized Hamming distance.
    ///
    /// # Arguments
    ///
    /// * `other` - The other SSP to compare with.
    pub fn hamming_distance(&self, other: &SSP) -> f32 {
        let n = self.data.len();
        let count = self
            .data
            .iter()
            .zip(other.data.iter())
            .filter(|(a, b)| a.signum() != b.signum())
            .count();
        count as f32 / n as f32
    }

    /// Returns a copy of the SSP data as a vector of `f32` values.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// The `SSPEncoder` converts an input sample (a point in some domain) into an SSP vector.
///
/// For simplicity, it assumes the input is a slice of `f32` values with length equal to `domain_dim`.
/// Internally, it stores a set of frequency means to compute phase values that are then converted
/// into cosine/sine pairs to form the SSP vector.
pub struct SSPEncoder {
    ssp_dim: usize,
    domain_dim: usize,
    /// Frequency means stored in row-major order.
    /// The value at index `i * (ssp_dim/2) + j` corresponds to the j-th frequency for the i-th input dimension.
    freq_means: Vec<f32>,
    lengthscale: f32,
}

impl SSPEncoder {
    /// Creates a new `SSPEncoder`.
    ///
    /// # Arguments
    ///
    /// * `ssp_dim` - The dimension of the SSP vector (must be even to allow cosine/sine pairs).
    /// * `domain_dim` - The dimensionality of the input sample.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Panics
    ///
    /// Panics if `ssp_dim` is not even.
    pub fn new(ssp_dim: usize, domain_dim: usize, rng: &mut impl Rng) -> Self {
        assert!(
            ssp_dim % 2 == 0,
            "ssp_dim must be even (to allow cosine/sine pairs)"
        );
        let half = ssp_dim / 2;
        let mut freq_means = Vec::with_capacity(domain_dim * half);
        let normal = Normal::new(0.0, 1.0).unwrap();
        for _ in 0..(domain_dim * half) {
            freq_means.push(normal.sample(rng));
        }
        SSPEncoder {
            ssp_dim,
            domain_dim,
            freq_means,
            lengthscale: 1.0,
        }
    }

    /// Updates the lengthscale parameter.
    ///
    /// The lengthscale is used to scale the dot products when computing phases.
    ///
    /// # Arguments
    ///
    /// * `ls` - The new lengthscale value.
    pub fn update_lengthscale(&mut self, ls: f32) {
        self.lengthscale = ls;
    }

    /// Encodes a single input sample into an SSP vector.
    ///
    /// The input sample is expected to have length equal to `domain_dim`.
    /// The encoder computes phase values using dot products with the frequency means,
    /// scales them by the lengthscale, and then converts each phase into a cosine and sine pair.
    ///
    /// # Arguments
    ///
    /// * `x` - A slice of `f32` values representing the input sample.
    ///
    /// # Returns
    ///
    /// A vector of `f32` values of length `ssp_dim` representing the encoded SSP.
    ///
    /// # Panics
    ///
    /// Panics if the length of `x` does not equal `domain_dim`.
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.domain_dim,
            "Input sample must have length equal to domain_dim"
        );
        let half = self.ssp_dim / 2;
        let mut phases = vec![0.0; half];

        // For each output frequency, compute the dot product between the input sample and the corresponding frequency mean vector.
        for j in 0..half {
            let mut dot = 0.0;
            for i in 0..self.domain_dim {
                // `freq_means` is stored in row-major order: row i, column j.
                dot += x[i] * self.freq_means[i * half + j];
            }
            phases[j] = dot / self.lengthscale;
        }

        // Form the SSP vector by taking the cosine and sine of each phase.
        let mut ssp = vec![0.0; self.ssp_dim];
        for j in 0..half {
            ssp[2 * j] = phases[j].cos();
            ssp[2 * j + 1] = phases[j].sin();
        }
        ssp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Tests that the SSPEncoder produces an SSP vector with the expected dimension.
    #[test]
    fn test_ssp_encoder() {
        let mut rng = StdRng::seed_from_u64(42);
        let encoder = SSPEncoder::new(2048, 1, &mut rng);
        let sample = [1.0f32];
        let encoded = encoder.encode(&sample);
        assert_eq!(encoded.len(), 2048);
    }

    /// Tests basic SSP operations: generation, bundling, binding, and cosine similarity.
    #[test]
    fn test_ssp_operations() {
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 2048;
        let a = SSP::generate(dim, &mut rng);
        let b = SSP::generate(dim, &mut rng);

        let bundled = a.bundle(&b);
        assert_eq!(bundled.data.len(), dim);

        let bound = a.bind(&b);
        assert_eq!(bound.data.len(), dim);

        let sim = a.cosine_similarity(&b);
        // Random SSP vectors should be nearly orthogonal.
        assert!(sim < 0.2);

        let self_sim = a.cosine_similarity(&a);
        assert!((self_sim - 1.0).abs() < 1e-6);
    }

    /// Tests that the Hamming distance between two SSPs is within the valid range [0, 1].
    #[test]
    fn test_hamming_distance() {
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 2048;
        let a = SSP::generate(dim, &mut rng);
        let b = SSP::generate(dim, &mut rng);
        let hd = a.hamming_distance(&b);
        assert!((0.0..=1.0).contains(&hd));
    }
}
