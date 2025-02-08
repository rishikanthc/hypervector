//! # FHRR Module
//!
//! This module implements the FHRR (Fourier Holographic Reduced Representation) VSA.
//! An FHRR hypervector is represented as a vector of angles (in radians). Each angle corresponds
//! to a unit‐complex number, i.e. `e^(i*theta)`. The FHRR type implements the [`VSA`] trait.
//!
//! In this implementation:
//! - **Generation:** Each angle is sampled uniformly from [0, 2π).
//! - **Bundling:** Two hypervectors are bundled by converting each angle into its complex representation,
//!   summing the complex numbers, and then taking the argument (phase) of the result. If the sum is nearly
//!   zero, a tie-breaker is used.
//! - **Binding:** Two hypervectors are bound by adding their angles element‑wise modulo 2π.
//! - **Cosine Similarity:** Defined as the average cosine of the differences between corresponding angles.
//! - **Hamming Distance:** Defined as `(1 - cosine_similarity) / 2`.
//! - **Conversion to Vec:** Returns the cosine (i.e. the real part) of each angle.

use rand::distributions::Uniform;
use rand::Rng;
use std::f32::consts::PI;

use crate::{TieBreaker, VSA};

/// An FHRR hypervector is represented as a vector of angles (in radians).
/// Each angle corresponds to a unit‐complex number `e^(i*theta)`.
#[derive(Debug, Clone, PartialEq)]
pub struct FHRR {
    /// The vector of angles (in radians) representing the hypervector.
    pub data: Vec<f32>,
}

impl FHRR {
    /// Helper function: Computes `x mod 2π` and returns a value in the range [0, 2π).
    ///
    /// # Arguments
    ///
    /// * `x` - A floating-point value.
    fn mod_2pi(x: f32) -> f32 {
        let two_pi = 2.0 * PI;
        let mut r = x % two_pi;
        if r < 0.0 {
            r += two_pi;
        }
        r
    }
}

impl VSA for FHRR {
    type Elem = f32; // Each element is an angle (in radians)

    /// Generate a random FHRR hypervector of dimension `dim` by sampling each angle uniformly from [0, 2π).
    ///
    /// # Arguments
    ///
    /// * `dim` - The number of angles in the hypervector.
    /// * `rng` - A mutable reference to a random number generator.
    fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        let uniform = Uniform::new(0.0, 2.0 * PI);
        let data = (0..dim).map(|_| rng.sample(uniform)).collect();
        FHRR { data }
    }

    /// Bundle two FHRR hypervectors.
    ///
    /// For each coordinate `i`, the angles are interpreted as complex numbers. The complex numbers
    /// are summed, and the bundled value is the phase (argument) of the sum. If the magnitude of the sum
    /// is nearly zero, a value is chosen based on the provided tie-breaker.
    ///
    /// # Arguments
    ///
    /// * `other` - The other FHRR hypervector to bundle with.
    /// * `tie_breaker` - The tie-breaking rule to use if the sum is nearly zero.
    /// * `rng` - A mutable reference to a random number generator.
    fn bundle(&self, other: &Self, tie_breaker: TieBreaker, rng: &mut impl Rng) -> Self {
        let dim = self.data.len();
        let mut result = Vec::with_capacity(dim);
        for i in 0..dim {
            let a = self.data[i];
            let b = other.data[i];
            // Convert the angles to their complex representation (cosine for real, sine for imaginary)
            let real_sum = a.cos() + b.cos();
            let imag_sum = a.sin() + b.sin();
            let magnitude = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
            let angle = if magnitude.abs() < 1e-6 {
                // If the sum is nearly zero, choose based on the tie-breaker.
                match tie_breaker {
                    TieBreaker::AlwaysPositive => 0.0,
                    TieBreaker::AlwaysNegative => PI,
                    TieBreaker::Random => rng.sample(Uniform::new(0.0, 2.0 * PI)),
                }
            } else {
                imag_sum.atan2(real_sum)
            };
            result.push(FHRR::mod_2pi(angle));
        }
        FHRR { data: result }
    }

    /// Bind two FHRR hypervectors by adding their angles element‑wise modulo 2π.
    ///
    /// # Arguments
    ///
    /// * `other` - The other FHRR hypervector to bind with.
    fn bind(&self, other: &Self) -> Self {
        let dim = self.data.len();
        let mut result = Vec::with_capacity(dim);
        for i in 0..dim {
            let sum = self.data[i] + other.data[i];
            result.push(FHRR::mod_2pi(sum));
        }
        FHRR { data: result }
    }

    /// Compute the cosine similarity between two FHRR hypervectors.
    ///
    /// Since each coordinate represents a phase, similarity is defined as the average cosine
    /// of the differences between corresponding angles.
    ///
    /// # Arguments
    ///
    /// * `other` - The other FHRR hypervector to compare with.
    fn cosine_similarity(&self, other: &Self) -> f32 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a - b).cos())
            .sum();
        sum / (n as f32)
    }

    /// Define the normalized Hamming distance as `(1 - cosine_similarity) / 2`.
    ///
    /// This yields 0 when the hypervectors are identical and 1 when they are opposites.
    ///
    /// # Arguments
    ///
    /// * `other` - The other FHRR hypervector.
    fn hamming_distance(&self, other: &Self) -> f32 {
        (1.0 - self.cosine_similarity(other)) / 2.0
    }

    /// Convert the FHRR hypervector to a plain `Vec<f32>`.
    ///
    /// This function returns the real part (cosine) of the corresponding unit-complex numbers.
    fn to_vec(&self) -> Vec<f32> {
        self.data.iter().map(|&theta| theta.cos()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TieBreaker, VSA};
    use rand::thread_rng;
    use std::f32::consts::PI;

    /// Tests that generating two FHRR hypervectors produces vectors of the correct dimension
    /// and that two independently generated vectors are unlikely to be identical.
    #[test]
    fn test_generate_consistency() {
        let mut rng = thread_rng();
        let a = FHRR::generate(100, &mut rng);
        let b = FHRR::generate(100, &mut rng);
        assert_eq!(a.data.len(), 100);
        assert_eq!(b.data.len(), 100);
        assert_ne!(a, b);
    }

    /// Tests that binding an FHRR hypervector with its inverse (i.e. negated angles modulo 2π)
    /// produces the identity hypervector (all angles 0 or 2π).
    #[test]
    fn test_bind_inverse() {
        let mut rng = thread_rng();
        let x = FHRR::generate(50, &mut rng);
        let x_inv = FHRR {
            data: x.data.iter().map(|&theta| FHRR::mod_2pi(-theta)).collect(),
        };
        let identity = x.bind(&x_inv);
        for angle in identity.data {
            assert!(
                (angle).abs() < 1e-5 || (angle - 2.0 * PI).abs() < 1e-5,
                "Angle {} is not close to 0 or 2π",
                angle
            );
        }
    }

    /// Tests that bundling two FHRR hypervectors is approximately commutative.
    #[test]
    fn test_bundle_commutative() {
        let mut rng = thread_rng();
        let x = FHRR::generate(100, &mut rng);
        let y = FHRR::generate(100, &mut rng);
        let b1 = x.bundle(&y, TieBreaker::AlwaysPositive, &mut rng);
        let b2 = y.bundle(&x, TieBreaker::AlwaysPositive, &mut rng);
        let sim = b1.cosine_similarity(&b2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Cosine similarity {} not close to 1 for bundled vectors",
            sim
        );
    }

    /// Tests that the cosine similarity of a hypervector with itself is 1.
    #[test]
    fn test_cosine_similarity_self() {
        let mut rng = thread_rng();
        let x = FHRR::generate(100, &mut rng);
        let sim = x.cosine_similarity(&x);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Self cosine similarity {} not equal to 1",
            sim
        );
    }

    /// Tests that the hamming distance of a hypervector with itself is 0.
    #[test]
    fn test_hamming_distance_self() {
        let mut rng = thread_rng();
        let x = FHRR::generate(100, &mut rng);
        let hd = x.hamming_distance(&x);
        assert!(
            hd.abs() < 1e-5,
            "Self hamming distance {} not equal to 0",
            hd
        );
    }

    /// Tests that converting an FHRR hypervector with all angles set to 0 returns a vector of ones.
    #[test]
    fn test_to_vec() {
        let x = FHRR {
            data: vec![0.0; 50],
        };
        let vec_f32 = x.to_vec();
        for &val in &vec_f32 {
            assert!((val - 1.0).abs() < 1e-5, "Expected 1.0 but found {}", val);
        }
    }
}
