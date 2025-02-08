// src/fhrr.rs

use rand::distributions::Uniform;
use rand::Rng;
use std::f32::consts::PI;

use crate::{TieBreaker, VSA};

/// An FHRR hypervector is represented as a vector of angles (in radians).
/// Each angle corresponds to a unit‐complex number e^(i*theta).
#[derive(Debug, Clone, PartialEq)]
pub struct FHRR {
    pub data: Vec<f32>, // angles in radians; each element represents e^(i*theta)
}

impl FHRR {
    /// Helper function: computes x mod 2π in the range [0, 2π)
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
    type Elem = f32; // each element is an angle (in radians)

    /// Generate a random FHRR hypervector of dimension `dim` by sampling each angle uniformly in [0, 2π).
    fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        let uniform = Uniform::new(0.0, 2.0 * PI);
        let data = (0..dim).map(|_| rng.sample(&uniform)).collect();
        FHRR { data }
    }

    /// Bundle two FHRR hypervectors.
    ///
    /// For each coordinate i, we interpret the angles as complex numbers,
    /// sum them, and then set the bundled value to be the phase (i.e., argument)
    /// of the sum. If the sum is (numerically) zero, we choose a value based on the tie-breaker.
    fn bundle(&self, other: &Self, tie_breaker: TieBreaker, rng: &mut impl Rng) -> Self {
        let dim = self.data.len();
        let mut result = Vec::with_capacity(dim);
        for i in 0..dim {
            let a = self.data[i];
            let b = other.data[i];
            // Convert to complex numbers (represented by their cosine and sine components)
            let real_sum = a.cos() + b.cos();
            let imag_sum = a.sin() + b.sin();
            let magnitude = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
            let angle = if magnitude.abs() < 1e-6 {
                // If the sum is (almost) zero, use the tie breaker.
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

    /// Bind two FHRR hypervectors by adding their angles element‐wise modulo 2π.
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
    /// Since each coordinate represents a phase, we define similarity as the average cosine of
    /// the differences of the angles.
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

    /// Define hamming distance as (1 - cosine_similarity) / 2.
    /// This gives 0 when the vectors are identical and 1 when they are opposites.
    fn hamming_distance(&self, other: &Self) -> f32 {
        (1.0 - self.cosine_similarity(other)) / 2.0
    }

    /// Convert the FHRR hypervector to a plain `Vec<f32>`.
    ///
    /// Here we return the real part of the corresponding unit-complex numbers.
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

    #[test]
    fn test_generate_consistency() {
        let mut rng = thread_rng();
        let a = FHRR::generate(100, &mut rng);
        let b = FHRR::generate(100, &mut rng);
        // Check that the dimensions are correct and that two independently generated hypervectors are (likely) different.
        assert_eq!(a.data.len(), 100);
        assert_eq!(b.data.len(), 100);
        // It is highly unlikely that a and b are exactly equal.
        assert_ne!(a, b);
    }

    #[test]
    fn test_bind_inverse() {
        let mut rng = thread_rng();
        let x = FHRR::generate(50, &mut rng);
        // Compute the inverse of x: for each angle, the inverse is (-theta) mod 2π.
        let x_inv = FHRR {
            data: x.data.iter().map(|&theta| FHRR::mod_2pi(-theta)).collect(),
        };
        // Binding x with its inverse should give the identity hypervector.
        // For FHRR, the identity is the hypervector with all angles equal to 0 (or equivalently 2π).
        let identity = x.bind(&x_inv);
        for angle in identity.data {
            // Allow for a small numerical tolerance.
            assert!(
                (angle).abs() < 1e-5 || (angle - 2.0 * PI).abs() < 1e-5,
                "Angle {} is not close to 0 or 2π",
                angle
            );
        }
    }

    #[test]
    fn test_bundle_commutative() {
        let mut rng = thread_rng();
        let x = FHRR::generate(100, &mut rng);
        let y = FHRR::generate(100, &mut rng);
        // Bundling should be (approximately) commutative.
        let b1 = x.bundle(&y, TieBreaker::AlwaysPositive, &mut rng);
        let b2 = y.bundle(&x, TieBreaker::AlwaysPositive, &mut rng);
        // Instead of expecting exact equality (since tie-breaking may introduce minor differences),
        // we check that their cosine similarity is nearly 1.
        let sim = b1.cosine_similarity(&b2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Cosine similarity {} not close to 1 for bundled vectors",
            sim
        );
    }

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

    #[test]
    fn test_to_vec() {
        // Create an FHRR hypervector with all angles set to 0.
        let x = FHRR {
            data: vec![0.0; 50],
        };
        let vec_f32 = x.to_vec();
        // Since cos(0) = 1, every element in the resulting vector should be 1.0.
        for &val in &vec_f32 {
            assert!((val - 1.0).abs() < 1e-5, "Expected 1.0 but found {}", val);
        }
    }
}
