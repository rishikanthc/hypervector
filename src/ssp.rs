use crate::{TieBreaker, VSA};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;

/// A spatial semantic pointer (SSP) is represented simply as a vector.
#[derive(Debug, Clone, PartialEq)]
pub struct SSP {
    pub data: Vec<f32>,
}

impl VSA for SSP {
    // Define the element type – here we use f32, which matches SSP’s data type.
    type Elem = f32;

    fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        SSP::generate(dim, rng)
    }

    // For bundling, we ignore the tie_breaker and RNG since SSP’s bundling is a simple addition.
    fn bundle(&self, other: &Self, _tie_breaker: TieBreaker, _rng: &mut impl Rng) -> Self {
        self.bundle(other)
    }

    fn bind(&self, other: &Self) -> Self {
        self.bind(other)
    }

    fn cosine_similarity(&self, other: &Self) -> f32 {
        self.cosine_similarity(other)
    }

    fn hamming_distance(&self, other: &Self) -> f32 {
        self.hamming_distance(other)
    }

    fn to_vec(&self) -> Vec<f32> {
        self.to_vec()
    }
}

impl SSP {
    /// Generate a random SSP of the given dimension.
    pub fn generate(dim: usize, rng: &mut impl Rng) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data = (0..dim).map(|_| normal.sample(rng)).collect();
        SSP { data }
    }

    /// Bundling is implemented here as element‐wise addition.
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

    /// Compute cosine similarity between two SSPs.
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

    /// Compute the fraction of elements that have different signs.
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

    /// Return a copy of the SSP data.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// The SSPEncoder converts an input sample (a point in some domain)
/// into an SSP vector. (For simplicity, we assume the input is a slice
/// of f32 values with length equal to `domain_dim`.)
pub struct SSPEncoder {
    ssp_dim: usize,
    domain_dim: usize,
    // For each of the ssp_dim/2 output frequencies, we store a vector (of length domain_dim)
    // in row-major order (i.e. freq_means[i * (ssp_dim/2) + j] is element (i, j)).
    freq_means: Vec<f32>,
    lengthscale: f32,
}

impl SSPEncoder {
    /// Create a new encoder.
    /// * `ssp_dim` must be even.
    /// * `domain_dim` is the dimensionality of the input.
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

    /// Update the lengthscale parameter.
    pub fn update_lengthscale(&mut self, ls: f32) {
        self.lengthscale = ls;
    }

    /// Encode a single input sample (of length `domain_dim`) into an SSP vector.
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.domain_dim,
            "Input sample must have length equal to domain_dim"
        );
        let half = self.ssp_dim / 2;
        let mut phases = vec![0.0; half];

        // For each output frequency, compute the dot product between the input sample and
        // the corresponding frequency mean vector.
        for j in 0..half {
            let mut dot = 0.0;
            for i in 0..self.domain_dim {
                // freq_means is stored in row-major order: row i, column j.
                dot += x[i] * self.freq_means[i * half + j];
            }
            phases[j] = dot / self.lengthscale;
        }

        // Form the SSP vector by taking cosine and sine of each phase.
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

    #[test]
    fn test_ssp_encoder() {
        let mut rng = StdRng::seed_from_u64(42);
        let encoder = SSPEncoder::new(2048, 1, &mut rng);
        let sample = [1.0f32];
        let encoded = encoder.encode(&sample);
        assert_eq!(encoded.len(), 2048);
    }

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
        // Random vectors should be nearly orthogonal.
        assert!(sim < 0.2);

        let self_sim = a.cosine_similarity(&a);
        assert!((self_sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 2048;
        let a = SSP::generate(dim, &mut rng);
        let b = SSP::generate(dim, &mut rng);
        let hd = a.hamming_distance(&b);
        assert!(hd >= 0.0 && hd <= 1.0);
    }
}
