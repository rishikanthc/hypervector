# Hypervector

[![Crates.io](https://img.shields.io/crates/v/hypervector)](https://crates.io/crates/hypervector)
[![Documentation](https://docs.rs/hypervector/badge.svg)](https://docs.rs/hypervector)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

Hypervector is a Rust crate that implements hyperdimensional vectors for hyperdimensional computing and Vector Symbolic Architectures (VSAs). It provides multiple implementations including:

- **MBAT:** Bipolar vectors (elements in {-1, +1}).
- **SSP:** Semantic Spatial Pointers.
- **FHRR:** Fourier Holographic Reduced Representations.

The crate offers a generic hypervector type with high-level operations such as generation, bundling, binding, similarity checks, and conversion to a plain vector. It also provides an object encoder to map JSON objects into hypervector representations.

## Features

- **Multiple VSA Implementations:** MBAT, SSP, and FHRR.
- **Generic Hypervector API:** Create, bundle, bind, and compare hypervectors.
- **Object Encoder:** Encode JSON objects into hypervectors using a codebook approach.
- **Reproducible Randomness:** Set a global RNG seed with `set_global_seed`.
- **Well-Documented and Tested:** Comprehensive tests and inline documentation.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
hypervector = "0.1.0"

Replace "0.1.0" with the current version if needed.

## Usage

Here is a basic example demonstrating how to generate a hypervector using the MBAT implementation and encode a JSON object:

```rust
use hypervector::{set_global_seed, Hypervector, TieBreaker};
use hypervector::mbat::MBAT;
use hypervector::encoder::ObjectEncoder;
use serde_json::json;

fn main() {
    // Set the global RNG seed for reproducibility.
    set_global_seed(42);

    // Generate a random MBAT hypervector of dimension 1000.
    let hv = Hypervector::<MBAT>::generate(1000);
    println!("Generated MBAT hypervector: {:?}", hv);

    // Create an object encoder for MBAT hypervectors.
    let mut encoder = ObjectEncoder::<MBAT>::new(1000, TieBreaker::AlwaysPositive);

    // Define a JSON object.
    let json_obj = json!({
        "firstName": "Alice",
        "lastName": "Smith",
        "isActive": true,
        "hobbies": ["reading", "cycling"]
    });

    // Encode the JSON object into a hypervector.
    let encoded = encoder.encode_object(&json_obj);
    println!("Encoded JSON object: {:?}", encoded);
}
```

You can also use SSP or FHRR by importing the corresponding modules (e.g. hypervector::ssp::SSP or hypervector::fhrr::FHRR).

## Documentation

Full API documentation is available on docs.rs/hypervector.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for details on how to get started.

## License

This project is licensed under either the MIT license or the Apache License (Version 2.0), at your option. See the LICENSE file for details.
