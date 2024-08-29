# Cognite: High-Performance ML Framework in Rust

Cognite is a high-performance machine learning framework implemented in Rust, designed to leverage GPU acceleration for efficient computation. It aims to provide a fast and memory-efficient alternative to existing ML frameworks.

## Features

- GPU-accelerated vector operations using OpenCL
- Efficient memory management with a custom memory pool
- Use of `Arc<T>` for O(1) cloning of tensors and GPU buffers
- Lazy initialization of global resources
- Singleton pattern for consistent memory pool access

## Current Status

The project is in its early stages. Currently implemented features include:

- Basic tensor structure
- GPU-accelerated vector addition
- Memory pool for efficient GPU buffer management
- Singleton pattern for global resource management

## Dependencies

- Rust (latest stable version)
- OpenCL
- ocl = "0.19"
- lazy_static = "1.4.0"
- ndarray = "0.15"
- num-traits = "0.2"

## Setup

1. Ensure you have Rust and OpenCL installed on your system.
2. Clone this repository:
   ```
   git clone https://github.com/faraaz-baig/cognite.git
   cd cognite
   ```
3. Build the project:
   ```
   cargo build
   ```

## Usage

Currently, the framework provides a GPU-accelerated vector addition operation. Here's a basic example of how to use it:

```rust
use cognite::gpu::add_vectors_gpu;

fn main() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    
    match add_vectors_gpu(&a, &b) {
        Ok(result) => println!("Result: {:?}", result),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

The `MEMORY_POOL` is automatically initialized and managed, so you don't need to worry about it in your code.

## Future Plans

- Implement more GPU-accelerated operations (subtraction, multiplication, etc.)
- Add support for matrix operations
- Implement basic machine learning algorithms (linear regression, logistic regression, etc.)
- Optimize memory usage and computation speed
- Add comprehensive test suite and benchmarks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.