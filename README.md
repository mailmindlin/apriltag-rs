# apriltag-rs

A pure Rust implementation of [AprilTag](https://april.eecs.umich.edu/software/apriltag), a visual fiducial system for robotics, augmented reality, and camera calibration. This library detects AprilTag markers in images and estimates their 3D pose.

This package tries to be generally compatible with [libapriltag](https://github.com/AprilRobotics/apriltag).

## Features

- **Pure Rust** — no C dependencies required (though C FFI bindings are available)
- **GPU acceleration** — optional OpenCL and wGPU backends for faster detection
- **Multiple tag families** — tag16h5, tag25h9, tag36h10, tag36h11, tagCircle21h7, tagCircle49h12, tagCustom48h12, tagStandard41h12, tagStandard52h13
- **Pose estimation** — estimate 3D tag pose from detections
- **Language bindings** — Python, C, and Java (JNI) FFI support
- **Parallel processing** — uses [Rayon](https://github.com/rayon-rs/rayon) for multithreaded detection

This package isn't (yet) on crates.io, but you can use it yourself with a git reference in `Cargo.toml`:
```
[dependencies]
apriltag_rs = { git = "https://github.com/mailmindlin/apriltag-rs" }
```

```toml
[dependencies]
apriltag-rs = { git = "https://github.com/mailmindlin/apriltag-rs" }
```

> **Note:** This crate requires Rust nightly.

## Quick Start

```rust
use apriltag_rs::{AprilTagDetector, DetectorBuilder, AprilTagFamily, ImageY8};

// Build a detector with the 36h11 tag family
let detector = DetectorBuilder::new()
    .add_family(AprilTagFamily::tag_36h11())
    .build()?;

// Load a grayscale image
let image: ImageY8 = /* your image here */;

// Detect tags
let detections = detector.detect(&image);

for det in detections.iter() {
    println!("Detected tag id={} at {:?}", det.id, det.center);
}
```

## Compile-time Features

Features are configured in `Cargo.toml` via `[features]`:

| Feature | Description |
|---------|-------------|
| `default-families` | Enables tag16h5, tag25h9, tag36h10, tag36h11 (included in `default`) |
| `all-families` | Enables all tag families |
| `opencl` | GPU acceleration via OpenCL |
| `wgpu` | GPU acceleration via wGPU |
| `cffi` | C FFI bindings compatible with [libapriltag](https://github.com/AprilRobotics/apriltag) |
| `python` | Python bindings via [PyO3](https://pyo3.rs) |
| `jni` | Java bindings via JNI |
| `debug` | Enable runtime debug profiling |
| `debug_ps` | Generate PostScript debug files |
| `opencv` | OpenCV integration |
| `bench` | Benchmarking support |

## GPU Acceleration

To use GPU acceleration, enable the `wgpu` or `opencl` feature and configure the detector:

```rust
use apriltag_rs::{DetectorBuilder, AccelerationRequest};

let detector = DetectorBuilder::new()
    .add_family(AprilTagFamily::tag_36h11())
    .accelerate(AccelerationRequest::Wgpu)
    .build()?;
```

## Examples

Run the detection example on an image:

```sh
cargo run --example detect --features tag36h11 -- path/to/image.jpg
```

Compare CPU vs GPU performance:

```sh
cargo run --example cpu_vs_gpu --features "tag36h11,wgpu" -- path/to/image.jpg
```

## Building

```sh
# Default (common tag families, CPU only)
cargo build

# With GPU support
cargo build --features wgpu

# With all tag families
cargo build --features all-families
```

## Python Bindings

Python bindings are built with [maturin](https://www.maturin.rs/). The package exposes a `Detector` class that accepts NumPy arrays directly.

```sh
# Install maturin
pip install maturin

# Build and install into the current virtualenv
maturin develop --features python

# Or build a wheel
maturin build --features python
```

Usage from Python:

```python
import numpy as np
from apriltag_rs import Detector, AprilTagFamily

detector = Detector(families=["tag36h11"])
# image must be a 2D uint8 NumPy array (grayscale)
detections = detector.detect(image)
for det in detections:
    print(f"Tag {det.id} at {det.center}")
```

## C Bindings

The `cffi` feature generates a C header compatible with [libapriltag](https://github.com/AprilRobotics/apriltag) using [cbindgen](https://github.com/mozilla/cbindgen). When the feature is enabled, the header is generated automatically during `cargo build`.

```sh
# Build the shared library with C bindings
cargo build --features cffi

# The generated header is written to target/apriltag-rs.h
```

You can then link against the resulting library and include the generated header in your C/C++ project.

## Acknowledgments

Based on the [AprilTag](https://github.com/AprilRobotics/apriltag) project by the APRIL Robotics Laboratory at the University of Michigan.
