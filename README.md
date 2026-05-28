# apriltag-rs

This package tries to be generally compatible with [libapriltag](https://github.com/AprilRobotics/apriltag).

# Installation

This package isn't (yet) on crates.io, but you can use it yourself with a git reference in `Cargo.toml`:
```
[dependencies]
apriltag_rs = { git = "https://github.com/mailmindlin/apriltag-rs" }
```

# Building from scratch

Compile with `cargo`:

```
cargo build --release
```

## Compile-time Features

### `cffi`
Generates C FFI interface that should be compatible with [libapriltag](https://github.com/AprilRobotics/apriltag).

### `cpython`
Python bindings and library

### `jni`
Genarate java bindings. Java library must be compiled separately.

### `debug`
Allow generating debug files (this is also still behind a runtime setting).

### `debug_ps`
Generate PostScript debug files (may have a large performance hit, even more than `debug`)

### `opencl`
Allows image preprocessing on with OpenCL (this feature also requires a runtime setting).

### `wgpu`
Allows image preprocessing and unionfind on with WGPU (this feature also requires a runtime setting).

I haven't seen huge speedups from GPU acceleration (in my very limited testing), but it does help to reduce CPU
usage.

### `approx_eq`
Derive approximate equality traits

### `compare_reference`
Compare output to native libapriltag.

This is mostly for debugging/validation, as it runs every step of the Apriltag detection
process twice (one in Rust, and one in libapriltag) and compares the two.

## Debug images

### debug_preprocess.pnm
### debug_quads_raw.pnm