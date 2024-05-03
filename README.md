# apriltag-rs

# Table of Contents


# Installation

This package tries to be compatible with 

# Building from scratch

Compile with `cargo`

## Compile-time Features

### Bindings
 `cffi`
Generates C FFI interface that should be compatible with [libapriltag](https://github.com/AprilRobotics/apriltag).

### `cpython`
Python bindings and library

### `jni`
Genarate java bindings

### `debug`
Allow generating debug information (this is still behind a runtime flag)

### `debug_ps`
Generate PostScript debug files (may be slow)

### `debug_svg`

### `extra_debug`

### `approx_eq`
Derive approximate equality traits (requires )

### `compare_reference`
Compare output to native libapriltag

## Debug images

### debug_preprocess.pnm
### debug_quads_raw.pnm