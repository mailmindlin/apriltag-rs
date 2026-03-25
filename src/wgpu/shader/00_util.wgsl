// Shared utilities for packed pixel manipulation
//
// WGSL has no native u8 type, so grayscale images are stored as packed u32
// values (4 pixels per word, little-endian byte order). This file provides
// helper functions for extracting and inserting individual pixel lanes,
// as well as a struct-based image abstraction (ImageY8) that bundles
// dimensions with the packed data array.

/// Alias: a u32 holding 4 packed 8-bit values (pixels or components)
alias u8x4 = u32;

// Bit shifts for each lane within a u8x4
const SHIFT_U8x4_0: u32 =  0u;
const SHIFT_U8x4_1: u32 =  8u;
const SHIFT_U8x4_2: u32 = 16u;
const SHIFT_U8x4_3: u32 = 24u;

/// Extract one 8-bit lane from a packed u8x4
@const @must_use fn u8x4_lane(packed: u8x4, lane: u32) -> u32 {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}
/// Extract one lane from each element of a vec2<u8x4>
@const @must_use fn u8x4_lane_x2(packed: vec2<u8x4>, lane: u32) -> vec2<u32> {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}
/// Extract one lane from each element of a vec3<u8x4>
@const @must_use fn u8x4_lane_x3(packed: vec3<u8x4>, lane: u32) -> vec3<u32> {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}
/// Extract one lane from each element of a vec4<u8x4>
@const @must_use fn u8x4_lane_x4(packed: vec4<u8x4>, lane: u32) -> vec2<u32> {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}

/// Unpack all 4 lanes of a u8x4 into a vec4<u32>
@const @must_use fn u8x4_unpack(packed: u32) -> vec4<u32> {
    return vec4(
        extractBits(packed, SHIFT_U8x4_0, 8u),
        extractBits(packed, SHIFT_U8x4_1, 8u),
        extractBits(packed, SHIFT_U8x4_2, 8u),
        extractBits(packed, SHIFT_U8x4_3, 8u),
    );
}

/// Pack a vec4<u32> (each component [0,255]) into a single u8x4
@const @must_use fn u8x4_pack(raw: vec4<u32>) -> u32 {
    return ((raw.x & 0xFFu) << SHIFT_U8x4_0)
         | ((raw.y & 0xFFu) << SHIFT_U8x4_1)
         | ((raw.z & 0xFFu) << SHIFT_U8x4_2)
         | ((raw.w & 0xFFu) << SHIFT_U8x4_3);
}

/// Dimensions of a packed grayscale image
struct ImageDims {
    width: u32,    // Image width in pixels
    height: u32,   // Image height in pixels
    stride: u32,   // Row stride in u32 words (= ceil(width/4))
}

/// A Y8 (8-bit grayscale) image stored as packed u8x4 words.
/// Used as a struct-of-arrays layout in storage buffers.
struct ImageY8 {
    dims: ImageDims,
    data: array<u8x4>,
}

/// Get image dimensions as a vec2
@const @must_use fn imageDimensions(image: ptr<storage, ImageY8, read>) -> vec2<u32> {
    return vec2(
        image.dims.width,
        image.dims.height,
    );
}

/// Load a single pixel from a packed Y8 image at the given (x, y) coordinates
@const @must_use fn imageLoad(image: ptr<storage, ImageY8, read>, coords: vec2<u32>) -> u32 {
    let cell_offset = (coords.x / 4u) + (coords.y * image.dims.stride);
    let cell = image.data[cell_offset];
    return u8x4_lane(cell, coords.x);
}

/// Load from packed Y8 image (read_write variant — appears incomplete)
@const @must_use fn imageStore4(image: ptr<storage, ImageY8, read_write>, coords: vec2<u32>, value: vec4<u32>) -> u32 {
    let cell_offset = (coords.x / 4u) + (coords.y * image.dims.stride);
    let cell = image.data[cell_offset];
    return u8x4_lane(cell, coords.x);
}