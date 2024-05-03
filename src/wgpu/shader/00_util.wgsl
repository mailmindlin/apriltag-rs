alias u8x4 = u32;
const SHIFT_U8x4_0: u32 =  0u;
const SHIFT_U8x4_1: u32 =  8u;
const SHIFT_U8x4_2: u32 = 16u;
const SHIFT_U8x4_3: u32 = 24u;
/// Get lane from u8x4
@const @must_use fn u8x4_lane(packed: u8x4, lane: u32) -> u32 {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}
/// Get lane from vec2<u8x4>
@const @must_use fn u8x4_lane_x2(packed: vec2<u8x4>, lane: u32) -> vec2<u32> {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}
/// Get lane from vec3<u8x4>
@const @must_use fn u8x4_lane_x3(packed: vec3<u8x4>, lane: u32) -> vec3<u32> {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}
/// Get lane from vec4<u8x4>
@const @must_use fn u8x4_lane_x4(packed: vec4<u8x4>, lane: u32) -> vec2<u32> {
    return extractBits(packed, (lane % 4u) * 8u, 8u);
}

@const @must_use fn u8x4_unpack(packed: u32) -> vec4<u32> {
    return vec4(
        extractBits(packed, SHIFT_U8x4_0, 8u),
        extractBits(packed, SHIFT_U8x4_1, 8u),
        extractBits(packed, SHIFT_U8x4_2, 8u),
        extractBits(packed, SHIFT_U8x4_3, 8u),
    );
}
@const @must_use fn u8x4_pack(raw: vec4<u32>) -> u32 {
    return ((raw.x & 0xFFu) << SHIFT_U8x4_0)
         | ((raw.y & 0xFFu) << SHIFT_U8x4_1)
         | ((raw.z & 0xFFu) << SHIFT_U8x4_2)
         | ((raw.w & 0xFFu) << SHIFT_U8x4_3);
}

struct ImageDims {
    width: u32,
    height: u32,
    stride: u32,
}

struct ImageY8 {
    dims: ImageDims,
    data: array<u8x4>,
}

@const @must_use fn imageDimensions(image: ptr<storage, ImageY8, read>) -> vec2<u32> {
    return vec2(
        image.dims.width,
        image.dims.height,
    );
}

/// Load from packed Y8 image
@const @must_use fn imageLoad(image: ptr<storage, ImageY8, read>, coords: vec2<u32>) -> u32 {
    let cell_offset = (coords.x / 4u) + (coords.y * image.dims.stride);
    let cell = image.data[cell_offset];
    return u8x4_lane(cell, coords.x);
}

/// Load from packed Y8 image
@const @must_use fn imageStore4(image: ptr<storage, ImageY8, read_write>, coords: vec2<u32>, value: vec4<u32>) -> u32 {
    let cell_offset = (coords.x / 4u) + (coords.y * image.dims.stride);
    let cell = image.data[cell_offset];
    return u8x4_lane(cell, coords.x);
}