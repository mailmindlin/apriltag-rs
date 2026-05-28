// Download texture to buffer
//
// Copies a texture_2d<u32> (one pixel per texel) into a packed u8x4 storage
// buffer. Each invocation reads up to 4 adjacent texels and packs them into
// a single u32 word. Handles the right edge of the image where fewer than 4
// pixels may remain.

@group(0) @binding(0) var img_src : texture_2d<u32>;                // Source texture (1 pixel per texel)
@group(0) @binding(1) var stride_dst: u32;                          // Row stride of destination buffer (in u32 words)
@group(0) @binding(2) var<storage, write> img_dst : array<u8x4>;    // Packed output buffer

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let width = textureDimensions(img_src).x;
    let base = global_invocation_id.xy * vec2(4u, 1u);
    let right = width - base.x * 4u; // Number of remaining pixels to the right edge

    // Load up to 4 pixels, guarding against the right edge
    var p0 = textureLoad(img_src, global_invocation_id.xy, 0u);
    var p1 = 0u;
    var p2 = 0u;
    var p3 = 0u;
    if (right > 1) {
        p1 = textureLoad(img_src, global_invocation_id.xy + vec2(1u, 0u), 0u).r;
        if (right > 2) {
            p2 = textureLoad(img_src, global_invocation_id.xy + vec2(2u, 0u), 0u).r;
            if (right > 3) {
                p3 = textureLoad(img_src, global_invocation_id.xy + vec2(3u, 0u), 0u).r;
            }
        }
    }

    // Pack 4 pixels into a single u32 (big-endian byte order)
    let val = ((p0.x & 0xFFu) << 24) | ((p1.x & 0xFFu) << 16) | ((p2.x & 0xFFu) << 8) | ((p3.x & 0xFFu) << 0);

    img_dst[dst_row_stride * global_invocation_id.y + global_invocation_id.x] = val;
}