// Step 2: Gaussian blur / sharpen (texture-based variant)
//
// Applies a separable Gaussian filter to the (possibly decimated) image.
// The filter kernel is provided as a 1D array of fixed-point weights in
// buf_filter; the 2D kernel is the outer product of this array with itself.
//
// Two entry points:
//   k02_gaussian_blur_filter  — standard Gaussian blur
//   k02_gaussian_sharp_filter — unsharp mask (2*original - blurred)
//
// Border handling: pixels near the image edge are convolved only along
// the axis that has enough room for the full kernel. Corner pixels are
// passed through unfiltered.

@group(0) @binding(0) var<storage, read> buf_filter : array<u32>;     // 1D Gaussian kernel (fixed-point)

@group(1) @binding(0) var img_src : texture_2d<u32>;                   // Input grayscale image
@group(1) @binding(1) var img_dst : texture_storage_2d<r8uint, write>; // Output filtered image

/// Read a single kernel weight
fn get_filter(x: u32) -> u32 {
    return buf_filter[x];
}

/// Saturating subtraction: returns max(x - y, 0)
fn sub_sat(x: u32, y: u32) -> u32 {
    return select(x - y, 0u, y >= x);
}

/// Apply 2D Gaussian blur at a single pixel coordinate.
/// Returns the blurred intensity value [0, 255].
///
/// Near image borders the convolution degrades gracefully:
///   - Both axes out of range: return the original pixel (no filtering)
///   - One axis out of range: convolve only along the valid axis (1D blur)
///   - Both axes in range: full 2D convolution
fn blur_coords(coords: vec2u) -> u32 {
    let ksz = arrayLength(&buf_filter);
    let dims = textureDimensions(img_src, 0);

    var res = 0u;
    let border = ksz/2u;
    let x_oob = coords.x < border || coords.x >= dims.x - border - 1u;
    let y_oob = coords.y < border || coords.y >= dims.y - border - 1u;

    // Corner: both axes too close to edge — pass through unfiltered
    if (x_oob && y_oob) {
        return textureLoad(img_src, coords, 0).r;
    }

    if (x_oob) {
        // Only convolve in y direction
        var sum = 127u; // rounding bias
        for (var u: u32 = 0u; u < ksz; u++) {
            let py = min(sub_sat(coords.y+u, border), dims.y);
            let k_u = get_filter(u);
            sum += k_u * textureLoad(img_src, vec2u(coords.x, py), 0).r;
        }
        return clamp(sum >> 8u, 0u, 255u);
    }

    if (y_oob) {
        // Only convolve in x direction
        var sum = 127u; // rounding bias
        for (var v: u32 = 0u; v < ksz; v++) {
            let px = min(sub_sat(coords.x+v, border), dims.x);
            let k_v = get_filter(v);
            sum += k_v * textureLoad(img_src, vec2u(px, coords.y), 0).r;
        }
        return clamp(sum >> 8u, 0u, 255u);
    }

    // Full 2D convolution: outer product of 1D kernel with itself
    var sum = (1u << 15u); // rounding bias for >>16
    for (var u: u32 = 0u; u < ksz; u++) {
        let py = min(sub_sat(coords.y+u, border), dims.y);
        let k_u = get_filter(u);
        for (var v: u32 = 0u; v < ksz; v++) {
            let px = min(sub_sat(coords.x + v, border), dims.x);
            let k_v = get_filter(v);
            let k = k_u * k_v;
            sum += k * textureLoad(img_src, vec2u(px, py), 0).r;
        }
    }
    return clamp(sum >> 16u, 0u, 255u);
}

/// Standard Gaussian blur — one invocation per pixel.
@compute
@workgroup_size($wg_width, $wg_height)
fn k02_gaussian_blur_filter(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if (any(global_invocation_id.xy >= textureDimensions(img_src, 0))) {
        return;
    }
    let res = blur_coords(global_invocation_id.xy);
    textureStore(img_dst, global_invocation_id.xy, vec4u(res));
}

/// Unsharp mask: enhances edges by computing (2 * original - blurred).
/// One invocation per pixel.
@compute
@workgroup_size($wg_width, $wg_height)
fn k02_gaussian_sharp_filter(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if (any(global_invocation_id.xy >= textureDimensions(img_src, 0))) {
        return;
    }
    let orig = textureLoad(img_src, global_invocation_id.xy, 0).r;
    let blurred = blur_coords(global_invocation_id.xy);
    let res = clamp(sub_sat(orig * 2u, blurred), 0u, 255u);
    textureStore(img_dst, global_invocation_id.xy, vec4u(res));
}
