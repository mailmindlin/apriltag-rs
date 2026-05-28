@group(0) @binding(0) var<storage, read> buf_filter : array<u32>;

@group(1) @binding(0) var img_src : texture_2d<u32>;
@group(1) @binding(1) var img_dst : texture_storage_2d<r8uint, write>;


fn get_filter(x: u32) -> u32 {
    return buf_filter[x];
}

fn sub_sat(x: u32, y: u32) -> u32 {
    if (y >= x) {
        return 0u;
    } else {
        return x - y;
    }
}

fn blur_coords(coords: vec2u) -> u32 {
    let ksz = arrayLength(&buf_filter);
    let dims = textureDimensions(img_src, 0);

    var res = 0u;
    let border = ksz/2u;
    let x_oob = coords.x < border || coords.x >= dims.x - border - 1u;
    let y_oob = coords.y < border || coords.y >= dims.y - border - 1u;
    if (x_oob && y_oob) {
        return textureLoad(img_src, coords, 0).r;
    }
    if (x_oob) {
        // Only convolve in y direction
        var sum = 127u;
        for (var u: u32 = 0u; u < ksz; u++) {
            let py = min(sub_sat(coords.y+u, border), dims.y);
            let k_u = get_filter(u);
            sum += k_u * textureLoad(img_src, vec2u(coords.x, py), 0).r;
        }
        return clamp(sum >> 8u, 0u, 255u);
    }
    if (y_oob) {
        // Only convolve in x direction
        var sum = 127u;
        for (var v: u32 = 0u; v < ksz; v++) {
            let px = min(sub_sat(coords.x+v, border), dims.x);
            let k_v = get_filter(v);
            sum += k_v * textureLoad(img_src, vec2u(px, coords.y), 0).r;
        }
        return clamp(sum >> 8u, 0u, 255u);
    }
    // Collect neighbor values and multiply with gaussian
    var sum = (1u << 15u);
    // Calculate the mask size based on sigma (larger sigma, larger mask)
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

@compute
@workgroup_size($wg_width, $wg_height)
fn k02_gaussian_blur_filter(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if (any(global_invocation_id.xy >= textureDimensions(img_src, 0))) {
        return;
    }
    let res = blur_coords(global_invocation_id.xy);
    textureStore(img_dst, global_invocation_id.xy, vec4u(res));
}

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
