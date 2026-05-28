struct Params {
    stride_src: u32,
    width_src: u32,
    height_src: u32,
    stride_dst: u32,
}

@group(0) @binding(0) var<storage, read> buf_filter : array<u32>;

@group(1) @binding(0) var<uniform> params: Params;
@group(1) @binding(1) var<storage, read> img_src : array<u32>; // Packed u8x4
@group(1) @binding(2) var<storage, write> img_dst : array<u32>; // Packed u8x4
@group(1) @binding(3) var<workgroup, write> row_buf: array<u32>;

fn unpack_pixel(packed: u32, off_x: u32) -> u32 {
	return extractBits(packed, (off_x % 4u) * 8u, 8u);
}

fn split_pixel(packed: u32) -> vec4<u32> {
    return vec4(
        unpack_pixel(packed, 0u),
        unpack_pixel(packed, 1u),
        unpack_pixel(packed, 2u),
        unpack_pixel(packed, 3u),
    );
}

fn get_pixel(x: u32, y: u32) -> u32 {
    let idx = (y * params.stride_src) + (x / 4u);
    return unpack_pixel()
}

fn get_pixel4(x: u32, y: u32) -> vec4u {
    let idx = (y * params.stride_src) + (x / 4u);
    return split_pixel(img_src[idx]);
}

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

fn convert_uchar_sat(sum: vec4u) -> u32 {
    let clamped = clamp(sum >> 16u, vec4(0u), vec4(255u));
    return (sum.w << 24u) | (sum.z << 16u) | (sum.y << 8u) | (sum.x << 0u);
}

@compute
@workgroup_size(1, 1)
fn k02_gaussian_blur_filter(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let x = global_invocation_id.x;
    let y = global_invocation_id.y;
    let ksz = arrayLength(&buf_filter);

    let res = 0u;
    if (x < ksz/2u) {
        res = get_pixel
    }
    
    // Collect neighbor values and multiply with gaussian
    var sum = vec4u(0u);
    // Calculate the mask size based on sigma (larger sigma, larger mask)
    for (var u: u32 = 0u; u < ksz; u++) {
        let py = min(sub_sat(y+u, ksz/2u), params.height_src);
        let k_u = get_filter(u);
        for (var v: u32 = 0u; v < ksz; v++) {
            let px = min(sub_sat(x + v, ksz/2u), params.width_src);
            let k_v = get_filter(v);
            let k = k_u * k_v;
            sum += vec4u(k) * get_pixel4(px, py);
        }
    }
    img_dst[y * params.stride_dst + x] = convert_uchar_sat(sum);
}

@compute
@workgroup_size(1, 1)
fn k02_gaussian_sharp_filter(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // let x = global_invocation_id.x;
    // let y = global_invocation_id.y;
    // let ksz = arraySize(buf_filter);
    
    // // Collect neighbor values and multiply with gaussian
    // uint sum = 0;
    // // float sum = 0.0f;
    // // Calculate the mask size based on sigma (larger sigma, larger mask)
    // for (size_t u = 0; u < ksz; u++) {
    //     size_t py = min(sub_sat(y+u, (size_t) ksz/2), (size_t) height_src);
    //     size_t row_base = py * stride_src;
    //     ushort k_u = (ushort) filter[u];
    //     for (uint v = 0; v < ksz; v++) {
    //         size_t px = min(sub_sat(x + v, (size_t) ksz/2), (size_t) width_src);
    //         ushort k = mul24((ushort) filter[v], k_u);
    //         sum += mul24(k, im_src[row_base + px]);
    //     }
    // }
    // // We might as well calculate this in fixed-point
    // uint value = sub_sat((((uint) im_src[y * stride_src + x]) << 16), sum);
    // im_dst[y * stride_dst + x] = convert_uchar_sat(value >> 16);
}
