@id(0) override stride_src: u32;
@id(1) override stride_dst: u32;
@group(0) @binding(0) var<uniform, read> filter : array<u32>; // Packed u8x4
@group(0) @binding(1) var<storage, read> img_src : array<u32>; // Packed u8x4
@group(0) @binding(2) var<storage, read_write> img_dst : array<u32>; // Packed u8x4

fn k02_gaussian_blur_filter(
        __global const uchar *im_src,
        __private const uint stride_src,
        __private const uint width_src,
        __private const uint height_src,
        __constant uchar *filter,
        __private const uint ksz,
        __global uchar *im_dst,
        __private const uint stride_dst
) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    
    // Collect neighbor values and multiply with gaussian
    uint sum = 0;
    // float sum = 0.0f;
    // Calculate the mask size based on sigma (larger sigma, larger mask)
    for (size_t u = 0; u < ksz; u++) {
        size_t py = min(sub_sat(y+u, (size_t) ksz/2), (size_t) height_src);
        size_t row_base = py * stride_src;
        ushort k_u = (ushort) filter[u];
        for (uint v = 0; v < ksz; v++) {
            size_t px = min(sub_sat(x + v, (size_t) ksz/2), (size_t) width_src);
            ushort k = mul24((ushort) filter[v], k_u);
            sum += mul24(k, im_src[row_base + px]);
        }
    }
    im_dst[y * stride_dst + x] = convert_uchar_sat(sum >> 16);
}

__kernel void k02_gaussian_sharp_filter(
        __global const uchar *im_src,
        __private const uint stride_src,
        __private const uint width_src,
        __private const uint height_src,
        __constant uchar *filter,
        __private const uint ksz,
        __global uchar *im_dst,
        __private const uint stride_dst
) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    
    // Collect neighbor values and multiply with gaussian
    uint sum = 0;
    // float sum = 0.0f;
    // Calculate the mask size based on sigma (larger sigma, larger mask)
    for (size_t u = 0; u < ksz; u++) {
        size_t py = min(sub_sat(y+u, (size_t) ksz/2), (size_t) height_src);
        size_t row_base = py * stride_src;
        ushort k_u = (ushort) filter[u];
        for (uint v = 0; v < ksz; v++) {
            size_t px = min(sub_sat(x + v, (size_t) ksz/2), (size_t) width_src);
            ushort k = mul24((ushort) filter[v], k_u);
            sum += mul24(k, im_src[row_base + px]);
        }
    }
    // We might as well calculate this in fixed-point
    uint value = sub_sat((((uint) im_src[y * stride_src + x]) << 16), sum);
    im_dst[y * stride_dst + x] = convert_uchar_sat(value >> 16);
}
