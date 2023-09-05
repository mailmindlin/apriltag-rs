__kernel void k02_gaussian_blur_filter(
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

    const size_t min_xy = ksz/2;
    const size_t max_y = (height_src - ksz) + ksz/2;
    const size_t max_x = (width_src - ksz) + ksz/2;
    
    // Calculate the mask size based on sigma (larger sigma, larger mask)
    uchar result = 0;
    if (y < min_xy || y >= max_y) {
        if (x < min_xy || x >= max_x) {
            // Corner (copy value)
            result = im_src[y * stride_src + x];
        } else {
            // Only convolve horizontally
            uint sum = 0;
            for (size_t v = 0; v < ksz; v++) {
                size_t px = x + v - ksz/2;
                sum += ((uint) im_src[y * stride_src + px]) * ((uint) filter[v]);
            }
            result = convert_uchar_sat(sum >> 8);
        }
    } else if (x < min_xy || x >= max_x) {
        // Only convolve vertically
        uint sum = 0;
        for (size_t u = 0; u < ksz; u++) {
            size_t py = y + u - ksz/2;
            sum += ((uint) im_src[py * stride_src + x]) * ((uint) filter[u]);
        }
        result = convert_uchar_sat(sum >> 8);
    } else {
        // Collect neighbor values and multiply with gaussian
        uint sum = 0;
        for (size_t u = 0; u < ksz; u++) {
            // Clamped y+(u-ksz/2)
            size_t py = min(sub_sat(y+u, (size_t) ksz/2), (size_t) height_src);
            size_t row_base = py * stride_src;
            ushort k_u = (ushort) filter[u];
            uint sum_row = 0;
            for (size_t v = 0; v < ksz; v++) {
                // Clamped x+(v-ksz/2)
                size_t px = min(sub_sat(x + v, (size_t) ksz/2), (size_t) width_src);
                // ushort k = mul24((ushort) filter[v], k_u);
                // sum += mul24(k, im_src[row_base + px]);
                sum_row += ((ushort) im_src[row_base + px]) * ((ushort) filter[v]);
            }
            sum += ((ushort) convert_uchar_sat(sum_row >> 8)) * ((ushort) k_u);
        }
        result = convert_uchar_sat(sum >> 8);
    }
    im_dst[y * stride_dst + x] = result;
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
