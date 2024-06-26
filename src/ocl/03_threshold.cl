#ifndef TILESZ
#define TILESZ 4
#endif

__kernel void k03_tile_minmax(
    __global const uchar *src,
    __private const uint src_stride,
    __global uchar2 *dst,
    __private const uint dst_stride
) {
    size_t tx = get_global_id(0);
    size_t ty = get_global_id(1);

    size_t base_src = (ty * TILESZ * src_stride) + (tx * TILESZ);

    uchar v_max = 0;
    uchar v_min = 255;
    for(size_t dy = 0; dy < TILESZ; dy++) {
        // size_t row_src = base_src + (dy * src_stride);
        for (size_t dx = 0; dx < TILESZ; dx++) {
            uchar v = src[(ty * TILESZ + dy) * src_stride + (tx * TILESZ + dx)];
            v_min = min(v_min, v);
            v_max = max(v_max, v);
        }
    }

    dst[(ty * dst_stride) + tx] = (uchar2)(v_min, v_max);
}

__kernel void k03_tile_blur(
    __global const uchar2 *im_src,
    __private const uint stride_src,
    __private const uint width_src,
    __private const uint height_src,
    __global uchar2 *im_dst,
    __private const uint stride_dst
) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    uchar v_min = 255;
    uchar v_max = 0;
    for (size_t i = sub_sat(y, (size_t) 1); i < min(y + 1, (size_t) height_src); i++) {
        size_t row = i * stride_src;
        for (size_t j = sub_sat(x, (size_t) 1); j < min(x + 1, (size_t) width_src); j++) {
            uchar2 v = im_src[row + j];
            v_min = min(v_min, v.s0);
            v_max = max(v_max, v.s1);
        }
    }
    
    im_dst[y * stride_dst + x] = (uchar2) (v_min, v_max);
}

/**
 * @param im_src Source image (2d array)
 * @param stride_src Source stride
 * @param width_src Source width
 * @param height_src Source height
 * @param im_minmax Min/max tiles (ceil(width_src/TILESZ), ceil(height_src/TILESZ))
 * @param stride_minmax Stride for im_minmax
 * @param min_white_black_diff
 * @param im_dst Output image (same dims as src)
 * @param stride_dst Stride for im_dst
 */
__kernel void k03_build_threshim(
    __global const uchar *im_src,
    __private const uint stride_src,
    __private const uint width_src,
    __private const uint height_src,
    __global const uchar2 *im_minmax,
    __private const uint stride_minmax,
    __private const uchar min_white_black_diff,
    __global uchar *im_dst,
    __private const uint stride_dst
) {
    size_t tx = get_global_id(0);
    size_t ty = get_global_id(1);

    const size_t tw = ((size_t) width_src) / TILESZ;
    const size_t th = ((size_t) height_src) / TILESZ;
    uchar2 v_minmax = im_minmax[(min(ty, th - 1) * stride_minmax) + min(tx, tw - 1)];
    const bool bottom_edge = (ty >= th) || (tx >= tw);
    const uchar v_min = v_minmax.s0;
    const uchar v_max = v_minmax.s1;

    const uchar delta = sub_sat(v_max, v_min);
    const uchar thresh = add_sat(v_min, (uchar) (delta / 2));
    for (size_t dy = 0; dy < TILESZ; dy++) {
        size_t y = (ty * TILESZ) + dy;
        if (y > height_src)
            continue;
        
        for (size_t dx = 0; dx < TILESZ; dx++) {
            size_t x = (tx * TILESZ) + dx;
            if (x > width_src)
                continue;
            
            uchar value;
            if (bottom_edge || delta >= min_white_black_diff) {
                // otherwise, actually threshold this tile.
                // argument for biasing towards dark; specular highlights
                // can be substantially brighter than white tag parts
                uchar v = im_src[y * stride_src + x];
                value = (v > thresh) ? 255 : 0;
            } else {
                // low contrast region? (no edges)
                value = 127;
            }
            im_dst[y * stride_dst + x] = value;
        }
    }
}
