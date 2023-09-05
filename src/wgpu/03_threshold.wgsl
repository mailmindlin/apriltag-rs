#ifndef TILESZ
#define TILESZ 4
#endif

__kernel void k03_tile_minmax(
        __global const uchar *src,
        __private const uint src_stride,
        __global uchar *dst,
        __private const uint dst_stride
    ) {
    int tx = get_global_id(0);
    int ty = get_global_id(1);

    int base_src = (ty * TILESZ * src_stride) + (tx * TILESZ);

    uchar v_max = 0;
    uchar v_min = 255;
    for(int dy = 0; dy < TILESZ; dy++) {
        int row_src = base_src + (dy * src_stride);
        for (int dx = 0; dx < TILESZ; dx++) {
            uchar v = src[row_src + dx];
            v_min = min(v_min, v);
            v_max = max(v_max, v);
        }
    }

    dst[(ty * dst_stride) + tx + 0] = v_min;
    dst[(ty * dst_stride) + tx + 1] = v_max;
}

__kernel void k03_tile_blur(
    __global const uchar2 *im_src,
    __private const uint stride_src,
    __private const uint width_src,
    __private const uint height_src,
    __global uchar2 *im_dst,
    __private const uint stride_dst
) {
    int tx = get_global_id(0);
    int ty = get_global_id(1);

    uchar v_min = 0;
    uchar v_max = 255;
    for (int i = sub_sat(ty, 1); i < min(ty + 1, (int) height_src); i++) {
        int row = i * stride_src + tx;
        for (int j = sub_sat(tx, 1); j < min(tx + 1, (int) width_src); j++) {
            uchar2 v = im_src[row + j];
            v_min = min(v_min, v.s0);
            v_max = max(v_max, v.s1);
        }
    }
    
    im_dst[ty * stride_dst + tx] = (uchar2) (v_min, v_max);
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

    uchar2 v_minmax = im_minmax[(ty * stride_minmax) + tx];
    uchar v_min = v_minmax.s0;
    uchar v_max = v_minmax.s1;

    uchar delta = v_max - v_min;
    uchar thresh = v_min + (delta / 2);
    for (size_t dy = 0; dy < TILESZ; dy++) {
        size_t y = (ty * TILESZ) + dy;
        if (y > height_src)
            continue;
        for (size_t dx = 0; dx < TILESZ; dx++) {
            size_t x = (tx * TILESZ) + dx;
            if (x > width_src)
                continue;

            size_t idx_sd = (y * stride_dst) + x;
            uchar value;
            if (delta < min_white_black_diff) {
                // low contrast region? (no edges)
                value = 127;
            } else {
                // otherwise, actually threshold this tile.

                // argument for biasing towards dark; specular highlights
                // can be substantially brighter than white tag parts
                uchar v = im_src[idx_sd];
                value = v > thresh ? 255 : 0;
            }
            im_dst[idx_sd] = value;
        }
    }
}
