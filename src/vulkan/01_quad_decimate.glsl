#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer ImageSrc {

	uchar data[];
};

layout(set = 0, binding = 1) buffer ImageDst {
	uint data[];
	
};

/// Downsample 3/2
static uchar downsample_32(uchar a, uchar b, uchar c, uchar d) {
	return (4 * ((ushort) a) + 2 * ((ushort) b) + 2 * ((ushort) c) + ((ushort) d)) / 9;
}

// quad_decimate but for ffactor == 1.5
__kernel void k01_filter_quad_decimate_32(
    __global const uchar *src,
    __private int stride_src,
	__private const uint src_width,
    __global uchar *dst,
	__private const int dst_stride
) {
    const size_t x = min(get_global_id(0), (size_t) src_width);
    const size_t y = get_global_id(1);

	// Input is 3x3
	size_t row = (y * 3 * stride_src) + (x * 3);
	uchar a = src[row + 0];
	uchar b = src[row + 1];
	uchar c = src[row + 2];

	row += stride_src;
	uchar d = src[row + 0];
	uchar e = src[row + 1];
	uchar f = src[row + 2];

	row += stride_src;
	uchar g = src[row + 0];
	uchar h = src[row + 1];
	uchar i = src[row + 2];

	// output is 2x2
	row = y * 2 * dst_stride + (x * 2);
	dst[row + 0] = downsample_32(a, b, d, e);
	dst[row + 1] = downsample_32(c, b, f, e);
	row += dst_stride;
	dst[row + 0] = downsample_32(g, h, d, e);
	dst[row + 1] = downsample_32(i, h, f, e);
}

// quad_decimate but for ffactor == 1.5
__kernel void k01_filter_quad_decimate(
    __global const uchar *src,
    __private uint src_stride,
	__private const uint src_width,
    __global uchar *dst,
	__private const uint dst_stride,
	__private const uint factor
) {
    const size_t x = min(get_global_id(0), (size_t) src_width);
    const size_t y = get_global_id(1);

	size_t src_idx = (y * factor * src_stride) + (x * factor);
	size_t dst_idx = (y * dst_stride) + x;
	dst[dst_idx] = src[src_idx];
	//TODO: replace with memcpy?
}
