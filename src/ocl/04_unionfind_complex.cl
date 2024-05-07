bool invalid_pixel(uchar pix) {
	return (pix == 127);
}
bool should_merge_pixels(uchar a, uchar b) {
	return a == b;
}

__kernel void k04_local_uf_merge_coarse(
	__global const uchar *src,
	__private const uint stride_src,
	__local uint *labels,
	__local uchar *dBuff,
	__global const uint *labelMap
) {
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	#ifdef LOCAL_LINEAR
		const size_t tid = get_local_linear_id();
	#else
		const size_t tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
	#endif
	labels[tid] = tid;
	dBuff[tid] = src[x + (y * stride_src)];
	barrier(CLK_LOCAL_MEM_FENCE);
	// Row scan (left)
	if (tid > 0 && should_merge_pixels(dBuff[tid], dBuff[tid - 1])) {
		labels[tid] = labels[tid - 1];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Column scan (up)
	if (tid > get_local_size(0) && should_merge_pixels(dBuff[tid], dBuff[tid - get_local_size(0)])) {
		labels[tid] = labels[tid - get_local_size(0)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Row-column unification
	uint temp = tid;
	while (temp != labels[temp]) {
		temp = labels[temp];
		labels[tid] = temp;
	}

	// Local union find
	if (tid > 0 && should_merge_pixels(dBuff[tid], dBuff[tid - 1])) {
		findAndUnion(labels, tid, tid - 1);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (dBuff[tid] == dBuff[tid - get_local_size(0)]) {
		findAndUnion(labels, tid, tid - get_local_size(0));
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Convert local index into global index

}

__kernel void k04_boundary_analysis(
	__global const uchar *src,
	__private const uint stride_src,
	__private const uint width_src,
	__private const uint height_src,
	__global uchar *labelMap,
) {
	size_t id = get_global_id(0);
// declare int id, hx, hy, vx, vy, pInLine, ph, pv
// 2: declare bool bh, bv
	size_t hx = id % width_src;
	size_t hy = id / (width_src * get_local_size(1));
	size_t pInLine = width_src / get_local_size(0);
	size_t vx = id % pInLine * get_local_size(0);
	size_t vy = id / pInLine;
	size_t ph = hx + hy * width_src;
	size_t pv = vx + vy * width_src;

	// (hx, hy) is within image dims
	bool bh = (hx < width_src) && (hy < height_src);
	// (vx, vy) is within image dims
	bool bv = (vx < width_src) && (vy < height_src);

	if (bh && src[hx + hy * stride_src] == src
4: // convert 1D global thread id to 2D cell id
5: hx ← id % imgW idth
6: hy ← id / (imgW idth ∗ blockDim.y)
7: pInLine ← imgW idth / blockDim.x
8: vx ← id % pInLine ∗ blockDim.x
9: vy ← id / pInLine
10: ph ← hx + hy ∗ imgW idth
11: pv ← vx + vy ∗ imgW idth
12: bh ← hx < imgW idth & hy < imgHeight
13: bv ← vx < imgW idth & vy < imgHeight
14: // boundary analysis along x-axis
15: if bh & image[hx, hy] == image[hx − 1, hy]
16: findAndUnion(LabelM ap, ph, ph − 1);
17: end if
18: // boundary analysis along y-axis
19: if bv & image[vx, vy] == image[vx, vy − imgW idth]
20: findAndUnion(LabelM ap, pv, pv − imgW idth);
21: end if
}
