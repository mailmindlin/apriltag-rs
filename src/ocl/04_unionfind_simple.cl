#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

uchar threshim_get_pixel(__global const uchar *image, size_t stride, size_t x, size_t y) {
	#ifdef THRESHIM_DENSE
		//TODO
		#error "threshim_dense not supported"
	#else
		return image[y * stride + x];
	#endif
}

#define MAX_LOOPS 999
#define INVALID_PIXEL 127

bool invalid_pixel(uchar pix) {
	return (pix == INVALID_PIXEL);
}

bool should_merge_pixels(uchar a, uchar b) {
	return a == b;
}

/** UnionFind element ID */
typedef uint uf_element_t;

/**
 * Computes the UnionFind element id for some coordinates
 */
uf_element_t uf_element(uint uf_width, uint x, uint y) {
	return (y * uf_width) + x;
}


#ifdef OPENCL_VERSION_2
	typedef volatile __global atomic_uint *unionfind_t;
	#define ocl_atomic_load_acquire(p) (atomic_load_explicit((p), memory_order_acquire))
	#define ocl_atomic_cmpxchg_lf(ptr, expected, desired) (atomic_compare_exchange_weak_explicit((ptr), (expected), (desired), memory_order_acq_rel, memory_order_acquire))
#else
	// Make do with OpenCL 1.2 relaxed atomics
	typedef volatile __global uint *unionfind_t;
	#define ocl_atomic_load_acquire(p) (atomic_add((p), 0))
	#define ocl_atomic_cmpxchg_lf(ptr, expected, desired) (atomic_cmpxchg((ptr), (expected), (desired)))
#endif


#define uf_parent(uf, element) (&(uf)[(element) * 2 + 0])
#define uf_size(uf, element) (&(uf)[(element) * 2 + 1])

uf_element_t uf_get_parent(unionfind_t uf, uf_element_t element) {
	return ocl_atomic_load_acquire(uf_parent(uf, element));
}

bool uf_set_parent(unionfind_t uf, uf_element_t element, uf_element_t cur_parent, uf_element_t new_parent) {
	return atomic_cmpxchg(uf_parent(uf, element), cur_parent, new_parent) == cur_parent;
}

// Get UnionFind group for id
uf_element_t uf_representative(unionfind_t uf, uf_element_t element) {
	uf_element_t parent = uf_get_parent(uf, element);
	uint i = 0;
	while (element != parent && i++ < MAX_LOOPS) {
		uint grandparent = uf_get_parent(uf, element);
		uint old_parent = ocl_atomic_cmpxchg_lf(uf_parent(uf, element), parent, grandparent);
		if (old_parent == parent) {
			// CMPXCHG success
			element = parent;
			parent = grandparent;
		} else {
			parent = old_parent;
		}
		// barrier(CLK_LOCAL_MEM_FENCE);
	}
	return element;
}
bool uf_connect(unionfind_t uf, uf_element_t a, uf_element_t b) {
	for (int i = 0; i < MAX_LOOPS; i++) {
		a = uf_representative(uf, a);
		b = uf_representative(uf, b);

		if (a == b) {
			return false;
		}

		// we don't perform "union by rank", but we perform a similar
		// operation (but probably without the same asymptotic guarantee):
		// We join trees based on the number of *elements* (as opposed to
		// rank) contained within each tree. I.e., we use size as a proxy
		// for rank.  In my testing, it's often *faster* to use size than
		// rank, perhaps because the rank of the tree isn't that critical
		// if there are very few nodes in it.
		// optimization idea: We could shortcut some or all of the tree
		// that is grafted onto the other tree. Pro: u32hose nodes were just
		// read and so are probably in cache. Con: it might end up being
		// wasted effort -- the tree might be grafted onto another tree in
		// a moment!
		uint a_size = ocl_atomic_load_acquire(uf_size(uf, a));
		uint b_size = ocl_atomic_load_acquire(uf_size(uf, b));

		if (a_size < b_size) {
			if (uf_set_parent(uf, b, b, a)) {
				atomic_add(uf_size(uf, a), b_size);
				return true;
			}
		} else {
			if (uf_set_parent(uf, a, a, b)) {
				atomic_add(uf_size(uf, b), a_size);
				return true;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void k04_unionfind_init(
	__global uint *uf_data,
	__private const uint width_src
) {
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	const uf_element_t idx = uf_element(width_src, x, y);
	*uf_parent(uf_data, idx) = idx;
	*uf_size(uf_data, idx) = 1;
}

__kernel void k04_unionfind_test(
	unionfind_t uf,
	__private const uint width_src
) {
	const size_t x = get_global_id(0);
	const size_t y = get_global_id(1);
	const uint idx = uf_element(width_src, x, y);
	if ((x != 0)) {
		// atomic_add(uf_parent(uf, idx), 1);
		const uf_element_t idx_left = uf_element(width_src, x - 1, y);
		uf_connect(uf, idx, idx_left);
		// uf_setparent(uf, idx, idx, idx_left);
	}
}

void smooth_uf(unionfind_t uf, uf_element_t current) {
	for (int i = 0; i < 1; i++) {
		uf_element_t parent = uf_get_parent(uf, current);
		uf_element_t grandparent = uf_get_parent(uf, parent);
		for (int j = 0; j < 2; j++) {
			grandparent = uf_get_parent(uf, parent);
		}
		if (parent == grandparent)
			break;
		if (uf_set_parent(uf, current, parent, grandparent)) {
			current = grandparent;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void k04_unionfind_flatten(unionfind_t uf_data, __private const uint width) {
	const size_t x = get_global_id(0);
	const size_t y = get_global_id(1);
	
	uf_element_t idx = uf_element(width, x, y);
	for (int i = 0; i < 1; i++) {
		const uf_element_t parent = uf_get_parent(uf_data, idx);
		if (parent != idx) {
			uint grandparent = parent;
			for (int j = 0; j < 5; j++)
				grandparent = uf_get_parent(uf_data, grandparent);
			
			if (parent != grandparent) {
				if (uf_set_parent(uf_data, idx, parent, grandparent))
					idx = grandparent;
			}
		}
	}
}

__kernel void k04_connected_components(
	__global const uchar *src,
	__private const uint stride_src,
	__private const uint width_src,
	__private const uint height_src,
	unionfind_t uf_data,
	__global uint *d
) {
	const size_t x = get_global_id(0) + 1;
	const size_t y = get_global_id(1);
	const uf_element_t idx = uf_element(width_src, x, y);
	if (get_global_id(0) == 0 && y == 0) {
		d[0] = get_local_size(0);
		d[1] = get_local_size(1);
	}

	const uchar value = (x >= width_src-1) || (y >= height_src) ? INVALID_PIXEL : threshim_get_pixel(src, stride_src, x, y);
	if (!invalid_pixel(value)) {
		// Left
		if (x > 0) {
			uchar left = threshim_get_pixel(src, stride_src, x - 1, y);
			if (should_merge_pixels(left, value)) {
				uf_element_t idx_l = uf_element(width_src, x - 1, y);
				uf_connect(uf_data, idx, idx_l);
			}
		}
		if (y > 0) {
			// Up
			uchar pix_up = threshim_get_pixel(src, stride_src, x, y - 1);
			if (should_merge_pixels(value, pix_up)) {
				uf_element_t idx_up = uf_element(width_src, x, y - 1);
				uf_connect(uf_data, idx, idx_up);
			}
			if (value == 255) {
				// Up-left
				if (x > 0) {
					uchar pix_ul = threshim_get_pixel(src, stride_src, x - 1, y - 1);
					if (should_merge_pixels(value, pix_ul)) {
						uf_element_t idx_ul = uf_element(width_src, x - 1, y - 1);
						uf_connect(uf_data, idx, idx_ul);
					}
				}
				// Up-right
				uchar pix_ur = threshim_get_pixel(src, stride_src, x + 1, y - 1);
				if (should_merge_pixels(value, pix_ur)) {
					uf_element_t idx_ur = uf_element(width_src, x + 1, y - 1);
					uf_connect(uf_data, idx, idx_ur);
				}
			}
		}

		// smooth_uf(uf_data, idx);
		// smooth_uf(uf_data, uf_element(uf_data, (x + width_src / 2) % width_src, y));
	}
}

__kernel void k04_connected_components_row(
	__global const uchar *src,
	__private const uint stride_src,
	__private const uint width_src,
	__private const uint height_src,
	unionfind_t uf_data,
	__global uint *d
) {
	const size_t y = get_global_id(0);

	if (get_global_id(0) == 0 && y == 0) {
		d[0] = get_local_size(0);
		d[1] = get_local_size(1);
	}

	const __global uchar *row_up = &src[(y - 1) * stride_src];
	const __global uchar *row = &src[y * stride_src];

	uchar v_0_m1 = row_up[0];
    uchar v_1_m1 = row_up[1];
    uchar v = row[0];

	for (uint x = 1; x < width_src - 1; x++) {
		uchar v_m1_m1 = v_0_m1;
        v_0_m1 = v_1_m1;
        v_1_m1 = row_up[x+1];
        uchar v_m1_0 = v;
        v = row[x];

		const uf_element_t idx = uf_element(width_src, x, y);

		if (!invalid_pixel(v)) {
			// Left
			if (should_merge_pixels(v_m1_0, v)) {
				uf_element_t idx_l = uf_element(width_src, x - 1, y);
				uf_connect(uf_data, idx, idx_l);
			}
			if (y > 0) {
				// Up
				if (should_merge_pixels(v, v_0_m1)) {
					uf_element_t idx_up = uf_element(width_src, x, y - 1);
					uf_connect(uf_data, idx, idx_up);
				}
				if (v == 255) {
					// Up-left
					if (x > 0) {
						if (should_merge_pixels(v, v_m1_m1)) {
							uf_element_t idx_ul = uf_element(width_src, x - 1, y - 1);
							uf_connect(uf_data, idx, idx_ul);
						}
					}
					// Up-right
					if (should_merge_pixels(v, v_1_m1)) {
						uf_element_t idx_ur = uf_element(width_src, x + 1, y - 1);
						uf_connect(uf_data, idx, idx_ur);
					}
				}
			}
			// smooth_uf(uf_data, idx);
			// if (y > 0)
			// smooth_uf(uf_data, uf_element(uf_data, (x + width_src / 2) % width_src, y - 1));
		}
	}
}

__kernel void k04_print_uf(
	__private const uint width_src,
	__global uint *uf,
	__private uint stride_dst,
	__global uchar *dst,
	__private uint stride_dst2,
	__global uchar *dst2
) {
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	const uint element = uf_element(width_src, x, y);

	uint rep0 = uf_representative(uf, element);
	uint parent = *uf_parent(uf, element);
	uint size = *uf_size(uf, element);
	uint c = clamp(rep0 % 255, (uint) 0, (uint) 255);
	dst[y * stride_dst + x] = (uchar) c;
	dst2[y * stride_dst + x] = size == 0 ? 0 : 255;
}