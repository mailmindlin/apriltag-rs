struct Params {
	uf_width: u32,
}
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read_write> parents : array<atomic<u32>>;
@group(0) @binding(2) var img_src : texture_2d<u32>;

const INVALID_PIXEL: u32 = 127u;
const MAX_LOOPS: u32 = 100u;

/// Computes the UnionFind element id for some coordinates
fn uf_element(pos: vec2u) -> u32 {
	return pos.y * params.uf_width + pos.x;
}

@compute
@workgroup_size(16, 16)
fn k04_unionfind_init(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let dims = textureDimensions(img_src);
	if (global_invocation_id.x > dims.x || global_invocation_id.y > dims.y) {
		return;
	}
	let idx = uf_element(global_invocation_id.xy);
	parents[idx * 2u] = idx;
}

fn uf_get_parent(element: u32) -> u32 {
	return atomicLoad(&parents[element * 2u]);
}

fn uf_add_count(parent: u32) {
	atomicAdd(&parents[parent * 2u + 1u], 1u);
}

// Get UnionFind group for id
fn uf_representative(element: u32) -> u32 {
	var result = element;

	var parent = uf_get_parent(result);
	var i = 0u;
	while (parent < result) {
		result = parent;
		parent = uf_get_parent(result);
		// parent = parents[element * 2u];
		// let exchange = atomicMin(&parents[element * 2u], grandparent);
		// if (exchange == parent) {
		// 	// CMPXCHG success
		// 	element = parent;
		// 	parent = grandparent;
		// } else {
		// 	parent = exchange;
		// }
		if (i >= MAX_LOOPS) {
			break;
		}
		i++;
	}

	atomicMin(&parents[element * 2u], result);

	// var current = element;
	// while (current > result) {
	// 	current = atomicMin(&parents[current * 2u], result);
	// }
	return result;
}

fn uf_connect(a_in: u32, b_in: u32) {
	var a = a_in;
	var b = b_in;
	for (var i: u32 = 0u; i < MAX_LOOPS; i++) {
		a = uf_representative(a);
		b = uf_representative(b);

		if (a == b) {
			break;
		}

		if (a < b) {
			if (atomicMin(&parents[a * 2u], b) == a) {
				break;
			}
		} else {
			if (atomicMin(&parents[b * 2u], a) == b) {
				break;
			}
		}
		// uint a_size = ocl_atomic_load_acquire(uf_size(uf, a));
		// uint b_size = ocl_atomic_load_acquire(uf_size(uf, b));

		// if (a_size < b_size) {
		// 	if (uf_set_parent(uf, b, b, a)) {
		// 		return;
		// 	}
		// } else {
		// 	if (uf_set_parent(uf, a, a, b)) {
		// 		atomic_add(uf_size(uf, b), a_size);
		// 		return;
		// 	}
		// }
		// barrier(CLK_LOCAL_MEM_FENCE);
	}
}

fn try_connect(value: u32, coords: vec2u, delta: vec2i) {
	if (value == INVALID_PIXEL) {
		return;
	}
	let x1a = vec2i(coords) + delta;
	if (x1a.x < 0 || x1a.y < 0) {
		return;
	}
	let x1 = vec2u(x1a);

	let v1 = textureLoad(img_src, x1, 0).r;
	if (value == v1) {
		let idx0 = uf_element(coords);
		let idx1 = uf_element(x1);

		let rep0 = uf_representative(idx0);
		let rep1 = uf_representative(idx1);
		// uf_connect(idx, idx1);
		atomicMin(&parents[idx0 * 2u], rep1);
		atomicMin(&parents[idx1 * 2u], rep0);
	}
}

@compute
@workgroup_size(16, 16)
fn k04_connected_components(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let coords = global_invocation_id.xy + vec2u(1u, 0u);
	let idx = uf_element(coords);
	let dims = textureDimensions(img_src);

	let value = select(INVALID_PIXEL, textureLoad(img_src, coords, 0).r, coords.x < dims.x - 2u && coords.y < dims.y);
	// if value != INVALID_PIXEL {
	// atomicMin(&parents[idx * 2u], 0u);
	// }
	
	try_connect(value, coords, vec2i(-1, 0));
	storageBarrier();
	try_connect(value, coords, vec2i(0, -1));
	storageBarrier();
	if (value == 255u) {
		try_connect(value, coords, vec2i(-1, -1));
	}
	storageBarrier();
	if (value == 255u) {
		try_connect(value, coords, vec2i(1, -1));
	}
	storageBarrier();
	// if (value != INVALID_PIXEL) {
	// 	// Left
	// 	if (coords.x > 0u) {
	// 		try_connect(value, coords, vec2i(-1, 0));
	// 	}
	// 	if (coords.y > 0u) {
	// 		// Up
	// 		try_connect(value, coords, vec2i(0, -1));
	// 		if (value == 255u) {
	// 			// Up-left
	// 			if (coords.x > 0u) {
	// 				try_connect(value, coords, vec2i(-1, -1));
	// 			}
	// 			try_connect(value, coords, vec2i(1, -1));
	// 		}
	// 	}
	// }
}

@compute
@workgroup_size(16, 16)
fn k04_count_groups(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let coords = global_invocation_id.xy;
	let idx = uf_element(coords);
	let dims = textureDimensions(img_src);

	if coords.x < dims.x && coords.y < dims.y {
		let rep = uf_representative(idx);
		parents[idx * 2u] = rep;
		// atomicMin(&parents[idx * 2u], rep);
		uf_add_count(rep);
	}
}