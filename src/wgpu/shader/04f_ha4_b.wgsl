@group(0) @binding(1) var<storage, read_write> labels : LabelsAtomic;
@group(0) @binding(2) var img_src : texture_2d<u32>;

const WARP_SIZE: u32 = 32;
const BLOCK_H: u32 = 4;

fn ha4_pixel(ignore: bool, pos: vec2u) -> bool {
	return select(pixel_high(pos, vec2(0)), false, ignore);
}
fn ha4_parent(elem: u32) -> u32 {
	return uf_size(elem) - 1u;
}
fn ha4_parent_set(elem: u32, parent: u32) {
	uf_size_set(elem, parent + 1u);
}
fn ha4_parent_idx(elem: u32) -> u32 {
	return uf_parent_idx(elem) + 1u;
}

fn label_adjust(cur: u32, other: u32) -> u32 {
	var current = cur;
	while current != other {
		let parent = atomicLoad(&labels.data[ha4_parent_idx(current)]);
		if current == parent {
			break;
		}
		current = parent;
	}
	return current;
}
fn merge(l1: u32, l2: u32) {
	var label_1 = label_adjust(l1, l2);
	var label_2 = label_adjust(l2, label_1);

	while (label_1 != label_2) {
		// Swap so label_2 <= label_1
		if (label_1 < label_2) {
			let tmp = label_1;
			label_1 = label_2;
			label_2 = tmp;
		}
		let actual = atomicMin(&labels.data[ha4_parent_idx(label_1)], label_2 + 1u) - 1u;
		if (label_1 == actual) {
			label_1 = label_2;
		} else {
			label_1 = actual;
		}
	}
}


fn start_distance(pixels: u32, tx: u32) -> u32 {
	return countLeadingZeros(~(pixels << (32u - tx)));
}

fn start_distance4(pixels: vec4<u32>, tx: u32) -> vec4<u32> {
	let shift = 32u - tx;
	let shifted = pixels << shift;
	return countLeadingZeros(~shifted);
}

fn wg_mask(wg_start: u32, width: u32) -> u32 {
	if wg_start > width {
		return 0u;
	}
	let cols = width - wg_start;
	var mask = 0xFFFFFFFFu;
	if cols < 32u {
		mask >>= 32u - cols;
	}
	return mask;
}


/** Bitmask of pixels for each row */
var<workgroup> wg_pixels: array<atomic<u32>, BLOCK_H>;

fn wg_pixels_set(warp_id: u32, ballot_mask: u32, pred: bool) {
	if pred {
		atomicOr(&wg_pixels[warp_id], ballot_mask);
	} else {
		// Initialized to zero, so we don't need to clear bits
		atomicAnd(&wg_pixels[warp_id], ~ballot_mask);
	}
}

fn pixels_rows(local_y: u32, mask: u32) -> vec4<u32> {
	let y0 = atomicLoad(&wg_pixels[0u]);
	let y1 = atomicLoad(&wg_pixels[1u]);
	let y2 = atomicLoad(&wg_pixels[2u]);
	let y3 = atomicLoad(&wg_pixels[3u]);
	// var y1 = 0u;
	// var y2 = 0u;
	// var y3 = 0u;
	// if local_y >= 1u {
	// 	y1 = atomicLoad(&wg_pixels[local_y - 1u]);
	// }
	// if local_y >= 2u {
	// 	y2 = atomicLoad(&wg_pixels[local_y - 2u]);
	// }
	// if local_y >= 3u {
	// 	y3 = atomicLoad(&wg_pixels[local_y - 3u]);
	// }
	return vec4(y0, y1, y2, y3) & mask;
}

// Label 
@compute
@workgroup_size(32, 4)
fn k04_ha4_StripLabel(
	@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
	@builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
	let tid = local_invocation_id.x;
	let wid = local_invocation_id.y;

	let dims = textureDimensions(img_src);
	let coords = vec2u(local_invocation_id.x, global_invocation_id.y);

	// Only used by Thread0
	var distance = vec4u(0u);

	for (var i = 0u; i < dims.x; i += WARP_SIZE) {
		let pos = coords + vec2u(i, 0u);
		// Do we ignore this loop?
		let ignore = !all(pos < dims);

		// Which lanes are enabled?
		let mask = wg_mask(i, dims.x);

		// Load current pixel
		let pix_c = ha4_pixel(ignore, pos);

		// Set the pixels in this warp
		let ballot_mask = Bit(tid);
		if pix_c {
			atomicOr(&wg_pixels[wid], ballot_mask);
		} else {
			atomicAnd(&wg_pixels[wid], ~ballot_mask);
		}
		workgroupBarrier();
		let pixels = pixels_rows(wid, mask);
		var sdist = start_distance4(pixels, tid);

		//Debug
		let labels_index = uf_element(pos);
		if (!ignore) {
			uf_parent_set(uf_element(pos), select(0u, 1u, pix_c));
		}

		// Vector of pixels above current pixel
		let pix = (pixels & ballot_mask) != 0u;

		// Set the rising edge as the start of a group
		if pix[wid] && (sdist[wid] == 0u) {
			ha4_parent_set(labels_index, labels_index - distance[wid]);
		}

		workgroupBarrier();

		// Get pixels bitset from row up
		let pixels_u = pixels_row(local_invocation_id.y, 1u, mask);

		if (local_invocation_id.x == 0u) {
			sdist = distance;

			let d = start_distance4(pixels, 32u);
			distance = d + select(vec4(0u), distance, d == 32u);
		}

		// Merge up
		if pix_c {
			var target_row = wid;
			while target_row > 0u && pix[target_row - 1u] {
				target_row -= 1u;
			}
			if sdist[wid] == 0u || sdist[target_row] == 0u {
				let label_c = labels_index;
			}
			
		}
		if (pix_c && pix_u && (sdist_c == 0u || sdist_u == 0u)) {
			let label_c = uf_element(pos) - sdist_c;
			let label_u = uf_element(pos - vec2u(0u, 1u)) - sdist_u;
			merge(label_c, label_u);
		}
	}
}

var<workgroup> wg_pixels_up: array<atomic<u32>, BLOCK_H>;

@compute
@workgroup_size(32, 4)
fn k04_ha4_StripMerge(
	@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>,
	@builtin(local_invocation_id) local_invocation_id: vec3<u32>
) {
	let coords = global_invocation_id.xy * vec2u(1u, BLOCK_H);
	let dims = textureDimensions(img_src);

	let mask = wg_mask(workgroup_id.x * WARP_SIZE, dims.x);
	let ignore = !(all(coords < dims) && coords.y > 0u);

	let pix_c = ha4_pixel(ignore, coords);
	let pix_u = ha4_pixel(ignore, coords - vec2u(0u, 1u));

	let shared_c = &wg_pixels[local_invocation_id.y];
	let shared_u = &wg_pixels_up[local_invocation_id.y];

	// Equivalent to CUDA __ballot_sync x2 (hopefully)
	let ballot_mask = Bit(local_invocation_id.x);
	if !ignore {
		if pix_c {
			atomicOr(shared_c, ballot_mask);
		} else {
			atomicAnd(shared_c, ~ballot_mask);
		}
		if pix_u {
			atomicOr(shared_u, ballot_mask);
		} else {
			atomicAnd(shared_u, ~ballot_mask);
		}
	}

	workgroupBarrier();

	if !ignore {
		let pixels_c = atomicLoad(shared_c) & mask;
		let pixels_u = atomicLoad(shared_u) & mask;

		if (pix_c && pix_u) {
			let sdist_c = start_distance(pixels_c, local_invocation_id.x);
			let sdist_u = start_distance(pixels_u, local_invocation_id.x);
			if (sdist_c == 0u || sdist_u == 0u) {
				let labels_index_c = uf_element(coords);
				let labels_index_u = uf_element(coords - vec2u(0u, 1u));
				merge(labels_index_c - sdist_c, labels_index_u - sdist_u);
			}
		}
	}
}

const WG_PIXELS: u32 = WARP_SIZE * BLOCK_H;
var<workgroup> wg_shared_pixels: array<u32, WG_PIXELS>;

@compute
@workgroup_size(32, 4)
fn k04_ha4_Relabeling(
	@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>,
	@builtin(local_invocation_id) local_invocation_id: vec3<u32>,
	@builtin(local_invocation_index) local_invocation_index: u32
) {
	let coords = global_invocation_id.xy;
	let dims = textureDimensions(img_src);

	// if local_invocation_index == 0u {
	// 	atomicStore(&wg_pixels, 0u);
	// }
	// workgroupBarrier();

	let ignore = !all(coords < dims);
	
	// __ballot_sync
	let mask = wg_mask(workgroup_id.x * WARP_SIZE, dims.x);
	let pix_c = ha4_pixel(ignore, coords);
	let ballot_mask = Bit(local_invocation_id.x);
	if pix_c {
		atomicOr(&wg_pixels[local_invocation_id.y], ballot_mask);
	} else {
		atomicAnd(&wg_pixels[local_invocation_id.y], ~ballot_mask);
	}
	workgroupBarrier();


	let labels_index = uf_element(coords);
	var sdist_c = 0u;
	if !ignore {
		let pixels_c = atomicLoad(&wg_pixels[local_invocation_id.y]) & mask;
		sdist_c = start_distance(pixels_c, local_invocation_id.x);
		var label = 0u;

		if pix_c && sdist_c == 0u {
			label = ha4_parent(labels_index);
			loop {
				let parent = ha4_parent(label);
				if label == parent {
					break;
				}
				label = parent;
			}
		}
		wg_shared_pixels[local_invocation_id.y + local_invocation_id.x * BLOCK_H] = label;
	}

	workgroupBarrier();

	if !ignore {
		let label = wg_shared_pixels[local_invocation_id.y + (local_invocation_id.x - sdist_c) * BLOCK_H];

		if pix_c {
			ha4_parent_set(labels_index, label);
		}
	}
}