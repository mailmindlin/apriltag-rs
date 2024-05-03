@group(0) @binding(1) var<storage, read_write> labels : LabelsAtomic;
@group(0) @binding(2) var img_src : texture_2d<u32>;

const WARP_SIZE: u32 = $WARP_SIZE;
const BLOCK_H: u32 = $BLOCK_H;

fn ha4_pixel(ignore: bool, pos: vec2u) -> bool {
	return select(pixel_low(pos, vec2(0)), false, ignore);
}
fn ha4_parent(elem: u32) -> u32 {
	let siz = uf_size(elem);
	if siz == 0u {
		return elem;
	} else {
		return siz - 1u;
	}
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
		if parent == 0u {
			break;
		}
		if current == parent - 1u {
			break;
		}
		current = parent - 1u;
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
	let shifted = pixels << vec4<u32>(shift);
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

fn pixels_row(local_y: u32, offset_up: u32, mask: u32) -> u32 {
	if offset_up > local_y {
		// Underflow
		return 0u;
	}
	return atomicLoad(&wg_pixels[local_y - offset_up]) & mask;
}

// Label 
@compute
@workgroup_size($WARP_SIZE, $BLOCK_H)
fn k04_ha4_StripLabel(
	@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
	@builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
	let dims = textureDimensions(img_src);
	let coords = vec2u(local_invocation_id.x, global_invocation_id.y);

	var distance_c = 0u;
	var distance_u = 0u;

	for (var i = 0u; i < dims.x; i += WARP_SIZE) {
		let pos = coords + vec2u(i, 0u);
		// Do we ignore this loop?
		let ignore = !all(pos < dims);

		// Which lanes are enabled?
		let mask = wg_mask(i, dims.x);

		// Load current pixel
		let pix_c = ha4_pixel(ignore, pos);

		// Set the pixels in this warp
		let ballot_mask = Bit(local_invocation_id.x);
		if pix_c {
			atomicOr(&wg_pixels[local_invocation_id.y], ballot_mask);
		} else {
			// Initialized to zero, so we don't need to clear bits
			atomicAnd(&wg_pixels[local_invocation_id.y], ~ballot_mask);
		}
		workgroupBarrier();
		let pixels_c = pixels_row(local_invocation_id.y, 0u, mask);
		var sdist_c = start_distance(pixels_c, local_invocation_id.x);

		// Set the rising edge as the start of a group
		if pix_c && (sdist_c == 0u) {
			let labels_index = uf_element(pos);
			var parent_idx = labels_index - distance_c;
			ha4_parent_set(labels_index, parent_idx);
		}

		workgroupBarrier();

		// Get pixels bitset from row up
		let pixels_u = pixels_row(local_invocation_id.y, 1u, mask);
		let pix_u = HasBit(pixels_u, local_invocation_id.x);
		var sdist_u = 0u;

		if (local_invocation_id.x == 0u) {
			sdist_c = distance_c;
			sdist_u = distance_u;

			let d_u = start_distance(pixels_u, 32u);
			distance_u = d_u + select(0u, distance_u, d_u == 32u);
			let d_c = start_distance(pixels_c, 32u);
			distance_c = d_c + select(0u, distance_c, d_c == 32u);
		} else {
			sdist_u = start_distance(pixels_u, local_invocation_id.x);
		}

		// Merge up
		if (pix_c && pix_u && (sdist_c == 0u || sdist_u == 0u)) {
			let label_c = uf_element(pos) - sdist_c;
			let label_u = uf_element(pos - vec2u(0u, 1u)) - sdist_u;
			merge(label_c, label_u);
		}
	}
}

var<workgroup> wg_pixels_up: array<atomic<u32>, BLOCK_H>;

@compute
@workgroup_size($WARP_SIZE, $BLOCK_H)
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

var<workgroup> wg_shared_pixels: array<array<u32, WARP_SIZE>, BLOCK_H>;

@compute
@workgroup_size($WARP_SIZE, $BLOCK_H)
fn k04_ha4_Relabeling(
	@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
	@builtin(workgroup_id) workgroup_id: vec3<u32>,
	@builtin(local_invocation_id) local_invocation_id: vec3<u32>
) {
	let coords = global_invocation_id.xy;
	let dims = textureDimensions(img_src);

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
	if (!ignore) && pix_c {
		let pixels_c = atomicLoad(&wg_pixels[local_invocation_id.y]) & mask;
		sdist_c = start_distance(pixels_c, local_invocation_id.x);

		var label = 0u;
		if sdist_c == 0u {
			label = ha4_parent(labels_index);
			loop {
				let parent = ha4_parent(label);
				if label == parent {
					break;
				}
				label = parent;
			}
		} else if sdist_c == 32u {
			label = labels_index;
		}
		wg_shared_pixels[local_invocation_id.y][local_invocation_id.x] = label;
	}

	workgroupBarrier();
	if !ignore {
		// uf_parent_set(labels_index, wg_shared_pixels[local_invocation_id.y][local_invocation_id.x]);
		// uf_parent_set(labels_index, sdist_c);
	}

	if (!ignore) && pix_c {
		let label = wg_shared_pixels[local_invocation_id.y][local_invocation_id.x - sdist_c];

		ha4_parent_set(labels_index, label);
	}
}