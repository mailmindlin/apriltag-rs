const PIX_HIG: u32 = 255u;
const PIX_INV: u32 = 127u;
const PIX_LOW: u32 = 0u;

const W_a: u32 = 0u;
const W_b: u32 = 1u;
const W_c: u32 = 2u;
const W_d: u32 = 3u;
const BIT_P: u32 = 4u;
const BIT_Q: u32 = 5u;
const BIT_R: u32 = 6u;
const BIT_S: u32 = 7u;
const B_a: u32 = 8u;
const B_b: u32 = 9u;
const B_c: u32 = 10u;
const B_d: u32 = 10u;
const B_Qc: u32 = 11u;
const B_Qd: u32 = 12u;
const B_Sb: u32 = 13u;
const B_Sd: u32 = 14u;

struct Params {
	uf_stride: u32,
}
@group(0) @binding(0) var<uniform> params : Params;

struct Labels {
	data: array<u32>,
}
struct LabelsAtomic {
	data: array<atomic<u32>>,
}

fn Bit(offset: u32) -> u32 {
	return (1u << offset);
}

fn HasBit(value: u32, offset: u32) -> bool {
	return (value & Bit(offset)) != 0u;
}

// Apply a signed offset to a vec2u
fn vec2u_offset(base: vec2u, offset: vec2i) -> vec2u {
	return vec2u(vec2i(base) + offset);
}

// Get the unionfind elementId for a position
fn uf_element(pos: vec2u) -> u32 {
	return pos.y * params.uf_stride + pos.x;
}

// Get the unionfind parent index for an elementId
fn uf_parent_idx(elem: u32) -> u32 {
	return elem * 2u;
}

// Get the direct unionfind parent for an elementId
fn uf_parent(elem: u32) -> u32 {
    return labels.data[uf_parent_idx(elem)];
}

fn uf_parent_set(elem: u32, value: u32) {
    labels.data[uf_parent_idx(elem)] = value;
}

fn uf_size(elem: u32) -> u32 {
	return labels.data[uf_parent_idx(elem) + 1u];
}

fn uf_size_set(elem: u32, size: u32) {
	labels.data[uf_parent_idx(elem) + 1u] = size;
}

fn uf_find(elem: u32) -> u32 {
    var current = elem;
    for (var i = 0u; i < 65536u; i++) {
        let parent = uf_parent(current);
        if current == parent {
            break;
        }
        current = parent;
    }
    return current;
}

fn pixel(pos: vec2u) -> u32 {
	return textureLoad(img_src, pos, 0).r;
}

fn pixel_high(pos: vec2u, offset: vec2i) -> bool {
	return pixel(vec2u_offset(pos, offset)) == PIX_HIG;
}

fn pixel_low(pos: vec2u, offset: vec2i) -> bool {
	return pixel(vec2u_offset(pos, offset)) == PIX_LOW;
}

fn pixel2(pos: vec2u) -> vec2u {
	return vec2u(
		pixel(pos),
		pixel(pos + vec2u(1u, 0u))
	);
}

fn pixel4(coords: vec2u) -> array<u32, 4> {
	let dims = textureDimensions(img_src);
	var buffer = array(0u, 0u, 0u, 0u);

	// Read pairs of consecutive values in memory at once
	if (coords.x + 1u < dims.x) {
		// This does not depend on endianness
		let chunk0 = pixel2(coords);
		buffer[0] = chunk0.x;
		buffer[1] = chunk0.y;
		
		if (coords.y + 1u < dims.y) {
			let chunk1 = pixel2(coords);
			buffer[2] = chunk1.x;
			buffer[3] = chunk1.y;
		}
	} else {
		buffer[0] = pixel(coords);

		if (coords.y + 1u < dims.y) {
			buffer[2] = pixel(coords + vec2u(0u, 1u));
		}
	}
	return buffer;
}

struct LabelData {
    white_father: u32,
    info: u32,
    // bf_a: u32,
    // bf_b: u32,
    // bf_c: u32,
    // bf_d: u32,
}


fn last_pixel_idx(coords: vec2u, dims: vec2u) -> u32 {
	var lpo = dims - vec2u(2u, 2u);
	if (coords.x + 1u < dims.x) {
		lpo = coords + vec2(1u, 0u);
	} else if (coords.y + 1u < dims.y) {
		lpo = coords + vec2(0u, 1u);
	}
	return uf_parent_idx(uf_element(lpo));
}

fn get_ld_info(coords: vec2u, dims: vec2u) -> u32 {
	let elem_c = uf_element(coords + vec2(0u, 1u));
	return uf_parent(elem_c);
}

fn set_ld(coords: vec2u, dims: vec2u, data: LabelData) {
	let elem_a = uf_element(coords);
	// let elem_b = uf_element(coords + vec2(1u, 0u));
	let elem_c = uf_element(coords + vec2(0u, 1u));
	// let elem_d = uf_element(coords + vec2(1u, 1u));

	// Info -> parent(c)
	// This works becaus we allocate the UF to have even rows
	uf_parent_set(elem_c, data.info);

	// Always store white_father in parent(a) for easy merging
	uf_parent_set(elem_a, data.white_father + 1u);

	// Store black_father(x) in size(x) because we're not using that field right now
	// if HasBit(data.info, B_a) {
	// 	labels.data[uf_parent_idx(elem_a) + 1u] = data.bf_a;
	// }
	// if HasBit(data.info, B_b) {
	// 	labels.data[uf_parent_idx(elem_b) + 1u] = data.bf_b;
	// }
	// if HasBit(data.info, B_c) {
	// 	labels.data[uf_parent_idx(elem_b) + 1u] = data.bf_c;
	// }
	// if HasBit(data.info, B_d) {
	// 	labels.data[uf_parent_idx(elem_b) + 1u] = data.bf_d;
	// }
}

fn get_ld(coords: vec2u, dims: vec2u) -> LabelData {
	let elem_a = uf_element(coords);
	// let elem_b = uf_element(coords + vec2(1u, 0u));
	let elem_c = uf_element(coords + vec2(0u, 1u));
	// let elem_d = uf_element(coords + vec2(1u, 1u));
	let info = uf_parent(elem_c);

	let wf = uf_parent(elem_a);
	// let bf_a = 0u;//uf_size(elem_a);
	// let bf_b = 0u;//uf_size(elem_b);
	// let bf_c = 0u;//uf_size(elem_c);
	// let bf_d = 0u;//uf_size(elem_d);

	return LabelData(wf, info,
		// bf_a, bf_b, bf_c, bf_d
	);
}