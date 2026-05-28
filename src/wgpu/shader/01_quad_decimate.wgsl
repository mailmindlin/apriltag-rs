// Step 1: Quad decimate (buffer-based variant)
//
// Downscales the image using packed u8x4 storage buffers. Contains two
// entry points:
//
//   k01_filter_quad_decimate_32 — Optimized 1.5x downscale (3x3 -> 2x2)
//     that processes 12 columns at a time in packed u32 words.
//
//   k01_filter_quad_decimate — General integer-factor downscale using
//     nearest-neighbor sampling, packing 4 output pixels per u32.

struct Params {
	stride_src: u32,  // Row stride of source image (in u32 words = 4 pixels each)
	stride_dst: u32,  // Row stride of destination image (in u32 words)
	width_dst: u32,   // Destination width (in u32 words)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> img_src : array<u32>;  // Packed u8x4 source
@group(0) @binding(2) var<storage, write> img_dst : array<u32>; // Packed u8x4 destination

/// Extract one 8-bit lane from each element of a vec3 of packed u32s.
/// Returns a vec3 of the unpacked values at the given lane index.
fn unpack_pixel_x3(packed: vec3u, off_x: u32) -> vec3u {
	return extractBits(packed, off_x * 8u, 8u);
}

/// Weighted downsample for 1.5x decimation.
/// Weights: a=4, b=2, c=2, d=1, total=9.
fn downsample_32(a: u32, b: u32, c: u32, d: u32) -> u32 {
	return min((4u * a + 2u * b + 2u * c + d) / 9u, 255u);
}

/// Process 3 columns (from 3 rows) into 2 output pixels (packed into a vec2).
/// Input: 3 columns x 3 rows -> output: 2 pixels x 2 rows.
/// The .x component holds the top row pair, .y holds the bottom row pair.
fn qd32_chunk(col1: vec3u, col2: vec3u, col3: vec3u) -> vec2u {
	return vec2(
		downsample_32(col1.x, col2.x, col1.y, col2.y) | (downsample_32(col3.x, col2.x, col3.y, col2.y) << 8u),
		downsample_32(col1.z, col2.z, col1.y, col2.y) | (downsample_32(col3.z, col2.z, col3.y, col2.y) << 8u),
	);
}

/// Write a packed u32 to the destination buffer at (id * 2 + offset).
fn write_dst_32(id: vec3u, dx: u32, dy: u32, val: u32) {
	let x = id.x * 2u + dx;
	let y = id.y * 2u + dy;
	let idx = y * params.stride_dst + x;
	img_dst[idx] = val;
}

/// 1.5x downscale (3:2 ratio) operating on packed u32 words.
/// Each invocation processes a 3-row x 12-column input region into
/// a 2-row x 8-column output region (i.e., 2 packed u32 output words).
@compute
@workgroup_size(1, 1)
fn k01_filter_quad_decimate_32(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let x = global_invocation_id.x; // 0..width(dst)/4
    let y = global_invocation_id.y; // 0..height(dst)/4

	if (x * 2u >= params.width_dst) {
		return;
	}

	// Row base addresses for 3 input rows
	let y0 = (y * 3u + 0u) * params.stride_src;
	let y1 = (y * 3u + 1u) * params.stride_src;
	let y2 = (y * 3u + 2u) * params.stride_src;

	// Process columns 0-2 (first packed u32 from each row)
	let chunk0 = vec3(
		img_src[y0 + (x * 3u + 0u)],
		img_src[y1 + (x * 3u + 0u)],
		img_src[y2 + (x * 3u + 0u)],
	);
	let dst0 = qd32_chunk(
		unpack_pixel_x3(chunk0, 0u), // Column 0
		unpack_pixel_x3(chunk0, 1u), // Column 1
		unpack_pixel_x3(chunk0, 2u), // Column 2
	);

	// Process columns 3-5 (straddling chunk0 lane 3 and chunk1 lanes 0-1)
	let chunk1 = vec3(
		img_src[y0 + (x * 3u + 1u)],
		img_src[y1 + (x * 3u + 1u)],
		img_src[y2 + (x * 3u + 1u)],
	);
	let dst1 = qd32_chunk(
		unpack_pixel_x3(chunk0, 3u), // Column 3
		unpack_pixel_x3(chunk1, 0u), // Column 4
		unpack_pixel_x3(chunk1, 1u), // Column 5
	);

	// Pack and write first 2 output words (4 output pixels per row, 2 rows)
	let dst01 = (dst1 << 16u) | (dst0 << 0u);
	write_dst_32(global_invocation_id, 0u, 0u, dst01.x);
	write_dst_32(global_invocation_id, 0u, 1u, dst01.y);

	if (x * 2u + 1u >= params.width_dst) {
		return;
	}

	// Process columns 6-11 for the second pair of output words
	let chunk2 = vec3(
		img_src[y0 + (x * 3u + 2u)],
		img_src[y1 + (x * 3u + 2u)],
		img_src[y2 + (x * 3u + 2u)],
	);
	let dst2 = qd32_chunk(
		unpack_pixel_x3(chunk1, 2u), // Column 6
		unpack_pixel_x3(chunk1, 3u), // Column 7
		unpack_pixel_x3(chunk2, 0u), // Column 8
	);
	let dst3 = qd32_chunk(
		unpack_pixel_x3(chunk2, 1u), // Column 9
		unpack_pixel_x3(chunk2, 2u), // Column 10
		unpack_pixel_x3(chunk2, 3u), // Column 11
	);

	let dst23 = dst2 | (dst3 << 16u);
	write_dst_32(global_invocation_id, 1u, 0u, dst23.x);
	write_dst_32(global_invocation_id, 1u, 1u, dst23.y);
}


/// Extract a single 8-bit pixel from a packed u32
fn unpack_pixel(packed: u32, off_x: u32) -> u32 {
	return extractBits(packed, (off_x % 4u) * 8u, 8u);
}

/// Read a single pixel from the packed source buffer
fn get_pixel(x: u32, y: u32) -> u32 {
	let idx = (y * params.stride_src) + (x / 4u);
	let value = img_src[idx];
	return unpack_pixel(value, x);
}

/// General integer-factor downscale using nearest-neighbor sampling.
/// Each invocation reads 4 source pixels (spaced by `factor`) and packs
/// them into a single u32 output word.
@compute
@workgroup_size(1, 1, 1)
fn k01_filter_quad_decimate(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
	let x = global_invocation_id.x;
	let y = global_invocation_id.y;
	if (x >= params.width_dst) {
		return;
	}

	// Sample 4 pixels at `factor` spacing and pack into one u32
	let p0 = get_pixel((x * 4u + 0u) * factor, y * factor);
	let p1 = get_pixel((x * 4u + 1u) * factor, y * factor);
	let p2 = get_pixel((x * 4u + 2u) * factor, y * factor);
	let p3 = get_pixel((x * 4u + 3u) * factor, y * factor);

	var d: u32;
	d = (p3 << 24u) | (p2 << 16u) | (p1 << 8u) | (p0 << 0u);

	let dst_idx = (y * params.stride_dst) + x;
	img_dst[dst_idx] = d;
}
