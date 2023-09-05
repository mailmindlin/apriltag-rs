static uchar downsample_32(uchar a, uchar b, uchar c, uchar d) {
	return (4 * a + 2 * b + 2 * c + d) / 9;
}

@id(0) override stride_src: u32;
@id(1) override stride_dst: u32;
@id(2) override factor: u32 = 0;
@group(0) @binding(0) var<storage, read> img_src : array<u32>; // Packed u8x4
@group(0) @binding(1) var<storage, read_write> img_dst : array<u32>; // Packed u8x4

fn get_pixel(x: u32, y: u32) -> u32 {
	let idx = (y * stride_src) + (x / 4);
	let value = img_src[idx];
	return extractBits(value, (x % 4) * 8, 8);
}

fn downsample_32(a: u32, b: u32, c: u32, d: u32) -> u32 {
	return min((4 * a + 2 * b + 2 * c + d) / 9, 255);
}

// Maps 3x4 to 2x3
fn qd32_chunk(col1: vec3u, col2: vec3u, col3: vec3u) -> vec2u {
	return vec2(
		downsample_32(col1.x, col2.x, col1.y, col2.y) | (downsample_32(col3.x, col2.x, col3.y, col2.y) << 8),
		downsample_32(col1.z, col2.z, col1.y, col2.y) | (downsample_32(col3.z, col2.z, col3.y, col2.y) << 8),
	)
}

@compute
@workgroup_size(64)
fn k01_filter_quad_decimate_32(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let x = global_invocation_id.x; // 0..width(dst)/4
    let y = global_invocation_id.y; // 0..height(dst)/4

	// We process a 3x12 size input
	let chunk0 = vec3(
		img_src[((y * 3 + 0) * stride_src) + (x * 3 + 0)],
		img_src[((y * 3 + 1) * stride_src) + (x * 3 + 0)],
		img_src[((y * 3 + 2) * stride_src) + (x * 3 + 0)],
	);
	let dst0 = qd32_chunk(
		extractBits(chunk0,  0, 8), // Column 0
		extractBits(chunk0,  8, 8), // Column 1
		extractBits(chunk0, 16, 8), // Column 2
	);

	let chunk1 = vec3(
		img_src[((y * 3 + 0) * stride_src) + (x * 3 + 1)],
		img_src[((y * 3 + 1) * stride_src) + (x * 3 + 1)],
		img_src[((y * 3 + 2) * stride_src) + (x * 3 + 1)],
	);
	let dst1 = qd32_chunk(
		extractBits(chunk0, 24, 8), // Column 3
		extractBits(chunk1,  0, 8), // Column 4
		extractBits(chunk1,  8, 8), // Column 5
	);
	// Write dst01
	let dst01 = dst0 | (dst1 << 16);
	img_dst[((y * 2 + 0) * stride_dst) + (x * 2 + 0)] = dst01.x;
	img_dst[((y * 2 + 1) * stride_dst) + (x * 2 + 0)] = dst01.y;

	let chunk2 = vec3(
		img_src[((y * 3 + 0) * stride_src) + (x * 3 + 2)],
		img_src[((y * 3 + 1) * stride_src) + (x * 3 + 2)],
		img_src[((y * 3 + 2) * stride_src) + (x * 3 + 2)],
	);
	let dst2 = qd32_chunk(
		extractBits(chunk1, 16, 8), // Column 6
		extractBits(chunk1, 24, 8), // Column 7
		extractBits(chunk2,  0, 8), // Column 8
	);
	let dst3 = qd32_chunk(
		extractBits(chunk2,  8, 8), // Column 9
		extractBits(chunk2, 16, 8), // Column 10
		extractBits(chunk2, 24, 8), // Column 11
	);

	// Write dst23
	let dst23 = dst2 | (dst3 << 16);
	img_dst[((y * 2 + 0) * stride_dst) + (x * 2 + 0)] = dst23.x;
	img_dst[((y * 2 + 1) * stride_dst) + (x * 2 + 0)] = dst23.y;
}

// quad_decimate but for ffactor != 1.5
@compute
@workgroup_size(64)
fn k01_filter_quad_decimate(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let x = global_invocation_id.x;
	let y = global_invocation_id.y;

	let p0 = get_pixel((x + 0) * factor, y * factor);
	let p1 = get_pixel((x + 1) * factor, y * factor);
	let p2 = get_pixel((x + 2) * factor, y * factor);
	let p3 = get_pixel((x + 3) * factor, y * factor);

	let d = (p0 << 24) | (p1 << 16) | (p2 << 8) | (p3 << 0);
	dst[]

	size_t src_idx = (y * factor * src_stride) + (x * factor);
	size_t dst_idx = (y * dst_stride) + x;
	dst[dst_idx] = src[src_idx];
	//TODO: replace with memcpy?
}
