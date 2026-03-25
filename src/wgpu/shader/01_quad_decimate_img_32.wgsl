// Step 1: Quad decimate — 1.5x downscale (texture-based variant)
//
// Downscales the image by factor 1.5 (i.e. 3x3 input -> 2x2 output) using
// a weighted average that gives the center pixel more influence than its
// neighbors. This is the special-case path for ffactor == 1.5.
//
// Weighting scheme (for each 2x2 output quad within a 3x3 input block):
//   corner pixel (a):  weight 4  (closest to output pixel)
//   edge pixels (b,c): weight 2  (adjacent)
//   diagonal pixel (d): weight 1 (furthest)
//   Total = 4 + 2 + 2 + 1 = 9

@group(0) @binding(0) var img_src : texture_2d<u32>;                   // Source grayscale image
@group(0) @binding(1) var img_dst : texture_storage_2d<r8uint, write>; // Decimated output (2/3 size)

/// Weighted downsample of a 2x2 region within a 3x3 block.
/// a = nearest corner, b/c = adjacent edges, d = diagonal center.
fn downsample_32(a: u32, b: u32, c: u32, d: u32) -> u32 {
	return min((4u * a + 2u * b + 2u * c + d) / 9u, 255u);
}

// One invocation per 3x3 input block, producing a 2x2 output block.
@compute
@workgroup_size($wg_width, $wg_height)
fn main(@builtin(global_invocation_id) GlobalId: vec3<u32>) {
	let dims = textureDimensions(img_src);

	// Map invocation to the 3x3 input block
	let base = GlobalId.xy * vec2u(3u, 3u);

	// Corresponding 2x2 output block
	var dst_base = GlobalId.xy * vec2u(2u, 2u);

	if (any(dst_base + vec2u(1u, 1u) >= textureDimensions(img_dst))) {
		return;
	}

	// Load all 9 pixels of the 3x3 input block:
	//   a b c
	//   d e f
	//   g h i
	var a = textureLoad(img_src, base + vec2(0u, 0u), 0).r;
	var b = textureLoad(img_src, base + vec2(1u, 0u), 0).r;
	var c = textureLoad(img_src, base + vec2(2u, 0u), 0).r;

	var d = textureLoad(img_src, base + vec2(0u, 1u), 0).r;
	var e = textureLoad(img_src, base + vec2(1u, 1u), 0).r;
	var f = textureLoad(img_src, base + vec2(2u, 1u), 0).r;

	var g = textureLoad(img_src, base + vec2(0u, 2u), 0).r;
	var h = textureLoad(img_src, base + vec2(1u, 2u), 0).r;
	var i = textureLoad(img_src, base + vec2(2u, 2u), 0).r;

	// Each output pixel is a weighted average of its nearest 2x2 sub-block,
	// with the center pixel (e) always contributing as the diagonal weight.
	textureStore(img_dst, dst_base + vec2u(0u, 0u), vec4u(downsample_32(a, b, d, e))); // top-left
	textureStore(img_dst, dst_base + vec2u(1u, 0u), vec4u(downsample_32(c, b, f, e))); // top-right
	textureStore(img_dst, dst_base + vec2u(0u, 1u), vec4u(downsample_32(g, h, d, e))); // bottom-left
	textureStore(img_dst, dst_base + vec2u(1u, 1u), vec4u(downsample_32(i, h, f, e))); // bottom-right
}