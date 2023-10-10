@group(0) @binding(0) var img_src : texture_2d<u32>;
@group(0) @binding(1) var img_dst : texture_storage_2d<r8uint, write>;

fn downsample_32(a: u32, b: u32, c: u32, d: u32) -> u32 {
	return min((4u * a + 2u * b + 2u * c + d) / 9u, 255u);
}

@compute
@workgroup_size($wg_width, $wg_height)
fn main(@builtin(global_invocation_id) GlobalId: vec3<u32>) {
	let dims = textureDimensions(img_src);
	
	// Input is 3x3
	let base = GlobalId.xy * vec2u(3u, 3u);

	// output is 2x2
	var dst_base = GlobalId.xy * vec2u(2u, 2u);

	if (any(dst_base + vec2u(1u, 1u) >= textureDimensions(img_dst))) {
		return;
	}

	var a = textureLoad(img_src, base + vec2(0u, 0u), 0).r;
	var b = textureLoad(img_src, base + vec2(1u, 0u), 0).r;
	var c = textureLoad(img_src, base + vec2(2u, 0u), 0).r;

	var d = textureLoad(img_src, base + vec2(0u, 1u), 0).r;
	var e = textureLoad(img_src, base + vec2(1u, 1u), 0).r;
	var f = textureLoad(img_src, base + vec2(2u, 1u), 0).r;

	var g = textureLoad(img_src, base + vec2(0u, 2u), 0).r;
	var h = textureLoad(img_src, base + vec2(1u, 2u), 0).r;
	var i = textureLoad(img_src, base + vec2(2u, 2u), 0).r;

	textureStore(img_dst, dst_base + vec2u(0u, 0u), vec4u(downsample_32(a, b, d, e)));
	textureStore(img_dst, dst_base + vec2u(1u, 0u), vec4u(downsample_32(c, b, f, e)));
	textureStore(img_dst, dst_base + vec2u(0u, 1u), vec4u(downsample_32(g, h, d, e)));
	textureStore(img_dst, dst_base + vec2u(1u, 1u), vec4u(downsample_32(i, h, f, e)));
}