const factor: u32 = $factor;

@group(0) @binding(0) var img_src : texture_2d<u32>;
@group(0) @binding(1) var img_dst : texture_storage_2d<r8uint, write>;

// quad_decimate but for ffactor != 1.5
@compute
@workgroup_size($wg_width, $wg_height)
fn main(@builtin(global_invocation_id) GlobalId: vec3<u32>) {
	let base = GlobalId.xy * vec2(factor);

	let p0 = textureLoad(img_src, base, 0);

	textureStore(img_dst, GlobalId.xy, p0);
}
