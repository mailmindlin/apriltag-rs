// Step 1: Quad decimate — integer factor (texture-based variant)
//
// Downscales the image by an integer factor (2, 3, 4, ...) using
// nearest-neighbor sampling (takes the top-left pixel of each block).
// This is the general-case path for ffactor != 1.5.
//
// $factor is substituted at shader creation time.

const factor: u32 = $factor;

@group(0) @binding(0) var img_src : texture_2d<u32>;                   // Source grayscale image
@group(0) @binding(1) var img_dst : texture_storage_2d<r8uint, write>; // Decimated output

// One invocation per output pixel.
@compute
@workgroup_size($wg_width, $wg_height)
fn main(@builtin(global_invocation_id) GlobalId: vec3<u32>) {
	// Map output pixel back to source coordinates
	let base = GlobalId.xy * vec2(factor);

	let p0 = textureLoad(img_src, base, 0);

	textureStore(img_dst, GlobalId.xy, p0);
}
