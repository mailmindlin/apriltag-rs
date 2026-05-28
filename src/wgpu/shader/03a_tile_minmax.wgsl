// Step 3a: Tile min/max computation
//
// Divides the source image into TILESZ x TILESZ tiles and computes the
// minimum and maximum pixel intensity within each tile. These values are
// used by later stages (03b, 03c) for adaptive thresholding.
//
// The max value is stored inverted (255 - v_max) so that a single min()
// operation on both channels in the blur pass (03b) simultaneously
// tightens the min and widens the max across neighboring tiles.

const TILESZ = 4u;

@group(1) @binding(0) var img_src : texture_2d<u32>;                       // Source grayscale image
@group(1) @binding(1) var img_tiles0 : texture_storage_2d<rg8uint, write>; // Output: (v_min, 255-v_max) per tile

// One invocation per tile.
@compute
@workgroup_size(1,1)
fn k03_tile_minmax(@builtin(global_invocation_id) tile_id: vec3<u32>) {
    let dims = textureDimensions(img_src);
    let pixel_base = tile_id.xy * vec2u(TILESZ);

    var v_min = 255u;
    var v_max = 0u;

    // Scan every pixel in the tile, clamping to image bounds
    for (var dy = 0u; dy < TILESZ; dy++) {
        for (var dx = 0u; dx < TILESZ; dx++) {
            let px = min(pixel_base + vec2u(dx, dy), dims - vec2u(1u));
            let v = textureLoad(img_src, px, 0).r;
            v_min = min(v_min, v);
            v_max = max(v_max, v);
        }
    }

    // Store v_max inverted so min(vec2) in the blur pass works on both channels
    textureStore(img_tiles0, tile_id.xy, vec4u(v_min, 255u - v_max, 0u, 0u));
}