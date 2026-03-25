// Step 3b: Tile min/max blur
//
// Applies a 3x3 min-filter over the tile min/max image from step 03a.
// This spreads each tile's intensity range to its neighbors, which smooths
// out the adaptive threshold boundaries and avoids harsh transitions between
// tiles with different contrast levels.
//
// Because v_max is stored inverted (255 - v_max), a single min() operation
// on both channels simultaneously tightens the min and widens the max.

// Input: tile min/max from step 03a
@group(1) @binding(1) var img_tiles0 : texture_2d<u32>;
// Output: blurred tile min/max
@group(1) @binding(2) var img_tiles1 : texture_storage_2d<rg8uint, write>;

/// Saturating subtraction: returns max(x - y, 0)
fn sub_sat(x: u32, y: u32) -> u32 {
    return select(x - y, 0u, y >= x);
}

// One invocation per tile. Reads a 3x3 neighborhood of tiles and takes
// the component-wise min of (v_min, 255-v_max) across all neighbors.
@compute
@workgroup_size(1,1)
fn k03_tile_blur(@builtin(global_invocation_id) coords: vec3<u32>) {
    let dims = textureDimensions(img_tiles0);
    var acc = vec2u(255u);

    // 3x3 neighborhood (clamped to image bounds)
    for (var i: u32 = sub_sat(coords.y, 1u); i < min(coords.y + 1u, dims.y); i++) {
        for (var j: u32 = sub_sat(coords.x, 1u); j < min(coords.x + 1u, dims.x); j++) {
            var v = textureLoad(img_tiles0, vec2u(j, i), 0).xy;
            acc = min(acc, v);
        }
    }

    textureStore(img_tiles1, coords.xy, vec4(acc.x, acc.y, 0u, 0u));
}