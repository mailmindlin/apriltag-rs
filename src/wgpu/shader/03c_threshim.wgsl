// Step 3c: Build threshold image
//
// Produces a binary (black/white/gray) image from the source grayscale image
// using adaptive thresholding. Each tile's local min/max (from step 03b) is
// used to compute a per-tile threshold at the midpoint of the intensity range.
//
// Output values:
//   255 = white (above threshold)
//     0 = black (at or below threshold)
//   127 = indeterminate (tile has too little contrast to classify)

const TILESZ = 4u;

struct Params {
    // Minimum difference between tile max and min intensity for the tile
    // to be considered high-contrast enough to threshold.
    min_white_black_diff: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var img_src : texture_2d<u32>;                   // Source grayscale image
// @group(1) @binding(1) var img_tiles0 : texture_storage_2d<rg8uint, read_write>;
@group(1) @binding(2) var img_tiles1 : texture_2d<u32>;                // Blurred tile min/max from step 03b
@group(1) @binding(3) var img_dst : texture_storage_2d<r8uint, write>; // Output threshold image

/// Saturating subtraction: returns max(x - y, 0)
fn sub_sat(x: u32, y: u32) -> u32 {
    if (y >= x) {
        return 0u;
    } else {
        return x - y;
    }
}

/// Saturating add clamped to [0, 255]
fn add_sat8(x: u32, y: u32) -> u32 {
    return min(x + y, 255u);
}

// One invocation per tile. Reads the tile's blurred min/max, computes a
// threshold, then classifies every pixel in the tile.
@compute
@workgroup_size(1,1)
fn k03_build_threshim(@builtin(global_invocation_id) t: vec3<u32>) {
    let dims = textureDimensions(img_src);
    let tdims = dims / vec2u(TILESZ);

    // Load this tile's min/max (v_max was stored inverted so min() could be
    // used in the blur pass, so we invert it back here).
    let v_minmax = textureLoad(img_tiles1, min(t.xy, tdims - vec2u(1u)), 0);
    let bottom_edge = (t.y >= tdims.y) || (t.x >= tdims.x);
    let v_min = v_minmax.x;
    let v_max = 255u - v_minmax.y;

    // Threshold is the midpoint of the tile's intensity range
    let delta = sub_sat(v_max, v_min);
    let thresh = add_sat8(v_min, delta / 2u);

    // Iterate over every pixel in this tile
    for (var dy: u32 = 0u; dy < TILESZ; dy++) {
        let y = (t.y * TILESZ) + dy;
        if (y > dims.y) {
            continue;
        }

        for (var dx: u32 = 0u; dx < TILESZ; dx++) {
            let x = (t.x * TILESZ) + dx;
            if (x > dims.x) {
                continue;
            }

            var value: u32;
            if (bottom_edge || delta >= params.min_white_black_diff) {
                // Tile has enough contrast (or is at the image border):
                // threshold the pixel. Biased toward dark because specular
                // highlights can be substantially brighter than white tag parts.
                let v = textureLoad(img_src, vec2u(x, y), 0).r;
                value = select(0u, 255u, v > thresh);
            } else {
                // Low contrast region — not enough dynamic range to decide
                // black vs white, so mark as indeterminate.
                value = 127u;
            }
            textureStore(img_dst, vec2u(x, y), vec4u(value));
        }
    }
}
