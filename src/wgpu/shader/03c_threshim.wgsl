const TILESZ = 4u;

struct Params {
    min_white_black_diff: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var img_src : texture_2d<u32>;
// @group(1) @binding(1) var img_tiles0 : texture_storage_2d<rg8uint, read_write>;
@group(1) @binding(2) var img_tiles1 : texture_2d<u32>;
@group(1) @binding(3) var img_dst : texture_storage_2d<r8uint, write>;

fn sub_sat(x: u32, y: u32) -> u32 {
    if (y >= x) {
        return 0u;
    } else {
        return x - y;
    }
}

fn add_sat8(x: u32, y: u32) -> u32 {
    return min(x + y, 255u);
}

@compute
@workgroup_size(1,1)
fn k03_build_threshim(@builtin(global_invocation_id) t: vec3<u32>) {
    let dims = textureDimensions(img_src);
    let tdims = dims / vec2u(TILESZ);
    let v_minmax = textureLoad(img_tiles1, min(t.xy, tdims - vec2u(1u)), 0);
    let bottom_edge = (t.y >= tdims.y) || (t.x >= tdims.x);
    let v_min = v_minmax.x;
    let v_max = 255u - v_minmax.y;

    let delta = sub_sat(v_max, v_min);
    let thresh = add_sat8(v_min, delta / 2u);
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
                // otherwise, actually threshold this tile.
                // argument for biasing towards dark; specular highlights
                // can be substantially brighter than white tag parts
                let v = textureLoad(img_src, vec2u(x, y), 0).r;
                value = select(0u, 255u, v > thresh);
            } else {
                // low contrast region? (no edges)
                value = 127u;
            }
            textureStore(img_dst, vec2u(x, y), vec4u(value));
        }
    }
}
