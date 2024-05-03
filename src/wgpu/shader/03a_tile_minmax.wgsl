const TILESZ = 4u;

@group(1) @binding(0) var img_src : texture_2d<u32>;
@group(1) @binding(1) var img_tiles0 : texture_storage_2d<rg8uint, write>;

@compute
@workgroup_size(1,1)
fn k03_tile_minmax(@builtin(global_invocation_id) tile_id: vec3<u32>) {
    let pixel_base = tile_id.xy * vec2u(TILESZ);

    var v_max = 0u;
    var v_min = 255u;
    for (var dy = 0u; dy < TILESZ; dy++) {
        for (var dx = 0u; dx < TILESZ; dx++) {
            let b = pixel_base + vec2u(dx, dy);
            let v = textureLoad(img_src, b, 0).r;
            v_min = min(v_min, v);
            v_max = max(v_max, v);
        }
    }

    // Invert v_max so we can use min(vec2) ops
    textureStore(img_tiles0, tile_id.xy, vec4u(v_min, 255u - v_max, 0u, 0u));
}