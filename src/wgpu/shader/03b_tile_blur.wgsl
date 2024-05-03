@group(1) @binding(1) var img_tiles0 : texture_2d<u32>;
@group(1) @binding(2) var img_tiles1 : texture_storage_2d<rg8uint, write>;

fn sub_sat(x: u32, y: u32) -> u32 {
    return select(x - y, 0u, y >= x);
}

@compute
@workgroup_size(1,1)
fn k03_tile_blur(@builtin(global_invocation_id) coords: vec3<u32>) {
    let dims = textureDimensions(img_tiles0);
    var acc = vec2u(255u);
    for (var i: u32 = sub_sat(coords.y, 1u); i < min(coords.y + 1u, dims.y); i++) {
        for (var j: u32 = sub_sat(coords.x, 1u); j < min(coords.x + 1u, dims.x); j++) {
            var v = textureLoad(img_tiles0, vec2u(j, i), 0).xy;
            acc = min(acc, v);
        }
    }
    
    textureStore(img_tiles1, coords.xy, vec4(acc.x, acc.y, 0u, 0u));
}