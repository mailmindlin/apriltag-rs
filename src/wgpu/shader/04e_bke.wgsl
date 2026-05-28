@group(0) @binding(1) var<storage, read_write> labels : Labels;
@group(0) @binding(2) var img_src : texture_2d<u32>;

fn update_parent(coords: vec2u, offset: vec2u, info: u32, white_father: u32, bit_w: u32) {
    let elem = uf_element(coords + offset);
    let parent = select(elem, white_father, HasBit(info, bit_w));
    uf_parent_set(elem, parent);
    // uf_size_set(elem, 0u);
}

// fn update_parent(coords: vec2u, offset: vec2u, info: u32, white_father: u32, black_father: u32, bit_w: u32, bit_b: u32) {
//     let elem = uf_element(coords + offset);
//     let idx = uf_parent_idx(elem);
//     var parent = elem;
//     if HasBit(info, bit_w) {
//         parent = white_father;
//     // } else if HasBit(info, bit_w) {
//     //     parent = black_father;
//     }
//     labels.data[idx] = elem;
//     // labels.data[idx + 1u] = 0u; // Clear size
// }

@compute
@workgroup_size(16, 16)
fn k04_bke_final(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let coords = global_invocation_id.xy * vec2u(2u);
    let dims = textureDimensions(img_src);

    let elem = uf_element(coords);

    if all(coords < dims) {
        let ld = get_ld(coords, dims);

        update_parent(coords, vec2u(0u, 0u), ld.info, ld.white_father, W_a);
        
        // update_parent(coords, vec2u(0u, 0u), ld.info, ld.white_father, ld.bf_a, W_a, B_a);

        if (coords.x + 1u < dims.x) {
            // update_parent(coords, vec2u(1u, 0u), ld.info, ld.white_father, ld.bf_b, W_b, B_b);
            update_parent(coords, vec2u(1u, 0u), ld.info, ld.white_father, W_b);

            if (coords.y + 1u < dims.y) {
                update_parent(coords, vec2u(0u, 1u), ld.info, ld.white_father, W_c);
                update_parent(coords, vec2u(1u, 1u), ld.info, ld.white_father, W_d);
                // update_parent(coords, vec2u(0u, 1u), ld.info, ld.white_father, ld.bf_c, W_c, B_c);
                // update_parent(coords, vec2u(1u, 1u), ld.info, ld.white_father, ld.bf_d, W_d, B_d);
            }
        } else {
            if (coords.y + 1u < dims.y) {
                // update_parent(coords, vec2u(0u, 1u), ld, W_c);
                update_parent(coords, vec2u(0u, 1u), ld.info, ld.white_father, W_c);
                // update_parent(coords, vec2u(0u, 1u), ld.info, ld.white_father, ld.bf_c, W_c, B_c);
            }
        }
    }
}