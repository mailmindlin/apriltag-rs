@group(0) @binding(1) var<storage, read_write> labels : LabelsAtomic;
@group(0) @binding(2) var img_src : texture_2d<u32>;

// Merges the UFTrees of a and b, linking one root to the other
fn Union(a0: u32, b0: u32) {
    var a = a0;
    var b = b0;
    loop {
        a = uf_find(a);
        b = uf_find(b);

        if (a < b) {
            let old = atomicMin(&labels.data[uf_parent_idx(b)], a);
            if old == b {
                return;
            }
            b = old;
        } else if (b < a) {
            let old = atomicMin(&labels.data[uf_parent_idx(a)], b);
            if old == a {
                return;
            }
            a = old;
        } else {
            return;
        }
    }
}

@compute
@workgroup_size(16, 16)
fn k04_bke_merge(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let coords = global_invocation_id.xy * vec2u(2u);
    let dims = textureDimensions(img_src);

    let labels_index = uf_element(coords);

    if all(coords < dims) {
        let info = get_ld_info(coords, dims);

        if (HasBit(info, BIT_Q)) {
            Union(labels_index, uf_element(coords - vec2u(0u, 2u)));
        }
        if (HasBit(info, BIT_R)) {
            Union(labels_index, uf_element(coords - vec2u(0u, 2u) + vec2u(2u, 0u)));
        }
        if (HasBit(info, BIT_S)) {
            Union(labels_index, uf_element(coords - vec2u(2u, 0u)));
        }
    }
}