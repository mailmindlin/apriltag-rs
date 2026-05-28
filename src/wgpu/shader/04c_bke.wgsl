@group(0) @binding(1) var<storage, read_write> labels : Labels;
@group(0) @binding(2) var img_src : texture_2d<u32>;

fn FindAndCompress(id: u32) -> u32 {
    var current = id;
    var parent = uf_parent(current);
    var i = 0u;
    while (current != parent) {
        current = parent;
        parent = uf_parent(current);
        uf_parent_set(id, current);
        if i > 1000u {
            break;
        }
        i++;
    }
	return current;
}

@compute
@workgroup_size(16, 16)
fn k04_bke_compression(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let coords = global_invocation_id.xy * vec2u(2u);
    let dims = textureDimensions(img_src);

    let labels_index = uf_element(coords);

	if all(coords < dims) {
        FindAndCompress(labels_index);
    }
}