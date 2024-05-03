@group(0) @binding(1) var<storage, read_write> labels : LabelsAtomic;
@group(0) @binding(2) var img_src : texture_2d<u32>;

@compute
@workgroup_size(16, 16)
fn k04_merge(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let coords = global_invocation_id.xy;
	let dims = textureDimensions(img_src);
	let elem = uf_element(coords);

    if all(coords < dims) {
        // let parent_w = Find(elem);
		let parent_w = uf_parent(elem);
        let parent_b = uf_size(elem);
        var parent = parent_w;
        // var parent = select(parent_w, parent_b - 1u, parent_b != 0u);
        if parent_b != 0u {
            parent = parent_b;
        }
        // let parent = select(elem, parent_b - 1u, parent_b != 0u);
        // let parent = parent_b;
        // Set parent to root (depth will be zero)
        uf_parent_set(elem, parent);
        uf_size_set(elem, 0u);
	}
}

@compute
@workgroup_size(16, 16)
fn k04_count(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let coords = global_invocation_id.xy;
	let elem = uf_element(coords);
	let dims = textureDimensions(img_src);

    if all(coords < dims) {
		let rep = uf_find(elem);

        // Set parent to root (depth will be zero)
        uf_parent_set(elem, rep);
        // Add 1 to parent's count
		atomicAdd(&labels.data[uf_parent_idx(rep) + 1u], 1u);
	}
}