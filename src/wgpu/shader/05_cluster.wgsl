@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> labels : Labels;
@group(0) @binding(2) var img_src : texture_2d<u32>;

struct HashMapKey {
    k0: atomic<u32>,
    k1: atomic<u32>,
    len: atomic<u32>,
}

struct HashMapBucket {
    bucket_offset: atomic<u32>
}

@group(1) @binding(0) var<storage, read_write> hm_keys: array<HashMapKey>;
@group(1) @binding(1) var<storage, read_write> hm_buckets: array<HashMapBucket>;

struct Pt {
    // Packed u16x2
    xy: u32,
    // Packed i16x2
    gxy: u32,
}

struct MaybePt {
    key: vec2u,
    value: Pt,
}

fn pack_u16x2(raw: vec2u) -> u32 {
    return (raw.x << 16) | raw.y;
}

fn pack_i16x2(raw: vec2i) -> u32 {
    return pack_u16x2(vec2u(raw) & 0xFFFFu);
}

fn make_pt(base: vec2u, offset: vec2i, v0: u32, v1: u32) -> Pt {
    let pos = vec2u(vec2i(base) + offset);
    let dv = i32(v1) - i32(v0);

    let xy = pack_u16x2(vec2u(vec2i(base * 2u) + offset));
    let gxy = pack_i16x2(offset * dv);
    return Pt(xy, gxy);
}

fn hash_key(key: vec2u) -> vec2u {
    let k0a = pack_u16x2(extractBits(key, 0, 16));
    let k0 = select(k0a, u32(-1), k0a == 0u);
    let k1 = pack_u16x2(extractBits(key, 16, 16));
    return vec2u(k0, k1);
}

fn try_cluster(base: vec2u, offset: vec2i, rep0: u32, v0: u32) -> MaybePt {
    let pos1 = vec2u(vec2i(base) + offset);
    let v1 = pixel(pos1);
    if (v1 != v0) {
        let idx1 = uf_element(pos1);
        let rep1 = uf_representative(idx1);
        if uf_size(rep1) > 24 {
            let pt = make_pt(base, offset, v0, v1);
            return MaybePt(false, pt);
        }
    }
    return MaybePt(true, Pt(0u, 0u));
}

fn valid_key(key: vec2u) -> bool {
    return key != vec2u(0u);
}

var<workgroup> wg_write_enables: array<atomic<u32>, 8>;
var<workgroup> wg_write_enable: u32;
var<workgroup> wg_write_idxs: array<vec2u, 256>;
fn wg_insert_hashmap(value: MaybePt, local_invocation_index: u32) {
    wg_write_idxs[local_invocation_index] = value.key;
    workgroupBarrier();
    let first_dup = local_invocation_index;
    let capacity = arrayLength(&hm_keys);
    if !valid_key(value.key) {
        return;
    }
    let k0 = value.key.x;
    var idx = k0 % capacity;
    loop {
        let res = atomicCompareExchangeWeak(&hm_keys[idx], 0u, k0);
        if res.exchanged {
            // Key inserted
            break;
        } else if (res.old_value == 0u) {
            continue;
        } else {
            // Bucket is full, try a different one
            idx = 2654435761u * idx % capacity;
        }
    }
    
}

// var<workgroup> write_enables: array<atomic<u32>, 8>;
// var<workgroup> wg_write_enable: u32;
// var<workgroup> write_idxs: array<vec2u, 256>;
// var<workgroup> wga_bucket_idx: atomic<u32>;
// var<workgroup> wg_bucket_idx: u32;

// fn wg_insert_keys(key_id: u32, local_invocation_index: u32) {
//     let key = write_idxs[key_id];
//     let capacity = arrayLength(hm_keys);
//     var probe_idx = (key.x % capacity + local_invocation_index) % capacity;
//     loop {
//         let fetch_k0 = atomicLoad(&hm_keys[probe_idx].k0);
//         if fetch_k0 != 0u {
//             atomicMin(&wg_bucket_idx, probe_idx);
//         }
//         workgroupBarrier();
//         if local_invocation_index == 0u {
//             wg_bucket_idx = atomicExchange(&wga_bucket_idx, u32(-1));
//         }
//         let bucket_idx = workgroupUniformLoad(&wg_bucket_idx);

//     }
// }

@compute
@workgroup_size(16, 16)
fn k05_cluster(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
	let coords = global_invocation_id.xy + vec2u(1u);
	let idx = uf_element(coords);
	let dims = textureDimensions(img_src);

    var ignore = (coords.x + 1u >= dims.x || coords.y >= dims.y);

    let rep0 = select(uf_representative(idx), 0u, ignore);
    if !ignore {
        let size0 = uf_size(rep0);
        ignore |= (size0 <= 24);
    }

    let p0 = try_cluster(coords, vec2( 1, 0), rep0, v0);
    atomicOr(&write_enables[local_invocation_index / 32u], 1u << (local_invocation_index % 32u));
    write_idxs[local_invocation_index] = p0.key;

    for (var wg_cluster: u32 = 0u; i < 8; i++) {
        // Skip iterations where we don't have any data
        workgroupBarrier();
        if local_invocation_index == 0u {
            wg_write_enable = atomicLoad(&write_enables[wg_cluster]);
            atomicStore(&wg_bucket_idx, u32(-1));
        }
        var write_enable = workgroupUniformLoad(&wg_write_enable);
        while write_enable != 0u {
            let i = firstTrailingBit(write_enable) + wg_cluster * 32u;
            let key = write_idxs[i];
            let capacity = arrayLength(hm_keys);
            var probe_idx = (key % capacity + local_invocation_index) % capacity;
            loop {
                let fetch_k0 = atomicLoad(&hm_keys[probe_idx].k0);
                if fetch_k0 != 0u {
                    atomicMin(&wg_bucket_idx, probe_idx);
                }
                workgroupBarrier();
                if local_invocation_index == 0u {
                    let bucket_idx = atomicLoad(&wg_bucket_idx);
                    if bucket_idx != u32(-1) {

                    }
                }
            }

        }
    }
    // let p1 = try_cluster(coords, vec2( 0, 1), rep0, v0);
    // let p2 = try_cluster(coords, vec2(-1, 1), rep0, v0);
    // let p3 = try_cluster(coords, vec2( 1, 1), rep0, v0);
}