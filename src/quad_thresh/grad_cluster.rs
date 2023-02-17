use rayon::prelude::*;

use crate::{util::{Image, mem::calloc}, ApriltagDetector};

use super::{uf::UnionFind, linefit::Pt};

struct ClusterHash {
    hash: u32,
    id: u64,
    data: Vec<CHArrayEntry>,//TODO
}

struct CHArrayEntry {
    id: usize,
    cluster: Vec<Pt>,
}


#[inline]
fn u64hash_2(x: u64) -> u32 {
    ((2654435761 * x) >> 32) as u32
    // return x as u32;
}

fn do_gradient_clusters(threshim: &Image, y0: usize, y1: usize, nclustermap: usize, uf: &UnionFind) -> Vec<ClusterHash> {
    // let nclustermap = 2*w*h - 1;
    let mut clustermap = calloc::<CHArrayEntry>(nclustermap);

    let mem_chunk_size = 2048;
    let mut mem_pools = Vec::<Box<[CHArrayEntry]>>::with_capacity(2*nclustermap/mem_chunk_size);
    let mut mem_pool_idx = 0;
    let mut mem_pool_loc = 0;
    mem_pools[mem_pool_idx] = calloc::<CHArrayEntry>(mem_chunk_size);

    for y in y0..y1 {
        for x in 1..(threshim.width-1) {
            let v0 = threshim[(x, y)];
            if v0 == 127 {
                continue;
            }

            // XXX don't query this until we know we need it?
            let rep0 = uf.get_representative(y * w + x);
            if uf.get_set_size(rep0) < 25 {
                continue;
            }

            // whenever we find two adjacent pixels such that one is
            // white and the other black, we add the point half-way
            // between them to a cluster associated with the unique
            // ids of the white and black regions.
            //
            // We additionally compute the gradient direction (i.e., which
            // direction was the white pixel?) Note: if (v1-v0) == 255, then
            // (dx,dy) points towards the white pixel. if (v1-v0) == -255, then
            // (dx,dy) points towards the black pixel. p.gx and p.gy will thus
            // be -255, 0, or 255.
            //
            // Note that any given pixel might be added to multiple
            // different clusters. But in the common case, a given
            // pixel will be added multiple times to the same cluster,
            // which increases the size of the cluster and thus the
            // computational costs.
            //
            // A possible optimization would be to combine entries
            // within the same cluster.

            fn do_conn(dx: isize, dy: isize) {
                let v1 = threshim[(y + dy, x + dx)];

                if v0 + v1 == 255 {
                    let rep1 = uf.get_representative((y + dy) * w + (x + dx));
                    if uf.get_set_size(rep1) > 24 {
                        let clusterid = if (rep0 < rep1) {
                            (rep1 << 32) + rep0
                        } else {
                            (rep0 << 32) + rep1
                        };

                        /* XXX lousy hash function */
                        let clustermap_bucket = u64hash_2(clusterid) % nclustermap;
                        let entry = clustermap[clustermap_bucket];
                        while (entry && entry.id != clusterid) {
                            entry = entry.next;
                        }
                        if !entry {
                            if mem_pool_loc == mem_chunk_size {
                                mem_pool_loc = 0;
                                mem_pool_idx++;
                                mem_pools[mem_pool_idx] = calloc(mem_chunk_size, sizeof(struct CHArrayEntry));
                            }
                            entry = mem_pools[mem_pool_idx] + mem_pool_loc;
                            mem_pool_loc++;

                            entry.id = clusterid;
                            entry.cluster = zarray_create(sizeof(struct pt));
                            entry.next = clustermap[clustermap_bucket];
                            clustermap[clustermap_bucket] = entry;
                        }

                        entry.cluster.push(Pt {
                            x: 2 * x + dx,
                            y: 2 * y + dy,
                            gx: dx * ((v1 - v0) as i32),
                            gy: dy * ((v1 - v0) as i32),
                        });
                    }
                }
            }

            // do 4 connectivity. NB: Arguments must be [-1, 1] or we'll overflow .gx, .gy
            do_conn(1, 0);
            do_conn(0, 1);

            // do 8 connectivity
            do_conn(-1, 1);
            do_conn(1, 1);
        }
    }
    // =======

    for i in 0..nclustermap {
        let start = clusters.len();
        for (struct CHArrayEntry *entry = clustermap[i]; entry; entry = entry.next) {
            clusters.push(ClusterHash {
                hash: u64hash_2(entry.id) % nclustermap,
                id: entry.id,
                data: entry.cluster,
            });
        }
        let end = clusters.len();

        // Do a quick bubblesort on the secondary key.
        let n = end - start;
        for j in 0..(n-1) {
            for k in 0..(n - j - 1) {
                let hash1 = &clusters[start + k];
                let hash2 = &clusters[start + k + 1];
                if clusters[start + k].id > clusters[start + k + 1].id {
                    clusters.swap(start + k, start + k + 1);
                }
            }
        }
    }

    clusters
}

fn merge_clusters(c1: Vec<ClusterHash>, c2: Vec<ClusterHash>) -> Vec<ClusterHash> {
    let mut ret = Vec::with_capacity(c1.len() + c2.len());

    let mut i1 = 0;
    let mut i2 = 0;
    let l1 = c1.len();
    let l2 = c2.len();

    while i1 < l1 && i2 < l2 {
        let h1 = &c1[i1];
        let h2 = &c2[i2];

        if h1.hash == h2.hash && h1.id == h2.id {
            h1.data.extend(h2.data.drain(..));
            ret.push(*h1);
            i1 += 1;
            i2 += 1;
        } else if h2.hash < h1.hash || (h2.hash == h1.hash && h2.id < h1.id) {
            ret.push(*h2);
            i2 += 1;
        } else {
            ret.push(*h1);
            i1 += 1;
        }
    }

    ret.extend(c1.into_iter().skip(i1));
    ret.extend(c2.into_iter().skip(i2));

    ret
}

pub(super) fn gradient_clusters(td: &ApriltagDetector, threshim: &Image, uf: &UnionFind) -> Vec<ClusterHash> {
    let nclustermap = 0.2*(threshim.len() as f64);

    let sz = threshim.height - 1;
    let chunksize = 1 + sz / td.params.nthreads;
    // struct cluster_task *tasks = malloc(sizeof(struct cluster_task)*(sz / chunksize + 1));

    let mut clusters_list = (1..sz)
        .into_par_iter()
        .step_by(chunksize)
        .map(|i| {
            let y0 = i;
            let y1 = std::cmp::min(sz, i + chunksize);
            let clusters = do_gradient_clusters(threshim, y0, y1, nclustermap, uf);
            clusters
        })
        .collect::<Vec<_>>();

    while clusters_list.len() > 1 {
        clusters_list
            .par_chunks(2)
            .map(|chunk| {
                let first = chunk[0];
                if let Some(second) = chunk.get(1) {
                    //TODO: prevent copy
                    merge_clusters(first, *second)
                } else {
                    first
                }
            })
            .collect::<Vec<_>>();
    }
    clusters_list[0]
}