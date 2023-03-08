use std::{collections::{HashMap, hash_map::Entry}, hash::Hash};

use rayon::prelude::*;

use crate::{util::{Image}, ApriltagDetector};

use super::{uf::{UnionFind2D, UnionFindId}, linefit::Pt};

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct ClusterId {
    rep0: UnionFindId,
    rep1: UnionFindId,
}

impl ClusterId {
    fn new(repa: UnionFindId, repb: UnionFindId) -> Self {
        let (rep0, rep1) = if repb < repa {
            (repa, repb)
        } else {
            (repb, repa)
        };
        Self {
            rep0,
            rep1,
        }
    }
}

struct ClusterEntry {
    id: ClusterId,
    data: Vec<Pt>,
}

#[inline]
fn u64hash_2(x: u64) -> u32 {
    ((2654435761 * x) >> 32) as u32
    // return x as u32;
}

fn do_gradient_clusters(threshim: &Image, y0: usize, y1: usize, nclustermap: usize, uf: &mut UnionFind2D) -> Vec<ClusterEntry> {
    // let nclustermap = 2*w*h - 1;
    let mut clustermap = HashMap::<ClusterId, Vec<Pt>>::with_capacity(nclustermap);

    for y in y0..y1 {
        for x in 1..(threshim.width-1) {
            let v0 = threshim[(x, y)];
            if v0 == 127 {
                continue;
            }

            // XXX don't query this until we know we need it?
            let rep0 = uf.get_representative(x, y);
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

            let offsets = [
                // do 4 connectivity. NB: Arguments must be [-1, 1] or we'll overflow .gx, .gy
                (1isize, 0isize),
                (0,1),
                // do 8 connectivity
                (-1,1),
                (1,1)
            ];
            for (dx, dy) in offsets {
                let off_x = (x as isize + dx) as usize;
                let off_y = (y as isize + dy) as usize;
                let v1 = threshim[(off_x, off_y)];

                if v0 + v1 == 255 {
                    let rep1 = uf.get_representative(off_x, off_y);
                    if uf.get_set_size(rep1) > 24 {
                        let clusterid = ClusterId::new(rep0, rep1);
                        let cluster = match clustermap.entry(clusterid){
                            Entry::Occupied(entry) => entry.into_mut(),
                            Entry::Vacant(entry) => entry.insert(Vec::new()),
                        };

                        let dv = (v1 as isize) - (v0 as isize);

                        cluster.push(Pt {
                            x: (2 * off_x).try_into().unwrap(),
                            y: (2 * off_y).try_into().unwrap(),
                            gx: (dx * dv).try_into().unwrap(),
                            gy: (dy * dv).try_into().unwrap(),
                            slope: 0.,//TODO?
                        });
                    }
                }
            }
        }
    }
    // =======
    let mut clusters = clustermap
        .into_iter()
        .map(|(id, data)| ClusterEntry { id, data })
        .collect::<Vec<_>>();

    clusters.sort_unstable_by_key(|e| e.id);
    clusters
}

fn merge_clusters(mut c1: Vec<ClusterEntry>, mut c2: Vec<ClusterEntry>) -> Vec<ClusterEntry> {
    /*let mut ret = Vec::with_capacity(c1.len() + c2.len());

    let mut i1 = 0;
    let mut i2 = 0;
    let l1 = c1.len();
    let l2 = c2.len();

    while i1 < l1 && i2 < l2 {
        let h1 = &c1[i1];
        let h2 = &c2[i2];
        match h1.cmp(h2) {
            Ordering::Equal => {
                h1.data.extend(h2.data.drain(..));
                ret.push(*h1);
                i1 += 1;
                i2 += 1;
            },
            Ordering::Greater => {
                ret.push(*h2);
                i2 += 1;
            },
            Ordering::Less => {
                ret.push(*h1);
                i1 += 1;
            },
        }
    }

    ret.extend(c1.into_iter().skip(i1));
    ret.extend(c2.into_iter().skip(i2));

    ret*/
    c1.extend(c2.drain(..));
    c1.sort_unstable_by_key(|c| c.id);
    todo!();
    c1
}

pub(super) fn gradient_clusters(td: &ApriltagDetector, threshim: &Image, uf: &mut UnionFind2D) -> Vec<Vec<Pt>> {
    let nclustermap = (0.2*(threshim.len() as f64)) as usize;

    let sz = threshim.height - 1;
    let chunksize = 1 + sz / td.params.nthreads;
    // struct cluster_task *tasks = malloc(sizeof(struct cluster_task)*(sz / chunksize + 1));

    (1..sz)
        // .into_par_iter()
        .step_by(chunksize)
        .map(|i| {
            let y0 = i;
            let y1 = std::cmp::min(sz, i + chunksize);
            let clusters = do_gradient_clusters(threshim, y0, y1, nclustermap, uf);
            clusters
        })
        //TODO: it might be more efficient to reduce adjacent clusters
        .fold(Vec::new(), merge_clusters)
        // Convert from ClusterEntry -> Vec<Pt>
        .into_iter()
        .map(|cluster| cluster.data)
        .collect()
}