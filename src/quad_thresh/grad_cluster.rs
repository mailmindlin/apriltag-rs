use std::{hash::Hash, collections::hash_map::RandomState};
use hashbrown::{HashMap, hash_map::Entry};

use rayon::prelude::*;

use crate::{util::image::ImageY8, detector::DetectorConfig};

use super::{unionfind::{UnionFindId, UnionFindStatic}, linefit::Pt};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub(super) struct ClusterId {
    rep0: UnionFindId,
    rep1: UnionFindId,
}

// impl Hash for ClusterId {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         let u = (self.rep0 as u64) << 32 | (self.rep1 as u64);
//         u.hash(state)
//     }
// }

impl ClusterId {
    const fn new(repa: UnionFindId, repb: UnionFindId) -> Self {
        let (rep0, rep1) = if repb < repa {
            (repa, repb)
        } else {
            (repb, repa)
        };
        // Self {
        //     value: (repa as u64) << 32 | (repb as u64)
        // }
        Self {
            rep0,
            rep1,
        }
    }
}

// #[derive(Clone, Copy, Default)]
// struct ClusterHasher(u64);

// impl BuildHasher for ClusterHasher {
//     type Hasher = Self;

//     fn build_hasher(&self) -> Self::Hasher {
//         *self
//     }
// }

// impl Hasher for ClusterHasher {
//     fn finish(&self) -> u64 {
//         self.0
//     }

//     fn write_u64(&mut self, i: u64) {
//         self.0 = self.0 * 2654435761 + i
//     }

//     fn write(&mut self, _bytes: &[u8]) {
//         todo!()
//     }
// }
type ClusterHasher = RandomState;

fn do_gradient_clusters(threshim: &ImageRefY8, y0: usize, y1: usize, clustermap: &mut Clusters, uf: &impl UnionFindStatic<(u32, u32), Id = u32>) {
    let width = threshim.width();
    for y in y0..y1 {
        for x in 1..(width-1) {
            let v0 = threshim[(x, y)];
            if v0 == 127 {
                continue;
            }

            // XXX don't query this until we know we need it?
            let (rep0, size0) = uf.get_set_static((x as _, y as _));
            if size0 < 25 {
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

            #[allow(non_upper_case_globals)]
            const offsets: [(isize, usize); 4] = [
                // do 4 connectivity. NB: Arguments must be [-1, 1] or we'll overflow .gx, .gy
                (1, 0),
                (0,1),
                // do 8 connectivity
                (-1,1),
                (1,1)
            ];
            for (dx, dy) in offsets {
                let off_x = (x as isize + dx) as usize;
                let off_y = y + dy;

                let v1 = threshim[(off_x, off_y)];
                if v0 != v1 {
                    let (rep1, size1) = uf.get_set_static((off_x as _, off_y as _));
                    if size1 > 24 {
                        let key = ClusterId::new(rep0, rep1);
                        let value: Pt = {
                            let dv = (v1 as i16) - (v0 as i16);
                            #[cfg(debug_assertions)]
                            let x = Pt {
                                x: (2 * x as isize + dx).try_into().unwrap(),
                                y: (2 * y + dy).try_into().unwrap(),
                                gx: (dx as i16 * dv),
                                gy: (dy as i16 * dv),
                                slope: 0.,//TODO?
                            };

                            #[cfg(not(debug_assertions))]
                            let x = Pt {
                                x: (2 * x as isize + dx) as _,
                                y: (2 * y + dy) as _,
                                gx: (dx as i16 * dv),
                                gy: (dy as i16 * dv),
                                slope: 0.,//TODO?
                            };

                            x
                        };

                        clustermap.entry(key)
                            .or_default()
                            .push(value);
                    }
                }
            }
        }
    }
    #[cfg(feature="extra_debug")]
    if !clustermap.is_empty() && false {
        println!("Found {} clusters on line {}..{}", clustermap.len(), y0, y1);
        for (cid, cluster) in clustermap.iter() {
            println!(" - {}/{} len {}", cid.rep0, cid.rep1, cluster.len());
        }
    }
}

pub(super) type Clusters = HashMap<ClusterId, Vec<Pt>, ClusterHasher>;

fn merge_clusters(c1: Clusters, c2: Clusters) -> Clusters {
    // Ensure c1 > c2 (fewer operations in next loop)
    let (mut c1, c2) = if c2.len() > c1.len() {
        (c2, c1)
    } else {
        (c1, c2)
    };

    for (k, v2) in c2.into_iter() {
        match c1.entry(k) {
            Entry::Occupied(mut e) => {
                let v1 = e.get_mut();
                // Pick the larger vector to keep
                let mut v2 = if v1.len() < v2.len() {
                    std::mem::replace(v1, v2)
                } else { v2 };
                
                v1.append(&mut v2);
            },
            Entry::Vacant(e) => {
                e.insert(v2);
            },
        }
    }
    c1
}

pub(crate) fn gradient_clusters(config: &DetectorConfig, threshim: &ImageRefY8, mut uf: (impl Sync + UnionFindStatic<(u32, u32), Id = u32>)) -> Clusters {
    let nclustermap = (0.2*(threshim.len() as f64)) as usize;

    let sz = threshim.height() - 1;
    if config.single_thread() && false {
        let mut clustermap = Clusters::with_capacity_and_hasher(nclustermap, ClusterHasher::default());
        do_gradient_clusters(threshim, 0, sz, &mut clustermap, &mut uf);
        clustermap
    } else {
        let chunksize = 1 + sz / config.nthreads;
        // struct cluster_task *tasks = malloc(sizeof(struct cluster_task)*(sz / chunksize + 1));

        (0..sz)
            .into_par_iter()
            .step_by(chunksize)
            .fold(|| Clusters::with_capacity_and_hasher(nclustermap, ClusterHasher::default()), |mut clustermap, i| {
                let y0 = i;
                let y1 = std::cmp::min(sz, i + chunksize);
                do_gradient_clusters(threshim, y0, y1, &mut clustermap, &uf);
                clustermap
            })
            //TODO: it might be more efficient to reduce adjacent clusters
            .reduce(|| Clusters::with_hasher(ClusterHasher::default()), merge_clusters)
    }
}

#[cfg(all(test, feature="foo"))]
mod test {
    use crate::quad_thresh::linefit::Pt;

    use super::{ClusterEntry, merge_clusters};

    fn random_point(seed: usize) -> Pt {
        Pt {
            x: seed as u16,
            y: seed.rotate_right(16) as u16,
            gx: seed.rotate_right(32) as i16,
            gy: seed.rotate_right(48) as i16,
            slope: 0.,
        }
    }

    #[test]
    fn merge_empty() {
        let e1 = ClusterEntry {
            id: super::ClusterId { rep0: 0, rep1: 0 },
            data: vec![random_point(0), random_point(1)],
        };
        let merged = merge_clusters(vec![e1], vec![]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_no_dedup() {
        let e1 = ClusterEntry {
            id: super::ClusterId { rep0: 0, rep1: 0 },
            data: vec![random_point(0), random_point(1)],
        };
        let e2 = ClusterEntry {
            id: super::ClusterId { rep0: 0, rep1: 1 },
            data: vec![random_point(2), random_point(3)],
        };
        let e3 = ClusterEntry {
            id: super::ClusterId { rep0: 0, rep1: 2 },
            data: vec![random_point(4), random_point(5)],
        };
        let e4 = ClusterEntry {
            id: super::ClusterId { rep0: 0, rep1: 3 },
            data: vec![random_point(6), random_point(7)],
        };
        let merged = merge_clusters(vec![e1, e2], vec![e3, e4]);
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn merge_dedup() {
        let id1 = super::ClusterId { rep0: 0, rep1: 0 };
        let id2 = super::ClusterId { rep0: 0, rep1: 1 };
        let id3 = super::ClusterId { rep0: 0, rep1: 2 };
        let e1 = ClusterEntry {
            id: id1.clone(),
            data: vec![random_point(0), random_point(1)],
        };
        let e2 = ClusterEntry {
            id: id2.clone(),
            data: vec![random_point(2), random_point(3)],
        };
        let e3 = ClusterEntry {
            id: id3.clone(),
            data: vec![random_point(4), random_point(5)],
        };
        let e4 = ClusterEntry {
            id: id1.clone(),
            data: vec![random_point(6), random_point(7)],
        };
        let merged = merge_clusters(vec![e1, e2], vec![e3, e4]);
        assert_eq!(merged.len(), 3);
        // Check that other two are still there
        assert!(merged.iter().find(|it| it.id == id2).is_some());
        assert!(merged.iter().find(|it| it.id == id3).is_some());
        
        let e_merged = merged.iter().find(|it| it.id == id1).unwrap();
        assert_eq!(e_merged.data.len(), 4);
    }
}