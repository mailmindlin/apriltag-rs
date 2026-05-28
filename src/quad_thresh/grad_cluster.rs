use std::{hash::Hash, collections::hash_map::RandomState};
use hashbrown::{HashMap, hash_map::Entry};

use rayon::prelude::*;

use crate::{dbg::debugln, detector::DetectorConfig, quad_thresh::MIN_CLUSTER_SIZE, util::image::ImageRefY8};

use super::{unionfind::{UnionFindId, UnionFindStatic}, linefit::Pt};

/// Canonical key identifying a pair of adjacent black/white connected regions.
///
/// During gradient clustering, every black/white pixel boundary contributes a
/// gradient point to the cluster that "belongs" to the region pair straddling
/// that edge. `ClusterId` uniquely names such a pair by storing the union-find
/// representative IDs of both regions in a normalized order (`rep0 >= rep1`),
/// so `ClusterId::new(a, b) == ClusterId::new(b, a)`.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub(crate) struct ClusterId {
	rep0: UnionFindId,
	rep1: UnionFindId,
}

// impl Hash for ClusterId {
// 	fn hash<H: Hasher>(&self, state: &mut H) {
// 		let u = (self.rep0 as u64) << 32 | (self.rep1 as u64);
// 		u.hash(state)
// 	}
// }

impl ClusterId {
	/// Construct a `ClusterId` from two region representatives, normalizing
	/// their order so that `new(a, b) == new(b, a)`.
	const fn new(repa: UnionFindId, repb: UnionFindId) -> Self {
		let (rep0, rep1) = if repb < repa {
			(repa, repb)
		} else {
			(repb, repa)
		};
		// The original representation stored both ids as a single u64 but
		// I don't think there's any reason for us to do that
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

/// Scan rows `y0..y1` of `threshim` for black/white pixel boundaries and
/// accumulate gradient points into `clustermap`.
///
/// For each pixel pair whose values differ (one black, one white) and whose
/// connected components both exceed [`MIN_CLUSTER_SIZE`], a [`Pt`] is added to
/// the cluster keyed by [`ClusterId`] of the two region representatives. The
/// point is placed at the sub-pixel midpoint between the two pixels, and its
/// gradient `(gx, gy)` points from black toward white (magnitude 255).
///
/// Connectivity is a hybrid 4/8 scheme: the right and below neighbours are
/// always checked (4-connectivity), and both diagonal-below neighbours are
/// checked while taking care to avoid emitting duplicate points when the
/// previous column already contributed a matching diagonal.
fn do_gradient_clusters(threshim: &ImageRefY8, y0: usize, y1: usize, clustermap: &mut Clusters, uf: &impl UnionFindStatic<(u32, u32), Id = u32>) {
	let width = threshim.width();
	for y in y0..y1 {
		let mut connected_last = false;

		for x in 1..(width-1) {
			let v0 = threshim[(x, y)];
			if v0 == 127 {
				connected_last = false;
				continue;
			}

			// XXX don't query this until we know we need it?
			let (rep0, size0) = uf.get_set_static((x as _, y as _));
			if size0 <= (MIN_CLUSTER_SIZE as u32) {
				connected_last = false;
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

			let mut DO_CONN = |dx: isize, dy: usize| {
				// NB: Arguments must be [-1, 1] or we'll overflow .gx, .gy
				debug_assert!(-1 <= dx && dx <= 1);
				debug_assert!(dy <= 1);

				let off_x = (x as isize + dx) as usize;
				let off_y = y + dy;

				let v1 = threshim[(off_x, off_y)];
				if v0 != v1 {
					let (rep1, size1) = uf.get_set_static((off_x as _, off_y as _));
					if size1 > (MIN_CLUSTER_SIZE as u32) {
						let key = ClusterId::new(rep0, rep1);
						let value: Pt = {
							let dv = (v1 as i16) - (v0 as i16);

							#[cfg(debug_assertions)]
							let pt_x = (2 * x as isize + dx).try_into().unwrap();
							#[cfg(debug_assertions)]
							let pt_y = (2 * y + dy).try_into().unwrap();

							#[cfg(not(debug_assertions))]
							let pt_x = (2 * x as isize + dx) as _;
							#[cfg(not(debug_assertions))]
							let pt_y = (2 * y + dy) as _;

							Pt {
								x: pt_x,
								y: pt_y,
								gx: (dx as i16 * dv),
								gy: (dy as i16 * dv),
								slope: 0.,//TODO?
							}
						};

						clustermap.entry(key)
							.or_default()
							.push(value);
						return true;
					}
				}
				false
			};
			// do 4 connectivity
			DO_CONN(1, 0);
			DO_CONN(0, 1);

			// do 8 connectivity
			// DO_CONN(-1,1);
			// DO_CONN(1, 1);
			
			if !connected_last {
				// Checking 1, 1 on the previous x, y, and -1, 1 on the current
				// x, y result in duplicate points in the final list.  Only
				// check the potential duplicate if adding this one won't
				// create a duplicate.
				DO_CONN(-1, 1);
			}
			connected_last = DO_CONN(1, 1);
		}
	}
	if !clustermap.is_empty() && false {
		debugln!("Found {} clusters on line {}..{}", clustermap.len(), y0, y1);
		for (cid, cluster) in clustermap.iter() {
			debugln!(" - {}/{} len {}", cid.rep0, cid.rep1, cluster.len());
		}
	}
}

/// Map from a region-pair [`ClusterId`] to the list of gradient [`Pt`]s lying
/// on the boundary between those two regions.
pub(crate) type Clusters = HashMap<ClusterId, Vec<Pt>, ClusterHasher>;

/// Merge two [`Clusters`] maps produced by separate row-range passes.
///
/// Entries that exist in only one map are moved as-is. Entries present in both
/// maps have their point lists concatenated. The larger of the two input maps
/// is used as the accumulator to minimise rehashing.
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

/// Build the full gradient cluster map from a thresholded image.
///
/// The image is divided into horizontal row-chunks and each chunk is processed
/// in parallel by [`do_gradient_clusters`]. The per-chunk [`Clusters`] maps are
/// then merged pairwise via [`merge_clusters`].
///
/// `uf` must be a completed union-find over the thresholded image so that
/// region representatives and sizes are stable before this call.
pub(crate) fn gradient_clusters(config: &DetectorConfig, threshim: &ImageRefY8, uf: impl Sync + UnionFindStatic<(u32, u32), Id = u32> ) -> Clusters {
	let nclustermap = (0.2*(threshim.num_pixels() as f64)) as usize;

	let sz = threshim.height() - 1;
	if config.single_thread() && false {
		let mut clustermap = Clusters::with_capacity_and_hasher(nclustermap, ClusterHasher::default());
		do_gradient_clusters(threshim, 0, sz, &mut clustermap, &uf);
		clustermap
	} else {
		let chunksize = 1 + sz / config.nthreads();
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

#[cfg(test)]
mod test {
	use crate::{quad_thresh::linefit::Pt, util::ImageY8};
	use super::{ClusterId, Clusters, ClusterHasher, do_gradient_clusters, merge_clusters};
	use super::super::unionfind::{UnionFind, UnionFindStatic};

	fn pt(seed: usize) -> Pt {
		Pt {
			x: seed as u16,
			y: seed.rotate_right(16) as u16,
			gx: seed.rotate_right(32) as i16,
			gy: seed.rotate_right(48) as i16,
			slope: 0.,
		}
	}

	fn make_clusters(entries: impl IntoIterator<Item = (ClusterId, Vec<Pt>)>) -> Clusters {
		let mut map = Clusters::with_hasher(ClusterHasher::default());
		for (k, v) in entries { map.insert(k, v); }
		map
	}

	fn image_from_rows(rows: &[&[u8]]) -> ImageY8 {
		let (h, w) = (rows.len(), rows[0].len());
		let mut img = ImageY8::zeroed_packed(w, h);
		for (y, row) in rows.iter().enumerate() {
			for (x, &v) in row.iter().enumerate() {
				img[(x, y)] = v;
			}
		}
		img
	}

	/// Minimal [`UnionFindStatic`] stub: returns a fixed `(representative, size)`
	/// per pixel without any path-compression or union logic.
	struct StubUF {
		width: usize,
		cells: Vec<(u32, u32)>,
	}

	impl StubUF {
		fn uniform(width: usize, height: usize, size: u32) -> Self {
			let n = width * height;
			Self { width, cells: (0..n as u32).map(|i| (i, size)).collect() }
		}

		fn set(&mut self, x: usize, y: usize, rep: u32, size: u32) {
			self.cells[y * self.width + x] = (rep, size);
		}
	}

	impl UnionFind<(u32, u32)> for StubUF {
		type Id = u32;
		fn index_to_id(&self, (x, y): (u32, u32)) -> u32 { y * self.width as u32 + x }
		fn get_set(&mut self, idx: (u32, u32)) -> (u32, u32) { self.get_set_static(idx) }
		fn connect_ids(&mut self, _a: u32, _b: u32) -> bool { false }
	}

	impl UnionFindStatic<(u32, u32)> for StubUF {
		fn get_set_static(&self, (x, y): (u32, u32)) -> (u32, u32) {
			self.cells[y as usize * self.width + x as usize]
		}
		fn get_set_hops(&self, _: (u32, u32)) -> usize { 1 }
	}

	fn run(img: &ImageY8, uf: &StubUF) -> Clusters {
		let mut clusters = Clusters::with_hasher(ClusterHasher::default());
		let threshim = img.as_ref();
		do_gradient_clusters(&threshim, 0, img.height() - 1, &mut clusters, uf);
		clusters
	}

	const BLACK: u8 = 0;
	const WHITE: u8 = 255;
	const GRAY:  u8 = 127;
	const BIG:  u32 = 30; // > MIN_CLUSTER_SIZE (24)
	const TINY: u32 = 1;  // < MIN_CLUSTER_SIZE (24)

	// --- merge_clusters ---

	#[test]
	fn merge_empty() {
		let c = make_clusters([(ClusterId::new(0, 0), vec![pt(0), pt(1)])]);
		let empty = Clusters::with_hasher(ClusterHasher::default());
		assert_eq!(merge_clusters(c, empty).len(), 1);
	}

	#[test]
	fn merge_no_dedup() {
		let c1 = make_clusters([
			(ClusterId::new(0, 0), vec![pt(0), pt(1)]),
			(ClusterId::new(0, 1), vec![pt(2), pt(3)]),
		]);
		let c2 = make_clusters([
			(ClusterId::new(0, 2), vec![pt(4), pt(5)]),
			(ClusterId::new(0, 3), vec![pt(6), pt(7)]),
		]);
		assert_eq!(merge_clusters(c1, c2).len(), 4);
	}

	#[test]
	fn merge_dedup() {
		let id1 = ClusterId::new(0, 0);
		let id2 = ClusterId::new(0, 1);
		let id3 = ClusterId::new(0, 2);
		let c1 = make_clusters([
			(id1, vec![pt(0), pt(1)]),
			(id2, vec![pt(2), pt(3)]),
		]);
		let c2 = make_clusters([
			(id3, vec![pt(4), pt(5)]),
			(id1, vec![pt(6), pt(7)]),
		]);
		let merged = merge_clusters(c1, c2);
		assert_eq!(merged.len(), 3);
		assert!(merged.contains_key(&id2));
		assert!(merged.contains_key(&id3));
		assert_eq!(merged[&id1].len(), 4);
	}

	// --- ClusterId ---

	#[test]
	fn cluster_id_symmetric() {
		assert_eq!(ClusterId::new(1, 2), ClusterId::new(2, 1));
		assert_ne!(ClusterId::new(1, 2), ClusterId::new(1, 3));
	}

	// --- do_gradient_clusters ---

	#[test]
	fn gradient_skips_gray() {
		let row = vec![GRAY; 6];
		let img = image_from_rows(&[&row, &row, &row, &row, &row]);
		let uf = StubUF::uniform(6, 5, BIG);
		assert!(run(&img, &uf).is_empty(), "gray-only image should produce no clusters");
	}

	#[test]
	fn gradient_filters_small_regions() {
		// Real black/white boundary, but every region is below MIN_CLUSTER_SIZE
		const W: usize = 6;
		let black = [BLACK; W];
		let white = [WHITE; W];
		let img = image_from_rows(&[&black, &black, &black, &white, &white, &white]);
		let uf = StubUF::uniform(W, 6, TINY);
		assert!(run(&img, &uf).is_empty(), "regions below MIN_CLUSTER_SIZE should be skipped");
	}

	#[test]
	fn gradient_horizontal_edge() {
		// Black rows 0-4, white rows 5-9. All boundary points are found via
		// DO_CONN(., 1): dy=1, dv=(255-0)=255, so gy = dy*dv = 255 for every point.
		const W: usize = 6;
		const H: usize = 10;
		let black = [BLACK; W];
		let white = [WHITE; W];
		let img = image_from_rows(&[
			&black, &black, &black, &black, &black,
			&white, &white, &white, &white, &white,
		]);
		let mut uf = StubUF::uniform(W, H, TINY);
		for y in 0..5 { for x in 0..W { uf.set(x, y, 0, BIG); } }
		for y in 5..H { for x in 0..W { uf.set(x, y, 1, BIG); } }

		let clusters = run(&img, &uf);
		assert_eq!(clusters.len(), 1, "one cluster for the black/white region pair");
		let pts = &clusters[&ClusterId::new(0, 1)];
		assert!(!pts.is_empty());
		for p in pts {
			assert_eq!(p.gy, 255, "gradient should point from black toward white (downward)");
		}
	}

	#[test]
	fn gradient_direction_flips_when_colors_swap() {
		// White rows 0-4, black rows 5-9. dv = (0-255) = -255, so gy = -255.
		const W: usize = 6;
		const H: usize = 10;
		let white = [WHITE; W];
		let black = [BLACK; W];
		let img = image_from_rows(&[
			&white, &white, &white, &white, &white,
			&black, &black, &black, &black, &black,
		]);
		let mut uf = StubUF::uniform(W, H, TINY);
		for y in 0..5 { for x in 0..W { uf.set(x, y, 1, BIG); } }
		for y in 5..H { for x in 0..W { uf.set(x, y, 0, BIG); } }

		let clusters = run(&img, &uf);
		assert_eq!(clusters.len(), 1);
		for p in &clusters[&ClusterId::new(0, 1)] {
			assert_eq!(p.gy, -255, "gradient should flip when white is above black");
		}
	}

	#[test]
	fn gradient_vertical_edge() {
		// Black columns 0-2, white columns 3-5. Boundary at x=2/3.
		// DO_CONN(1,0) and DO_CONN(1,1) both fire at x=2 with dx=1 and dv=255,
		// giving gx = dx*dv = 255 for every boundary point.
		const W: usize = 6;
		const H: usize = 10;
		let row: Vec<u8> = (0..W).map(|x| if x < 3 { BLACK } else { WHITE }).collect();
		let rows: Vec<&[u8]> = (0..H).map(|_| row.as_slice()).collect();
		let img = image_from_rows(&rows);
		let mut uf = StubUF::uniform(W, H, TINY);
		for y in 0..H {
			for x in 0..3 { uf.set(x, y, 0, BIG); }
			for x in 3..W { uf.set(x, y, 1, BIG); }
		}

		let clusters = run(&img, &uf);
		assert_eq!(clusters.len(), 1, "one cluster for the black/white region pair");
		for p in &clusters[&ClusterId::new(0, 1)] {
			assert_eq!(p.gx, 255, "gradient should point rightward (from black toward white)");
		}
	}
}