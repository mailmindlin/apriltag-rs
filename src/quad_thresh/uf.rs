use crate::{ApriltagDetector, util::Image, quad_thresh::APRILTAG_TASKS_PER_THREAD_TARGET};
use rayon::prelude::*;

pub(super) type UnionFindId = u32;

pub(super) struct UnionFind {
	maxid: UnionFindId,
	data: Vec<UnionFindNode>,
}

/// Single-element node in UnionFind
#[derive(Debug)]
struct UnionFindNode {
	/// the parent of this node. If a node's parent is its own index,
	/// then it is a root.
	parent: u32,
	/// for the root of a connected component, the number of components
	/// connected to it. For intermediate values, it's not meaningful.
	size: u32,
}

impl UnionFind {
	/// Create new UnionFind with given capacity
	pub fn create(maxid: UnionFindId) -> Self {
		let mut result = UnionFind {
			maxid,
			data: Vec::with_capacity(maxid as usize),
		};

		for i in 0..=maxid {
			result.data.push(UnionFindNode {
				parent: i,
				size: 1,
			});
		}

		result
	}

	/// Get node for ID
	#[inline]
	fn get(&mut self, id: UnionFindId) -> &mut UnionFindNode {
		&mut self.data[id as usize]
	}

	// Get UnionFind group for id
	pub fn get_representative<'a>(&'a mut self, id: UnionFindId) -> UnionFindId {
		// chase down the root
		let root = {
			let mut root = id;
			loop {
				let parent = self.get(root).parent;
				if parent != root {
					root = parent;
				} else {
					break;
				}
			}
			root
		};
	
		// go back and collapse the tree.
		let mut current = id;
		loop {
			let parent = self.get(current).parent;
			if parent == root {
				break;
			}
			self.get(current).parent = root;
			current = parent;
		}
	
		root
	}

	pub(crate) fn get_set_size(&mut self, id: u32) -> u32 {
		let repid = self.get_representative(id);
		self.data[repid as usize].size
	}

	pub(crate) fn connect<'a>(&'a mut self, aid: u32, bid: u32) -> UnionFindId {
		let a_idx = self.get_representative(aid);
		let b_idx = self.get_representative(bid);
	
		if a_idx == b_idx {
			return a_idx;
		}
	
		// we don't perform "union by rank", but we perform a similar
		// operation (but probably without the same asymptotic guarantee):
		// We join trees based on the number of *elements* (as opposed to
		// rank) contained within each tree. I.e., we use size as a proxy
		// for rank.  In my testing, it's often *faster* to use size than
		// rank, perhaps because the rank of the tree isn't that critical
		// if there are very few nodes in it.
		// optimization idea: We could shortcut some or all of the tree
		// that is grafted onto the other tree. Pro: u32hose nodes were just
		// read and so are probably in cache. Con: it might end up being
		// wasted effort -- the tree might be grafted onto another tree in
		// a moment!
		let a_size = self.data[a_idx as usize].size;
		let b_size = self.data[b_idx as usize].size;

		if a_size > b_size {
			self.data[b_idx as usize].parent = a_idx as UnionFindId;
			self.data[a_idx as usize].size += b_size;
			// aroot
			a_idx
		} else {
			self.data[a_idx as usize].parent = b_idx as UnionFindId;
			self.data[b_idx as usize].size += a_size;
			b_idx
		}
	}
}

pub(crate) struct UnionFind2D {
	width: u32,
	#[cfg(debug_assertions)]
	height: u32,
	inner: UnionFind,
}

impl UnionFind2D {
	pub fn new(width: usize, height: usize) -> Self {
		let len = usize::checked_mul(width, height)
			.expect("Dimension overflow")
			.try_into()
			.expect("Dimension overflow");
		let width = width.try_into().unwrap();
		#[cfg(debug_assertions)]
		let height = height.try_into().unwrap();
		Self {
			width,
			#[cfg(debug_assertions)]
			height,
			inner: UnionFind::create(len)
		}
	}

	fn index_to_id<T>(&self, idx: (T, T)) -> UnionFindId where UnionFindId: TryFrom<T> {
		fn force_from<T>(src: T) -> UnionFindId where UnionFindId: TryFrom<T> {
			match UnionFindId::try_from(src) {
				Ok(v) => v,
				Err(_) => panic!("Index out-of-bounds")
			}
		}

		let x = force_from(idx.0);
		debug_assert!(x < self.width);
		let y = force_from(idx.1);
		debug_assert!(y < self.height);
		(y * self.width) + x
	}

	fn connect<T>(&mut self, a: (T, T), b: (T, T)) -> UnionFindId where UnionFindId: TryFrom<T> {
		let aid = self.index_to_id(a);
		let bid = self.index_to_id(b);
		self.inner.connect(aid, bid)
	}

	fn index_to_id_checked<T>(&self, idx: (T, T)) -> Result<UnionFindId, <UnionFindId as TryFrom<T>>::Error> where UnionFindId: TryFrom<T> {
		let x = UnionFindId::try_from(idx.0)?;
		debug_assert!(x < self.width);
		let y = UnionFindId::try_from(idx.1)?;
		debug_assert!(y < self.height);
		Ok((y * self.width) + x)
	}

	fn connect_checked<T>(&mut self, a: (T, T), b: (T, T)) -> Result<UnionFindId, <UnionFindId as TryFrom<T>>::Error> where UnionFindId: TryFrom<T> {
		let aid = self.index_to_id_checked(a)?;
		let bid = self.index_to_id_checked(b)?;
		let joined_id = self.inner.connect(aid, bid);
		Ok(joined_id)
	}

	pub fn get_representative<T>(&mut self, x: T, y: T) -> UnionFindId where UnionFindId: TryFrom<T> {
		let id = self.index_to_id((x, y));
		self.inner.get_representative(id)
	}

	pub fn get_set_size(&mut self, id: UnionFindId) -> u32 {
		self.inner.get_set_size(id)
	}
}


fn do_unionfind_first_line(uf: &mut UnionFind2D, im: &Image) {
	let w = im.width;
	for x in 1..(w-1) {
		let v0 = im[(x, 0)];
        if v0 == 127 {
            continue;
		}
		let v1 = im[(x - 1, 0)];
		if v0 == v1 {
			uf.connect((x, 0), (x - 1, 0));
		}
    }
}

fn do_unionfind_line2(uf: &mut UnionFind2D, im: &Image, y: usize) {
    assert!(y > 0);
	assert_eq!(im.width, uf.width as usize);
	debug_assert_eq!(im.height, uf.height as usize);

    let mut v_0_m1 = im[(0, y-1)];
    let mut v_1_m1 = im[(1, y-1)];
    let mut v = im[(0,y)];

	let w = im.width;

	for x in 1..(w - 1) {
        let v_m1_m1 = v_0_m1;
        v_0_m1 = v_1_m1;
        v_1_m1 = im[(x + 1, y - 1)];
        let v_m1_0 = v;
        v = im[(x, y)];

        if v == 127 {
			continue;
		}

        // (dx,dy) pairs for 8 connectivity:
        // (-1, -1)    (0, -1)    (1, -1)
        // (-1, 0)    (REFERENCE)
		if im[(x - 1, y)] == v {
			uf.connect((x, y), (x - 1, y));
		}

        if x == 1 || !((v_m1_0 == v_m1_m1) && (v_m1_m1 == v_0_m1)) {
			if im[(x, y - 1)] == v {
				uf.connect((x, y), (x, y - 1));
			}
        }

        if v == 255 {
            if x == 1 || !(v_m1_0 == v_m1_m1 || v_0_m1 == v_m1_m1) {
				if im[(x - 1, y - 1)] == v {
					uf.connect((x, y), (x - 1, y - 1));
				}
            }
            if v_0_m1 != v_1_m1 {
				if im[(x + 1, y - 1)] == v {
					uf.connect((x, y), (x + 1, y - 1));
				}
            }
        }
    }
}

pub(super) fn connected_components(td: &ApriltagDetector, threshim: &Image) -> UnionFind2D {
    let mut uf = UnionFind2D::new(threshim.width, threshim.height);

    if td.params.nthreads <= 1 {
        do_unionfind_first_line(&mut uf, threshim);
		for y in 1..threshim.height {
            do_unionfind_line2(&mut uf, threshim, y);
        }
    } else {
        do_unionfind_first_line(&mut uf, threshim);

        let chunksize = 1 + threshim.height / (APRILTAG_TASKS_PER_THREAD_TARGET * td.params.nthreads);
		td.wp.install(|| {
			(1..threshim.height).into_par_iter()
				.step_by(chunksize)
				.for_each(|i| {
					let y0 = i;
					let y1 = std::cmp::min(threshim.height, i + chunksize - 1);

					for y in y0..y1 {
						//TODO: how do we make UF parallel?
						do_unionfind_line2(&mut uf, threshim, y);
					}
				});
		});

        // XXX stitch together the different chunks.
		for i in (1..threshim.height).step_by(chunksize) {
			do_unionfind_line2(&mut uf, threshim, i);
		}
    }
    return uf;
}