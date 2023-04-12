use std::sync::atomic::{AtomicU32, Ordering::SeqCst};

use crate::{ApriltagDetector, util::{Image, image::Luma}};

pub(super) type UnionFindId = u32;

pub(super) struct UnionFind {
	data: Vec<Entry>,
}

/// Single-element node in UnionFind
#[derive(Debug)]
struct Entry {
	/// the parent of this node. If a node's parent is its own index,
	/// then it is a root.
	parent: AtomicU32,
	/// for the root of a connected component, the number of components
	/// connected to it. For intermediate values, it's not meaningful.
	size: AtomicU32,
}

impl Entry {
	fn new(id: u32) -> Self {
		Self::with_size(id, 1)
	}
	fn with_size(id: u32, rank: u32) -> Self {
		Self {
			parent: AtomicU32::new(id),
			size: AtomicU32::new(rank),
		}
	}
}

impl UnionFind {
	/// Create new UnionFind with given capacity
	pub fn create(maxid: UnionFindId) -> Self {
		let mut result = UnionFind {
			data: Vec::with_capacity(maxid as usize),
		};

		for i in 0..=maxid {
			result.data.push(Entry::new(i));
		}

		result
	}

	#[inline]
	fn parent(&self, id: UnionFindId) -> UnionFindId {
		self.data[id as usize].parent.load(SeqCst)
	}

	/// Get node for ID
	#[inline]
	fn get(&self, id: UnionFindId) -> &Entry {
		&self.data[id as usize]
	}

	// Get UnionFind group for id
	pub fn get_representative(&self, mut element: UnionFindId) -> UnionFindId {
		// chase down the root
		let mut parent = self.parent(element);
		while element != parent {
			let grandparent = self.parent(parent);
			match self.data[element as usize].parent.compare_exchange(parent, grandparent, SeqCst, SeqCst) {
				Ok(_) => {
					element = parent;
					parent = grandparent;
				},
				Err(new_parent) => {
					parent = new_parent;
				}
			}
		}
		element
	}

	fn change_parent(&self, element: UnionFindId, cur_parent: UnionFindId, next_parent: UnionFindId) -> bool {
		self.data[element as usize].parent.compare_exchange(cur_parent, next_parent, SeqCst, SeqCst).is_ok()
	}

	pub(crate) fn get_set_size(&self, id: UnionFindId) -> u32 {
		let repid = self.get_representative(id);
		self.data[repid as usize].size.load(SeqCst)
	}

	pub(crate) fn connect(&self, mut a: u32, mut b: u32) -> bool {
		loop {
			a = self.get_representative(a);
			b = self.get_representative(b);

			if a == b {
				return false;
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
			let a_size = self.data[a as usize].size.load(SeqCst);
			let b_size = self.data[b as usize].size.load(SeqCst);

			if a_size > b_size {
				if self.change_parent(b, b, a) {
					self.data[a as usize].size.fetch_add(b_size, SeqCst);
					return true;
				}
			} else {
				if self.change_parent(a, a, b) {
					self.data[b as usize].size.fetch_add(a_size, SeqCst);
					return true;
				}
			}
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
		#[cfg(debug_assertions)]
		debug_assert!(y < self.height);
		(y * self.width) + x
	}

	fn connect<T>(&self, a: (T, T), b: (T, T)) -> bool where UnionFindId: TryFrom<T> {
		let aid = self.index_to_id(a);
		let bid = self.index_to_id(b);
		self.inner.connect(aid, bid)
	}

	fn index_to_id_checked<T>(&self, idx: (T, T)) -> Result<UnionFindId, <UnionFindId as TryFrom<T>>::Error> where UnionFindId: TryFrom<T> {
		let x = UnionFindId::try_from(idx.0)?;
		debug_assert!(x < self.width);
		let y = UnionFindId::try_from(idx.1)?;
		#[cfg(debug_assertions)]
		debug_assert!(y < self.height);
		Ok((y * self.width) + x)
	}

	fn connect_checked<T>(&self, a: (T, T), b: (T, T)) -> Result<bool, <UnionFindId as TryFrom<T>>::Error> where UnionFindId: TryFrom<T> {
		let aid = self.index_to_id_checked(a)?;
		let bid = self.index_to_id_checked(b)?;
		let joined_id = self.inner.connect(aid, bid);
		Ok(joined_id)
	}

	pub fn get_representative<T>(&self, x: T, y: T) -> UnionFindId where UnionFindId: TryFrom<T> {
		let id = self.index_to_id((x, y));
		self.inner.get_representative(id)
	}

	pub fn get_set_size(&self, id: UnionFindId) -> u32 {
		self.inner.get_set_size(id)
	}
}


fn do_unionfind_line2(uf: &mut UnionFind2D, im: &impl Image<Luma<u8>>, y: usize) {
    assert!(y > 0);
	assert_eq!(im.width(), uf.width as usize);
	#[cfg(debug_assertions)]
	debug_assert_eq!(im.height(), uf.height as usize);

    let mut v_0_m1 = im[(0, y-1)];
    let mut v_1_m1 = im[(1, y-1)];
    let mut v = im[(0,y)];

	let w = im.width();

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

pub(super) fn connected_components(_td: &ApriltagDetector, threshim: &impl Image<Luma<u8>>) -> UnionFind2D {
    let mut uf = UnionFind2D::new(threshim.width(), threshim.height());

	fn do_unionfind_first_line(uf: &mut UnionFind2D, im: &impl Image<Luma<u8>>) {
		for x in 1..(im.width()-1) {
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
	do_unionfind_first_line(&mut uf, threshim);
	//TODO: parallelism
	for y in 1..threshim.height() {
		do_unionfind_line2(&mut uf, threshim, y);
	}

    /*if td.params.nthreads <= 1 {
		for y in 1..threshim.height {
            do_unionfind_line2(&mut uf, threshim, y);
        }
    } else {
        let chunksize = 1 + threshim.height / (APRILTAG_TASKS_PER_THREAD_TARGET * td.params.nthreads);
		td.wp.install(|| {
			// each task will process [y0, y1). Note that this attaches
            // each cell to the right and down, so row y1 *is* potentially modified.
            //
            // for parallelization, make sure that each task doesn't touch rows
            // used by another thread.
			(1..threshim.height).into_par_iter()
				.step_by(chunksize)
				.fold(|| UnionFind2D::new(threshim.width, threshim.height), |acc, i| {
					let y0 = i;
					let y1 = std::cmp::min(threshim.height, i + chunksize - 1);
					for y in y0..y1 {
						//TODO: how do we make UF parallel?
						do_unionfind_line2(&mut uf, threshim, y);
					}
					(y0, y1, )
				})
		});

        // XXX stitch together the different chunks.
		for i in (1..threshim.height).step_by(chunksize) {
			do_unionfind_line2(&mut uf, threshim, i - 1);
		}
    }*/
    return uf;
}