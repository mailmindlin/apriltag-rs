use crate::{ApriltagDetector, util::Image, quad_thresh::APRILTAG_TASKS_PER_THREAD_TARGET};
use rayon::prelude::*;

pub(super) struct UnionFind {
	maxid: u32,
	data: Vec<UnionFindNode>,
}

struct UnionFindNode {
	/// the parent of this node. If a node's parent is its own index,
	/// then it is a root.
	parent: u32,
	/// for the root of a connected component, the number of components
	/// connected to it. For intermediate values, it's not meaningful.
	size: u32,
}

impl UnionFind {
	pub fn create(maxid: u32) -> Self {
		let result = UnionFind {
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

	#[inline]
	fn get(&self, id: u32) -> &mut UnionFindNode {
		&mut self.data[id as usize]
	}

	pub fn get_representative(&mut self, id: u32) -> u32 {
		let mut root = id;
	
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
		self.get(repid).size
	}

	pub(crate) fn connect(&mut self, aid: u32, bid: u32) -> u32 {
		let aroot = self.get_representative(aid);
		let broot = self.get_representative(bid);
	
		if aroot == broot {
			return aroot;
		}
	
		// we don't perform "union by rank", but we perform a similar
		// operation (but probably without the same asymptotic guarantee):
		// We join trees based on the number of *elements* (as opposed to
		// rank) contained within each tree. I.e., we use size as a proxy
		// for rank.  In my testing, it's often *faster* to use size than
		// rank, perhaps because the rank of the tree isn't that critical
		// if there are very few nodes in it.
		let asize = self.get(aroot).size;
		let bsize = self.get(broot).size;
	
		// optimization idea: We could shortcut some or all of the tree
		// that is grafted onto the other tree. Pro: u32hose nodes were just
		// read and so are probably in cache. Con: it might end up being
		// wasted effort -- the tree might be grafted onto another tree in
		// a moment!
		if asize > bsize {
			self.get(broot).parent = aroot;
			self.get(aroot).size += bsize;
			aroot
		} else {
			self.get(aroot).parent = broot;
			self.get(broot).size += asize;
			broot
		}
	}
}


fn do_unionfind_first_line(uf: &mut UnionFind, im: &Image) {
	let w = im.width;
	for x in 1..(w-1) {
		let v = im[(x, 0)];
        if v == 127 {
            continue;
		}

		if im[(x - 1, 0)] == v {
			uf.connect(x as u32, x as u32 - 1);
		}
    }
}

fn do_unionfind_line2(uf: &mut UnionFind, im: &Image, y: usize) {
    assert!(y > 0);

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
			uf.connect((y*w + x) as u32, (y*w + x - 1) as u32);
		}

        if x == 1 || !((v_m1_0 == v_m1_m1) && (v_m1_m1 == v_0_m1)) {
			if im[(x, y - 1)] == v {
				uf.connect((y*w + x) as u32, ((y - 1)*w + x) as u32);
			}
        }

        if v == 255 {
            if x == 1 || !(v_m1_0 == v_m1_m1 || v_0_m1 == v_m1_m1) {
				if im[(x - 1, y - 1)] == v {
					uf.connect((y*w + x) as u32, ((y - 1)*w + x - 1) as u32);
				}
            }
            if !(v_0_m1 == v_1_m1) {
				if im[(x + 1, y - 1)] == v {
					uf.connect((y*w + x) as u32, ((y - 1)*w + x + 1) as u32);
				}
            }
        }
    }
}

pub(super) fn connected_components(td: &ApriltagDetector, threshim: &Image) -> UnionFind {
	let ts = threshim.stride;
    let mut uf = UnionFind::create(threshim.len() as u32);

    if td.params.nthreads <= 1 {
        do_unionfind_first_line(&mut uf, threshim);
		for y in 1..threshim.height {
            do_unionfind_line2(&mut uf, threshim, y);
        }
    } else {
        do_unionfind_first_line(&mut uf, threshim);

        let chunksize = 1 + threshim.height / (APRILTAG_TASKS_PER_THREAD_TARGET * td.params.nthreads);
		td.wp.scope(|sc| {
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