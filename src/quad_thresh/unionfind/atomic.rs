use std::sync::atomic::{AtomicU32, Ordering};

use super::{UnionFind, UnionFindId, UnionFindAtomic, UnionFindStatic};

pub(super) struct UnionFindConcurrent {
	pub(super) data: Vec<Entry>,
}

const ORDER: Ordering = Ordering::AcqRel;
const ORDER_L: Ordering = Ordering::Acquire;

/// Single-element node in UnionFind
#[derive(Debug)]
pub(super) struct Entry {
	/// the parent of this node. If a node's parent is its own index,
	/// then it is a root.
	pub(super) parent: AtomicU32,
	/// for the root of a connected component, the number of components
	/// connected to it. For intermediate values, it's not meaningful.
	pub(super) size: AtomicU32,
}

impl Entry {
	pub(super) fn new(id: u32) -> Self {
		Self::with_size(id, 1)
	}
	fn with_size(id: u32, rank: u32) -> Self {
		Self {
			parent: AtomicU32::new(id),
			size: AtomicU32::new(rank),
		}
	}
}


impl UnionFind<UnionFindId> for UnionFindConcurrent {
    #[inline(always)]
    fn get_set(&mut self, element: UnionFindId) -> (UnionFindId, u32) {
        let repid = self.get_representative_mut(element);
        let size = *self.data[repid as usize].size.get_mut();

        (repid, size)
    }

	#[inline(always)]
	fn connect(&mut self, a: UnionFindId, b: UnionFindId) -> bool {
		self.connect_ids(a, b)
	}

    // #[inline(always)]
    fn connect_ids(&mut self, a: UnionFindId, b: UnionFindId) -> bool {
        let a = self.get_representative_mut(a);
		let b = self.get_representative_mut(b);

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
		let a_size = *self.data[a as usize].size.get_mut();
		let b_size = *self.data[b as usize].size.get_mut();

		if a_size > b_size {
			*self.data[b as usize].parent.get_mut() = a;
			*self.data[a as usize].size.get_mut() += b_size;
		} else {
			*self.data[a as usize].parent.get_mut() = b;
			*self.data[b as usize].size.get_mut() += a_size;
		}
		true
    }

    #[inline(always)]
    fn index_to_id(&self, idx: UnionFindId) -> UnionFindId {
        idx
    }
}

impl UnionFindStatic<u32> for UnionFindConcurrent {
    fn get_set_static(&self, index: u32) -> (UnionFindId, u32) {
        let repid = self.get_representative(index);
        let size = self.get(repid).size.load(ORDER_L);
        (repid, size)
    }
}

impl UnionFindAtomic<u32> for UnionFindConcurrent {
    fn connect_ids_atomic(&self, mut a: UnionFindId, mut b: UnionFindId) -> bool {
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
			let a_size = self.data[a as usize].size.load(ORDER_L);
			let b_size = self.data[b as usize].size.load(ORDER_L);

			if a_size > b_size {
				if self.change_parent(b, b, a) {
					self.data[a as usize].size.fetch_add(b_size, ORDER);
					return true;
				}
			} else {
				if self.change_parent(a, a, b) {
					self.data[b as usize].size.fetch_add(a_size, ORDER);
					return true;
				}
			}
		}
    }
}


impl UnionFindConcurrent {
	/// Create new UnionFind with given capacity
	pub fn new(maxid: UnionFindId) -> Self {
		let mut result = UnionFindConcurrent {
			data: Vec::with_capacity(maxid as usize),
		};

		for i in 0..=maxid {
			result.data.push(Entry::new(i));
		}

		result
	}

    // Get UnionFind group for id
	fn get_representative(&self, mut element: UnionFindId) -> UnionFindId {
		// chase down the root
		let mut parent = self.parent(element);
		while element != parent {
			let grandparent = self.parent(parent);
			match self.get(element).parent.compare_exchange(parent, grandparent, ORDER, ORDER_L) {
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

    fn get_representative_mut(&mut self, mut element: UnionFindId) -> UnionFindId {
        // chase down the root
		let mut root = *self.parent_mut(element);
        if root != element {
            loop {
                let grandparent = *self.parent_mut(root);
                if grandparent == root {
                    break;
                }
                root = grandparent;
            }
    
            while element != root {
                element = std::mem::replace(self.parent_mut(element), root);
            }
        }
        root
    }

	#[inline]
	fn parent(&self, id: UnionFindId) -> UnionFindId {
		self.data[id as usize].parent.load(ORDER_L)
	}

    #[inline(always)]
    fn parent_mut(&mut self, id: UnionFindId) -> &mut UnionFindId {
        self.data[id as usize].parent.get_mut()
    }

	/// Get node for ID
	#[inline]
	fn get(&self, id: UnionFindId) -> &Entry {
		&self.data[id as usize]
	}

	fn change_parent(&self, element: UnionFindId, cur_parent: UnionFindId, next_parent: UnionFindId) -> bool {
		self.get(element).parent.compare_exchange(cur_parent, next_parent, ORDER, Ordering::Acquire).is_ok()
	}
}