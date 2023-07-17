use std::sync::atomic::AtomicU32;

use raw_parts::RawParts;

use super::{UnionFind, UnionFindId, atomic::UnionFindConcurrent, UnionFindStatic};

pub(super) struct UnionFindReference {
	data: Box<[Entry]>,
}

/// Single-element node in UnionFind
#[derive(Debug)]
struct Entry {
	/// the parent of this node. If a node's parent is its own index,
	/// then it is a root.
	pub(super) parent: u32,
	/// for the root of a connected component, the number of components
	/// connected to it. For intermediate values, it's not meaningful.
	pub(super) size: u32,
}

impl Entry {
	fn new(id: u32) -> Self {
		Self::with_size(id, 1)
	}
	fn with_size(id: u32, rank: u32) -> Self {
		Self {
			parent: id,
			size: rank,
		}
	}
}

impl UnionFindStatic<UnionFindId> for UnionFindReference {
    fn get_set_static(&self, index: UnionFindId) -> (UnionFindId, u32) {
        let repid = self.find_representative(index);
		let size = self.data[repid as usize].size;
		(repid, size)
    }
}

impl UnionFind<UnionFindId> for UnionFindReference {
    fn get_set(&mut self, index: UnionFindId) -> (UnionFindId, u32) {
        let repid = self.get_representative(index);
		let size = self.data[repid as usize].size;
		(repid, size)
    }

	#[inline(always)]
	fn connect(&mut self, a: UnionFindId, b: UnionFindId) -> bool {
		self.connect_ids(a, b)
	}

    fn connect_ids(&mut self, a: UnionFindId, b: UnionFindId) -> bool {
		let a = self.get_representative(a);
		let b = self.get_representative(b);

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
		let a_size = self.data[a as usize].size;
		let b_size = self.data[b as usize].size;

		if a_size > b_size {
			self.data[b as usize].parent = a;
			self.data[a as usize].size += b_size;
		} else {
			self.data[a as usize].parent = b;
			self.data[b as usize].size += a_size;
		}
		true
    }

	#[inline(always)]
    fn index_to_id(&self, idx: UnionFindId) -> UnionFindId {
        idx
    }
}


impl UnionFindReference {
	pub fn from_concurrent(src: UnionFindConcurrent) -> Self {
		use super::atomic::Entry as AtomicEntry;
		// Check that the layouts of Entry and AtomicEntry are the same
		{
			assert_eq!(std::mem::size_of::<AtomicEntry>(), std::mem::size_of::<Entry>());
			assert_eq!(std::mem::align_of::<AtomicEntry>(), std::mem::align_of::<Entry>());
			
			let E_A: AtomicEntry = AtomicEntry { parent: AtomicU32::new(0), size: AtomicU32::new(0) };
			let E_S: Entry = Entry { parent: 0, size: 0 };
			fn ptrdif<T, U>(a: *const T, b: *const U) -> isize {
				unsafe { a.byte_offset_from(b) }
			}
			assert_eq!(ptrdif(&E_A, &E_A.parent), ptrdif(&E_S, &E_S.parent), "Parent align mismatch");
			assert_eq!(ptrdif(&E_A, &E_A.size), ptrdif(&E_S, &E_S.size));
		}

		let RawParts { ptr, length, capacity } = RawParts::from_vec(src.data);
		let ptr: *mut Entry = unsafe { std::mem::transmute(ptr) };
		Self {
			data: unsafe { Vec::from_raw_parts(ptr, length, capacity) }.into_boxed_slice()
		}
	}
	/// Create new UnionFind with given capacity
	pub fn new(maxid: UnionFindId) -> Self {
		let mut data = Vec::with_capacity(maxid as usize);
		for i in 0..=maxid {
			data.push(Entry::new(i));
		}

		Self { data: data.into_boxed_slice() }
	}


	fn find_representative(&self, element: UnionFindId) -> UnionFindId {
		// chase down the root
		let mut root = element;
		while self.data[root as usize].parent != root {
			root = self.data[root as usize].parent;
		}
		root
	}

    // Get UnionFind group for id
	fn get_representative(&mut self, mut element: UnionFindId) -> UnionFindId {
		// chase down the root
		let root = self.find_representative(element);
	
		// go back and collapse the tree.
		while self.data[element as usize].parent != root {
			element = std::mem::replace(&mut self.data[element as usize].parent, root);
		}
		root
	}
}