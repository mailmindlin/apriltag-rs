use super::{UnionFind, UnionFindAtomic, atomic::UnionFindConcurrent, UnionFindStatic, reference::UnionFindReference};

type UnionFindId = u32;

pub(super) struct UnionFind2D<U: UnionFind<UnionFindId>> {
	width: u32,
	#[cfg(debug_assertions)]
	height: u32,
	inner: U,
}

impl<U: UnionFind<u32>> UnionFind<(u32, u32)> for UnionFind2D<U> {
    fn index_to_id(&self, (x, y): (u32, u32)) -> UnionFindId {
		#[cfg(debug_assertions)]
		debug_assert!(y < self.height);
		(y * self.width) + x
	}

    fn get_set(&mut self, index: (u32, u32)) -> (UnionFindId, u32) {
		let id = self.index_to_id(index);
		self.inner.get_set(id)
    }

    fn connect(&mut self, a: (u32, u32), b: (u32, u32)) -> bool {
        let a_id = self.index_to_id(a);
        let b_id = self.index_to_id(b);
        self.inner.connect(a_id, b_id)
    }

	#[inline(always)]
	fn connect_ids(&mut self, a: super::UnionFindId, b: super::UnionFindId) -> bool {
		self.inner.connect(a, b)
	}
}

impl<U: UnionFindStatic<u32>> UnionFindStatic<(u32, u32)> for UnionFind2D<U> {
    fn get_set_static(&self, index: (u32, u32)) -> (super::UnionFindId, u32) {
		let id = self.index_to_id(index);
        self.inner.get_set_static(id)
    }

    fn get_set_hops(&self, index: (u32, u32)) -> usize {
		let id = self.index_to_id(index);
        self.inner.get_set_hops(id)
    }
}

impl<U: UnionFindAtomic<u32>> UnionFindAtomic<(u32, u32)> for UnionFind2D<U> {
    fn connect_ids_atomic(&self, a: UnionFindId, b: UnionFindId) -> bool {
        self.inner.connect_atomic(a, b)
    }
}

impl UnionFind2D<UnionFindReference> {
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
			inner: UnionFindReference::new(len)
		}
	}

	pub fn from_concurrent(src: UnionFind2D<UnionFindConcurrent>) -> Self {
		Self {
			width: src.width,
			#[cfg(debug_assertions)]
			height: src.height,
			inner: UnionFindReference::from_concurrent(src.inner),
		}
	}
}

impl UnionFind2D<UnionFindConcurrent> {
    pub fn new_concurrent(width: usize, height: usize) -> Self {
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
			inner: UnionFindConcurrent::new(len)
		}
	}
}