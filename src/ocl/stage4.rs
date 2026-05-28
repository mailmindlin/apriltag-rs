use crate::{util::image::ImageDimensions, quad_thresh::unionfind::{UnionFindStatic, UnionFind}};
use ocl::{Kernel as OclKernel, Error as OclError};
use super::{OclBufferState, OclCore, buffer::OclBufferMapped};

pub(super) struct OclUnionFindInit {
}

impl OclUnionFindInit {
	pub(super) fn new(_core: &OclCore) -> Self {
		Self {
		}
	}

	pub(super) fn result_dims(&self, src_dims: &ImageDimensions) -> ImageDimensions {
		let uf_width = src_dims.width * 2;
		ImageDimensions {
			width: uf_width,
			height: src_dims.height,
			stride: uf_width.next_multiple_of(32),
		}
	}

	pub(super) fn make_kernel(&self, core: &OclCore, dims: &ImageDimensions, dst: &OclBufferMapped<u32>) -> Result<OclKernel, OclError> {
		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_kernel.clone());

		builder.name("k04_unionfind_init");
		builder.arg(dst.buf());
		builder.arg(dims.width as u32);

		builder.global_work_size((dims.width, dims.height));
		builder.build()
	}
}

pub(super) struct OclConnectedComponents {
}

impl OclConnectedComponents {
	pub(super) fn new(_core: &OclCore) -> Self {
		Self {
		}
	}

	pub(super) fn make_kernel(&self, core: &OclCore, src: &OclBufferState<u8>, uf: &OclBufferMapped<u32>, temp: &OclBufferState<u32>) -> Result<OclKernel, OclError> {
		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_kernel.clone());

		builder.name("k04_connected_components_row");
		builder.arg(src.buf());
		builder.arg(src.stride() as u32);
		builder.arg(src.width() as u32);
		builder.arg(src.height() as u32);
		builder.arg(uf.buf());
		builder.arg(temp.buf());

		// builder.global_work_size(((src.width() - 2).next_multiple_of(1), src.height().next_multiple_of(1)));
		// builder.global_work_size(((src.width() - 2).next_multiple_of(16), src.height().next_multiple_of(16)));
		builder.global_work_size(src.height());
		let kernel = builder.build().expect("sdf");
		Ok(kernel)
	}
}

pub(super) struct OclUfFlatten {
}

impl OclUfFlatten {
	pub(super) fn new(_core: &OclCore) -> Self {
		Self {
		}
	}

	pub(super) fn make_kernel(&self, core: &OclCore, uf: &OclBufferMapped<u32>) -> Result<OclKernel, OclError> {
		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_init.clone());

		builder.name("k04_unionfind_flatten");
		builder.arg(uf.buf());
		builder.arg(uf.width() as u32 / 2);

		builder.global_work_size((uf.width() / 2, uf.height()));
		let kernel = builder.build().expect("sdf");
		Ok(kernel)
	}
}

pub(super) struct WrappedUnionFind {
	pub(super) data: Box<[u32]>,
}

impl WrappedUnionFind {
	fn parent(&self, idx: u32) -> u32 {
		self.data[idx as usize * 2 + 0]
	}
	fn set_size(&self, idx: u32) -> u32 {
		self.data[idx as usize * 2 + 1]
	}
}

impl UnionFind<u32> for WrappedUnionFind {
	type Id = u32;

	fn get_set(&mut self, index: u32) -> (Self::Id, u32) {
		self.get_set_static(index)
	}

	fn index_to_id(&self, idx: u32) -> Self::Id {
		idx
	}

	fn connect_ids(&mut self, _a: Self::Id, _b: Self::Id) -> bool {
		panic!("Not supported")
	}
}

impl UnionFindStatic<u32> for WrappedUnionFind {
	fn get_set_static(&self, index: u32) -> (u32, u32) {
		// We can detect cycles by tracking the lowest index encountered
		let mut lc = u32::MAX;

		let mut current = index;
		let mut hops = 0;
		loop {
			let parent = self.parent(current);
			
			if parent == current || hops > 1000 {
				return (current, self.set_size(current));
			}
			lc = std::cmp::min(current, lc);
			current = parent;
			hops += 1;
		}
	}

	fn get_set_hops(&self, index: u32) -> usize {
		// We can detect cycles by tracking the lowest index encountered
		let mut lc = u32::MAX;

		let mut current = index;
		let mut hops = 0;
		loop {
			let parent = self.parent(current);
			#[cfg(feature="debug")]
			if hops > 990 {
				println!("{parent} {} {lc}", self.set_size(current));
			}
			// debug_assert!(hops < 1000);
			if hops > 10000 {
				println!();
				return usize::MAX;
			}
			if parent == current || lc == current {
				return hops;
			}
			lc = std::cmp::min(current, lc);
			current = parent;
			hops += 1;
		}
	}
}