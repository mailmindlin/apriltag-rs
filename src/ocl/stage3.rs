use crate::{util::image::ImageDimensions, DetectorConfig, quad_thresh::threshold::TILESZ};
use ocl::{Kernel as OclKernel, Error as OclError, prm::Uchar2};
use super::{OclBufferState, OclCore, OclStage};

pub(super) struct OclTileMinmax {}

impl OclTileMinmax {
	pub(super) fn new(_core: &OclCore) -> Self {
		Self {}
	}
}

impl OclStage for OclTileMinmax {
	type S = u8;
	type R = Uchar2;
	type K<'a> = OclKernel;
	fn result_dims(&self, src_dims: &ImageDimensions) -> ImageDimensions {
		// Two bytes
		let tw = src_dims.width.div_floor(TILESZ);
		let th = src_dims.height.div_floor(TILESZ);
		let ts = tw.next_multiple_of(16);
		ImageDimensions {
			width: tw,
			height: th,
			stride: ts,
		}
	}

	fn make_kernel(&self, core: &OclCore, src: &OclBufferState<Self::S>, dst: &OclBufferState<Self::R>) -> Result<OclKernel, OclError> {
		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_kernel.clone());

		builder.name("k03_tile_minmax");
		builder.arg(src.buf());
		builder.arg(src.stride() as u32);
		builder.arg(dst.buf());
		builder.arg(dst.stride() as u32);

		builder.global_work_size((dst.width(), dst.height()));
		builder.build()
	}
}

pub(super) struct OclTileBlur {}

impl OclTileBlur {
	pub(super) fn new(_core: &OclCore) -> Self {
		Self {}
	}
}

impl OclStage for OclTileBlur {
	type S = Uchar2;
	type R = Uchar2;
	type K<'a> = OclKernel;
	fn result_dims(&self, src_dims: &ImageDimensions) -> ImageDimensions {
		*src_dims
	}

	fn make_kernel(&self, core: &OclCore, src: &OclBufferState<Self::S>, dst: &OclBufferState<Self::R>) -> Result<OclKernel, OclError> {
		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_kernel.clone());

		builder.name("k03_tile_blur");
		builder.arg(src.buf());
		builder.arg(src.stride() as u32);
		builder.arg(src.width() as u32);
		builder.arg(src.height() as u32);
		builder.arg(dst.buf());
		builder.arg(dst.stride() as u32);

		builder.global_work_size((src.width(), src.height()));
		builder.build()
	}
}

pub(super) struct OclThreshold {
	min_white_black_diff: u8,
}

impl OclThreshold {
	pub(super) fn new(_core: &OclCore, config: &DetectorConfig) -> Self {
		Self {
			min_white_black_diff: config.qtp.min_white_black_diff,
		}
	}

	pub(super) fn result_dims(&self, src_dims: &ImageDimensions) -> ImageDimensions {
		ImageDimensions { width: src_dims.width, height: src_dims.height, stride: src_dims.width.next_multiple_of(16) }
	}

	pub(super) fn make_kernel(&self, core: &OclCore, src: &OclBufferState<u8>, src_tiled: &OclBufferState<Uchar2>, dst: &OclBufferState<u8>) -> Result<OclKernel, OclError> {
		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_kernel.clone());

		builder.name("k03_build_threshim");
		builder.arg(src.buf());
		builder.arg(src.stride() as u32);
		builder.arg(src.width() as u32);
		builder.arg(src.height() as u32);
		builder.arg(src_tiled.buf());
		builder.arg(src_tiled.stride() as u32);
		builder.arg(self.min_white_black_diff as u8);
		builder.arg(dst.buf());
		builder.arg(dst.stride() as u32);
		
		builder.global_work_size((src.width().div_ceil(TILESZ), src.height().div_ceil(TILESZ)));
		// builder.local_work_size((8, 8));
		builder.build()
	}
}