use ocl::{Kernel as OclKernel, Error as OclError, Buffer as OclBuffer, SpatialDims};

use crate::{util::{image::ImageDimensions, pool::{KeyedPool, PoolGuard}}, detector::config::QuadDecimateMode};

use super::{OclBufferState, OclCore, OclStage};

pub(super) struct OclQuadDecimate {
	mode: QuadDecimateMode,
	kcache: KeyedPool<(), OclKernel>,
	pub(super) download: bool,
}

impl OclQuadDecimate {
	pub(super) fn new(core: &OclCore, qdm: Option<QuadDecimateMode>, download: bool) -> Option<Self> {
		let mode = qdm?;
		let kcache = KeyedPool::default();

		let kernel = {
			let mut builder = OclKernel::builder();
			builder.program(&core.program);
			builder.queue(core.queue_kernel.clone());
	
			builder.arg_named("src", None::<&OclBuffer<u8>>);
			builder.arg_named("src_stride", 0u32);
			builder.arg_named("src_width", 0u32);
			builder.arg_named("dst", None::<&OclBuffer<u8>>);
			builder.arg_named("dst_stride", 0u32);
			match mode {
				QuadDecimateMode::ThreeHalves => {
					builder.name("k01_filter_quad_decimate_32");
				},
				QuadDecimateMode::Scaled(factor) => {
					builder.name("k01_filter_quad_decimate");
					builder.arg(factor.get() as u32);
				},
			}
			builder.build().unwrap()
		};

		kcache.offer((), kernel);
		Some(Self {
			mode,
			kcache,
			download,
		})
	}
}

impl OclStage for OclQuadDecimate {
	type S = u8;
	type R = u8;
	type K<'a> = PoolGuard<'a, (), OclKernel>;
	fn result_dims(&self, src_dims: &ImageDimensions) -> ImageDimensions {
		let (width, height) = match self.mode {
			QuadDecimateMode::ThreeHalves => {
				let swidth = src_dims.width / 3 * 2;
				let sheight = src_dims.height / 3 * 2;
				assert_eq!(swidth % 2, 0, "Input dimension must be multiple of two");
				assert_eq!(sheight % 2, 0, "Input dimension must be multiple of two");
				(swidth, sheight)
			},
			QuadDecimateMode::Scaled(factor) => {
				let factor = factor.get() as usize;

				let swidth = 1 + (src_dims.width - 1) / factor;
				let sheight = 1 + (src_dims.height - 1) / factor;
				assert!((swidth - 1) * factor <= src_dims.width);
				assert!((sheight - 1) * factor <= src_dims.height);
				(swidth, sheight)
			},
		};
		// Make output stride a multiple of 16 so we can set the local dims
		let stride = width.next_multiple_of(if width >= 256 { 32 } else { 16 });
		ImageDimensions { width, height, stride }
	}

	fn make_kernel<'a>(&'a self, core: &OclCore, src: &OclBufferState<Self::S>, dst: &OclBufferState<Self::R>) -> Result<Self::K<'a>, OclError> {
		assert!(dst.dims.stride % 16 == 0, "Output stride must be multiple of 16");

		let gws: SpatialDims = match self.mode {
			QuadDecimateMode::ThreeHalves => {
				(dst.width() / 2, dst.height() / 2)
			},
			QuadDecimateMode::Scaled(_) => {
				(dst.width(), dst.height())
			},
		}.into();
		
		let key = ();
		let kernel = match self.kcache.try_borrow(key) {
			Some(mut cached) => {
				cached.set_arg("src", src.buf())?;
				cached.set_arg("src_stride", src.stride() as u32)?;
				cached.set_arg("src_width", src.width() as u32)?;
				cached.set_arg("dst", dst.buf())?;
				cached.set_arg("dst_stride", dst.stride() as u32)?;
				cached.set_default_global_work_size(gws);
				// cached.set_default_local_work_size((16, 1).into());
				cached
			},
			None => {
				let mut builder = OclKernel::builder();
				builder.program(&core.program);
				builder.queue(core.queue_kernel.clone());

				builder.arg_named("src", src.buf());
				builder.arg_named("src_stride", src.stride() as u32);
				builder.arg_named("src_width", src.width() as u32);
				builder.arg_named("dst", dst.buf());
				builder.arg_named("dst_stride", dst.stride() as u32);
				
				builder.global_work_size(gws);
				let lws_x = if dst.stride() % 64 == 0 {
					64
				} else if dst.stride() % 32 == 0 {
					32
				} else {
					16
				};
				builder.local_work_size((lws_x, 1));

				match self.mode {
					QuadDecimateMode::ThreeHalves => {
						builder.name("k01_filter_quad_decimate_32");
						
					},
					QuadDecimateMode::Scaled(factor) => {
						builder.name("k01_filter_quad_decimate");
						builder.arg(factor.get() as u32);
					},
				}

				let kernel = builder.build()?;
				self.kcache.offer(key, kernel)
			}
		};
	
		Ok(kernel)
	}
}