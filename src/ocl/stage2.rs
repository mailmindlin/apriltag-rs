use ocl::{Kernel as OclKernel, Error as OclError, Buffer as OclBuffer, MemFlags, builders::KernelBuilder};

use crate::{util::image::ImageDimensions, detector::quad_sigma_kernel};

use super::{OclBufferState, OclCore, OclStage};

pub(super) struct OclQuadSigma {
	quad_sigma: f32,
	filter: OclBuffer<u8>,
	pub(super) download: bool,
	// kbuilder: RwLock<KernelBuilder<'static>>,
}

impl OclQuadSigma {
	pub(super) fn new(core: &OclCore, quad_sigma: f32, download: bool) -> Option<Self> {
		// Upload quad_sigma filter
		let quad_sigma_kernel = quad_sigma_kernel(quad_sigma)?;
		let cl_filter = OclBuffer::builder()
			.queue(core.queue_write.clone())
			.flags(MemFlags::new().host_write_only().read_only())
			.len(quad_sigma_kernel.len())
			.copy_host_slice(&quad_sigma_kernel)
			.build()
			.unwrap();
		Some(Self {
			quad_sigma,
			filter: cl_filter,
			download,
			// kbuilder: Self::kernel_builder(core, quad_sigma, &cl_filter),
		})
	}

	fn kernel_builder<'a>(core: &'a OclCore, quad_sigma: f32, filter: &'a OclBuffer<u8>) -> KernelBuilder<'a> {
		// Make quad_sigma kernel
		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_kernel.clone());

		builder.arg_named("src", None::<&OclBuffer<u8>>);
		builder.arg_named("src_stride", 0u32);
		builder.arg_named("src_width", 0u32);
		builder.arg_named("src_height", 0u32);
		builder.arg(filter);                 // filter
		builder.arg(filter.len() as u32); // ksz
		builder.arg_named("dst", None::<&OclBuffer<u8>>);
		builder.arg_named("dst_stride", 0u32);

		builder.global_work_size((1, 1));

		if quad_sigma > 0. {
			builder.name("k02_gaussian_blur_filter");
		} else {
			builder.name("k02_gaussian_sharp_filter");
		}
		builder
	}
}

impl OclStage for OclQuadSigma {
	type S = u8;
	type R = u8;
	type K<'a> = OclKernel;
	fn result_dims(&self, src_dims: &ImageDimensions) -> ImageDimensions {
		*src_dims
	}

	fn make_kernel(&self, core: &OclCore, src: &OclBufferState<Self::S>, dst: &OclBufferState<Self::R>) -> Result<OclKernel, OclError> {
		// Make quad_sigma kernel
		// let mut kernel = Self::kernel_builder(core, self.quad_sigma, &self.filter).build()?;

		// kernel.set_arg("src", src.buf())?;                   // im_src
		// kernel.set_arg("src_stride", src.stride() as u32)?;    // stride_src
		// kernel.set_arg("src_width", src.width() as u32)?;     // width_src
		// kernel.set_arg("src_height", src.height() as u32)?;    // height_src
		// kernel.set_arg("dst", dst.buf())?;                        // im_dst
		// kernel.set_arg("dst_stride", dst.stride() as u32)?;    // stride_dst

		// kernel.set_default_global_work_size((src.width(), src.height()).into());

		let mut builder = OclKernel::builder();
		builder.program(&core.program);
		builder.queue(core.queue_kernel.clone());

		builder.arg(src.buf());
		builder.arg(src.stride() as u32);
		builder.arg(src.width() as u32);
		builder.arg(src.height() as u32);
		builder.arg(&self.filter);
		builder.arg(self.filter.len() as u32);
		builder.arg(dst.buf());
		builder.arg(dst.stride() as u32);

		builder.global_work_size((src.width(), src.height()));

		if self.quad_sigma > 0. {
			builder.name("k02_gaussian_blur_filter");
		} else {
			builder.name("k02_gaussian_sharp_filter");
		}
		builder.build()
	}
}