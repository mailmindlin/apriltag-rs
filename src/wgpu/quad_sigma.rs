use wgpu::{util::DeviceExt, BufferUsages};

use crate::detector::quad_sigma_kernel;

use super::{WImage, WContext, WStage};


pub(super) struct WQuadSigma {
	bg_layout: wgpu::BindGroupLayout,
	compute_pipeline: wgpu::ComputePipeline,
	filter: wgpu::Buffer,
}

impl WQuadSigma {
	pub fn new(device: &wgpu::Device, quad_sigma: f32) -> Option<Self> {
		let kernel = quad_sigma_kernel(quad_sigma)?;
		let filter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("quad_sigma_filter"), contents: &kernel, usage: BufferUsages::UNIFORM });

		Some(Self {
			bg_layout: todo!(),
			compute_pipeline: todo!(),
			filter,
		})
	}
}

impl WStage for WQuadSigma {
    fn src_alignment(&self) -> usize {
        todo!()
    }

    fn debug_name(&self) -> &'static str {
        "quad_sigma"
    }

    fn apply(&self, ctx: &mut WContext<'_>, src: WImage) -> Result<WImage, super::WGPUError> {
        let filter = self.filter;

		#[cfg(feature="debug")]
		ctx.encoder.push_debug_group("quad_sigma");
		
		// Create output buffer
		let dst = ctx.temp_buffer(src.dims.width, src.dims.height, 0, false);
		ctx.tp.stamp("quad_sigma buffer");

		// Make quad_sigma kernel
		let kernel = {
			let mut builder = OclKernel::builder();
			builder.program(&self.program);
			builder.queue(self.queue_kernel.clone());

			builder.arg(prev.buf);                   // im_src
			builder.arg(prev.dims.stride as u32);    // stride_src
			builder.arg(prev.dims.width as u32);     // width_src
			builder.arg(prev.dims.height as u32);    // height_src
			builder.arg(quad_sigma_filter);          // filter
			builder.arg(quad_sigma_filter.len() as u32); // ksz
			builder.arg(&buf);                        // im_dst
			builder.arg(prev.dims.stride as u32);    // stride_dst

			builder.global_work_size((prev.dims.width, prev.dims.height));

			if config.quad_sigma > 0. {
				builder.name("k02_gaussian_blur_filter");
			} else {
				builder.name("k02_gaussian_sharp_filter");
			}
			builder.build()?
		};
		tp.stamp("quad_sigma kernel");

		let mut evt = OclEvent::empty();
		let kcmd = kernel
			.cmd()
			.queue(&self.queue_kernel)
			.enew(&mut evt)
			.ewait(prev.event.as_ref());

		unsafe { kcmd.enq() }?;

		tp.stamp("quad_sigma");

		ctx.encoder.pop_debug_group();

		#[cfg(feature="debug")]
		ctx.encoder.pop_debug_group();

		dst
    }
}