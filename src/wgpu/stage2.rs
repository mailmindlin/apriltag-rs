use std::{borrow::Cow, num::NonZeroU64, mem::size_of};

use wgpu::{util::DeviceExt, BufferUsages, BindGroupEntry};

use crate::detector::quad_sigma_kernel;

use super::{GpuBufferY8, GpuStageContext, GpuStage, GpuContext, WGPUError, util::DataStore};

const PROG_QUAD_SIGMA: &str = include_str!("./02_quad_sigma.wgsl");

pub(super) struct GpuQuadSigma {
	bg_layout: wgpu::BindGroupLayout,
	compute_pipeline: wgpu::ComputePipeline,
	const_bg: wgpu::BindGroup,
}

impl GpuQuadSigma {
	pub fn new(context: &GpuContext, quad_sigma: f32) -> Option<Self> {
		let filter_buf = {
			let kernel_u8 = quad_sigma_kernel(quad_sigma)?;
			let mut kernel_u32 = vec![];
			for v in kernel_u8 {
				kernel_u32.push(v);
				kernel_u32.push(0u8);
				kernel_u32.push(0u8);
				kernel_u32.push(0u8);
			}
			
			context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("quad_sigma_filter"), contents: &kernel_u32, usage: BufferUsages::STORAGE })
		};

		let cs_module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("02_quad_sigma"),
			source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROG_QUAD_SIGMA)),
		});

		let const_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("quad_sigma->const_bgl"),
			entries: &[
				// Filter
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Storage { read_only: true },
						has_dynamic_offset: false,
						min_binding_size: Some(NonZeroU64::new(filter_buf.size()).unwrap()),
					},
					count: None,
				},
			]
		});

		let const_bg = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("quad_sigma->const_bg"),
			layout: &const_bgl,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: filter_buf.as_entire_binding(),
				}
			],
		});

		let param_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("quad_sigma->param_bgl"),
			entries: &[
				// Params
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: Some(NonZeroU64::new(size_of::<[u32; 4]>() as _).unwrap()),
					},
					count: None,
				},
				// img_src
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Storage { read_only: true },
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				},
				// img_dst
				wgpu::BindGroupLayoutEntry {
					binding: 2,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Storage { read_only: false },
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				},
			],
		});

		let entry_point = if quad_sigma > 0. {
			"k02_gaussian_blur_filter"
		} else {
			"k02_gaussian_sharp_filter"
		};

		let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("quad_sigma pipeline_layout"),
			bind_group_layouts: &[
				&const_bgl,
				&param_bgl,
			],
			push_constant_ranges: &[],
		});

		let compute_pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("quad_sigma compute_pipeline"),
			layout: Some(&pipeline_layout),
			module: &cs_module,
			entry_point,
		});

		Some(Self {
			bg_layout: param_bgl,
			compute_pipeline,
			const_bg,
		})
	}
}

impl GpuStage for GpuQuadSigma {
	type Source = GpuBufferY8;
	type Data = wgpu::BindGroup;
	type Output = GpuBufferY8;

    fn src_alignment(&self) -> usize {
        size_of::<u32>()
    }

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WGPUError> {
		// #[cfg(feature="debug")]
		// ctx.encoder.push_debug_group("quad_sigma");

		// Create output buffer
		let dst = ctx.dst_buffer(src.dims.width, src.dims.height, size_of::<u32>(), false)?;
		ctx.tp.stamp("quad_sigma buffer");

		let bind_group = {
			let params = [
				// Constant 0: stride_src
				(src.dims.stride / size_of::<u32>()) as u32,
				// Constant 1: width_src
				(src.dims.width) as u32,
				// Constant 2: height_src
				src.dims.height as u32,
				// Constant 3: stride_dst
				dst.dims.stride as u32,
			];
			println!("Params: {params:?}");
			let buf_params = ctx.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some("quad_decimate params"),
				contents: bytemuck::cast_slice(&params),
				usage: BufferUsages::UNIFORM,
			});
			
	
			ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("quad_decimate bindgroup"),
				layout: &self.bg_layout,
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: buf_params.as_entire_binding(),
					},
					BindGroupEntry {
						binding: 1,
						resource: src.buffer.as_entire_binding(),
					},
					BindGroupEntry {
						binding: 2,
						resource: dst.buffer.as_entire_binding(),
					},
				],
			})
		};

		// let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("quad_sigma:cp") });
		ctx.cpass.set_pipeline(&self.compute_pipeline);

		ctx.cpass.set_bind_group(0, &self.const_bg, &[]);
		ctx.cpass.set_bind_group(1, temp.store(bind_group), &[]);

		let wg_x = src.dims.width as u32;
		let wg_y = src.dims.height as u32;
		println!("Dispatch wg=({wg_x}, {wg_y})");
		ctx.cpass.dispatch_workgroups(wg_x, wg_y, 1);

		#[cfg(feature="debug")]
		if ctx.config.debug() {
			ctx.cpass.insert_debug_marker("quad_sigma");
			// drop(cpass);

			// ctx.encoder.pop_debug_group();
		}
		ctx.tp.stamp("quad_sigma");

		Ok(dst)
    }
}