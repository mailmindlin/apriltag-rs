use std::{borrow::Cow, mem::size_of, num::NonZeroU64};

use wgpu::{ComputePipelineDescriptor, BindGroupEntry, util::{DeviceExt, BufferInitDescriptor}, BufferUsages};

use crate::{detector::config::QuadDecimateMode, wgpu::buffer::GpuImageLike};

use super::{util::ConstBuilder, buffer::{GpuStage, GpuBufferY8}, GpuStageContext, GpuContext, WGPUError};

const PROG_QUAD_DECIMATE: &str = include_str!("./01_quad_decimate.wgsl");

pub(super) struct GpuQuadDecimate {
	mode: QuadDecimateMode,
	local_dims: (u32, u32),
	bg_layout: wgpu::BindGroupLayout,
	compute_pipeline: wgpu::ComputePipeline,
}


impl GpuQuadDecimate {
	pub(super) fn new(context: &GpuContext, mode: QuadDecimateMode) -> Self {
		let local_dims = (1, 1);

		let program = {
			let factor = match mode {
				QuadDecimateMode::ThreeHalves => 0,
				QuadDecimateMode::Scaled(f) => f.get(),
			};
			let mut constants = ConstBuilder::default();
			constants.set_u32("factor", factor);
			constants.set_u32("wg_width", local_dims.0);
			constants.set_u32("wg_height", local_dims.1);
			constants.finish(PROG_QUAD_DECIMATE)
		};
		let cs_module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("01_quad_decimate"),
			source: wgpu::ShaderSource::Wgsl(Cow::Owned(program)),
		});

		let bg_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("bgl:quad_decimate"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: Some(NonZeroU64::new(size_of::<[u32; 3]>() as _).unwrap()),
					},
					count: None,
				},
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

		let entry_point = match mode {
			QuadDecimateMode::Scaled(_) => "k01_filter_quad_decimate",
			QuadDecimateMode::ThreeHalves => "k01_filter_quad_decimate_32",
		};

		let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("quad_decimate:pipeline_layout"),
			bind_group_layouts: &[
				&bg_layout,
			],
			push_constant_ranges: &[],
		});

		let compute_pipeline = context.device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: Some("quad_decimate:compute_pipeline"),
			layout: Some(&pipeline_layout),
			module: &cs_module,
			entry_point,
		});

		Self {
			mode,
			local_dims,
			compute_pipeline,
			bg_layout,
		}
	}
}

impl GpuStage for GpuQuadDecimate {
	type Source = GpuBufferY8;
	type Data = wgpu::BindGroup;
	type Output = GpuBufferY8;
	fn src_alignment(&self) -> usize {
		match self.mode {
			QuadDecimateMode::ThreeHalves => size_of::<u32>() * 3,
			QuadDecimateMode::Scaled(factor) => size_of::<u32>() * (factor.get() as usize),
		}
	}

	fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut Option<Self::Data>) -> Result<Self::Output, WGPUError> {
		let (swidth, sheight) = match self.mode {
			QuadDecimateMode::ThreeHalves => {
				let swidth = src.dims.width / 3 * 2;
				let sheight = src.dims.height / 3 * 2;
				assert_eq!(swidth % 2, 0, "Output dimension must be multiple of two");
				assert_eq!(sheight % 2, 0, "Output dimension must be multiple of two");
				(swidth, sheight)
			},
			QuadDecimateMode::Scaled(factor) => {
				let factor = factor.get() as usize;
				let swidth = 1 + (src.dims.width - 1) / factor;
				let sheight = 1 + (src.dims.height - 1) / factor;

				assert!((swidth - 1) * factor <= src.dims.width);
				assert!((sheight - 1) * factor <= src.dims.height);
				(swidth, sheight)
			},
		};

		// #[cfg(feature="debug")]
		// ctx.encoder.push_debug_group("quad_decimate");
		let dst = ctx.dst_buffer(swidth, sheight, size_of::<u32>(), false)?;

		let args_bg = {
			let params = [
				// Constant 0: stride_src
				(src.stride() / size_of::<u32>()) as u32,
				// Constant 1: stride_dst
				(dst.stride() / size_of::<u32>()) as u32,
				// Constant 2: factor
				// factor as u32,
				// Constant 3: width_dst
				dst.width() as u32,
			];
			println!("Params: {params:?}");
			let buf_params = ctx.device().create_buffer_init(&BufferInitDescriptor {
				label: Some("quad_decimate params"),
				contents: bytemuck::cast_slice(&params),
				usage: BufferUsages::UNIFORM,
			});

			ctx.tp.stamp("quad_decimate buffer");

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

		{
			// let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("quad_decimate:cp") });
			ctx.cpass.set_pipeline(&self.compute_pipeline);
	
			*temp = Some(args_bg);
			ctx.cpass.set_bind_group(0, temp.as_ref().unwrap(), &[]);
			
			let (local_width, local_height) = match self.mode {
				QuadDecimateMode::ThreeHalves => {
					(size_of::<u32>() * 2 * (self.local_dims.0 as usize), 2 * (self.local_dims.1 as usize))
				},
				QuadDecimateMode::Scaled(_) => {
					(size_of::<u32>() * 1 * (self.local_dims.0 as usize), 1 * (self.local_dims.1 as usize))
				}
			};
	
			let wg_x = swidth.div_ceil(local_width) as u32;
			let wg_y = sheight.div_ceil(local_height) as u32;
			println!("Dispatch local=({local_width}, {local_height}) wg=({wg_x}, {wg_y}) global=({swidth}, {sheight})");
			ctx.cpass.dispatch_workgroups(wg_x, wg_y, 1);
	
			#[cfg(feature="debug")]
			if ctx.config.debug() {
				ctx.cpass.insert_debug_marker("quad_decimate");
				// ctx.encoder.pop_debug_group();
			}
		}
		ctx.tp.stamp("quad_decimate");

		Ok(dst)
	}
}