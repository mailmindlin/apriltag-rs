use std::{num::NonZeroU64, mem::size_of};

use wgpu::{util::DeviceExt, BufferUsages, BindGroupEntry};

use crate::{detector::quad_sigma_kernel, wgpu::util::GpuImageLike, DetectorBuildError};

use super::{util::{ComputePipelineDescriptor, DataStore, GpuTextureY8, ProgramBuilder}, GpuContext, GpuStage, GpuStageContext, WgpuDetectError};

const PROG_QUAD_SIGMA: &str = include_str!("./shader/02_quad_sigma_img.wgsl");

pub(super) struct GpuQuadSigma {
	local_dims: (u32, u32),
	bg_layout: wgpu::BindGroupLayout,
	compute_pipeline: wgpu::ComputePipeline,
	const_bg: wgpu::BindGroup,
}

impl GpuQuadSigma {
	pub async fn new(context: &GpuContext, quad_sigma: f32) -> Result<Option<Self>, DetectorBuildError> {
		let local_dims = (64, 1);
		let filter_buf = {
			let kernel_u8 = match quad_sigma_kernel(quad_sigma) {
				Some(kernel) => kernel,
				None => return Ok(None)
			};

			let mut kernel_u32 = vec![];
			for v in kernel_u8 {
				kernel_u32.push(v);
				kernel_u32.push(0u8);
				kernel_u32.push(0u8);
				kernel_u32.push(0u8);
			}
			
			context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some("quad_sigma_filter"),
				contents: &kernel_u32,
				usage: BufferUsages::STORAGE //TODO: uniform?
			})
		};

		let cs_module = {
			let mut program = ProgramBuilder::new("02_quad_sigma");
			program.set_u32("wg_width", local_dims.0);
			program.set_u32("wg_height", local_dims.1);
			program.append(PROG_QUAD_SIGMA);
			program.build(&context.device).await?
		};

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
				// img_src
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Uint,
						view_dimension: wgpu::TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				// img_dst
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::StorageTexture {
						access: wgpu::StorageTextureAccess::WriteOnly,
						format: wgpu::TextureFormat::R8Uint,
						view_dimension: wgpu::TextureViewDimension::D2,
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

		let compute_pipeline = cs_module.create_compute_pipeline(ComputePipelineDescriptor {
			label: Some("quad_sigma compute_pipeline"),
			layout: Some(&pipeline_layout),
			entry_point,
		}).await?;

		Ok(Some(Self {
			local_dims,
			bg_layout: param_bgl,
			compute_pipeline,
			const_bg,
		}))
	}
}

impl GpuStage for GpuQuadSigma {
	type Source = GpuTextureY8;
	type Data = wgpu::BindGroup;
	type Output = GpuTextureY8;

    fn src_alignment(&self) -> usize {
        size_of::<u32>()
    }

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WgpuDetectError> {
		// Create output buffer
		let dst = ctx.dst_texture(src.width(), src.height());

		let params_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("quad_sigma->params_bg"),
			layout: &self.bg_layout,
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::TextureView(&src.as_view()),
				},
				BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(&dst.as_view()),
				},
			],
		});

		// let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("quad_sigma:cp") });

		ctx.cpass.set_pipeline(&self.compute_pipeline);

		ctx.cpass.set_bind_group(0, &self.const_bg, &[]);
		ctx.cpass.set_bind_group(1, temp.store(params_bg), &[]);

		let wg_x = (src.width()).div_ceil(self.local_dims.0 as usize) as u32;
		let wg_y = (src.height()).div_ceil(self.local_dims.1 as usize) as u32;
		#[cfg(feature="extra_debug")]
		println!("Dispatch wg=({wg_x}, {wg_y})");
		ctx.cpass.dispatch_workgroups(wg_x, wg_y, 1);

		#[cfg(feature="debug")]
		if ctx.config.debug() {
			ctx.cpass.insert_debug_marker("quad_sigma");
		}
		ctx.tp.stamp("quad_sigma");

		Ok(dst)
    }
}