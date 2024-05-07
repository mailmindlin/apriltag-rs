use std::{mem::size_of, num::NonZeroU32};

use wgpu::BindGroupEntry;

use crate::{detector::config::QuadDecimateMode, wgpu::util::{ProgramBuilder, GpuImageLike}, DetectorConfig, DetectorBuildError, util::image::Luma};

use super::{util::{ComputePipelineDescriptor, DataStore, GpuStage, GpuTextureY8}, GpuContext, GpuStageContext, WgpuDetectError};

const PROG_QUAD_DECIMATE_32: &str = include_str!("./shader/01_quad_decimate_img_32.wgsl");
const PROG_QUAD_DECIMATE_F: &str = include_str!("./shader/01_quad_decimate_img_f.wgsl");

pub(super) enum GpuQuadDecimate {
	Factor {
		factor: NonZeroU32,
		local_dims: (u32, u32),
		args_bgl: wgpu::BindGroupLayout,
		compute_pipeline: wgpu::ComputePipeline,
	},
	ThreeHalves {
		local_dims: (u32, u32),
		args_bgl: wgpu::BindGroupLayout,
		compute_pipeline: wgpu::ComputePipeline,
	},
}

impl GpuQuadDecimate {
	pub(super) async fn new(context: &GpuContext, config: &DetectorConfig) -> Result<Option<Self>, DetectorBuildError> {
		let res = match config.quad_decimate_mode() {
			None => return Ok(None),
			Some(QuadDecimateMode::Scaled(factor)) => Self::new_factor(context, factor).await,
			Some(QuadDecimateMode::ThreeHalves) => Self::new_32(context, config).await,
		};
		Ok(Some(res?))
	}

	async fn new_32(context: &GpuContext, config: &DetectorConfig) -> Result<Self, DetectorBuildError> {
		if let Err(e) = config.source_dimensions.check(3.., 3..) {
			return Err(DetectorBuildError::InvalidSourceDimensions(e));
		}

		let local_dims = (64, 1);
		let cs_module = {
			let mut builder = ProgramBuilder::new("01_quad_decimate");
			builder.set_u32("wg_width", local_dims.0);
			builder.set_u32("wg_height", local_dims.1);

			builder.append(PROG_QUAD_DECIMATE_32);
			builder.build(&context.device).await?
		};

		let args_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("quad_decimate->bgl"),
			entries: &[
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

		let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("quad_decimate->pipeline_layout"),
			bind_group_layouts: &[
				&args_bgl,
			],
			push_constant_ranges: &[],
		});

		let compute_pipeline = cs_module.create_compute_pipeline(ComputePipelineDescriptor {
			label: Some("quad_decimate->compute_pipeline"),
			layout: Some(&pipeline_layout),
			entry_point: "main",
		}).await?;

		Ok(Self::ThreeHalves {
			local_dims,
			args_bgl,
			compute_pipeline,
		})
	}

	pub(super) async fn new_factor(context: &GpuContext, factor: NonZeroU32) -> Result<Self, DetectorBuildError> {
		let local_dims = (64, 1);
		let cs_module = {
			let mut builder = ProgramBuilder::new("01_quad_decimate");
			builder.set_u32("wg_width", local_dims.0);
			builder.set_u32("wg_height", local_dims.1);
			builder.set_u32("factor", factor.get());
			builder.append(PROG_QUAD_DECIMATE_F);
			builder.build(&context.device).await?
		};

		let args_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("bgl:quad_decimate"),
			entries: &[
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

		let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("quad_decimate->pipeline_layout"),
			bind_group_layouts: &[
				&args_bgl,
			],
			push_constant_ranges: &[],
		});

		let compute_pipeline = cs_module.create_compute_pipeline(ComputePipelineDescriptor {
			label: Some("quad_decimate->compute_pipeline"),
			layout: Some(&pipeline_layout),
			entry_point: "main",
		}).await?;

		Ok(Self::Factor {
			factor,
			local_dims,
			args_bgl,
			compute_pipeline,
		})
	}
}

impl GpuStage for GpuQuadDecimate {
    type Source = GpuTextureY8;
    type Output = GpuTextureY8;
	type Data = wgpu::BindGroup;
	fn src_alignment(&self) -> usize {
		match self {
			GpuQuadDecimate::ThreeHalves { .. } => 3,
			GpuQuadDecimate::Factor { factor, .. } => size_of::<u32>() * (factor.get() as usize),
		}
	}

	fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WgpuDetectError> {
		let (dst, local_width, local_height) = match self {
			Self::Factor { factor, local_dims, args_bgl, compute_pipeline } => {
				let factor = factor.get() as usize;
				let swidth = 1 + (src.width() - 1) / factor;
				let sheight = 1 + (src.height()- 1) / factor;

				assert!((swidth - 1) * factor <= src.width());
				assert!((sheight - 1) * factor <= src.height());

				let dst = ctx.dst_texture::<Luma<u8>>(swidth, sheight);

				let args_bg = ctx.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some("quad_decimate->args_bg"),
					layout: &args_bgl,
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

				ctx.cpass.set_pipeline(compute_pipeline);
				ctx.cpass.set_bind_group(0, temp.store(args_bg), &[]);

				let local_width = 1 * (local_dims.0 as usize);
				let local_height = 1 * (local_dims.1 as usize);

				(dst, local_width, local_height)
			},
			Self::ThreeHalves { local_dims, args_bgl, compute_pipeline } => {
				let swidth = src.width() / 3 * 2;
				let sheight = src.height() / 3 * 2;
				assert_eq!(swidth % 2, 0, "Output dimension must be multiple of two");
				assert_eq!(sheight % 2, 0, "Output dimension must be multiple of two");

				let dst = ctx.dst_texture(swidth, sheight);

				let args_bg = ctx.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some("quad_decimate->args_bg"),
					layout: args_bgl,
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

				ctx.cpass.set_pipeline(compute_pipeline);
				ctx.cpass.set_bind_group(0, temp.store(args_bg), &[]);

				let local_width = 2 * (local_dims.0 as usize);
				let local_height = 2 * (local_dims.1 as usize);
		
				let wg_x = swidth.div_ceil(local_width) as u32;
				let wg_y = sheight.div_ceil(local_height) as u32;
				#[cfg(feature="extra_debug")]
				println!("Dispatch local=({local_width}, {local_height}) wg=({wg_x}, {wg_y}) global=({swidth}, {sheight})");
				ctx.cpass.dispatch_workgroups(wg_x, wg_y, 1);

				(dst, local_width, local_height)
			}
		};

		let wg_x = dst.width().div_ceil(local_width) as u32;
		let wg_y = dst.height().div_ceil(local_height) as u32;
		// #[cfg(feature="extra_debug")]
		// println!("Dispatch local=({local_width}, {local_height}) wg=({wg_x}, {wg_y}) global=({swidth}, {sheight})");
		ctx.cpass.dispatch_workgroups(wg_x, wg_y, 1);

		#[cfg(feature="debug")]
		if ctx.config.debug() {
			ctx.cpass.insert_debug_marker("quad_decimate");
		}
		ctx.tp.stamp("quad_decimate");

		Ok(dst)
	}
}