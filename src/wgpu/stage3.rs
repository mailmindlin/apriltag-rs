use std::{borrow::Cow, num::NonZeroU64, mem::size_of};

use wgpu::{util::DeviceExt, BufferUsages, BindGroupEntry};

use crate::{DetectorConfig, wgpu::buffer::GpuImageLike, quad_thresh::threshold::TILESZ};

use super::{GpuStageContext, GpuStage, GpuContext, WGPUError, util::DataStore, buffer::GpuTexture};

const PROG_THRESHIM1: &str = include_str!("./03a_tile_minmax.wgsl");
const PROG_THRESHIM2: &str = include_str!("./03b_tile_blur.wgsl");
const PROG_THRESHIM3: &str = include_str!("./03c_threshim.wgsl");

pub(super) struct GpuThreshim {
	const_bg: wgpu::BindGroup,
	tile_minmax: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
	tile_blur: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
	threshim: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

impl GpuThreshim {
	pub fn new(context: &GpuContext, config: &DetectorConfig) -> Self {
		let params_buf = {
			let data = [
				config.qtp.min_white_black_diff as u32,
			];
			context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("threshim->params"), contents: bytemuck::bytes_of(&data), usage: BufferUsages::UNIFORM })
		};

		let const_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("threshim->const_bgl"),
			entries: &[
				// Params
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: Some(NonZeroU64::new(size_of::<[u32; 1]>() as _).unwrap()),
					},
					count: None,
				},
			]
		});

		let const_bg = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("threshim->const_bg"),
			layout: &const_bgl,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: params_buf.as_entire_binding(),
				}
			],
		});

		let texture_u32 = wgpu::BindingType::Texture {
			sample_type: wgpu::TextureSampleType::Uint,
			view_dimension: wgpu::TextureViewDimension::D2,
			multisampled: false,
		};

		let storage_rg8 = wgpu::BindingType::StorageTexture {
			access: wgpu::StorageTextureAccess::WriteOnly,
			format: wgpu::TextureFormat::Rg8Uint,
			view_dimension: wgpu::TextureViewDimension::D2,
		};

		let tile_minmax = {
			let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
				label: Some("03a_threshim"),
				source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROG_THRESHIM1)),
			});

			let params_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("threshim->tile_minmax_bgl"),
				entries: &[
					// img_src
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: texture_u32.clone(),
						count: None,
					},
					// img_tiles0
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: storage_rg8.clone(),
						count: None,
					},
				],
			});

			let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("threshim->pl_tile_minmax"),
				bind_group_layouts: &[
					&const_bgl,
					&params_bgl,
				],
				push_constant_ranges: &[],
			});

			let cp_minmax = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
				label: Some("threshim->cp_tile_minmax"),
				layout: Some(&pipeline_layout),
				module: &shader,
				entry_point: "k03_tile_minmax",
			});
			(cp_minmax, params_bgl)
		};

		let tile_blur = {
			let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
				label: Some("03b_tile_blur"),
				source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROG_THRESHIM2)),
			});

			let params_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("threshim->bgl_tile_blur"),
				entries: &[
					// img_tiles0
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: texture_u32.clone(),
						count: None,
					},
					// img_tiles1
					wgpu::BindGroupLayoutEntry {
						binding: 2,
						visibility: wgpu::ShaderStages::COMPUTE,
						ty: wgpu::BindingType::StorageTexture {
							access: wgpu::StorageTextureAccess::WriteOnly,
							format: wgpu::TextureFormat::Rg8Uint,
							view_dimension: wgpu::TextureViewDimension::D2,
						},
						count: None,
					},
				],
			});

			let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("threshim->pl_tile_blur"),
				bind_group_layouts: &[
					&const_bgl,
					&params_bgl,
				],
				push_constant_ranges: &[],
			});

			let cp_minmax = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
				label: Some("threshim->cp_tile_blur"),
				layout: Some(&pipeline_layout),
				module: &shader,
				entry_point: "k03_tile_blur",
			});
			(cp_minmax, params_bgl)
		};

		let threshim = {
			let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
				label: Some("03c_threshim"),
				source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROG_THRESHIM3)),
			});

			let params_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("threshim->param_bgl3"),
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
					// img_tiles1
					wgpu::BindGroupLayoutEntry {
						binding: 2,
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
						binding: 3,
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
				label: Some("threshim->pl_threshim"),
				bind_group_layouts: &[
					&const_bgl,
					&params_bgl,
				],
				push_constant_ranges: &[],
			});

			let cp_minmax = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
				label: Some("threshim->cp_threshim"),
				layout: Some(&pipeline_layout),
				module: &shader,
				entry_point: "k03_build_threshim",
			});
			(cp_minmax, params_bgl)
		};

		Self {
			const_bg,
			tile_minmax,
			tile_blur,
			threshim,
		}
	}
}

pub(super) struct GpuThreshimExtra {
	bg_minmax: wgpu::BindGroup,
	bg_blur: wgpu::BindGroup,
	bg_threshim: wgpu::BindGroup,
	pub(super) tile_minmax: GpuTexture<[u8; 2]>,
	pub(super) tile_blur: GpuTexture<[u8; 2]>,
}


impl GpuStage for GpuThreshim {
	type Source = GpuTexture;
	type Data = GpuThreshimExtra;
	type Output = GpuTexture;

    fn src_alignment(&self) -> usize {
        size_of::<u32>()
    }

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WGPUError> {
		// Create output buffer
		let dst = ctx.dst_texture(src.width(), src.height());

		let tw = src.width() / TILESZ;
		let th = src.height() / TILESZ;
		let tiles0 = ctx.context.temp_texture::<[u8; 2]>(tw, th, true);
		let tiles1 = ctx.context.temp_texture::<[u8; 2]>(tw, th, true);

		ctx.cpass.set_bind_group(0, &self.const_bg, &[]);


		let GpuThreshimExtra { bg_minmax, bg_blur, bg_threshim, .. } = {
			let bg1 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("threshold->tile_minmax->bindgroup"),
				layout: &self.tile_minmax.1,
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&src.as_view()),
					},
					BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::TextureView(&tiles0.as_view()),
					},
				],
			});

			let bg2 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("threshold->tile_minmax->bindgroup"),
				layout: &self.tile_blur.1,
				entries: &[
					BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::TextureView(&tiles0.as_view()),
					},
					BindGroupEntry {
						binding: 2,
						resource: wgpu::BindingResource::TextureView(&tiles1.as_view()),
					},
				],
			});

			let bg3 = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("threshold->tile_minmax->bindgroup"),
				layout: &self.threshim.1,
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&src.as_view()),
					},
					BindGroupEntry {
						binding: 2,
						resource: wgpu::BindingResource::TextureView(&tiles1.as_view()),
					},
					BindGroupEntry {
						binding: 3,
						resource: wgpu::BindingResource::TextureView(&dst.as_view()),
					},
				],
			});
			temp.store(GpuThreshimExtra {
				bg_minmax: bg1,
				bg_blur: bg2,
				bg_threshim: bg3,
				tile_minmax: tiles0.clone(),
				tile_blur: tiles1.clone(),
			})
		};
		
		#[cfg(feature="extra_debug")]
		println!("Dispatch wg=({tw}, {th})");
		ctx.cpass.set_bind_group(1, bg_minmax, &[]);
		ctx.cpass.set_pipeline(&self.tile_minmax.0);
		ctx.cpass.dispatch_workgroups(tw as u32, th as u32, 1);

		ctx.cpass.set_bind_group(1, bg_blur, &[]);
		ctx.cpass.set_pipeline(&self.tile_blur.0);
		ctx.cpass.dispatch_workgroups(tw as u32, th as u32, 1);

		ctx.cpass.set_bind_group(1, bg_threshim, &[]);
		ctx.cpass.set_pipeline(&self.threshim.0);
		ctx.cpass.dispatch_workgroups(src.width() as u32, src.height() as u32, 1);

		#[cfg(feature="debug")]
		if ctx.config.debug() {
			ctx.cpass.insert_debug_marker("threshim");
		}
		ctx.tp.stamp("threshim");

		Ok(dst)
    }
}