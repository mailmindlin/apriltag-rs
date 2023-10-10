use std::{borrow::Cow, num::NonZeroU64, mem::size_of};

use wgpu::{util::DeviceExt, BufferUsages, BindGroupEntry, TextureFormat};

use crate::{DetectorConfig, wgpu::buffer::GpuImageLike, quad_thresh::threshold::TILESZ};

use super::{GpuStageContext, GpuStage, GpuContext, WGPUError, util::DataStore, buffer::GpuTextureU};

const PROG_THRESHIM1: &str = include_str!("./03a_tile_minmax.wgsl");
const PROG_THRESHIM2: &str = include_str!("./03b_tile_blur.wgsl");
const PROG_THRESHIM3: &str = include_str!("./03c_threshim.wgsl");

pub(super) struct GpuConnectedComponents {
	const_bg: wgpu::BindGroup,
	tile_minmax: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
	tile_blur: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
	threshim: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

impl GpuConnectedComponents {
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

		Self {
			const_bg,
			tile_minmax,
			tile_blur,
			threshim,
		}
	}
}

impl GpuStage for GpuConnectedComponents {
	type Source = GpuTextureU;
	type Data = (wgpu::BindGroup, wgpu::BindGroup);
	type Output = GpuTextureU;

    fn src_alignment(&self) -> usize {
        size_of::<u32>()
    }

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WGPUError> {
		// Create output buffer
		let dst = ctx.dst_texture_u32(src.width(), src.height());

		let tw = src.width() / TILESZ;
		let th = src.height() / TILESZ;
		let tiles0 = ctx.context.temp_texture(tw, th, false, TextureFormat::Rg8Uint);
		let tiles1 = ctx.context.temp_texture(tw, th, false, TextureFormat::Rg8Uint);
		ctx.tp.stamp("threshim buffer");

		ctx.cpass.set_bind_group(0, &self.const_bg, &[]);


		let (bg1, bg2, bg3) = {
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
			temp.store((bg1, bg2, bg3))
		};
		ctx.tp.stamp("stage3 bindgroups");
		
		println!("Dispatch wg=({tw}, {th})");
		ctx.cpass.set_bind_group(1, bg1, &[]);
		ctx.cpass.set_pipeline(&self.tile_minmax.0);
		ctx.cpass.dispatch_workgroups(tw as u32, th as u32, 1);
		ctx.tp.stamp("tile_minmax kernel");

		ctx.cpass.set_bind_group(1, bg2, &[]);
		ctx.cpass.set_pipeline(&self.tile_blur.0);
		ctx.cpass.dispatch_workgroups(tw as u32, th as u32, 1);
		ctx.tp.stamp("tile_blur kernel");

		ctx.cpass.set_bind_group(1, bg3, &[]);
		ctx.cpass.set_pipeline(&self.threshim.0);
		ctx.cpass.dispatch_workgroups(tw as u32, th as u32, 1);
		ctx.tp.stamp("threshim kernel");

		#[cfg(feature="debug")]
		if ctx.config.debug() {
			ctx.cpass.insert_debug_marker("threshim");
		}
		ctx.tp.stamp("threshim");

		Ok(dst)
    }
}