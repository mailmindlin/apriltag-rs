use std::{num::NonZeroU64, mem::size_of};

use futures::future::try_join3;
use wgpu::{BindGroupEntry, PipelineLayoutDescriptor};

use crate::DetectorConfig;

use super::{util::{buffer_traits::GpuBuffer, ComputePipelineDescriptor, DataStore, GpuBuffer1, GpuImageLike, GpuTextureY8, ProgramBuilder}, GpuContext, GpuStage, GpuStageContext, WgpuBuildError, WgpuDetectError};

pub(super) struct GpuConnectedComponents {
	args_bgl: wgpu::BindGroupLayout,
	/// Kernel to initialize UnionFind structure
	uf_init: wgpu::ComputePipeline,
	/// CCL kernel
	uf_connect: wgpu::ComputePipeline,
	/// Count set size kernel
	uf_count: wgpu::ComputePipeline,
}

impl GpuConnectedComponents {
	async fn new(context: &GpuContext, _config: &DetectorConfig) -> Result<Self, WgpuBuildError> {
		// let cs_init = {
		// 	let mut prog = ProgramBuilder::new("uf_init");
		// 	prog.append(include_str!("./04a_unionfind_init.wgsl"));
		// 	prog.build(&context.device)
		// };
		let cs_build = {
			let mut prog = ProgramBuilder::new("uf_build");
			prog.append(include_str!("./shader/04_unionfind.wgsl"));
			prog.build(&context.device).await?
		};

		let args_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("uf->bgl"),
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
				// Parents
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Storage { read_only: false },
						has_dynamic_offset: false,
						min_binding_size: Some(NonZeroU64::new(size_of::<[u32; 1]>() as _).unwrap()),
					},
					count: None,
				},
				// img_src
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
			]
		});

		let cp_layout = context.device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("uf->cpl"),
			bind_group_layouts: &[&args_bgl],
			push_constant_ranges: &[]
		});

		let cp_init = cs_build.create_compute_pipeline(ComputePipelineDescriptor {
			label: Some("uf_init->cp"),
			layout: Some(&cp_layout),
			entry_point: "k04_unionfind_init",
		});

		let cp_build = cs_build.create_compute_pipeline(ComputePipelineDescriptor {
			label: Some("uf_build->cp"),
			layout: Some(&cp_layout),
			entry_point: "k04_connected_components",
		});

		let cp_count = cs_build.create_compute_pipeline(ComputePipelineDescriptor {
			label: Some("uf_count->cp"),
			layout: Some(&cp_layout),
			entry_point: "k04_count_groups",
		});

		let (uf_init, uf_connect, uf_count) = try_join3(cp_init, cp_build, cp_count).await?;

		Ok(Self {
			args_bgl,
			uf_init,
			uf_connect,
			uf_count,
		})
	}
}

impl GpuStage for GpuConnectedComponents {
	type Source = GpuTextureY8;
	type Data = wgpu::BindGroup;
	type Output = GpuBuffer1<u32>;

    fn src_alignment(&self) -> usize {
        size_of::<u32>()
    }

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WgpuDetectError> {
		let params = ctx.params_buffer(&[
			src.width() as u32,
		]);
		// Create output buffer
		let uf_parents = ctx.dst_buffer1::<u32>(src.width() * src.height() * 2)?;

		let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("uf->bg"),
			layout: &self.args_bgl,
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: params.as_entire_binding(),
				},
				BindGroupEntry {
					binding: 1,
					resource: uf_parents.as_binding(),
				},
				BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::TextureView(&src.as_view()),
				},
			]
		});

		ctx.cpass.set_bind_group(0, temp.store(bg), &[]);
		ctx.tp.stamp("stage4 bindgroups");
		
		{
			let tw = src.width().div_ceil(16) as u32;
			let th = src.height().div_ceil(16) as u32;
			println!("Dispatch wg=({tw}, {th}) uf_init");
			ctx.cpass.set_pipeline(&self.uf_init);
			ctx.cpass.dispatch_workgroups(tw, th, 1);
			ctx.tp.stamp("uf_init kernel");
		}

		{
			let tw = (src.width() - 2).div_ceil(16) as u32;
			let th = src.height().div_ceil(16) as u32;
			println!("Dispatch wg=({tw}, {th}) uf_connect");
			ctx.cpass.set_pipeline(&self.uf_connect);
			for _ in 0..25 {
				ctx.cpass.dispatch_workgroups(tw, th, 1);
			}
			ctx.tp.stamp("uf_connect kernel");
		}

		{
			let tw = src.width().div_ceil(16) as u32;
			let th = src.height().div_ceil(16) as u32;
			println!("Dispatch wg=({tw}, {th}) uf_count");
			ctx.cpass.set_pipeline(&self.uf_count);
			ctx.cpass.dispatch_workgroups(tw, th, 1);
			ctx.tp.stamp("uf_count kernel");
		}

		#[cfg(feature="debug")]
		if ctx.config.debug() {
			ctx.cpass.insert_debug_marker("unionfind");
		}
		ctx.tp.stamp("unionfind");

		Ok(uf_parents)
    }
}