use std::{num::NonZeroU64, mem::size_of};

use futures::future::{try_join, try_join3};
use wgpu::{BindGroupEntry, PipelineLayoutDescriptor};

use crate::{DetectorBuildError, DetectorConfig};

use super::{util::{buffer_traits::GpuBuffer, ComputePipelineDescriptor, DataStore, GpuBuffer1, GpuImageLike, GpuTextureY8, ProgramBuilder}, GpuContext, GpuStage, GpuStageContext, WgpuDetectError};

const WG_WIDTH: usize = 16;
const WG_HEIGHT: usize = 16;

const WARP_SIZE: u32 = 32;
const BLOCK_H: u32 = 4;

pub(super) struct GpuBke {
	args_bgl: wgpu::BindGroupLayout,
	/// Kernel to initialize UnionFind structure
	cp_bke_init: wgpu::ComputePipeline,
	/// CCL kernel
	cp_bke_compression: wgpu::ComputePipeline,
	/// Count set size kernel
	cp_bke_merge: wgpu::ComputePipeline,
	cp_bke_final: wgpu::ComputePipeline,
	cp_ha4_strip_label: wgpu::ComputePipeline,
	cp_ha4_strip_merge: wgpu::ComputePipeline,
	cp_ha4_relabeling: wgpu::ComputePipeline,
	cp_merge: wgpu::ComputePipeline,
	cp_count: wgpu::ComputePipeline,
}

impl GpuBke {
	pub async fn new(context: &GpuContext, _config: &DetectorConfig) -> Result<Self, DetectorBuildError> {
		let args_bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("bke->bgl"),
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
				// Labels
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
			label: Some("bke->cpl"),
			bind_group_layouts: &[&args_bgl],
			push_constant_ranges: &[]
		});

		const HEADER_BKE: &str = include_str!("./shader/04a_bke.wgsl");

		let cp_bke_init = {
			let name = "bke_init";
			let shader = {
				let mut prog = ProgramBuilder::new(name);
				prog.append(HEADER_BKE);
				prog.append(include_str!("./shader/04b_bke.wgsl"));
				prog.build(&context.device).await?
			};
			shader.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some(name),
				layout: Some(&cp_layout),
				entry_point: "k04_bke_init",
			}).await?
		};

		let cp_bke_compression = {
			let name = "bke_compression";
			let cs = {
				let mut prog = ProgramBuilder::new(name);
				prog.append(HEADER_BKE);
				prog.append(include_str!("./shader/04c_bke.wgsl"));
				prog.build(&context.device).await?
			};
			cs.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some(name),
				layout: Some(&cp_layout),
				entry_point: "k04_bke_compression",
			}).await?
		};

		let cp_bke_merge = {
			let name = "bke_merge";
			let cs = {
				let mut prog = ProgramBuilder::new(name);
				prog.append(HEADER_BKE);
				prog.append(include_str!("./shader/04d_bke.wgsl"));
				prog.build(&context.device).await?
			};
			cs.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some(name),
				layout: Some(&cp_layout),
				entry_point: "k04_bke_merge",
			}).await?
		};

		let cp_bke_final = {
			let name = "bke_final";
			let cs = {
				let mut prog = ProgramBuilder::new(name);
				prog.append(HEADER_BKE);
				prog.append(include_str!("./shader/04e_bke.wgsl"));
				prog.build(&context.device).await?
			};
			cs.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some(name),
				layout: Some(&cp_layout),
				entry_point: "k04_bke_final",
			}).await?
		};

		let (
			cp_ha4_strip_label,
			cp_ha4_strip_merge,
			cp_ha4_relabeling,
		) = {
			let cs_ha4 = {
				let mut prog = ProgramBuilder::new("04f_ha4.wgsl");
				prog.append(HEADER_BKE);
				prog.set_u32("BLOCK_H", BLOCK_H);
				prog.set_u32("WARP_SIZE", WARP_SIZE);
				prog.append(include_str!("./shader/04f_ha4.wgsl"));
				prog.build(&context.device).await?
			};
	
			let cp_ha4_strip_label = cs_ha4.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some("ha4_strip_label"),
				layout: Some(&cp_layout),
				entry_point: "k04_ha4_StripLabel",
			});
	
			let cp_ha4_strip_merge = cs_ha4.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some("ha4_strip_merge"),
				layout: Some(&cp_layout),
				entry_point: "k04_ha4_StripMerge",
			});
	
			let cp_ha4_relabeling = cs_ha4.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some("ha4_relabeling"),
				layout: Some(&cp_layout),
				entry_point: "k04_ha4_Relabeling",
			});

			try_join3(
				cp_ha4_strip_label,
				cp_ha4_strip_merge,
				cp_ha4_relabeling
			).await?
		};

		let (cp_merge, cp_count) = {
			let cs_count = {
				let mut prog = ProgramBuilder::new("ccl_count");
				prog.append(HEADER_BKE);
				prog.append(include_str!("./shader/04g_count.wgsl"));
				prog.build(&context.device).await?
			};
			let cp_merge = cs_count.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some("ccl_merge"),
				layout: Some(&cp_layout),
				entry_point: "k04_merge",
			});
			let cp_count = cs_count.create_compute_pipeline(ComputePipelineDescriptor {
				label: Some("ccl_count"),
				layout: Some(&cp_layout),
				entry_point: "k04_count",
			});
			try_join(cp_merge, cp_count).await?
		};

		Ok(Self {
			args_bgl,
			cp_bke_init,
			cp_bke_compression,
			cp_bke_merge,
			cp_bke_final,
			cp_ha4_strip_label,
			cp_ha4_strip_merge,
			cp_ha4_relabeling,
			cp_merge,
			cp_count,
		})
	}

	fn dispatch_bke<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &GpuTextureY8) {
		// Compute grid 
		let grid_x: u32 = src.width()
			.div_ceil(2) // Number of 2x2 blocks
			.div_ceil(WG_WIDTH) // Number of workgroups over 2x2 blocks
			.try_into().unwrap();
		let grid_y: u32 = src.height()
			.div_ceil(2) // Number of 2x2 blocks
			.div_ceil(WG_HEIGHT) // Number of workgroups over 2x2 blocks
			.try_into().unwrap();
		
		ctx.cpass.set_pipeline(&self.cp_bke_init);
		ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		ctx.tp.stamp("bke_init kernel");

		// if true {
		// 	ctx.cpass.set_pipeline(&self.cp_bke_compression);
		// 	ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		// 	ctx.tp.stamp("bke_compression kernel");
		// }

		// if true {
		// 	ctx.cpass.set_pipeline(&self.cp_bke_merge);
		// 	ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		// 	ctx.tp.stamp("bke_merge kernel");
		// }
		// if true {
		// 	ctx.cpass.set_pipeline(&self.cp_bke_compression);
		// 	ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		// 	ctx.tp.stamp("bke_compression kernel");
		// }
		// if true {
		// 	ctx.cpass.set_pipeline(&self.cp_bke_final);
		// 	ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		// 	ctx.tp.stamp("bke_final kernel");
		// }

	}

	fn dispatch_ha4<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &GpuTextureY8) {
		if true {
			// StripLabel
			let grid_x = 1;
			let grid_y = src.height()
				.div_ceil(BLOCK_H as usize) as u32;
			ctx.cpass.set_pipeline(&self.cp_ha4_strip_label);
			ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		}

		if true {
			let grid_x = src.width()
				.div_ceil(WARP_SIZE as usize) as u32;
			let grid_y = src.height()
				.div_ceil(BLOCK_H as usize)
				.div_ceil(BLOCK_H as usize) as u32;
			ctx.cpass.set_pipeline(&self.cp_ha4_strip_merge);
			ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		}

		if true {
			let grid_x = src.width()
				.div_ceil(WARP_SIZE as usize) as u32;
			let grid_y = src.height()
				.div_ceil(BLOCK_H as usize) as u32;
			ctx.cpass.set_pipeline(&self.cp_ha4_relabeling);
			ctx.cpass.dispatch_workgroups(grid_x, grid_y, 1);
		}
	}

	fn dispatch_count<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &GpuTextureY8) {
		// `Count` operates on individual pixels
		let tw = src.width().div_ceil(16) as u32;
		let th = src.height().div_ceil(16) as u32;

		ctx.cpass.set_pipeline(&self.cp_merge);
		ctx.cpass.dispatch_workgroups(tw, th, 1);

		ctx.cpass.set_pipeline(&self.cp_count);
		ctx.cpass.dispatch_workgroups(tw, th, 1);
		ctx.tp.stamp("uf_count kernel");
	}
}

impl GpuStage for GpuBke {
	type Source = GpuTextureY8;
	type Data = wgpu::BindGroup;
	type Output = GpuBuffer1<u32>;

    fn src_alignment(&self) -> usize {
        size_of::<u32>()
    }

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WgpuDetectError> {
		// We guarauntee that the UnionFind buffer has a height multiple of two
		let (uf_width, uf_height) = if src.height() % 2 == 0 {
			(src.width(), src.height())
		} else {
			(src.width(), src.height().next_multiple_of(2))
		};

		let params = ctx.params_buffer(&[
			uf_width.try_into().expect("Width overflow")
		]);

		let uf_parents = ctx.dst_buffer1::<u32>(uf_width * uf_height * 2)?;

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
		ctx.tp.stamp("BKE bindgroups");

		self.dispatch_bke(ctx, src);
		// self.dispatch_ha4(ctx, src);
		// self.dispatch_count(ctx, src);

		#[cfg(feature="debug")]
		if ctx.config.debug() {
			ctx.cpass.insert_debug_marker("bke");
		}
		ctx.tp.stamp("bke");

		Ok(uf_parents)
    }
}