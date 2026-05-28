#![cfg(feature="wgpu")]

#[cfg(test)]
mod test;
mod util;
mod stage1_img;
mod stage2_img;
mod stage3;
mod stage4;
mod stage4_bke;
mod error;

use futures::executor::block_on;
use futures::future::{join, join4, try_join4};
use wgpu::util::DeviceExt;
use wgpu::wgc::api::Metal;
use wgpu::{BufferUsages, ComputePass};

use self::util::debug::{DebugTargets, DebugImageGenerator};
use crate::dbg::{debug_enabled, debugln};
#[cfg(feature="debug")]
use crate::{dbg::debug_images, quad_thresh::debug_unionfind};
use crate::detector::{Preprocessor, ImageDimensionError};

use crate::quad_thresh::{gradient_clusters, unionfind::{UnionFindStatic, UnionFind, UnionFind2D}};
use crate::util::{image::{ImageRefY8, ImageDimensions}, multiple::lcm};
use crate::{DetectorBuildError, DetectorConfig, util::ImageY8, DetectError, TimeProfile};
use self::stage1_img::GpuQuadDecimate;
use self::stage2_img::GpuQuadSigma;
use self::stage3::GpuThreshim;
use self::stage4_bke::GpuBke;
use self::util::{DataStore, GpuBuffer1, GpuBuffer2, GpuBufferFetch, GpuContext, GpuImageDownload, GpuImageLike, GpuPixel, GpuStage, GpuTexture, GpuTextureY8, GpuTimestampQueries};
pub use self::error::{WgpuDetectError, WgpuBuildError};

struct GpuStageContext<'params, 'pass> {
	pub(super) context: &'params GpuContext,
	pub(super) tp: &'params mut TimeProfile,
	pub(super) config: &'params DetectorConfig,
	pub(super) next_read: bool,
	pub(super) next_align: usize,
	pub(super) cpass: ComputePass<'pass>,
	pub(super) stage_name: &'static str,
	pub(super) queries: &'params mut GpuTimestampQueries,
}

impl<'params, 'pass> GpuStageContext<'params, 'pass> {
	fn device(&self) -> &wgpu::Device {
		&self.context.device
	}

	fn params_buffer(&self, values: &[u32]) -> wgpu::Buffer {
		let res = self.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("params"),
			contents: bytemuck::cast_slice(values),
			usage: BufferUsages::UNIFORM,
		});
		res
	}

	/// Create intermediate (temporary) 2D GPU buffer
	fn tmp_buffer2<P>(&self, width: usize, height: usize, align: usize, label: &'static str) -> Result<GpuBuffer2<P>, WgpuDetectError> {
		self.context.temp_buffer2(width, height, align, false, label)
	}

	/// Create output 1D GPU buffer
	fn dst_buffer1<E>(&self, length: usize) -> Result<GpuBuffer1<E>, WgpuDetectError> {
		self.context.temp_buffer1(length, self.next_read, self.stage_name)
	}

	/// Create output 2D GPU buffer.
	/// The row stride is aligned to `lcm(next_align, align)` so the buffer
	/// satisfies both this stage's write alignment and the next stage's
	/// `src_alignment()` requirement (see `GpuStage` docs).
	fn dst_buffer2<P>(&self, width: usize, height: usize, align: usize) -> Result<GpuBuffer2<P>, WgpuDetectError> {
		let align = lcm(self.next_align, align);
		self.context.temp_buffer2(width, height, align, self.next_read, self.stage_name)
	}

	/// Create output GPU texture
	fn dst_texture<P: GpuPixel>(&self, width: usize, height: usize) -> GpuTexture<P> {
		self.context.temp_texture(width, height, self.next_read, self.stage_name)
	}
}

pub(crate) struct WGPUDetector {
	context: GpuContext,
	quad_decimate: Option<GpuQuadDecimate>,
	quad_sigma: Option<GpuQuadSigma>,
	threshim: GpuThreshim,
	unionfind: GpuBke,
}

impl WGPUDetector {
	/// Create new detector
	pub(super) async fn new_async(config: &DetectorConfig) -> Result<Self, DetectorBuildError> {
		let context = GpuContext::new(config).await?;

		let (
			quad_decimate,
			quad_sigma,
			threshim,
			unionfind,
		) = try_join4(
			GpuQuadDecimate::new(&context, config),
			GpuQuadSigma::new(&context, config.quad_sigma),
			GpuThreshim::new(&context, config),
			GpuBke::new(&context, config)
		).await?;

		Ok(Self {
			context,
			quad_decimate,
			quad_sigma,
			threshim,
			unionfind,
		})
	}

	/// Create new detector (blocking call)
	pub(super) fn new(config: &DetectorConfig) -> Result<Self, DetectorBuildError> {
		block_on(Self::new_async(config))
	}

	/// Get information about the GPU device being used.
	pub(crate) fn device_info(&self) -> super::detector::GpuDeviceInfo {
		let info = &self.context.adapter_info;
		super::detector::GpuDeviceInfo {
			accelerator: "wgpu".into(),
			backend: format!("{:?}", info.backend),
			name: info.name.clone(),
			device_type: format!("{:?}", info.device_type),
			vendor: format!("{:#x}", info.vendor),
			driver: info.driver.clone(),
		}
	}

	/// Upload source image to GPU
	fn upload_texture(&self, downloadable: bool, image: &ImageRefY8, label: &'static str) -> Result<GpuTextureY8, ImageDimensionError> {
		self.context.upload_texture(downloadable, image, label)
	}

	/// Upload source image to GPU
	fn upload_image(&self, downloadable: bool, row_alignment: usize, image: &ImageRefY8) -> GpuBuffer2<u8> {
		self.context.upload_image(downloadable, row_alignment, image)
	}

	/// Generate preprocessing
	fn apply_preprocess(&self, config: &DetectorConfig, tp: &mut TimeProfile, gpu_src: GpuTextureY8, debug: &mut DebugImageGenerator, cpass: ComputePass, queries: &mut GpuTimestampQueries) -> Result<(GpuTextureY8, GpuTextureY8), WgpuDetectError> {
		let gpu_last = gpu_src;
		let downloads = DebugTargets::new(self, config);

		// Data for stages (must outlive context)
		let mut quad_decimate_bg = Default::default();
		let mut quad_sigma_bg = Default::default();
		let mut data_threshim = Default::default();

		let mut ctx = GpuStageContext {
			tp,
			config,
			context: &self.context,
			cpass,
			next_align: 4,
			next_read: downloads.download_decimate,
			queries,
			stage_name: "quad_decimate",
		};

		/// Apply a [GpuStage] if present
		fn opt_stage<'s: 'c, 'c, S: GpuStage<Source = D, Output = D>, D>(stage: &'s Option<S>, gpu_last: D, ctx: &mut GpuStageContext<'_, 'c>, temp: &'s mut DataStore<S::Data>) -> Result<D, WgpuDetectError> {
			match stage.as_ref() {
				Some(stage) => stage.apply(ctx, &gpu_last, temp),
				// Identity
				None => Ok(gpu_last),
			}
		}

		let gpu_last = opt_stage(&self.quad_decimate, gpu_last, &mut ctx, &mut quad_decimate_bg)
			.expect("Error in quad_decimate");

		#[cfg(feature="debug")]
		debug.register(&gpu_last, debug_images::DECIMATE);

		ctx.next_read = downloads.download_sigma;
		let gpu_last = opt_stage(&self.quad_sigma, gpu_last, &mut ctx, &mut quad_sigma_bg)
			.expect("Error in quad_simga");

		#[cfg(feature="debug")]
		debug.register(&gpu_last, debug_images::PREPROCESS);

		ctx.next_read = true;
		let gpu_threshim = self.threshim.apply(&mut ctx, &gpu_last, &mut data_threshim)
				.expect("Error in threshim");
		#[cfg(feature="debug")]
		{
			debug.register(&gpu_threshim.threshim, debug_images::THRESHOLD);
			debug.register_minmax(&gpu_threshim.tile_minmax, debug_images::TILE_MIN, debug_images::TILE_MAX);
			debug.register_minmax(&gpu_threshim.tile_blur, debug_images::BLUR_MIN, debug_images::BLUR_MAX);
		}
		
		Ok((gpu_last, gpu_threshim.threshim))
	}

	fn apply_ccl(&self, config: &DetectorConfig, tp: &mut TimeProfile, _debug: &mut DebugImageGenerator, gpu_threshim: GpuTextureY8, cpass: ComputePass, queries: &mut GpuTimestampQueries) -> Result<GpuBuffer1<u32>, WgpuDetectError> {
		let mut data_unionfind = Default::default();
		let mut ctx = GpuStageContext {
			context: &self.context,
			tp,
			config,
			next_read: true,
			next_align: 4,
			cpass,
			queries,
			stage_name: "UnionFind",
		};

		let gpu_uf = self.unionfind.apply(&mut ctx, &gpu_threshim, &mut data_unionfind)
			.expect("Error in unionfind");

		Ok(gpu_uf)
	}
}

impl Preprocessor for WGPUDetector {
	fn preprocess(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<(ImageY8, ImageY8), DetectError> {
		let downloads = DebugTargets::new(self, config);

		let mut debug = DebugImageGenerator::new(config.generate_debug_image());

		// Upload source image to GPU
		let gpu_src = self.upload_texture(downloads.download_src, &image.as_ref(), "preprocess_src")
			.map_err(DetectError::BadSourceImageDimensions)?;

		// Check that our image wasn't modified
		#[cfg(feature = "wgpu_validate")]
		{
			let cpu_src = block_on(gpu_src.download_image(&self.context)).unwrap();
			assert_eq!(image.as_ref(), cpu_src.as_ref());
			drop(cpu_src);
		}

		// #[cfg(feature="debug")]
		// config.debug_image(debug_images::SOURCE, |mut f| self.debug_image(&mut f, &gpu_src));

		let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("AprilTag wgpu preprocess") });
		let mut queries = self.context.make_queries(32);
		let (gpu_quad, gpu_threshim) = {
			let cpass = encoder.begin_compute_pass(&queries.make_cpd("apriltag preprocess"));
			self.apply_preprocess(config, tp, gpu_src, &mut debug, cpass, &mut queries)?
		};
		queries.resolve(&mut encoder);
		self.context.submit([encoder.finish()]);
		tp.stamp("GPU submit");

		// These have to execute after queue.submit()
		#[cfg(feature="debug")]
		if config.generate_debug_image() {
			debug.write_debug_images(&self.context, config);
			tp.stamp("GPU fetch debug");
		}
		async fn download_results(context: &GpuContext, gpu_quad: &GpuTextureY8, gpu_threshim: &GpuTextureY8) -> Result<(ImageY8, ImageY8), WgpuDetectError> {
			let fut_quad = gpu_quad.download_image(context);
			let fut_threshim = gpu_threshim.download_image(context);
			let (res_quad, res_threshim) = join(fut_quad, fut_threshim).await;
			Ok((res_quad?, res_threshim?))
		}
		let (quad_im, threshim) = block_on(download_results(&self.context, &gpu_quad, &gpu_threshim))?;
		tp.stamp("GPU fetch results");
		Ok((quad_im, threshim))
	}

	fn cluster(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<crate::detector::ClusterResult, DetectError> {
		let downloads = DebugTargets::new(self, config);

		let mut debug = DebugImageGenerator::new(config.generate_debug_image());

		let gpu_src = self.upload_texture(
			downloads.download_src || cfg!(feature = "wgpu_validate"),
			&image.as_ref(),
			"cluster_src"
		).map_err(DetectError::BadSourceImageDimensions)?;

		// Check that our image wasn't modified
		#[cfg(feature = "wgpu_validate")]
		{
			let cpu_src = block_on(gpu_src.download_image(&self.context)).unwrap();
			assert_eq!(image.as_ref(), cpu_src.as_ref(), "GPU source upload/download round-trip mismatch");
			drop(cpu_src);
		}

		// #[cfg(feature="debug")]
		// config.debug_image(debug_images::SOURCE, |mut f| self.debug_image(&mut f, &gpu_src));

		let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
		let mut queries = self.context.make_queries(4);

		queries.write_timestamp(&mut encoder, "preprocess->start".into());
		let (gpu_quad, gpu_threshim) = {
			let cpass = encoder.begin_compute_pass(&queries.make_cpd("apriltag preprocess"));
			self.apply_preprocess(config, tp, gpu_src, &mut debug, cpass, &mut queries)?
		};
		queries.write_timestamp(&mut encoder, "preprocess->end".into());
		let uf_dims = ImageDimensions {
			width: gpu_threshim.width() as _,
			height: gpu_threshim.height() as _,
			stride: 0,
		};
		tp.stamp("GPU submit preprocess");
		queries.write_timestamp(&mut encoder, "CCL->start".into());
		let gpu_uf = {
			let cpass = encoder.begin_compute_pass(&queries.make_cpd("apriltag CCL"));
			self.apply_ccl(config, tp, &mut debug, gpu_threshim.clone(), cpass, &mut queries)?
		};
		queries.write_timestamp(&mut encoder, "CCL->end".into());
		queries.resolve(&mut encoder);

		self.context.submit([encoder.finish()]);
		tp.stamp("GPU submit ccl");

		// This has to execute after queue.submit()
		#[cfg(feature="debug")]
		debug.write_debug_images(&self.context, config);

		async fn download_results(context: &GpuContext, gpu_quad: GpuTextureY8, gpu_threshim: GpuTextureY8, gpu_uf: GpuBuffer1<u32>, queries: GpuTimestampQueries) -> Result<(ImageY8, ImageY8, Box<[u32]>, Option<TimeProfile>), WgpuDetectError> {
			let fut_quad = gpu_quad.download_image(context);
			let fut_threshim = gpu_threshim.download_image(context);
			let fut_uf = gpu_uf.fetch_buffer(context);
			let (
				res_quad,
				res_threshim,
				res_uf,
				queries_tp
			) = join4(
				fut_quad,
				fut_threshim,
				fut_uf,
				queries.wait_for_results(context),
			).await;
			let gpu_tp = queries_tp?;
			Ok((res_quad?, res_threshim?, res_uf?, gpu_tp))
		}
		let (quad_im, threshim, unionfind, gpu_tp) = block_on(download_results(&self.context, gpu_quad, gpu_threshim, gpu_uf, queries))?;
		tp.stamp("GPU fetch results");

		#[cfg(feature = "wgpu_validate")]
		{
			use crate::quad_thresh::threshold::threshold;

			// Compare GPU threshim against CPU reference applied to the same (GPU-decimated+sigma) image.
			let cpu_threshim = threshold(config, &mut TimeProfile::default(), quad_im.as_ref()).unwrap();
			assert_eq!(cpu_threshim.as_ref(), threshim.as_ref(),
				"GPU threshim does not match CPU reference (threshold stage bug)");

			// Assert the CCL count pass ran: if every size field is 0 the count
			// shader was never dispatched (incomplete CCL implementation).
			let all_zero_sizes = unionfind.chunks_exact(2).all(|c| c[1] == 0);
			assert!(!all_zero_sizes,
				"GPU CCL: all component sizes are 0 — count pass is not running (dispatch_count is commented out)");
		}
		
		if debug_enabled() {
			debugln!("Uf width: {}", uf_dims.width);
			let print_row = |y| {
				let off = 0;
				let n = 64;
				let threshim_line: String = (0..n).map(|i| {
					match threshim[(i + off, y)] {
						0 => '-',
						255 => '#',
						_ => '_',
					}
				}).collect();
				let base = uf_dims.width * y + off;
				let uf_line: String = (0..n).map(|i| {
					#[allow(clippy::identity_op)]
					let par = &unionfind[(base + i)*2+0];
					let siz = &unionfind[(base + i)*2+1];
					format!("({par},{siz}), ")
				}).collect();
				debugln!("Threshim data: {threshim_line}");
				debugln!("Uf data: {uf_line}");
			};
			for i in 0..5 {
				print_row(i);
			}
		}

		struct SimpleUnionFind(Box<[u32]>);

		impl UnionFind<u32> for SimpleUnionFind {
			type Id = u32;

			fn get_set(&mut self, index: u32) -> (Self::Id, u32) {
				self.get_set_static(index)
			}

			fn index_to_id(&self, idx: u32) -> Self::Id {
				idx
			}

			fn connect_ids(&mut self, _a: Self::Id, _b: Self::Id) -> bool {
				todo!()
			}
		}

		impl UnionFindStatic<u32> for SimpleUnionFind {
			fn get_set_static(&self, mut index: u32) -> (Self::Id, u32) {
				for _ in 0..1000 {
					let parent = self.0[index as usize * 2];
					// if parent == 0 {
					// 	return (index, 0);
					// } else if parent - 1 == index {
					// 	return (index, self.0[index as usize * 2 + 1]);
					// }
					if parent == index {
						return (index, self.0[index as usize * 2 + 1]);
					}
					index = parent;
				}
				(index, 0)
			}

			fn get_set_hops(&self, mut index: u32) -> usize {
				for hops in 0..1000 {
					let parent = self.0[index as usize * 2];
					if parent == index {
						return hops;
					}
					index = parent;
				}
				usize::MAX
			}
		}
		let inner = SimpleUnionFind(unionfind);
		if debug_enabled() {
			for i in 0..5 {
				debugln!("Parent({i}) = {:?}", inner.get_set_static(i));
			}
		}
		let uf2 = UnionFind2D::wrap(uf_dims.width, uf_dims.height, inner);

		#[cfg(feature="debug")]
		debug_unionfind(config, tp, &uf_dims, &uf2);

		#[cfg(feature = "wgpu_validate")]
		{
			use std::collections::HashMap;
			use crate::quad_thresh::unionfind::connected_components;

			let cpu_uf = connected_components(config, &threshim);
			let mut gpu_to_cpu: HashMap<u32, u32> = HashMap::new();
			let mut cpu_to_gpu: HashMap<u32, u32> = HashMap::new();
			let mut max_gpu_hops: usize = 0;

			for y in 0..(uf_dims.height as u32) {
				for x in 0..(uf_dims.width as u32) {
					let (g_rep, _) = uf2.get_set_static((x, y));
					let (c_rep, _) = cpu_uf.get_set_static((x, y));

					if let Some(&expected_cpu) = gpu_to_cpu.get(&g_rep) {
						assert_eq!(expected_cpu, c_rep,
							"GPU CCL partition mismatch at ({x},{y}): GPU component {g_rep} maps to CPU component {expected_cpu}, but this pixel has CPU component {c_rep}");
					} else {
						gpu_to_cpu.insert(g_rep, c_rep);
					}

					if let Some(&expected_gpu) = cpu_to_gpu.get(&c_rep) {
						assert_eq!(expected_gpu, g_rep,
							"GPU CCL partition mismatch at ({x},{y}): CPU component {c_rep} maps to GPU component {expected_gpu}, but this pixel has GPU component {g_rep}");
					} else {
						cpu_to_gpu.insert(c_rep, g_rep);
					}

					max_gpu_hops = max_gpu_hops.max(uf2.get_set_hops((x, y)));
				}
			}
			assert!(max_gpu_hops <= 2,
				"GPU CCL: union-find chain depth is {max_gpu_hops} (expected ≤ 2 after BKE compression)");
		}

		let clusters = gradient_clusters(config, &threshim.as_ref(), uf2);
		Ok((quad_im, clusters, gpu_tp))
	}
}