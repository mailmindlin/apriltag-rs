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
use wgpu::{BufferUsages, ComputePass};

use crate::dbg::debug_images;
use crate::detector::{Preprocessor, ImageDimensionError};

use crate::quad_thresh::{debug_unionfind, gradient_clusters};
use crate::quad_thresh::unionfind::{UnionFindStatic, UnionFind, UnionFind2D};
use crate::util::image::{ImageRefY8, ImageDimensions};
use crate::util::multiple::lcm;
use crate::wgpu::util::debug::DebugTargets;
use crate::wgpu::util::{GpuStage, GpuBuffer1, GpuBufferFetch, GpuImageLike, GpuTimestampQueries};
use crate::DetectorBuildError;
use crate::{DetectorConfig, util::ImageY8, DetectError, TimeProfile};
use self::stage1_img::GpuQuadDecimate;
use self::stage2_img::GpuQuadSigma;
use self::stage3::GpuThreshim;
use self::stage4_bke::GpuBke;
use self::util::debug::DebugImageGenerator;
use self::util::{GpuPixel, GpuTexture, GpuBuffer2, GpuTextureY8, GpuImageDownload};
pub(self) use self::util::GpuContext;
pub use self::error::{WgpuDetectError, WgpuBuildError};

pub(self) struct GpuStageContext<'a> {
	pub(super) context: &'a GpuContext,
	pub(super) tp: &'a mut TimeProfile,
	pub(super) config: &'a DetectorConfig,
	pub(super) next_read: bool,
	pub(super) next_align: usize,
	pub(super) cpass: ComputePass<'a>,
	pub(super) stage_name: Option<&'static str>,
	pub(super) queries: &'a mut GpuTimestampQueries,
}

impl<'a> GpuStageContext<'a> {
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
	fn tmp_buffer2<P>(&self, width: usize, height: usize, align: usize, label: Option<&'static str>) -> Result<GpuBuffer2<P>, WgpuDetectError> {
		self.context.temp_buffer2(width, height, align, false, label)
	}

	/// Create output 1D GPU buffer
	fn dst_buffer1<E>(&self, length: usize) -> Result<GpuBuffer1<E>, WgpuDetectError> {
		self.context.temp_buffer1(length, self.next_read, self.stage_name)
	}

	/// Create output 2D GPU buffer
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

	/// Upload source image to GPU
	fn upload_texture(&self, downloadable: bool, image: &ImageRefY8) -> Result<GpuTextureY8, ImageDimensionError> {
		self.context.upload_texture(downloadable, image)
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
			stage_name: Some("quad_decimate"),
		};

		let gpu_last = match &self.quad_decimate {
			None => gpu_last,
			Some(qd) =>
				qd.apply(&mut ctx, &gpu_last, &mut quad_decimate_bg)
					.expect("Error in quad_decimate"),
		};
		debug.register(&gpu_last, debug_images::DECIMATE);

		ctx.next_read = downloads.download_sigma;
		let gpu_last = match &self.quad_sigma {
			None => gpu_last,
			Some(qd) => qd.apply(&mut ctx, &gpu_last, &mut quad_sigma_bg)
				.expect("quad_simga"),
		};
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
			stage_name: Some("UnionFind"),
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
		let gpu_src = self.upload_texture(downloads.download_src, &image.as_ref())
			.map_err(|e| DetectError::BadSourceImageDimensions(e))?;

		// Check that our image wasn't modified
		#[cfg(all(debug_assertions, feature="extra_debug"))]
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

	fn cluster(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<(ImageY8, crate::quad_thresh::Clusters), DetectError> {
		let downloads = DebugTargets::new(self, config);

		let mut debug = DebugImageGenerator::new(config.generate_debug_image());

		let gpu_src = self.upload_texture(downloads.download_src, &image.as_ref())
			.map_err(|e| DetectError::BadSourceImageDimensions(e))?;

		// Check that our image wasn't modified
		#[cfg(all(debug_assertions, feature="extra_debug"))]
		{
			let cpu_src = block_on(gpu_src.download_image(&self.context)).unwrap();
			assert_eq!(image.as_ref(), cpu_src.as_ref());
			drop(cpu_src);
		}

		// #[cfg(feature="debug")]
		// config.debug_image(debug_images::SOURCE, |mut f| self.debug_image(&mut f, &gpu_src));

		let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
		let mut queries = self.context.make_queries(32);

		let (gpu_quad, gpu_threshim) = {
			let cpass = encoder.begin_compute_pass(&queries.make_cpd("apriltag preprocess"));
			self.apply_preprocess(config, tp, gpu_src, &mut debug, cpass, &mut queries)?
		};
		let uf_dims = ImageDimensions {
			width: gpu_threshim.width() as _,
			height: gpu_threshim.height() as _,
			stride: 0,
		};
		tp.stamp("GPU submit preprocess");
		let gpu_uf = {
			let cpass = encoder.begin_compute_pass(&queries.make_cpd("apriltag CCL"));
			self.apply_ccl(config, tp, &mut debug, gpu_threshim.clone(), cpass, &mut queries)?
		};
		queries.resolve(&mut encoder);

		self.context.submit([encoder.finish()]);
		tp.stamp("GPU submit ccl");

		// This has to execute after queue.submit()
		#[cfg(feature="debug")]
		debug.write_debug_images(&self.context, config);

		async fn download_results(context: &GpuContext, gpu_quad: GpuTextureY8, gpu_threshim: GpuTextureY8, gpu_uf: GpuBuffer1<u32>, queries: GpuTimestampQueries) -> Result<(ImageY8, ImageY8, Box<[u32]>), WgpuDetectError> {
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
			if let Some(tp) = queries_tp? {
				println!("GPU TP: {tp}");
			}
			Ok((res_quad?, res_threshim?, res_uf?))
		}
		let (quad_im, threshim, unionfind) = block_on(download_results(&self.context, gpu_quad, gpu_threshim, gpu_uf, queries))?;
		tp.stamp("GPU fetch results");
		
		println!("Uf width: {}", uf_dims.width);
		let print_row = |y| {
			print!("Threshim data: ");
			let off = 0;
			let n = 64;
			for i in 0..n {
				let pix = threshim[(i + off, y)];
				match pix {
					0 => print!("-"),
					255 => print!("#"),
					_ => print!("_"),
				}
			}
			println!();
			print!("Uf data: ");
			let base = uf_dims.width * y + off;
			for i in 0..n {
				let par = &unionfind[(base + i)*2+0];
				let siz = &unionfind[(base + i)*2+1];
				print!("({:},{}), ", par, siz);
				// print!("{}, ", siz);
			}
			println!();
		};
		for i in 0..5 {
			print_row(i);
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
						// return (index, self.0[index as usize * 2 + 1]);
						return (index, 10);
					}
					index = parent;
				}
				return (index, 0);
			}

			fn get_set_hops(&self, mut index: u32) -> usize {
				let mut hops = 0;
				for _ in 0..1000 {
					let parent = self.0[index as usize * 2];
					if parent == index {
						return hops;
					}
					index = parent;
					hops += 1;
				}
				return usize::MAX;
			}
		}
		let inner = SimpleUnionFind(unionfind);
		for i in 0..5 {
			println!("Parent({i}) = {:?}", inner.get_set_static(i));
		}
		let uf2 = UnionFind2D::wrap(uf_dims.width, uf_dims.height, inner);

		#[cfg(feature="debug")]
		debug_unionfind(config, tp, &uf_dims, &uf2);
		let clusters = gradient_clusters(config, &threshim.as_ref(), uf2);
		Ok((quad_im, clusters))
	}
}