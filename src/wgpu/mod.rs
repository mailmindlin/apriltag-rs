#![cfg(feature="wgpu")]

#[cfg(test)]
mod test;
// mod stage1;
// mod stage2;
mod util;
mod stage1_img;
mod buffer;
mod stage2_img;
mod stage3;
// mod stage4;

use std::mem::size_of;
use futures::executor::block_on;
use futures::future;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use wgpu::{
	RequestAdapterOptions, PowerPreference, BufferUsages, BufferDescriptor, TextureUsages, TextureDescriptor, ComputePass,
};

use crate::detector::Preprocessor;
use crate::detector::config::GpuAccelRequest;
use crate::util::image::{ImageWritePNM, ImageRefY8, Luma};
use crate::util::mem::SafeZero;
use crate::util::multiple::lcm;
use crate::wgpu::buffer::{GpuStage, GpuImageLike, split_minmax};
use crate::DetectorBuildError;
use crate::{DetectorConfig, util::{ImageY8, image::ImageDimensions}, DetectError, TimeProfile};

use self::buffer::{GpuTexture, GpuImageDownload, GpuBuffer, GpuPixel};
// use self::stage1::GpuQuadDecimate;
use self::stage1_img::GpuQuadDecimate;
use self::stage2_img::GpuQuadSigma;
use self::stage3::GpuThreshim;
// use self::stage2::WQuadSigma;

pub(self) struct GpuContext {
	device: wgpu::Device,
	queue: wgpu::Queue,
	mappable_primary: bool,
}

impl GpuContext {
	fn temp_buffer<P: GpuPixel + SafeZero>(&self, width: usize, height: usize, align: usize, read: bool) -> Result<GpuBuffer<P>, WGPUError> {
		let usage  = if read {
			if self.mappable_primary {
				BufferUsages::STORAGE | BufferUsages::MAP_READ
			} else {
				BufferUsages::STORAGE | BufferUsages::COPY_SRC
			}
		} else {
			BufferUsages::STORAGE
		};

		debug_assert_ne!(align, 0);
		//TODO: relevant?
		debug_assert_eq!(align % size_of::<u32>(), 0);


		let dims = ImageDimensions {
			width,
			height,
			stride: (width * P::CHANNEL_COUNT).next_multiple_of(align),
		};
		let buffer = self.device.create_buffer(&BufferDescriptor {
			label: None,
			size: ((dims.stride * height)) as u64,
			usage,
			mapped_at_creation: false,
		});
		Ok(GpuBuffer::new(dims, buffer))
	}

	fn temp_texture<P: GpuPixel>(&self, width: usize, height: usize, read: bool) -> GpuTexture<P> {
		let usage  = if read {
			TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC
		} else {
			TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING
		};

		let format = <P as GpuPixel>::GPU_FORMAT;

		let tex = self.device.create_texture(&TextureDescriptor {
			label: None,
			size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format,
			usage,
			view_formats: &[format],
		});
		let res = GpuTexture::new(tex);
		debug_assert_eq!(res.width(), width);
		debug_assert_eq!(res.height(), height);
		res
	}

	fn upload_texture(&self, downloadable: bool, image: &ImageRefY8) -> GpuTexture {
		let usage = if downloadable {
			TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST
		} else {
			TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST
		};

		let size = wgpu::Extent3d {
			width: image.width() as _,
			height: image.height() as _,
			depth_or_array_layers: 1,
		};

		let tex = self.device.create_texture(&wgpu::TextureDescriptor {
			label: None,
			size: size.clone(),
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::R8Uint,
			usage,
			view_formats: &[
				wgpu::TextureFormat::R8Uint,
			],
		});
		self.queue.write_texture(
			tex.as_image_copy(),
			image.container(),
			wgpu::ImageDataLayout {
				offset: 0,
				bytes_per_row: Some(image.stride() as _),
				rows_per_image: Some(image.height() as _),
			},
			size
		);

		let res = GpuTexture::new(tex);
		debug_assert_eq!(res.width(), image.width());
		debug_assert_eq!(res.height(), image.height());

		res
	}

	fn upload_image(&self, downloadable: bool, row_alignment: usize, image: &ImageRefY8) -> GpuBuffer<Luma<u8>> {
		let usage  = if downloadable {
			BufferUsages::STORAGE | BufferUsages::COPY_SRC
		} else {
			BufferUsages::STORAGE
		};
		let label = None;
		//TODO: make async?
		let (buffer, dims) = if (image.stride() % row_alignment == 0) && image.width() > (image.stride() * 3 / 2) {
			// Upload without copying
			// We need stride to be a multiple of 4 because we're packing it as u8x4
			let buffer = self.device.create_buffer_init(&BufferInitDescriptor {
				label,
				contents: image.data,
				usage
			});
			(buffer, *image.dimensions())
		} else {
			let dims = ImageDimensions {
				width: image.dimensions().width,
				height: image.dimensions().height,
				stride: image.dimensions().width.next_multiple_of(row_alignment),
			};
			let buffer = self.device.create_buffer(&BufferDescriptor {
				label,
				size: (dims.height * dims.stride) as u64,
				usage,
				mapped_at_creation: true,
			});
			// Copy image data
			{
				let mut view = buffer.slice(..).get_mapped_range_mut();
				for y in 0..image.height() {
					let src = image.row(y).as_slice();
					let dst = &mut view[y * dims.stride..y*dims.stride + dims.width];
					dst.copy_from_slice(src);
				}
			}
			buffer.unmap();
			(buffer, dims)
		};

		GpuBuffer::new(dims, buffer)
	}
}

pub(self) struct GpuStageContext<'a> {
	pub(super) context: &'a GpuContext,
	pub(super) tp: &'a mut TimeProfile,
	pub(super) config: &'a DetectorConfig,
	pub(super) next_read: bool,
	pub(super) next_align: usize,
	pub(super) cpass: ComputePass<'a>,
}

impl<'a> GpuStageContext<'a> {
	fn device(&self) -> &wgpu::Device {
		&self.context.device
	}

	fn dst_buffer<P: GpuPixel + SafeZero>(&self, width: usize, height: usize, align: usize, read: bool) -> Result<GpuBuffer<P>, WGPUError> {
		let align = lcm(self.next_align, align);
		self.context.temp_buffer(width, height, align, read || self.next_read)
	}

	fn dst_texture<P: GpuPixel>(&self, width: usize, height: usize) -> GpuTexture<P> {
		self.context.temp_texture(width, height, self.next_read)
	}
}

#[derive(Debug)]
pub enum WGPUError {

}

/// Find GPU device
async fn find_device(mode: &GpuAccelRequest) -> Result<GpuContext, DetectorBuildError> {
	async fn find_adapter() -> Result<wgpu::Adapter, DetectorBuildError> {
		let inst = wgpu::Instance::default();
		let mut opts = RequestAdapterOptions::default();
		opts.power_preference = PowerPreference::HighPerformance;
		match inst.request_adapter(&opts).await {
			Some(adapter) => Ok(adapter),
			None => return Err(DetectorBuildError::GpuNotAvailable),
		}
	}
	let adapter = find_adapter().await?;
	assert!(adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS), "Needs compute shaders");

	let device_features = {
		let adapter_features = adapter.features();
		// let adapter_info = adapter.get_info();
		// println!("{adapter:?} {adapter_features:?} {adapter_info:?}");

		// Enable MAPPABLE_PRIMARY_BUFFERS
		let request_features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PIPELINE_STATISTICS_QUERY | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
		adapter_features.intersection(request_features)
	};
	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				label: Some("AprilTag GPU"),
				features: device_features,
				limits: wgpu::Limits::downlevel_defaults(),
			},
			None,
		)
		.await
		.map_err(|_| DetectorBuildError::GpuNotAvailable)?;
	
	// println!("Device features: {:?}", device.features());

	let features = device.features();
	let mappable_primary = features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
	
	Ok(GpuContext { device, queue, mappable_primary })
}

pub(crate) struct WGPUDetector {
	context: GpuContext,
	quad_decimate: Option<GpuQuadDecimate>,
	quad_sigma: Option<GpuQuadSigma>,
	threshim: GpuThreshim,
}

impl WGPUDetector {
	/// Create new detector
	async fn new_async(config: &DetectorConfig) -> Result<Self, DetectorBuildError> {
		let context = find_device(&config.gpu).await?;

		let quad_decimate = GpuQuadDecimate::new(&context, config)?;
		let quad_sigma = GpuQuadSigma::new(&context, config.quad_sigma);
		let threshim = GpuThreshim::new(&context, config);

		Ok(Self {
			context,
			quad_decimate,
			quad_sigma,
			threshim,
		})
	}

	pub(super) fn new(config: &DetectorConfig) -> Result<Self, DetectorBuildError> {
		block_on(Self::new_async(config))
	}

	#[cfg(feature="debug")]
	fn debug_image(&self, file: &mut std::fs::File, image: &(impl GpuImageDownload<Luma<u8>, Context = GpuContext> + Sync)) -> Result<(), std::io::Error> {
		let host_img = block_on(image.download_image(&self.context))?;
		host_img.write_pnm(file)
	}

	/// Upload source image to GPU
	fn upload_texture(&self, downloadable: bool, image: &ImageRefY8) -> GpuTexture {
		self.context.upload_texture(downloadable, image)
	}

	/// Upload source image to GPU
	fn upload_image(&self, downloadable: bool, row_alignment: usize, image: &ImageRefY8) -> GpuBuffer<Luma<u8>> {
		self.context.upload_image(downloadable, row_alignment, image)
	}
}

impl Preprocessor for WGPUDetector {
	fn preprocess(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<(ImageY8, ImageY8), DetectError> {
		// Figure out which buffers we might want to download
		let mut download_src = false;
		let mut download_decimate = false;
		let mut download_sigma = false;
		if config.debug() {
			download_src = true;
			download_decimate = true;
			download_sigma = true;
		} else if self.quad_sigma.is_some() {
			download_sigma = true;
		} else if self.quad_decimate.is_some() {
			download_decimate = true;
		} else {
			download_src = true;
		}

		let gpu_src = self.upload_texture(download_src, &image.as_ref());

		#[cfg(debug_assertions)]
		{
			let cpu_src = block_on(gpu_src.download_image(&self.context)).unwrap();
			assert_eq!(image.as_ref(), cpu_src.as_ref());
			drop(cpu_src);
		}

		#[cfg(feature="debug")]
		config.debug_image("00_debug_src.pnm", |mut f| self.debug_image(&mut f, &gpu_src));

		let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

		let (gpu_quad, gpu_threshim) = {
			// Data for stages (must outlive context)
			let mut quad_decimate_bg = Default::default();
			let mut quad_sigma_bg = Default::default();
			let mut data_threshim = Default::default();

			let mut ctx = GpuStageContext {
				tp,
				config,
				context: &self.context,
				cpass: encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("apriltag cp") }),
				next_align: 0,
				next_read: true,
			};

			let gpu_last = gpu_src;

			ctx.next_read = download_decimate;
			let gpu_last = match self.quad_decimate.as_ref() {
				None => gpu_last,
				Some(qd) =>
					qd.apply(&mut ctx, &gpu_last, &mut quad_decimate_bg)
						.expect("Error in quad_decimate"),
			};
			let gpu_decimated = gpu_last.clone();

			ctx.next_read = download_sigma;
			let gpu_last = match self.quad_sigma.as_ref() {
				None => gpu_last,
				Some(qd) => qd.apply(&mut ctx, &gpu_last, &mut quad_sigma_bg)
					.expect("quad_simga"),
			};
			let gpu_quad = gpu_last.clone();

			ctx.next_read = true;
			let gpu_threshim = self.threshim.apply(&mut ctx, &gpu_last, &mut data_threshim)
					.expect("Error in threshim");
			
			drop(ctx);
			let sub_idx = self.context.queue.submit([encoder.finish()]);
			tp.stamp("GPU submit");

			// These have to execute after queue.submit()
			#[cfg(feature="debug")]
			if config.debug() {
				config.debug_image("00a_debug_decimate.pnm", |mut f| self.debug_image(&mut f, &gpu_decimated));
				config.debug_image("01_debug_preprocess.pnm", |mut f| self.debug_image(&mut f, &gpu_quad));
				config.debug_image("02_debug_threshold.pnm", |mut f| self.debug_image(&mut f, &gpu_threshim));
				config.debug_image("02a_tile_minmax_min.pnm", |mut f| {
					let gpu_tile_minmax = &Option::as_ref(data_threshim.as_ref()).unwrap().tile_minmax;
					let (cpu_tile_min, cpu_tile_max) = split_minmax(&self.context, gpu_tile_minmax)?;
					cpu_tile_min.write_pnm(&mut f)?;
					config.debug_image("02b_tile_minmax_max.pnm", |mut f| cpu_tile_max.write_pnm(&mut f));
					Ok(())
				});

				config.debug_image("02c_tile_minmax_blur_min.pnm", |mut f| {
					let gpu_blur_minmax = &Option::as_ref(data_threshim.as_ref()).unwrap().tile_blur;
					let (cpu_blur_min, cpu_blur_max) = split_minmax(&self.context, gpu_blur_minmax)?;
					cpu_blur_min.write_pnm(&mut f)?;
					config.debug_image("02d_tile_minmax_blur_max.pnm", |mut f| cpu_blur_max.write_pnm(&mut f));
					Ok(())
				});
				tp.stamp("GPU fetch debug");
			}

			(gpu_quad, gpu_threshim)
		};
		async fn download_results(context: &GpuContext, gpu_quad: &GpuTexture, gpu_threshim: &GpuTexture) -> Result<(ImageY8, ImageY8), std::io::Error> {
			let fut_quad = gpu_quad.download_image(context);
			let fut_threshim = gpu_threshim.download_image(context);
			let (res_quad, res_threshim) = future::join(fut_quad, fut_threshim).await;
			Ok((res_quad?, res_threshim?))
		}
		let (quad_im, threshim) = block_on(download_results(&self.context, &gpu_quad, &gpu_threshim)).unwrap();
		tp.stamp("GPU fetch results");
		Ok((quad_im, threshim))
	}

	fn cluster(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<(ImageY8, crate::quad_thresh::Clusters), DetectError> {
		todo!()
	}
}