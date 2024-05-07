mod texture;
mod async_buffer;
mod buffer;
pub(super) mod buffer_traits;
pub(super) mod debug;
mod program;
mod query;
pub(super) mod dev_select;

use std::mem::size_of;

use wgpu::{util::{BufferInitDescriptor, DeviceExt}, BufferDescriptor, BufferUsages, CommandBuffer, DeviceLostReason, SubmissionIndex, TextureDescriptor, TextureUsages};

use crate::{detector::ImageDimensionError, util::image::{ImageDimensions, ImageRefY8}, wgpu::{error::WgpuBuildError, util::dev_select::select_adapter}, DetectorConfig};

use self::dev_select::request_device;

use super::{GpuStageContext, WgpuDetectError};
pub(super) use program::{ProgramBuilder, ComputePipelineDescriptor};
pub(super) use texture::{GpuPixel, GpuTexture, GpuTextureY8};
pub(super) use buffer::{GpuBuffer1, GpuBuffer2};
pub(super) use buffer_traits::{GpuBufferFetch, GpuImageDownload, GpuImageLike};
pub(super) use query::GpuTimestampQueries;

pub(crate) struct GpuContext {
	pub(in super::super) device: wgpu::Device,
	queue: wgpu::Queue,
	// === GPU info ===
	/// Can 
	mappable_primary: bool,
	/// Maximum allowed size for 2d texture
	max_texture_dimension_2d: u32,
	/// Maximum allowed workgroup X
	max_compute_workgroup_size_x: u32,
	/// Maximum allowed workgroup Y
	max_compute_workgroup_size_y: u32,
	max_compute_invocations_per_workgroup: u32,
	max_compute_workgroups_per_dimension: u32,
}

fn device_lost_callback(reason: DeviceLostReason, msg: String) {
	println!("wgpu device lost: {reason:?} {msg}");
}
impl GpuContext {
	/// Attaches to a device
	pub(super) async fn new(config: &DetectorConfig) -> Result<Self, WgpuBuildError> {
		let adapter = select_adapter(config).await?;
		println!("Selected wgpu adapter: {:?}", adapter.get_info());

		let (device, queue) = request_device(adapter, config).await?;

		device.set_device_lost_callback(device_lost_callback);
		
		let dev_features = device.features();
		let mappable_primary = dev_features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
		let dev_limits = device.limits();
		
		Ok(Self {
			device,
			queue,
			mappable_primary,
			max_texture_dimension_2d: dev_limits.max_texture_dimension_2d,
			max_compute_workgroup_size_x: dev_limits.max_compute_workgroup_size_x,
			max_compute_workgroup_size_y: dev_limits.max_compute_workgroup_size_y,
			max_compute_invocations_per_workgroup: dev_limits.max_compute_invocations_per_workgroup,
			max_compute_workgroups_per_dimension: dev_limits.max_compute_workgroups_per_dimension,
		})
	}

	pub(super) fn check_texture_dims(&self, width: usize, height: usize) -> Result<(), ImageDimensionError> {
		let maximum = self.max_texture_dimension_2d as usize;
		if width > maximum {
			return Err(ImageDimensionError::WidthTooBig { actual: width, maximum });
		}
		if height > maximum {
			return Err(ImageDimensionError::HeightTooBig { actual: height, maximum });
		}

		Ok(())
	}

	/// Submit commands to queue
	pub(super) fn submit(&self, command_buffers: impl IntoIterator<Item = CommandBuffer>) -> SubmissionIndex {
		self.queue.submit(command_buffers)
	}

    fn buffer_usage(&self, read: bool) -> BufferUsages {
        if read {
			if self.mappable_primary {
				BufferUsages::STORAGE | BufferUsages::MAP_READ
			} else {
				BufferUsages::STORAGE | BufferUsages::COPY_SRC
			}
		} else {
			BufferUsages::STORAGE
		}
    }

	pub(super) fn temp_buffer1<E>(&self, length: usize, read: bool, label: Option<&'static str>) -> Result<GpuBuffer1<E>, WgpuDetectError> {
		let usage  = self.buffer_usage(read);

		self.temp_buffer1_usage(length, usage, label)
	}

	fn temp_buffer1_usage<E>(&self, length: usize, usage: BufferUsages, label: Option<&'static str>) -> Result<GpuBuffer1<E>, WgpuDetectError> {
		let buffer = self.device.create_buffer(&BufferDescriptor {
			label,
			size: (length * std::mem::size_of::<E>()) as u64,
			usage,
			mapped_at_creation: false,
		});
		Ok(GpuBuffer1::new(buffer, length))
	}

    pub(super) fn temp_buffer2<P>(&self, width: usize, height: usize, align: usize, read: bool, label: Option<&'static str>) -> Result<GpuBuffer2<P>, WgpuDetectError> {
		let usage  = self.buffer_usage(read);

		debug_assert_ne!(align, 0);
		//TODO: relevant?
		debug_assert_eq!(align % size_of::<P>(), 0);

		let dims = ImageDimensions {
			width,
			height,
			stride: (width * size_of::<P>()).next_multiple_of(align),
		};
		let buffer = self.device.create_buffer(&BufferDescriptor {
			label,
			size: ((dims.stride * height)) as u64,
			usage,
			mapped_at_creation: false,
		});
		Ok(GpuBuffer2::new(dims, buffer))
	}

	pub(super) fn temp_texture<P: GpuPixel>(&self, width: usize, height: usize, read: bool, label: Option<&'static str>) -> GpuTexture<P> {
		let usage  = if read {
			TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC
		} else {
			TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING
		};

		let format = <P as GpuPixel>::GPU_FORMAT;

		let tex = self.device.create_texture(&TextureDescriptor {
			label,
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

	/// Upload a CPU image to GPU as a texture
	pub(super) fn upload_texture(&self, downloadable: bool, image: &ImageRefY8) -> Result<GpuTextureY8, ImageDimensionError> {
		let usage = if downloadable {
			TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST
		} else {
			TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST
		};

		self.check_texture_dims(image.width(), image.height())?;

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

		Ok(res)
	}

	/// Upload CPU image as to GPU as a buffer
	pub(super) fn upload_image(&self, downloadable: bool, row_alignment: usize, image: &ImageRefY8) -> GpuBuffer2<u8> {
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

		GpuBuffer2::new(dims, buffer)
	}

	pub(super) fn make_queries(&self, count: u32) -> GpuTimestampQueries {
		if self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
			GpuTimestampQueries::new(self, count as _).unwrap()
		} else {
			GpuTimestampQueries::empty()
		}
	}
}

pub(super) trait GpuStage {
    type Data;
    type Source;
    type Output;

    fn src_alignment(&self) -> usize;

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WgpuDetectError>;
}

pub(super) struct DataStore<T>(Option<T>);

impl<T> DataStore<T> {
    pub(super) fn store<'a>(&'a mut self, value: T) -> &'a T {
        self.0 = Some(value);
        self.0.as_ref().unwrap()
    }
}

impl<T> AsRef<Option<T>> for DataStore<T> {
    fn as_ref(&self) -> &Option<T> {
        &self.0
    }
}

impl<T> Default for DataStore<T> {
    fn default() -> Self {
        Self(None)
    }
}