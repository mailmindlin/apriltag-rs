#![cfg(feature="debug")]

use std::io::ErrorKind;

use futures::executor::block_on;
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

use crate::{util::{image::ImageWritePNM, ImageBuffer, ImageY8}, wgpu::{error::GpuBufferFetchError, util::{buffer_traits::GpuImageLike, texture::GpuTexture, GpuBufferFetch}, GpuContext}, DetectorConfig};

use super::{GpuTextureY8, GpuImageDownload};

/// Calculate which targets should be downloaded from the GPU for debugging
pub(in super::super) struct DebugTargets {
	pub(crate) download_src: bool,
	pub(crate) download_decimate: bool,
	pub(crate) download_sigma: bool,
}

impl DebugTargets {
	pub(crate) fn new(det: &super::super::WGPUDetector, config: &DetectorConfig) -> Self {
		// Figure out which buffers we might want to download
		let mut download_src = false;
		let mut download_decimate = false;
		let mut download_sigma = false;
		if config.debug() {
			download_src = true;
			download_decimate = true;
			download_sigma = true;
		} else if det.quad_sigma.is_some() {
			download_sigma = true;
		} else if det.quad_decimate.is_some() {
			download_decimate = true;
		} else {
			download_src = true;
		}
		Self {
			download_src,
			download_decimate,
			download_sigma,
		}
	}
}

enum DebugEntry {
	Y8 {
		path: &'static str,
		image: GpuTextureY8,
	},
	MinMax {
		min_path: &'static str,
		max_path: &'static str,
		image: GpuTexture<[u8; 2]>,
	}
}
pub(in super::super) struct DebugImageGenerator {
	#[cfg(feature="debug")]
	debug: bool,
	#[cfg(feature="debug")]
	values: Vec<DebugEntry>,
}

impl DebugImageGenerator {
	pub(in super::super) const fn new(debug: bool) -> Self {
		Self {
			#[cfg(feature="debug")]
			debug,
			#[cfg(feature="debug")]
			values: Vec::new(),
		}
	}
	pub(in super::super) fn register(&mut self, image: &GpuTextureY8, path: &'static str) {
		if self.debug {
			self.values.push(DebugEntry::Y8 { path, image: image.clone() });
		}
	}

	pub(in super::super) fn register_minmax(&mut self, image: &GpuTexture<[u8; 2]>, min_path: &'static str, max_path: &'static str) {
		if self.debug {
			self.values.push(DebugEntry::MinMax { min_path, max_path, image: image.clone() });
		}
	}

	pub(in super::super) fn write_debug_images(self, context: &GpuContext, config: &DetectorConfig) {
		for entry in self.values.into_iter() {
			match entry {
				DebugEntry::Y8 { path, image } => {
					config.debug_image(path, |mut f| {
						let img_cpu = block_on(image.download_image(context)).unwrap();
						img_cpu.write_pnm(&mut f)
					});
				},
				DebugEntry::MinMax { min_path, max_path, image } => {
                    config.debug_image(min_path, |mut f| {
                        let (img_min, img_max) = split_minmax(context, &image)?;
                        config.debug_image(max_path, |mut f| img_max.write_pnm(&mut f));
                        img_min.write_pnm(&mut f)
                    });
				}
			}
		}
	}
}

fn fetch_minmax(ctx: &GpuContext, gpu_tex: &GpuTexture<[u8; 2]>) -> Result<ImageBuffer<[u8; 2]>, std::io::Error> {
    match block_on(gpu_tex.fetch_buffer(ctx)) {
        Ok(cpu_buf) => {
			let cpu_img =ImageBuffer::<[u8; 2]>::wrap(cpu_buf, gpu_tex.width(), gpu_tex.height(), (gpu_tex.width() * 2).next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT as usize));
			Ok(cpu_img)
		},
        Err(GpuBufferFetchError::NotReadable) => Err(std::io::Error::new(ErrorKind::PermissionDenied, "Buffer is not readable")),
        Err(GpuBufferFetchError::BufferMap(_)) => Err(std::io::Error::new(ErrorKind::Interrupted, "Could not map buffer")),
    }
}

fn split_minmax(ctx: &GpuContext, gpu_tex: &GpuTexture<[u8; 2]>) -> Result<(ImageY8, ImageY8), std::io::Error> {
    let cpu_img = fetch_minmax(ctx, gpu_tex)?;
    let mut im_min = ImageY8::zeroed(gpu_tex.width(), gpu_tex.height());
    let mut im_max = ImageY8::zeroed(gpu_tex.width(), gpu_tex.height());
    for y in 0..cpu_img.height() {
        for x in 0..cpu_img.width() {
            let v = cpu_img[(x, y)];
            im_min[(x, y)] = v[0];
            im_max[(x, y)] = 255 - v[1];
        }
    }
    Ok((im_min, im_max))
}