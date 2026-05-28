use std::{sync::Arc, marker::PhantomData};

use async_trait::async_trait;
use bytemuck::Pod;
use wgpu::{TextureFormat, SubmissionIndex, COPY_BYTES_PER_ROW_ALIGNMENT, BufferUsages, TextureUsages, CommandBuffer};

use crate::{util::{image::{ImageDimensions, Luma, Pixel}, mem::SafeZero, ImageBuffer}, wgpu::{error::GpuBufferFetchError, util::GpuBuffer2}};

use super::{buffer_traits::{GpuAwaitable, GpuImageLike}, GpuBufferFetch, GpuContext, GpuImageDownload};

pub(crate) trait GpuPixel: Pixel {
    const GPU_FORMAT: wgpu::TextureFormat;
}

impl GpuPixel for Luma<u8> {
    const GPU_FORMAT: wgpu::TextureFormat = TextureFormat::R8Uint;
}

impl GpuPixel for [u8; 2] {
    const GPU_FORMAT: wgpu::TextureFormat = TextureFormat::Rg8Uint;
}

#[derive(Clone)]
pub(crate) struct GpuTexture<P: GpuPixel> {
    pub(super) buffer: Arc<wgpu::Texture>,
    pub(crate) index: Option<SubmissionIndex>,
    pixel: PhantomData<P>,
}
pub(crate) type GpuTextureY8 = GpuTexture<Luma<u8>>;

impl<P: GpuPixel> GpuTexture<P> {
    pub(crate) fn new(buffer: wgpu::Texture) -> Self {
        debug_assert_eq!(P::GPU_FORMAT, buffer.format(), "Format mismatch");

        Self {
            buffer: Arc::new(buffer),
            index: None,
            pixel: PhantomData,
        }
    }

    /// Get as a GPU texture (for bindings)
    pub(crate) fn as_texture(&self) -> &wgpu::Texture {
        &self.buffer
    }

    pub(crate) fn as_view(&self) -> wgpu::TextureView {
        self.buffer.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        })
    }

    fn copy_to_buffer(&self, context: &GpuContext) -> (GpuBuffer2<P::Subpixel>, CommandBuffer) {
        let size_spx = std::mem::size_of::<P::Subpixel>();

        let buf = {
            let usage  = BufferUsages::COPY_DST | BufferUsages::MAP_READ;

            assert_eq!(self.stride() % size_spx, 0, "I'm not sure what happens here");
    
            let dims = ImageDimensions {
                width: self.width(),
                height: self.height(),
                stride: self.stride() / size_spx,
            };
            let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: ((dims.stride * self.height())) as u64,
                usage,
                mapped_at_creation: false,
            });
            GpuBuffer2::<P::Subpixel>::new(dims, buffer)
        };

        let cmds = {
            let mut enc = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_texture_to_buffer(
                self.buffer.as_image_copy(),
                buf.as_image_copy(),
                wgpu::Extent3d {
                    width: self.width() as u32,
                    height: self.height() as u32,
                    depth_or_array_layers: 1,
                }
            );
            enc.finish()
        };
        (buf, cmds)
    }
}

impl<P: GpuPixel> GpuImageLike for GpuTexture<P> {
    fn width(&self) -> usize {
        self.buffer.width() as usize
    }

    fn height(&self) -> usize {
        self.buffer.height() as usize
    }

    fn stride(&self) -> usize {
        (self.width() * P::CHANNEL_COUNT * std::mem::size_of::<P::Subpixel>())
            .next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT as usize)
    }
}

impl<P: GpuPixel> GpuAwaitable for GpuTexture<P> {
    type Context = GpuContext;

    fn submission_index(&self) -> Option<&SubmissionIndex> {
        self.index.as_ref()
    }

    fn poll(&self, _context: &Self::Context) {
    }
}

#[async_trait]
impl<P: GpuPixel> GpuBufferFetch<P::Subpixel> for GpuTexture<P> where P: Sync, P::Subpixel: Send + Sync + SafeZero + Pod {
    async fn fetch_buffer(&self, context: &Self::Context) -> Result<Box<[P::Subpixel]>, GpuBufferFetchError> {
        let buf = self.as_texture();

        let usage = buf.usage();
        if usage.contains(TextureUsages::COPY_SRC) {
            let (mut buf, cmd_buf) = self.copy_to_buffer(context);
            let sub_idx = context.queue.submit([cmd_buf]);
            buf.index = Some(sub_idx);
            buf.fetch_buffer(context).await
        // } else if self.buffer.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        //     todo!()
        } else {
            return Err(GpuBufferFetchError::NotReadable);
        }
    }
}

// enum GpuImageMapGuard<'a, P: Pixel> {
//     MappedBuffer(),
//     Owned,
// }

// #[async_trait]
// impl<P: GpuPixel> GpuImageMap<P> for GpuTexture<P> where P: Send + Sync + 'static {
//     type Guard<'a> = GpuImageMapGuard<'a, P>;
//     async fn map_image<'a>(&self, context: &Self::Context) -> Result<Self::Guard<'a>, GpuBufferFetchError> {
//         let buf = self.as_texture();
//         let size_spx = std::mem::size_of::<P::Subpixel>();

//         let usage = buf.usage();
//         if usage.contains(TextureUsages::COPY_SRC) {
//             let (buf, cmd_buf) = self.copy_to_buffer(context);
//             let sub_idx = context.queue.submit([cmd_buf]);
//             buf.index = Some(sub_idx);
//             buf.fetch_buffer(context).await
//         // } else if self.buffer.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING) {
//         //     todo!()
//         } else {
//             return Err(GpuBufferFetchError::NotReadable);
//         }
//     }
// }



#[async_trait]
impl<P: GpuPixel> GpuImageDownload<P> for GpuTexture<P> where P: Sync, P::Subpixel: Pod + SafeZero + Send + Sync {
    async fn download_image(&self, context: &Self::Context) -> Result<ImageBuffer<P>, GpuBufferFetchError> {
        let host_buf = self.fetch_buffer(context).await?;
		let host_img = ImageBuffer::wrap(host_buf, self.width(), self.height(), self.stride());
		Ok(host_img)
    }
}