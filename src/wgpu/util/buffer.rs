use std::{marker::PhantomData, borrow::Borrow};

use async_trait::async_trait;
use bytemuck::Pod;
use wgpu::{SubmissionIndex, BufferUsages};

use crate::{util::{image::{ImageDimensions, Pixel}, mem::SafeZero, ImageBuffer}, wgpu::{error::GpuBufferFetchError, GpuContext}};

use super::{async_buffer::{read_mappable_buffer, MappedBufferGuard}, buffer_traits::{GpuAwaitable, GpuBuffer, GpuBufferFetch, GpuImageLike}, GpuImageDownload};

/// 1d buffer on GPU
pub(crate) struct GpuBuffer1<E> {
    /// GPU buffer handle
    pub(super) buffer: wgpu::Buffer,
    pub(super) index: Option<SubmissionIndex>,
    /// Number of elements
    len: usize,
    pixel: PhantomData<E>,
}

impl<E> GpuBuffer1<E> {
    pub(super) fn new(buffer: wgpu::Buffer, length: usize) -> Self {
        Self {
            buffer,
            index: None,
            len: length,
            pixel: PhantomData,
        }
    }
}

impl<E> GpuBuffer for GpuBuffer1<E> {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<E> GpuAwaitable for GpuBuffer1<E> {
    type Context = GpuContext;

    fn poll(&self, _context: &Self::Context) {
        // It's not clear if this wait is necessary
        // if let Some(sub_idx) = self.index.as_ref() {
        //     context.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(sub_idx.clone()));
        // }
    }

    fn submission_index(&self) -> Option<&SubmissionIndex> {
        self.index.as_ref()
    }
}

#[async_trait]
impl<E> GpuBufferFetch<E> for GpuBuffer1<E> where E: Send + Sync + SafeZero + Pod {
    async fn fetch_buffer(&self, context: &Self::Context) -> Result<Box<[E]>, GpuBufferFetchError> {
        let usage = self.buffer.usage();
        let res = if usage.contains(wgpu::BufferUsages::MAP_READ) {
            // Map buffer directly
            read_mappable_buffer::<E>(&context.device, &self.buffer, self.index.clone()).await
        } else if usage.contains(BufferUsages::COPY_SRC) {
            self.poll(context);

            // Copy to new buffer that *can* be read
            let tmp = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.buffer.size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false
            });

            let cmd_buf = {
                let mut enc = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Download") });
                enc.copy_buffer_to_buffer(&self.buffer, 0, &tmp, 0, self.buffer.size());
                enc.finish()
            };
            let sub_idx = context.queue.submit([cmd_buf]);
            read_mappable_buffer(&context.device, &tmp, Some(sub_idx)).await
        } else {
            return Err(GpuBufferFetchError::NotReadable);
        }?;

        // Truncate remaining on row
        if res.len() > self.len {
            let mut res = res.into_vec();
            res.truncate(self.len);
            Ok(res.into_boxed_slice())
        } else {
            Ok(res)
        }
    }
}


pub(crate) struct GpuBuffer2<E> {
    buffer: wgpu::Buffer,
    pub(super) dims: ImageDimensions,
    pub(super) index: Option<SubmissionIndex>,
    pixel: PhantomData<E>,
}

impl<E> GpuBuffer for GpuBuffer2<E> {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<E> GpuBuffer2<E> {
    pub(crate) fn new(dims: ImageDimensions, buffer: wgpu::Buffer) -> Self {
        Self {
            dims,
            buffer,
            index: None,
            pixel: PhantomData,
        }
    }
    pub(super) fn as_image_copy(&self) -> wgpu::ImageCopyBuffer {
        let size_elem = std::mem::size_of::<E>();
        wgpu::ImageCopyBuffer {
            buffer: self.buffer(),
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some((self.dims.stride * size_elem) as _),
                rows_per_image: Some(self.dims.height as _),
            }
        }
    }
}

impl<E> GpuImageLike for GpuBuffer2<E> {
    fn width(&self) -> usize {
        self.dims.width
    }

    fn height(&self) -> usize {
        self.dims.height
    }

    fn stride(&self) -> usize {
        self.dims.stride * std::mem::size_of::<E>()
    }
}

impl<E> GpuAwaitable for GpuBuffer2<E> {
    type Context = GpuContext;

    fn submission_index(&self) -> Option<&SubmissionIndex> {
        self.index.as_ref()
    }

    fn poll(&self, _context: &Self::Context) {
        // It's not clear if this wait is necessary
        // if let Some(sub_idx) = self.index.as_ref() {
        //     context.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(sub_idx.clone()));
        // }
    }
}


#[async_trait]
impl<E> GpuBufferFetch<E> for GpuBuffer2<E> where E: Sync + Pod + SafeZero {
    async fn fetch_buffer(&self, context: &Self::Context) -> Result<Box<[E]>, GpuBufferFetchError> {
        let usage = self.buffer.usage();
        let res = if usage.contains(BufferUsages::MAP_READ) {
            // Map buffer directly
            read_mappable_buffer::<E>(&context.device, &self.buffer, self.index.clone()).await
        } else if usage.contains(BufferUsages::COPY_SRC) {
            self.poll(context);

            // Copy to new buffer that *can* be read
            let tmp = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.buffer.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false
            });

            let cmd_buf = {
                let mut enc = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Download") });
                enc.copy_buffer_to_buffer(&self.buffer, 0, &tmp, 0, self.buffer.size());
                enc.finish()
            };
            let sub_idx = context.queue.submit([cmd_buf]);
            read_mappable_buffer(&context.device, &tmp, Some(sub_idx)).await
        } else {
            return Err(GpuBufferFetchError::NotReadable);
        }?;

        Ok(res)
    }
}

#[async_trait]
impl<P: Pixel> GpuImageDownload<P> for GpuBuffer2<P::Subpixel> where P: Send + Sync, P::Subpixel: SafeZero + Pod + Sync {
    async fn download_image(&self, context: &Self::Context) -> Result<ImageBuffer<P>, GpuBufferFetchError> {
        let host_buf: Box<[P::Subpixel]> = self.fetch_buffer(context).await?;
		let host_img = ImageBuffer::<P>::wrap(host_buf, self.width(), self.height(), self.stride());
		Ok(host_img)
    }
}

// enum MappedBuffer2Guard<'a, E: Pod> {
//     Direct(MappedBufferGuard<'a, &'a wgpu::Buffer, E>),
//     Copied(MappedBufferGuard<'a, wgpu::Buffer, E>),
// }

// impl<'a, E: Pod> Borrow<[E]> for MappedBuffer2Guard<'a, E> {
//     fn borrow(&self) -> &[E] {
//         match self {
//             MappedBuffer2Guard::Direct(g) => g.borrow(),
//             MappedBuffer2Guard::Copied(g) => g.borrow(),
//         }
//     }
// }

// #[async_trait]
// impl<E: Pod> GpuBufferMap<E> for GpuBuffer2<E> where E: Send + Sync {
//     type Guard<'a> = MappedBuffer2Guard<'a, E>;
//     /// Map buffer data to CPU
//     async fn map_buffer<'a>(&'a self, context: &'a <Self as GpuAwaitable>::Context) -> Result<Self::Guard<'a>, GpuBufferFetchError> {
//         let usage = self.buffer.usage();
//         if usage.contains(BufferUsages::MAP_READ) {
//             // Map buffer directly
//             let res = map_buffer::<E>(&context.device, &self.buffer, self.index.clone()).await?;
//             Ok(MappedBuffer2Guard::Direct(res))
//         } else if usage.contains(BufferUsages::COPY_SRC) {
//             self.poll(context);

//             // Copy to new buffer that *can* be read
//             let tmp = context.device.create_buffer(&wgpu::BufferDescriptor {
//                 label: None,
//                 size: self.buffer.size(),
//                 usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
//                 mapped_at_creation: false
//             });

//             let cmd_buf = {
//                 let mut enc = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Download") });
//                 enc.copy_buffer_to_buffer(&self.buffer, 0, &tmp, 0, self.buffer.size());
//                 enc.finish()
//             };
//             let sub_idx = context.queue.submit([cmd_buf]);
//             let res = map_buffer(&context.device, tmp, Some(sub_idx)).await?;
//             Ok(MappedBuffer2Guard::Copied(res))
//         } else {
//             Err(GpuBufferFetchError::NotReadable)
//         }
//     }
// }