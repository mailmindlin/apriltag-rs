use std::borrow::Borrow;

use async_trait::async_trait;
use wgpu::SubmissionIndex;

use crate::{util::{image::{ImageDimensions, ImageRef, Pixel}, ImageBuffer}, wgpu::error::GpuBufferFetchError};


pub(crate) enum SpatialDims {
    One(usize),
    Two(ImageDimensions),
}

pub(crate) trait GpuBuffer {
    /// Get underlying wgpu buffer
    fn buffer(&self) -> &wgpu::Buffer;
    
    /// Return the binding view of the entire buffer.
    fn as_binding(&self) -> wgpu::BindingResource {
        self.buffer().as_entire_binding()
    }
}


pub(crate) trait GpuImageLike {
    /// Image name
    fn label(&self) -> Option<&str> {
        None
    }
    /// Image width (pixels)
    fn width(&self) -> usize;
    /// Image height (pixels)
    fn height(&self) -> usize;
    /// Stride (bytes)
    fn stride(&self) -> usize {
        self.width()
    }
}

pub(crate) trait GpuAwaitable {
    type Context: Sync;
    /// Get submission index to wait on (if available)
    fn submission_index(&self) -> Option<&SubmissionIndex>;
    
    fn poll(&self, context: &Self::Context);
}

/// Gpu types where we can get fetch raw data
#[async_trait]
pub(crate) trait GpuBufferMap<E: 'static>: GpuAwaitable {
    type Guard<'a>: Borrow<[E]> where Self: 'a;
    /// Map buffer data to CPU
    async fn map_buffer<'a>(&'a self, context: &'a <Self as GpuAwaitable>::Context) -> Result<Self::Guard<'a>, GpuBufferFetchError>;
}

/// Gpu types where we can get fetch raw data
#[async_trait]
pub(crate) trait GpuBufferFetch<E>: GpuAwaitable {
    /// Download buffer data
    async fn fetch_buffer(&self, context: &<Self as GpuAwaitable>::Context) -> Result<Box<[E]>, GpuBufferFetchError>;
}

#[async_trait]
pub(crate) trait GpuImageMap<P: Pixel + 'static>: GpuImageLike + GpuAwaitable {
    type Guard<'a>: Borrow<ImageRef<'a, P>>;

    async fn map_image<'a>(&self, context: &Self::Context) -> Result<Self::Guard<'a>, GpuBufferFetchError>;
}

#[async_trait]
pub(crate) trait GpuImageDownload<P: Pixel>: GpuImageLike + GpuAwaitable {
    /// Download image to CPU
    async fn download_image(&self, context: &<Self as GpuAwaitable>::Context) -> Result<ImageBuffer<P>, GpuBufferFetchError>;
}

// Default implementation when we can just fetch the buffer
// #[async_trait]
// impl<V: GpuBufferFetch<P::Subpixel> + GpuImageLike, P: Pixel> GpuImageDownload<P> for V {
//     async fn download_image(&self, context: &Self::Context) -> Result<ImageBuffer<P>, GpuBufferFetchError> {
//         let host_buf: Box<P::Subpixel> = self.fetch_buffer(context).await?;
// 		let host_img = ImageBuffer::<P>::wrap(host_buf, self.width(), self.height(), self.stride());
// 		Ok(host_img)
//     }
// }