use async_trait::async_trait;
use bytemuck::Pod;
use futures::{channel::oneshot::Receiver, Future};
use wgpu::{SubmissionIndex, BufferUsages, COPY_BYTES_PER_ROW_ALIGNMENT, TextureFormat};
use std::{io::Error as IOError, task::Poll, sync::Arc, marker::PhantomData};
use crate::util::{image::{ImageDimensions, Pixel, Luma}, ImageY8, mem::{SafeZero, calloc}, ImageBuffer};

use super::{GpuStageContext, WGPUError, GpuContext, util::DataStore};

#[must_use]
pub(crate) struct AsyncBufferView<'a> {
    buffer_slice: wgpu::BufferSlice<'a>,
    device: &'a wgpu::Device,
    receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl<'a> AsyncBufferView<'a> {
    fn new(buffer_slice: wgpu::BufferSlice<'a>, device: &'a wgpu::Device) -> Self {
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).expect("Couldn't notify mapping")
        });

        AsyncBufferView {
            buffer_slice,
            device,
            receiver,
        }
    }
}

impl<'a> Future for AsyncBufferView<'a> {
    type Output = Result<wgpu::BufferView<'a>, IOError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.device.poll(wgpu::MaintainBase::Poll);
        match self.receiver.try_recv() {
            Ok(Some(Ok(_))) => Poll::Ready(Ok(self.buffer_slice.get_mapped_range())),
            Ok(Some(Err(e))) => Poll::Ready(Err(IOError::new(std::io::ErrorKind::InvalidData, e))),
            // Ok(None) => todo!("Future Ok(None)"),
            Ok(None) | Err(_) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

async fn read_mappable_buffer<E: SafeZero + Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer, sub_idx: Option<&SubmissionIndex>) -> Result<Box<[E]>, IOError> {
    let buf_view = AsyncBufferView::new(buffer.slice(..), device).await?;

    let mut data = calloc::<E>(buffer.size() as usize);
    let buf_slice = bytemuck::cast_slice::<u8, E>(&buf_view);
    data.copy_from_slice(buf_slice);
    drop(buf_view);
    buffer.unmap();

    Ok(data)
}


pub(super) trait GpuImageLike{
    fn label(&self) -> Option<&str> {
        None
    }
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    /// Stride (bytes)
    fn stride(&self) -> usize {
        self.width()
    }
}

#[async_trait]
pub(super) trait GpuImageDownload<P: GpuPixel>: GpuImageLike {
    type Context: Sync;
    fn poll(&self, context: &Self::Context);
    async fn fetch_buffer(&self, context: &Self::Context) -> Result<Box<[P::Subpixel]>, std::io::Error>;
    async fn download_image(&self, context: &Self::Context) -> Result<ImageBuffer<P>, std::io::Error> {
        let host_buf = self.fetch_buffer(context).await?;
        // println!("Got buffer: {}", host_buf.len());
        // println!("Dims: {} ({})x{} = {}", self.width(), self.stride(), self.height(), self.stride() * self.height());
		let host_img = ImageBuffer::wrap(host_buf, self.width() as usize, self.height() as usize, self.stride() as usize);
		Ok(host_img)
    }
}

pub(super) struct GpuBuffer<P: GpuPixel = Luma<u8>> {
    pub(super) dims: ImageDimensions,
    pub(super) buffer: wgpu::Buffer,
    pub(super) index: Option<SubmissionIndex>,
    pixel: PhantomData<P>,
}

impl<P: GpuPixel> GpuBuffer<P> {
    pub(super) fn new(dims: ImageDimensions, buffer: wgpu::Buffer) -> Self {
        Self {
            dims,
            buffer,
            index: None,
            pixel: PhantomData,
        }
    }
}

impl<P: GpuPixel> GpuImageLike for GpuBuffer<P> {
    fn width(&self) -> usize {
        self.dims.width
    }

    fn height(&self) -> usize {
        self.dims.height
    }

    fn stride(&self) -> usize {
        self.dims.stride * std::mem::size_of::<P::Subpixel>()
    }
}

#[cfg(feature="debug")]
pub(super) fn split_minmax(ctx: &GpuContext, gpu_tex: &GpuTexture<[u8; 2]>) -> Result<(ImageY8, ImageY8), std::io::Error> {
    fn fetch_minmax(ctx: &GpuContext, gpu_tex: &GpuTexture<[u8; 2]>) -> Result<ImageBuffer<[u8; 2]>, std::io::Error> {
        let cpu_buf = futures::executor::block_on(gpu_tex.fetch_buffer(ctx))?;
        let cpu_img = ImageBuffer::<[u8; 2]>::wrap(cpu_buf, gpu_tex.width(), gpu_tex.height(), (gpu_tex.width() * 2).next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT as usize));
        Ok(cpu_img)
    }
    let cpu_img = fetch_minmax(ctx, gpu_tex)?;
    let mut im_min: ImageBuffer<crate::util::image::Luma<u8>, Box<[u8]>> = ImageY8::zeroed(gpu_tex.width(), gpu_tex.height());
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

#[async_trait]
impl<P: GpuPixel> GpuImageDownload<P> for GpuBuffer<P> where P::Subpixel: SafeZero + Pod, P: Sync {
    type Context = GpuContext;

    fn poll(&self, context: &Self::Context) {
        // It's not clear if this wait is necessary
        // if let Some(sub_idx) = self.index.as_ref() {
        //     context.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(sub_idx.clone()));
        // }
    }
    
    async fn fetch_buffer(&self, context: &Self::Context) -> Result<Box<[P::Subpixel]>, std::io::Error> {
        if !self.buffer.usage().contains(wgpu::BufferUsages::MAP_READ) {
            self.poll(context);

			// Copy to new buffer that can be read
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
			read_mappable_buffer(&context.device, &tmp, Some(sub_idx).as_ref()).await
		} else {
			read_mappable_buffer(&context.device, &self.buffer, self.index.as_ref()).await
		}
    }
}

pub(super) trait GpuPixel: Pixel {
    const GPU_FORMAT: wgpu::TextureFormat;
}

impl GpuPixel for Luma<u8> {
    const GPU_FORMAT: wgpu::TextureFormat = TextureFormat::R8Uint;
}

impl GpuPixel for [u8; 2] {
    const GPU_FORMAT: wgpu::TextureFormat = TextureFormat::Rg8Uint;
}

#[derive(Clone)]
pub(super) struct GpuTexture<P: GpuPixel = Luma<u8>> {
    pub(super) buffer: Arc<wgpu::Texture>,
    pub(super) index: Option<SubmissionIndex>,
    pixel: PhantomData<P>,
}

impl<P: GpuPixel> GpuTexture<P> {
    pub(super) fn new(buffer: wgpu::Texture) -> Self {
        debug_assert_eq!(P::GPU_FORMAT, buffer.format(), "Format mismatch");

        Self {
            buffer: Arc::new(buffer),
            index: None,
            pixel: PhantomData,
        }
    }

    fn as_texture(&self) -> &wgpu::Texture {
        &self.buffer
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

#[async_trait]
impl<P: GpuPixel> GpuImageDownload<P> for GpuTexture<P> where P: Send + Sync, P::Subpixel: Pod + SafeZero {
    type Context = GpuContext;

    fn poll(&self, context: &Self::Context) {
        // It's not clear if this wait is necessary
        // if let Some(sub_idx) = self.index.as_ref() {
        //     context.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(sub_idx.clone()));
        // }
    }

    async fn fetch_buffer(&self, context: &Self::Context) -> Result<Box<[P::Subpixel]>, std::io::Error> {
        let buf = self.as_texture();
        let size_spx = std::mem::size_of::<P::Subpixel>();

        if buf.usage().contains(wgpu::TextureUsages::COPY_SRC) {
            let mut buf: GpuBuffer<P> = {
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
                GpuBuffer {
                    dims,
                    buffer,
                    index: None,
                    pixel: PhantomData
                }
            };

            let cmd_buf = {
                let mut enc = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.copy_texture_to_buffer(
                    self.buffer.as_image_copy(),
                    wgpu::ImageCopyBuffer {
                        buffer: &buf.buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some((buf.dims.stride * size_spx) as _),
                            rows_per_image: Some(self.height() as _),
                        }
                    },
                    wgpu::Extent3d {
                        width: self.width() as u32,
                        height: self.height() as u32,
                        depth_or_array_layers: 1,
                    }
                );
                enc.finish()
            };
            let sub_idx = context.queue.submit([cmd_buf]);
            buf.index = Some(sub_idx);
            buf.fetch_buffer(context).await
        // } else if self.buffer.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        //     todo!()
        } else {
            return Err(IOError::new(std::io::ErrorKind::InvalidInput, "Buffer is not readable"));
        }
    }
}
impl<P: GpuPixel> GpuTexture<P> {
    pub(super) fn as_view(&self) -> wgpu::TextureView {
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
}

pub(super) trait GpuStage {
    type Data;
    type Source;
    type Output;

    fn src_alignment(&self) -> usize;

    fn apply<'a, 'b: 'a>(&'b self, ctx: &mut GpuStageContext<'a>, src: &Self::Source, temp: &'b mut DataStore<Self::Data>) -> Result<Self::Output, WGPUError>;
}