use std::ops::DerefMut;

use ocl::{Event as OclEvent, Buffer as OclBuffer, OclPrm, Queue, Error as OclError};

use crate::{util::{ImageBuffer, image::{Pixel, ImageDimensions, Luma, ImageRefY8}, ImageY8, mem::{calloc, SafeZero}}, TimeProfile};
use std::ops::Deref;

use ocl::{Program, Context, Device, MemFlags};

pub(super) struct OclCore {
	pub(super) device: Device,
	pub(super) context: Context,
	pub(super) program: Program,
	/// Queue for uploading the image
	pub(super) queue_write: Queue,
	/// Queue for executing kernels
	pub(super) queue_kernel: Queue,
	/// Queue for initializing data
	pub(super) queue_init: Queue,
	/// Queue for reading images back for debugging
	pub(super) queue_read_debug: Queue,
}

impl OclCore {
	fn temp_buffer<E: OclPrm>(&self, dims: &ImageDimensions, read: bool) -> Result<OclBuffer<E>, OclError> {
        let read = read | true;
        let flags = {
            let flags = MemFlags::new()
                .read_write();
            if read {
                // We want to read from this buffer for post-processing debug
                flags.host_read_only()
                    .alloc_host_ptr()
            } else {
                flags.host_no_access()
            }
        };

        OclBuffer::<E>::builder()
            .context(&self.context)
            .flags(flags)
            .len((dims.stride, dims.height))
            .build()
    }

    pub(super) fn temp_bufferstate<E: OclPrm>(&self, dims: &ImageDimensions, read: bool) -> Result<OclBufferState<E>, OclError> {
        let buf = self.temp_buffer(dims, read)?;
        Ok(OclBufferState { buf, dims: *dims, event: None })
    }

	pub(super) fn mapped_buffer<E: OclPrm + SafeZero>(&self, dims: ImageDimensions) -> Result<OclBufferMapped<E>, OclError> {
		let backing = calloc::<E>(dims.stride * dims.height);

        let flags = {
            let flags = MemFlags::new()
                .read_write();
			// We want to read from this buffer for post-processing debug
			flags.host_read_only()
				.alloc_host_ptr()
        };

        let buf = OclBuffer::<E>::builder()
            .context(&self.context)
            .flags(flags)
            .len((dims.stride, dims.height))
            .build()?;
		Ok(OclBufferMapped {
			buf,
			dims,
			event: None,
			backing,
		})
	}

    /// Create new temporary image
    fn temp_img<P: Pixel>(&self, dims: &ImageDimensions, read: bool) -> Result<OclImageState<'static, P>, OclError> where <P as Pixel>::Subpixel: SafeZero + OclPrm {
        let read = read | true;
        let flags = {
            let flags = MemFlags::new()
                .read_write();
            if read {
                // We want to read from this buffer for post-processing debug
                flags.host_read_only()
            } else {
                flags.host_no_access()
            }
        };

        // assert!(dims.stride % P::size_bytes() == 0);

        let builder = OclBuffer::<<P as Pixel>::Subpixel>::builder()
            .context(&self.context)
            .flags(flags)
            .len((dims.stride, dims.height));

        if read {
            let image = ImageBuffer::<P>::zeroed_with_stride(dims.width, dims.height, dims.stride);
            let buf = unsafe { builder.use_host_slice(image.data.deref()) }
                .build()?;
            Ok(OclImageState { buffer: OclBufferState { buf, dims: *dims, event: None }, backing: OclImageBacking::Buffer(image) })
        } else {
            let buf = builder.build()?;
            Ok(OclImageState { buffer: OclBufferState { buf, dims: *dims, event: None }, backing: OclImageBacking::None })
        }
    }
	
	/// Upload source image to OpenCL device
    pub(super) fn upload_image<'a>(&self, download: bool, tp: &mut TimeProfile, image: ImageRefY8<'a>) -> Result<OclImageState<'a>, OclError> {
        tp.stamp("Buffer start");
        let cl_src = {
            let flags = {
                let flags = MemFlags::new()
                    .read_only();
                if download {
                    // Host read/write
                    flags
                } else {
                    flags.host_write_only()
                }
            };

            let builder = OclBuffer::<u8>::builder()
                .queue(self.queue_write.clone())
                .len((image.stride(), image.height()))
                .flags(flags)
                .copy_host_slice(image.data.deref());
            
            // let builder = unsafe { builder.use_host_slice(&image.data) };
            builder.build()?
        };
        
        //TODO: Async quues
        tp.stamp("Buffer upload");

        Ok(OclImageState {
            buffer: OclBufferState {
                buf: cl_src,
                dims: *image.dimensions(),
                event: None,
            },
            backing: OclImageBacking::None
        })
    }
	pub(super) fn fetch_ocl_buffer<E: OclPrm + SafeZero>(&self, image: &impl OclBufferLike<E>) -> Result<Box<[E]>, std::io::Error> {
		let buf_flags = image.buf().flags().expect("Get imagebuf flags");

		let host_buf = if true || !buf_flags.contains(MemFlags::new().alloc_host_ptr()) {
			let mut host_buf = calloc::<E>(image.buf().len());
			let read_cmd: ocl::builders::BufferReadCmd<'_, '_, E> = image.buf()
				.cmd()
				.queue(&self.queue_read_debug)
				.read(host_buf.deref_mut())
				.ewait(image.event());
			read_cmd
				.enq()
				.expect("Host read");
			host_buf
		} else {
			let map_cmd = image.buf()
				.cmd()
				.queue(&self.queue_read_debug)
				.map()
				.ewait(image.event())
				.read();
			
			#[allow(unused)]
			let res = unsafe { map_cmd.enq() }.unwrap();

			todo!("memmap")
		};
		Ok(host_buf)
	}

	pub(super) fn download_image(&self, image: &OclBufferState) -> Result<ImageY8, std::io::Error> {
		let host_buf = self.fetch_ocl_buffer(image)?;
		
		let host_img = ImageY8::wrap(host_buf, image.width(), image.height(), image.stride());
		Ok(host_img)
	}
}

pub(super) trait OclAwaitable {
	fn event(&self) -> Option<&OclEvent>;
}

pub(super) trait OclBufferLike<R: OclPrm>: OclAwaitable {
	fn buf(&self) -> &OclBuffer<R>;
	fn event_mut(&mut self) -> &mut Option<OclEvent>;
}

impl OclAwaitable for Option<&OclEvent> {
	fn event(&self) -> Option<&OclEvent> {
		*self
	}
}
pub(super) enum OclImageBacking<'a, P: Pixel> {
	None,
	Ref(&'a ImageBuffer<P>),
	Buffer(ImageBuffer<P>)
}

#[derive(Clone)]
pub(super) struct OclBufferMapped<E: OclPrm = u8> {
	buf: OclBuffer<E>,
	dims: ImageDimensions,
	event: Option<OclEvent>,
	backing: Box<[E]>,
}
impl<E: OclPrm> OclAwaitable for OclBufferMapped<E> {
	fn event(&self) -> Option<&OclEvent> {
		self.event.as_ref()
	}
}
impl<E: OclPrm> OclBufferLike<E> for OclBufferMapped<E> {
    fn buf(&self) -> &OclBuffer<E> {
        &self.buf
    }
    fn event_mut(&mut self) -> &mut Option<OclEvent> {
        &mut self.event
    }
}
impl<E: OclPrm> OclBufferMapped<E> {
	pub(super) fn with_event(self, event: OclEvent) -> Self {
		Self {
			buf: self.buf,
			dims: self.dims,
			backing: self.backing,
			event: Some(event),
		}
	}
	pub(super) fn width(&self) -> usize {
		self.dims.width
	}
	pub(super) fn height(&self) -> usize {
		self.dims.height
	}
	pub(super) fn stride(&self) -> usize {
		self.dims.stride
	}
	pub(super) fn buf(&self) -> &OclBuffer<E> {
		&self.buf
	}
}


#[derive(Clone)]
pub(super) struct OclBufferState<E: OclPrm = u8> {
	buf: OclBuffer<E>,
	pub(super) dims: ImageDimensions,
	pub(super) event: Option<OclEvent>,
}
impl<E: OclPrm> OclAwaitable for OclBufferState<E> {
	fn event(&self) -> Option<&OclEvent> {
		self.event.as_ref()
	}
}
impl<E: OclPrm> OclBufferLike<E> for OclBufferState<E> {
    fn buf(&self) -> &OclBuffer<E> {
        &self.buf
    }
    fn event_mut(&mut self) -> &mut Option<OclEvent> {
        &mut self.event
    }
}
impl<E: OclPrm> OclBufferState<E> {
	pub(super) fn with_event(&self, event: OclEvent) -> Self {
		Self {
			buf: self.buf.clone(),
			dims: self.dims.clone(),
			event: Some(event),
		}
	}
	pub(super) const fn width(&self) -> usize {
		self.dims.width
	}
	pub(super) const fn height(&self) -> usize {
		self.dims.height
	}
	pub(super) const fn stride(&self) -> usize {
		self.dims.stride
	}
	pub(super) const fn buf(&self) -> &OclBuffer<E> {
		&self.buf
	}
}

pub(super) struct OclImageState<'a, P: Pixel = Luma<u8>> where <P as Pixel>::Subpixel: OclPrm {
	pub(super) buffer: OclBufferState<<P as Pixel>::Subpixel>,
	pub(super) backing: OclImageBacking<'a, P>,
}
impl<'a, P: Pixel> OclAwaitable for OclImageState<'a, P> where <P as Pixel>::Subpixel: OclPrm {
	fn event(&self) -> Option<&OclEvent> {
		self.buffer.event.as_ref()
	}
}

impl<'a, P: Pixel> OclImageState<'a, P> where <P as Pixel>::Subpixel: OclPrm {
	pub(super) fn width(&self) -> usize {
		self.buffer.dims.width
	}
	pub(super) fn height(&self) -> usize {
		self.buffer.dims.height
	}
	pub(super) fn stride(&self) -> usize {
		self.buffer.dims.stride
	}
	pub(super) fn buf(&self) -> &OclBuffer<<P as Pixel>::Subpixel> {
		&self.buffer.buf
	}
}

impl<'a> OclImageState<'a> {
	pub(super) fn read_image(&self, queue: &Queue) -> Result<ImageY8, OclError> {
		let mut host_buf = calloc::<u8>(self.buffer.buf.len());
		let read_cmd = self.buffer.buf
			.cmd()
			.queue(&queue)
			.read(host_buf.deref_mut());
		let read_cmd = if let Some(evt) = &self.buffer.event {
			read_cmd.ewait(evt)
		} else { read_cmd };

		read_cmd.enq()?;

		Ok(ImageY8::wrap(host_buf, self.buffer.dims.width, self.buffer.dims.height, self.buffer.dims.stride))
	}
	pub(super) fn into_image(self, tp: &mut TimeProfile, queue: &Queue) -> Result<ImageY8, OclError> {
		match self.backing {
			OclImageBacking::None => self.read_image(queue),
			OclImageBacking::Ref(_imref) => self.read_image(queue),
			OclImageBacking::Buffer(imbuf) => {
				let mcmd = self.buffer.buf.map()
					.ewait(self.buffer.event.as_ref())
					.read()
					.queue(queue);
				tp.stamp("Start read");
				let mut result = unsafe { mcmd.enq() }?;
				tp.stamp("Mapped");
				assert_eq!(result.as_ptr(), imbuf.data.as_ptr());
				result.unmap()
					.queue(queue)
					.enq()?;
				tp.stamp("Unmap");
				Ok(imbuf)
			},
		}
	}
}