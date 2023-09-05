#![cfg(feature="opencl")]

use std::ops::{Deref, DerefMut};
use std::pin::Pin;

use ocl::prm::Uchar2;
use ocl::{
    Platform as OclPlatform,
    Error as OclError,
    Device as OclDevice,
    Program, Context,
    Queue,
    Buffer as OclBuffer,
    CommandQueueProperties,
    MemFlags,
    Kernel as OclKernel,
    Event as OclEvent, DeviceType, OclPrm, SpatialDims, MapFlags,
};

use crate::detector::config::QuadDecimateMode;
use crate::quad_thresh::threshold;
use crate::util::ImageBuffer;
use crate::util::image::{ImageWritePNM, Pixel, Luma};
use crate::{OpenClMode, DetectorBuildError};
use crate::util::mem::{calloc, SafeZero};
use crate::{DetectorConfig, util::{ImageY8, image::ImageDimensions}, DetectError, detector::quad_sigma_kernel, TimeProfile};

const PROG_QUAD_DECIMATE: &str = include_str!("./01_quad_decimate.cl");
const PROG_QUAD_SIGMA: &str = include_str!("./02_quad_sigma.cl");
const PROG_THRESHOLD: &str = include_str!("./03_threshold.cl");
// const PROG_UNIONFIND: &str = include_str!("./04_unionfind.cl");

// trait SafeCallback<V>: Sized {
//     fn callback<T>(&self, callback: T) -> Result<(), OclError> where T: Send + Fn(V, Result<(), OclError>) -> ();
// }

// impl SafeCallback<OclEvent> for OclEvent {
//     fn callback<T>(&self, callback: T) -> Result<(), OclError> where T: Send + Fn(OclEvent, Result<(), OclError>) -> () {
//         extern "C" fn callback_thunk(evt: *mut c_void, status: i32, data: *mut c_void) {
//             let event = unsafe { OclEvent::from_raw(evt) };
//             let data = Box::from_raw(data as *mut _ as *mut (dyn Fn(OclEvent, Result<(), OclError>) -> () + Send));
            
//         }
//         let data: Box<dyn Fn(OclEvent, Result<(), OclError>) -> () + Send> = Box::new(callback);
//         unsafe {
//             self.set_callback(callback_thunk, Box::into_raw(data) as *mut c_void)?
//         };
//         Ok(())
//     }
// }

impl Into<SpatialDims> for ImageDimensions {
    fn into(self) -> SpatialDims {
        SpatialDims::Two(self.width, self.height)
    }
}

impl SafeZero for Uchar2 {}

pub(crate) struct OpenCLDetector {
    pub(crate) mode: OpenClMode,
    // platform: OclPlatform,
    context: Context,
    program: Program,
    queue_write: Queue,
    queue_kernel: Queue,
    queue_read_debug: Queue,
    quad_sigma_filter: Option<OclBuffer<u8>>,
}

enum OclBacking<'a, P: Pixel> {
    None,
    Ref(&'a ImageBuffer<P>),
    Buffer(ImageBuffer<P>)
}

struct OclBufferState<E: OclPrm = u8> {
    buf: OclBuffer<E>,
    dims: ImageDimensions,
    event: Option<OclEvent>,
}
struct OclImageState<'a, P: Pixel = Luma<u8>> where <P as Pixel>::Subpixel: OclPrm {
    buffer: OclBufferState<<P as Pixel>::Subpixel>,
    backing: OclBacking<'a, P>,
}

impl<'a> OclImageState<'a> {
    fn read_image(&self, queue: &Queue) -> Result<ImageY8, OclError> {
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
    fn into_image(self, tp: &mut TimeProfile, queue: &Queue) -> Result<ImageY8, OclError> {
        match self.backing {
            OclBacking::None => self.read_image(queue),
            OclBacking::Ref(imref) => self.read_image(queue),
            OclBacking::Buffer(imbuf) => {
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

fn find_device(mode: &OpenClMode) -> Result<(OclPlatform, OclDevice), DetectorBuildError> {
    let (devtype, idx, prefer) = match mode {
        OpenClMode::Disabled => panic!(),
        OpenClMode::Prefer => (None, 0, true),
        OpenClMode::Required => (None, 0, false),
        OpenClMode::PreferDeviceIdx(idx) => (None, *idx, true),
        OpenClMode::RequiredDeviceIdx(idx) => (None, *idx, false),
        OpenClMode::PreferGpu => (Some(DeviceType::GPU), 0, true),
        OpenClMode::RequiredGpu => (Some(DeviceType::GPU), 0, false),
    };
    for platform in OclPlatform::list() {
        let devs = match OclDevice::list(platform, devtype) {
            Ok(devs) => devs,
            Err(e) => {
                if prefer {
                    #[cfg(feature="debug")]
                    eprintln!("OpenCLDevice::list error: {e:?}");
                    continue;
                } else {
                    return Err(DetectorBuildError::OpenCLError(e));
                }
            }
        };
        if devs.is_empty() {
            continue;
        }
        let device = devs[idx % devs.len()];
        if let Ok(true) = device.is_available() {
            return Ok((platform, device));
        }
    }
    Err(DetectorBuildError::OpenCLNotAvailable)
}

impl OpenCLDetector {
    pub(super) fn new(config: &DetectorConfig, mode: OpenClMode) -> Result<Self, DetectorBuildError> {
        let (platform, device) = find_device(&mode)?;

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        let make_queue_ooo = || {
            let out_of_order = Some(CommandQueueProperties::new()
                
            );
            Queue::new(&context, device, out_of_order)
                // .or_else(|e| {
                //     eprintln!("Not out of order: {e:?}");
                //     Queue::new(&context, device, None)
                // })
        };
        
        // Try to make out-of-order queues
        let queue_write = make_queue_ooo()?;
        let queue_kernel = make_queue_ooo()?;
        let queue_read_debug = make_queue_ooo()?;
        
        let program = {
            let mut builder = Program::builder();
            if config.quad_decimate > 1. {
                builder.source(PROG_QUAD_DECIMATE);
            }
            if config.quad_sigma != 0. {
                builder.source(PROG_QUAD_SIGMA);
            }
            builder.source(PROG_THRESHOLD);
            builder.cmplr_def("TILESZ", threshold::TILESZ as _);

            // builder.source(PROG_UNIONFIND);
            
            let res = builder.build(&context);
            #[cfg(feature="debug")]
            if let Err(OclError::OclCore(ocl::OclCoreError::ProgramBuild(e))) = &res {
                let buf = format!("{e:?}");
                let msg = buf.strip_prefix("BuildLog(").unwrap_or(&buf);
                let msg = msg.replace("\\n", "\n");
                println!("Build error: {}", msg);
            }
            res?
        };

        // Upload quad_sigma filter
        let quad_sigma_filter = match quad_sigma_kernel(config.quad_sigma) {
            Some(kernel) => {
                let cl_filter = OclBuffer::builder()
                    .queue(queue_write.clone())
                    .flags(MemFlags::new().host_write_only().read_only())
                    .len(kernel.len())
                    .copy_host_slice(&kernel)
                    .build()
                    .unwrap();
                Some(cl_filter)
            },
            None => None,
        };

        Ok(Self {
            mode,
            // platform,
            context,
            program,
            queue_kernel,
            queue_write,
            queue_read_debug,
            quad_sigma_filter,
        })
    }

    fn fetch_ocl_buffer<E: OclPrm + SafeZero>(&self, image: &OclBufferState<E>) -> Result<Box<[E]>, std::io::Error> {
        let buf_flags = image.buf.flags().expect("Get imagebuf flags");

        let host_buf = if true || !buf_flags.contains(MemFlags::new().alloc_host_ptr()) {
            let mut host_buf = calloc::<E>(image.buf.len());
            let read_cmd = image.buf
                .cmd()
                .queue(&self.queue_kernel)
                .read(host_buf.deref_mut())
                .ewait(image.event.as_ref());
            read_cmd
                .enq()
                .expect("Host read");
            host_buf
        } else {
            let map_cmd = image.buf
                .cmd()
                .queue(&self.queue_read_debug)
                .map()
                .ewait(image.event.as_ref().cloned())
                .read();
            
            #[allow(unused)]
            let res = unsafe { map_cmd.enq() }.unwrap();

            todo!("memmap")
        };
        Ok(host_buf)
    }

    fn download_image(&self, image: &OclImageState) -> Result<ImageY8, std::io::Error> {
        let host_buf = self.fetch_ocl_buffer(&image.buffer)?;
        
        let host_img = ImageY8::wrap(host_buf, image.buffer.dims.width, image.buffer.dims.height, image.buffer.dims.stride);
        Ok(host_img)
    }

    #[cfg(feature="debug")]
    fn debug_ocl_image(&self, file: &mut std::fs::File, image: &OclImageState) -> Result<(), std::io::Error> {
        use crate::util::image::ImageWritePNM;
        let host_img = self.download_image(image)?;
        
        host_img.write_pnm(file)
    }

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

        let builder = OclBuffer::<<P as Pixel>::Subpixel>::builder()
            .context(&self.context)
            .flags(flags)
            .len((dims.stride, dims.height));

        if read {
            let image = ImageBuffer::<P>::zeroed_with_stride(dims.width, dims.height, dims.stride);
            let buf = unsafe { builder.use_host_slice(image.data.deref()) }
                .build()?;
            Ok(OclImageState { buffer: OclBufferState { buf, dims: *dims, event: None }, backing: OclBacking::Buffer(image) })
        } else {
            let buf = builder.build()?;
            Ok(OclImageState { buffer: OclBufferState { buf, dims: *dims, event: None }, backing: OclBacking::None })
        }
    }

    /// Upload source image to OpenCL device
    fn upload_image<'a>(&self, download: bool, tp: &mut TimeProfile, image: &'a ImageY8) -> Result<OclImageState<'a>, OclError> {
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
        // let mut evt = OclEvent::empty();
        // cl_src
        //     .cmd()
        //     .queue(&self.queue_write)
        //     .write(image.data.deref())
        //     .enew(&mut evt)
        //     .enq()?;
        tp.stamp("Buffer upload");

        Ok(OclImageState {
            buffer: OclBufferState {
                buf: cl_src, dims: *image.dimensions(), event: None
            },
            backing: OclBacking::None
        })
    }

    /// Downsample image
    fn quad_decimate<'a>(&self, config: &DetectorConfig, download: bool, tp: &mut TimeProfile, prev: OclImageState<'a>) -> Result<OclImageState<'a>, OclError> {
        let mode = match config.quad_decimate_mode() {
            None => return Ok(prev),
            Some(mode) => mode,
        };
        let (swidth, sheight, factor) = match mode {
            QuadDecimateMode::ThreeHalves => {
                let swidth = prev.buffer.dims.width / 3 * 2;
                let sheight = prev.buffer.dims.height / 3 * 2;
                assert_eq!(swidth % 2, 0, "Input dimension must be multiple of two");
                assert_eq!(sheight % 2, 0, "Input dimension must be multiple of two");
                (swidth, sheight, 1)
            },
            QuadDecimateMode::Scaled(factor) => {
                let factor = factor.get() as usize;

                let swidth = 1 + (prev.buffer.dims.width - 1) / factor;
                let sheight = 1 + (prev.buffer.dims.height - 1) / factor;
                // println!("dims = {}x{}, sdims ={swidth}x{sheight} factor={factor}", prev.dims.width, prev.dims.height);
                assert!((swidth - 1) * factor <= prev.buffer.dims.width);
                assert!((sheight - 1) * factor <= prev.buffer.dims.height);
                (swidth, sheight, factor)
            },
        };

        let sstride = swidth.next_multiple_of(16);
        
        let buf_dims = ImageDimensions { width: swidth, height: sheight, stride: sstride };
        let mut res = self.temp_img(&buf_dims, download)?;
        tp.stamp("quad_decimate buffer");

        let kernel = {
            let mut builder = OclKernel::builder();
            builder.program(&self.program);
            builder.queue(self.queue_kernel.clone());

            builder.arg(prev.buffer.buf);
            builder.arg(prev.buffer.dims.stride as u32);
            builder.arg(&res.buffer.buf);
            builder.arg(sstride as u32);

            match mode {
                QuadDecimateMode::ThreeHalves => {
                    builder.name("k01_filter_quad_decimate_32");
                    builder.global_work_size((swidth / 2, sheight / 2));
                },
                QuadDecimateMode::Scaled(_) => {
                    builder.name("k01_filter_quad_decimate");
                    builder.arg(factor as u32);
                    builder.global_work_size((swidth, sheight));
                },
            }
            builder.build()?
        };
        tp.stamp("quad_decimate kernel");

        let mut evt = OclEvent::empty();
        let kcmd = kernel
            .cmd()
            .queue(&self.queue_kernel)
            .enew(&mut evt)
            .ewait(prev.buffer.event.as_ref())
            ;

        unsafe { kcmd.enq() }?;
        tp.stamp("quad_decimate");

        res.buffer.event = Some(evt);
        Ok(res)
    }

    fn quad_sigma<'a>(&self, config: &DetectorConfig, download: bool, tp: &mut TimeProfile, prev: OclImageState<'a>) -> Result<OclImageState<'a>, OclError> {
        let quad_sigma_filter = if let Some(quad_sigma_filter) = &self.quad_sigma_filter {
            quad_sigma_filter
        } else {
            return Ok(prev);
        };
        
        // Create output buffer (always read by host)
        let mut res = self.temp_img(&prev.buffer.dims, download)?;
        tp.stamp("quad_sigma buffer");

        // Make quad_sigma kernel
        let kernel = {
            let mut builder = OclKernel::builder();
            builder.program(&self.program);
            builder.queue(self.queue_kernel.clone());

            builder.arg(prev.buffer.buf);                   // im_src
            builder.arg(prev.buffer.dims.stride as u32);    // stride_src
            builder.arg(prev.buffer.dims.width as u32);     // width_src
            builder.arg(prev.buffer.dims.height as u32);    // height_src
            builder.arg(quad_sigma_filter);          // filter
            builder.arg(quad_sigma_filter.len() as u32); // ksz
            builder.arg(&res.buffer.buf);                        // im_dst
            builder.arg(prev.buffer.dims.stride as u32);    // stride_dst

            builder.global_work_size((prev.buffer.dims.width, prev.buffer.dims.height));

            if config.quad_sigma > 0. {
                builder.name("k02_gaussian_blur_filter");
            } else {
                builder.name("k02_gaussian_sharp_filter");
            }
            builder.build()?
        };
        tp.stamp("quad_sigma kernel");

        let mut evt = OclEvent::empty();
        let kcmd = kernel
            .cmd()
            .queue(&self.queue_kernel)
            .enew(&mut evt)
            .ewait(prev.buffer.event.as_ref());

        unsafe { kcmd.enq() }?;

        tp.stamp("quad_sigma");

        res.buffer.event = Some(evt);
        Ok(res)
    }

    fn tile_minmax(&self, tp: &mut TimeProfile, prev: &OclBufferState) -> Result<OclBufferState<Uchar2>, OclError> {
        // Create output buffer
        let tile_dims = {
            // Two bytes
            let tw = prev.dims.width.div_floor(threshold::TILESZ);
            let th = prev.dims.height.div_floor(threshold::TILESZ);
            let ts = tw.next_multiple_of(16);
            ImageDimensions {
                width: tw,
                height: th,
                stride: ts,
            }
        };
        let buf_minmax = self.temp_buffer(&tile_dims, true)?;
        tp.stamp("tile_minmax buffer");
        

        // Make quad_sigma kernel
        let kernel_minmax = {
            let mut builder = OclKernel::builder();
            builder.program(&self.program);
            builder.queue(self.queue_kernel.clone());

            builder.name("k03_tile_minmax");
            builder.arg(&prev.buf);
            builder.arg(prev.dims.stride as u32);
            builder.arg(&buf_minmax);
            builder.arg(tile_dims.stride as u32);

            builder.global_work_size((tile_dims.width, tile_dims.height));
            builder.build()?
        };
        tp.stamp("tile_minmax kernel");

        let evt_minmax = {
            let mut evt = OclEvent::empty();
            let kcmd = kernel_minmax
                .cmd()
                .queue(&self.queue_kernel)
                .enew(&mut evt)
                .ewait(prev.event.as_ref());

            unsafe { kcmd.enq() }?;
            // drop(kernel_minmax);
            evt
        };
        tp.stamp("tile_minmax");

        Ok(OclBufferState {
            buf: buf_minmax,
            dims: tile_dims,
            event: Some(evt_minmax),
        })
    }

    fn tile_blur(&self, tp: &mut TimeProfile, prev: OclBufferState<Uchar2>) -> Result<OclBufferState<Uchar2>, OclError> {
        let buf_minmax_blur = self.temp_buffer(&prev.dims, true)?;
        tp.stamp("tile_blur buffer");

        // Make blur kernel
        let kernel_blur = {
            let mut builder = OclKernel::builder();
            builder.program(&self.program);
            builder.queue(self.queue_kernel.clone());

            builder.name("k03_tile_blur");
            builder.arg(&prev.buf);
            builder.arg(prev.dims.stride as u32);
            builder.arg(prev.dims.width as u32);
            builder.arg(prev.dims.height as u32);
            builder.arg(&buf_minmax_blur);
            builder.arg(prev.dims.stride as u32);

            builder.global_work_size(prev.dims);
            builder.build()?
        };
        tp.stamp("tile_blur kernel");

        let evt_blur = {
            let mut evt = OclEvent::empty();
            let kcmd = kernel_blur
                .cmd()
                .queue(&self.queue_kernel)
                .enew(&mut evt)
                .ewait(prev.event.as_ref());

            unsafe { kcmd.enq() }?;
            evt
        };
        tp.stamp("tile_blur");
        Ok(OclBufferState { buf: buf_minmax_blur, dims: prev.dims, event: Some(evt_blur) })
    }

    fn threshim(&self, config: &DetectorConfig, tp: &mut TimeProfile, prev: &OclImageState) -> Result<OclImageState, OclError> {
        let tiled_raw = self.tile_minmax(tp, &prev.buffer)?;

        fn split_image(me: &OpenCLDetector, src: &OclBufferState<Uchar2>) -> Result<(ImageY8, ImageY8), std::io::Error> {
            let host_buf = me.fetch_ocl_buffer(&src)?;
            let mut img_min = ImageY8::zeroed_packed(src.dims.width, src.dims.height);
            let mut img_max = ImageY8::zeroed_packed(src.dims.width, src.dims.height);
            for y in 0..src.dims.height {
                for x in 0..src.dims.width {
                    let elem: [u8;2] = host_buf[y * src.dims.stride + x].into();
                    img_min[(x, y)] = elem[0];
                    img_max[(x, y)] = elem[1];
                }
            }
            Ok((img_min, img_max))
        }

        config.debug_image("tile_minmax_min.pnm", |mut f| {
            let (img_min, img_max) = split_image(self, &tiled_raw)?;
            config.debug_image("tile_minmax_max.pnm", |mut f| img_max.write_pnm(&mut f));
            img_min.write_pnm(&mut f)
        });

        let tiled = self.tile_blur(tp, tiled_raw)?;
        config.debug_image("tile_minmax_blur_min.pnm", |mut f| {
            let (img_min, img_max) = split_image(self, &tiled)?;
            config.debug_image("tile_minmax_blur_max.pnm", |mut f| img_max.write_pnm(&mut f));
            img_min.write_pnm(&mut f)
        });

        let mut res = self.temp_img(&prev.buffer.dims, true)?;
        tp.stamp("threshim buffer");

        // Make threshold kernel
        let kernel = {
            let mut builder = OclKernel::builder();
            builder.program(&self.program);
            builder.queue(self.queue_kernel.clone());

            builder.name("k03_build_threshim");
            builder.arg(&prev.buffer.buf);
            builder.arg(prev.buffer.dims.stride as u32);
            builder.arg(prev.buffer.dims.width as u32);
            builder.arg(prev.buffer.dims.height as u32);
            builder.arg(tiled.buf);
            builder.arg(tiled.dims.stride);
            builder.arg(config.qtp.min_white_black_diff);
            builder.arg(&res.buffer.buf);
            builder.arg(prev.buffer.dims.stride);

            builder.global_work_size((prev.buffer.dims.width, prev.buffer.dims.height));
            builder.build()?
        };
        tp.stamp("threshim kernel");

        let evt = {
            let mut evt = OclEvent::empty();
            let kcmd = kernel
                .cmd()
                .queue(&self.queue_kernel)
                .enew(&mut evt);

            // Wait for previous kernel(s)
            let mut wait_events = Vec::<OclEvent>::new();
            if let Some(prev_event) = &prev.buffer.event {
                wait_events.push(prev_event.clone());
            }
            if let Some(tile_event) = tiled.event {
                wait_events.push(tile_event);
            }
            let evl: &[OclEvent] = &wait_events;
            let kcmd = kcmd.ewait(evl);

            unsafe { kcmd.enq() }?;
            drop(kernel);
            evt
        };
        res.buffer.event = Some(evt);

        if config.qtp.deglitch {
            tp.stamp("build_threshim");
            panic!("Deglitch not supported in OpenCL");
        }
        tp.stamp("threshim");

        Ok(res)
    }

    pub(crate) fn preprocess(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: &ImageY8) -> Result<(ImageY8, ImageY8), DetectError> {
        // Figure out which buffers we might want to download
        let mut download_src = false;
        let mut download_decimate = false;
        let mut download_sigma = false;
        if config.debug() {
            download_src = true;
            download_decimate = true;
            download_sigma = true;
        } else if config.do_quad_sigma() {
            download_sigma = true;
        } else if config.quad_decimate_mode().is_some() {
            download_decimate = true;
        } else {
            download_src = true;
        }

        let img_src = self.upload_image(download_src, tp, image)
            .expect("Error in upload_image");
        #[cfg(feature="debug")]
        config.debug_image("debug_src.pnm", |mut f| self.debug_ocl_image(&mut f, &img_src));

        let decimated = self.quad_decimate(config, download_decimate, tp, img_src)
            .expect("Error in quad_decimate");
        #[cfg(feature="debug")]
        config.debug_image("debug_decimate.pnm", |mut f| self.debug_ocl_image(&mut f, &decimated));

        let quad_im = self.quad_sigma(config, download_sigma, tp, decimated)
            .expect("Error in quad_sigma");
        #[cfg(feature="debug")]
        config.debug_image("debug_preprocess.pnm", |mut f| self.debug_ocl_image(&mut f, &quad_im));

        let threshim = self.threshim(config, tp, &quad_im)
            .expect("Error in threshold");
        #[cfg(feature="debug")]
        config.debug_image("debug_threshold.pnm", |mut f| self.debug_ocl_image(&mut f, &threshim));

        let quad_im_host = quad_im.into_image(tp, &self.queue_read_debug)
            .expect("Unable to read quad_im");
        tp.stamp("read quad_im");
        let threshim_host = threshim.into_image(tp, &self.queue_read_debug)
            .expect("Unable to read threshim");
        tp.stamp("read threshim");
        Ok((quad_im_host, threshim_host))
    }
}

#[cfg(test)]
mod test {
    use std::{hint::black_box, num::NonZeroU32};

    use rand::{thread_rng, Rng};

    use crate::{util::{ImageY8, ImageBuffer}, DetectorConfig, TimeProfile, detector::{config::QuadDecimateMode, quad_sigma_cpu}, quad_thresh::threshold::{tile_minmax_cpu, TILESZ}};

    use super::OpenCLDetector;

    fn random_image(width: usize, height: usize) -> ImageY8 {
        let mut rng = thread_rng();
        let mut result = ImageY8::zeroed_packed(width, height);
        for y in 0..height {
            for x in 0..width {
                result[(x, y)] = rng.gen();
            }
        }
        result
    }

    #[test]
    fn test_create() {
        let config = DetectorConfig::default();
        let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
        black_box(ocl);
    }

    /// Test that we can upload and download an image without error
    #[test]
    fn test_upload_download() {
        let mut config = DetectorConfig::default();
        config.debug = false;
        let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();

        let mut tp = TimeProfile::default();
        let img_cpu = random_image(128, 128);
        let img_gpu = ocl.upload_image(true, &mut tp, &img_cpu)
            .unwrap();

        let img_cpu2 = ocl.download_image(&img_gpu)
            .unwrap();
        assert_eq!(img_cpu, img_cpu2);
    }

    /// Test GPU quad_decimate with ffactor=1.5 (special case)
    #[test]
    fn test_quad_decimate_32() {
        let mut config = DetectorConfig::default();
        config.debug = false;
        config.quad_decimate = 1.5;
        assert_eq!(config.quad_decimate_mode(), Some(QuadDecimateMode::ThreeHalves));

        let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();

        let mut tp = TimeProfile::default();
        let img_cpu = random_image(128, 128);
        let img_gpu = ocl.upload_image(false, &mut tp, &img_cpu)
            .unwrap();

        // Decimate on GPU
        let img_gpu = ocl.quad_decimate(&config, true, &mut tp, img_gpu)
            .unwrap();

        // Decimate on CPU
        let img_cpu1 = img_cpu.decimate_three_halves();

        // Download GPU image
        let img_cpu2 = ocl.download_image(&img_gpu)
            .unwrap();
        assert_eq!(img_cpu1, img_cpu2);
    }

    /// Test GPU quad_decimate with ffactor=3.0
    #[test]
    fn test_quad_decimate_3() {
        let mut config = DetectorConfig::default();
        config.debug = false;
        config.quad_decimate = 3.0;
        assert_eq!(config.quad_decimate_mode(), Some(QuadDecimateMode::Scaled(NonZeroU32::new(3).unwrap())));

        let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();

        let mut tp = TimeProfile::default();
        let img_cpu = random_image(128, 128);
        let img_gpu = ocl.upload_image(false, &mut tp, &img_cpu)
            .unwrap();

        // Decimate on GPU
        let img_gpu = ocl.quad_decimate(&config, true, &mut tp, img_gpu)
            .unwrap();

        // Decimate on CPU
        let img_cpu1 = img_cpu.decimate(config.quad_decimate);

        // Download GPU image
        let img_cpu2 = ocl.download_image(&img_gpu)
            .unwrap();
        assert_eq!(img_cpu1, img_cpu2);
    }

    #[test]
    fn test_quad_sigma_blur() {
        let mut config = DetectorConfig::default();
        config.debug = false;
        config.quad_sigma = 0.8;
        assert!(config.do_quad_sigma());

        let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
        assert!(ocl.quad_sigma_filter.is_some());
        let mut tp = TimeProfile::default();
        // let img_cpu = random_image(4, 4);
        let img_cpu = random_image(128, 128);
        let img_gpu = ocl.upload_image(false, &mut tp, &img_cpu)
            .unwrap();

        // Blur on GPU
        let img_gpu = ocl.quad_sigma(&config, true, &mut tp, img_gpu)
            .unwrap();

        // Blur on CPU
        let img_cpu1 = {
            let mut img_cpu1 = img_cpu.clone();
            // img_cpu1.gaussian_blur(config.quad_sigma as _, ksz);
            quad_sigma_cpu(&mut img_cpu1, config.quad_sigma);
            img_cpu1
        };

        // Download GPU image
        let img_cpu2 = ocl.download_image(&img_gpu)
            .unwrap();
        assert_eq!(img_cpu1, img_cpu2);
    }

    #[test]
    #[ignore=""]
    fn test_quad_sigma_sharp() {
        let mut config = DetectorConfig::default();
        config.debug = false;
        config.quad_sigma = -0.8;
        assert!(config.do_quad_sigma());

        let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
        assert!(ocl.quad_sigma_filter.is_some());
        let mut tp = TimeProfile::default();
        let img_cpu = random_image(128, 128);
        let img_gpu = ocl.upload_image(false, &mut tp, &img_cpu)
            .unwrap();

        // Sharpen on GPU
        let img_gpu = ocl.quad_sigma(&config, true, &mut tp, img_gpu)
            .unwrap();

        // Sharpen on CPU
        let img_cpu1 = {
            let mut img_cpu1 = img_cpu.clone();
            // img_cpu1.gaussian_blur(config.quad_sigma as _, ksz);
            quad_sigma_cpu(&mut img_cpu1, config.quad_sigma);
            img_cpu1
        };

        // Download GPU image
        let img_cpu2 = ocl.download_image(&img_gpu)
            .unwrap();
        assert_eq!(img_cpu1, img_cpu2);
    }

    #[test]
    fn test_tile_minmax() {
        let mut config = DetectorConfig::default();
        config.debug = true;

        let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
        let mut tp = TimeProfile::default();
        let img_cpu = random_image(128, 128);
        let img_gpu = ocl.upload_image(false, &mut tp, &img_cpu)
            .unwrap();

        // Tile on GPU
        let img_gpu = ocl.tile_minmax(&mut tp, &img_gpu.buffer)
            .unwrap();
        
        // Tile on CPU
        let img_cpu1 = tile_minmax_cpu::<TILESZ>(&img_cpu);

        // Download GPU image
        let img_cpu2 = {
            let buf_cpu2 = ocl.fetch_ocl_buffer(&img_gpu)
                .unwrap();
            let mut dst = ImageBuffer::<[u8; 2]>::zeroed_with_stride(img_gpu.dims.width, img_gpu.dims.height, img_gpu.dims.stride);
            for i in 0..buf_cpu2.len() {
                dst.data[i * 2 + 0] = buf_cpu2[i][0];
                dst.data[i * 2 + 1] = buf_cpu2[i][1];
            }
            dst
        };
        assert_eq!(img_cpu1, img_cpu2);
    }
}