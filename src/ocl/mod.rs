#![cfg(feature="opencl")]

mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod buffer;
mod test;

use std::borrow::Borrow;
use std::marker::PhantomData;
use std::time::{Instant, Duration};

use ocl::core::OpenclVersion;
use ocl::enums::{ProfilingInfo, DeviceInfoResult};
use ocl::prm::{Uchar2, Uint};
use ocl::{
    Platform as OclPlatform,
    Error as OclError,
    Device as OclDevice,
    Program, Context,
    Queue,
    CommandQueueProperties,
    Kernel as OclKernel,
    Event as OclEvent, DeviceType, OclPrm, SpatialDims, EventList,
};
use rayon::Scope;

use crate::detector::config::GpuAccelRequest;
use crate::ocl::stage2::OclQuadSigma;
use crate::ocl::stage4::WrappedUnionFind;
use crate::quad_thresh::{gradient_clusters, Clusters, debug_unionfind};
use crate::quad_thresh::threshold::{self};
use crate::quad_thresh::unionfind::UnionFind2D;
use crate::util::image::{ImageWritePNM, ImageRefY8};
use crate::util::pool::PoolGuard;
use crate::DetectorBuildError;
use crate::util::mem::SafeZero;
use crate::{DetectorConfig, util::{ImageY8, image::ImageDimensions}, DetectError, TimeProfile};

use self::buffer::{OclAwaitable, OclBufferState, OclImageState, OclBufferMapped, OclBufferLike, OclCore};
use self::stage1::OclQuadDecimate;
use self::stage3::{OclTileMinmax, OclTileBlur, OclThreshold};
use self::stage4::{OclConnectedComponents, OclUnionFindInit, OclUfFlatten};

const PROG_QUAD_DECIMATE: &str = include_str!("./01_quad_decimate.cl");
const PROG_QUAD_SIGMA: &str = include_str!("./02_quad_sigma.cl");
const PROG_THRESHOLD: &str = include_str!("./03_threshold.cl");
const PROG_UNIONFIND: &str = include_str!("./04_unionfind_simple.cl");

trait OclStage {
    type S: OclPrm;
    type R: OclPrm;
    type K<'a>: Borrow<OclKernel> + 'a where Self: 'a;
    fn result_dims(&self, src_dims: &ImageDimensions) -> ImageDimensions;
    
    fn make_kernel<'a>(&'a self, core: &OclCore, src_buf: &OclBufferState<Self::S>, dst_dims: &OclBufferState<Self::R>) -> Result<Self::K<'a>, OclError>;
}

impl From<OclError> for DetectError {
    fn from(_: OclError) -> Self {
        Self::OpenCLError
    }
}

impl Into<SpatialDims> for ImageDimensions {
    fn into(self) -> SpatialDims {
        SpatialDims::Two(self.width, self.height)
    }
}

impl SafeZero for Uchar2 {}
impl SafeZero for Uint {}

pub(crate) struct OpenCLDetector {
    core: OclCore,
    download_src: bool,
    quad_decimate: Option<OclQuadDecimate>,
    quad_sigma: Option<OclQuadSigma>,
    tile_minmax: OclTileMinmax,
    tile_blur: OclTileBlur,
    threshold: OclThreshold,
    unionfind_init: OclUnionFindInit,
    connected_components: OclConnectedComponents,
    uf_flatten: OclUfFlatten,
}

// struct KernelBuffersG<'a, R: OclPrm = u8> {
//     buffer: OclBufferState<R>,
//     kernel: PoolGuard<'a, (), OclKernel>,
// }
struct KernelBuffers<R: OclPrm = u8, B: OclBufferLike<R> = OclBufferState<R>, K: Borrow<OclKernel> = OclKernel> {
    buffer: B,
    kernel: K,
    p: PhantomData<R>
}
struct Kernels<'a> {
    quad_decimate: Option<KernelBuffers<u8, OclBufferState, PoolGuard<'a, (), OclKernel>>>,
    quad_sigma: Option<KernelBuffers>,
    tile_minmax: KernelBuffers<Uchar2>,
    tile_blur: KernelBuffers<Uchar2>,
    threshold: KernelBuffers<u8>,
    unionfind_init: KernelBuffers<u32, OclBufferMapped<u32>>,
    connected_components: KernelBuffers<u32>,
    uf_flatten: OclKernel,
}

fn combined_events<'a>(bases: &[&'a dyn OclAwaitable]) -> EventList {
    let mut wait_events = EventList::new();
    for base in bases {
        if let Some(evt) = base.event() {
            wait_events.push(evt.clone());
        }
    }
    wait_events
}

fn find_device(mode: &GpuAccelRequest) -> Result<(OclPlatform, OclDevice), DetectorBuildError> {
    let (devtype, idx, prefer) = match mode {
        GpuAccelRequest::Disabled => panic!(),
        GpuAccelRequest::Prefer => (None, 0, true),
        GpuAccelRequest::Required => (None, 0, false),
        GpuAccelRequest::PreferDeviceIdx(idx) => (None, *idx, true),
        GpuAccelRequest::RequiredDeviceIdx(idx) => (None, *idx, false),
        GpuAccelRequest::PreferGpu => (Some(DeviceType::GPU), 0, true),
        GpuAccelRequest::RequiredGpu => (Some(DeviceType::GPU), 0, false),
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
        println!("OpenCL version: {:?}", device.version());
        println!("OpenCL extensions: {:?}", device.info(ocl::enums::DeviceInfo::Extensions));
        println!("OpenCL built in kernels: {:?}", device.info(ocl::enums::DeviceInfo::BuiltInKernels));
        println!("OpenCL profile: {:?}", device.info(ocl::enums::DeviceInfo::Profile));
        if let Ok(true) = device.is_available() {
            return Ok((platform, device));
        }
    }
    Err(DetectorBuildError::GpuNotAvailable)
}

fn split_minmax(core: &OclCore, buf: &OclBufferState<Uchar2>) -> (ImageY8, ImageY8) {
    let mut im_min = ImageY8::zeroed(buf.width(), buf.height());
    let mut im_max = ImageY8::zeroed(buf.width(), buf.height());
    let buf_cpu = core.fetch_ocl_buffer(buf).unwrap();
    for y in 0..buf.height() {
        for x in 0..buf.width() {
            let v = buf_cpu[y * buf.stride() + x];
            im_min[(x, y)] = v[0];
            im_max[(x, y)] = v[1];
        }
    }
    (im_min, im_max)
}

impl OpenCLDetector {
    pub(super) fn new(config: &DetectorConfig) -> Result<Self, DetectorBuildError> {
        let (platform, device) = find_device(&config.gpu)?;

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        
        let (queue_write, queue_kernel, queue_init, queue_read_debug) = {
            let available_props = match device.info(ocl::enums::DeviceInfo::QueueProperties) {
                Ok(DeviceInfoResult::QueueProperties(props)) => props,
                Ok(_) => return Err(DetectorBuildError::OpenCLError("Unable to get queue props".into())),
                Err(e) => return Err(DetectorBuildError::OpenCLError(e)),
            };
            
            let props = CommandQueueProperties::new();
            #[cfg(feature="debug")]
            let props = if !config.debug {
                props
            } else if !available_props.contains(CommandQueueProperties::PROFILING_ENABLE) {
                eprintln!("OpenCL: Profiling unavailable");
                props
            } else {
                props.profiling()
            };

            // Try to make out-of-order queues
            if available_props.contains(CommandQueueProperties::OUT_OF_ORDER_EXEC_MODE_ENABLE) {
                let props = Some(props.out_of_order());
                let queue = Queue::new(&context, device, props)?;
                (queue.clone(), queue.clone(), queue.clone(), queue.clone())
            } else {
                let props = Some(props);
                let queue_write = Queue::new(&context, device, props)?;
                let queue_kernel = Queue::new(&context, device, props)?;
                let queue_init = Queue::new(&context, device, props)?;
                let queue_debug = Queue::new(&context, device, props)?;
                (queue_write, queue_kernel, queue_init, queue_debug)
            }
        };

        let opencl_version = device
            .version()
            .map_err(|e| DetectorBuildError::OpenCLError(e.into()))?;
        
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

            if opencl_version >= OpenclVersion::new(2, 0) {
                builder.cmplr_def("OPENCL_VERSION_2", 1);
            }
            builder.source(PROG_UNIONFIND);
            
            let res = builder.build(&context);
            #[cfg(feature="debug")]
            if let Err(OclError::OclCore(ocl::OclCoreError::ProgramBuild(e))) = &res {
                let buf = format!("{e:?}");
                let msg = buf.strip_prefix("BuildLog(").unwrap_or(&buf);
                let msg = msg.replace("\\n", "\n");
                eprintln!("Build error: {}", msg);
            }
            res?
        };

        #[cfg(feature="debug")]
        if config.debug {
            println!("Build status: {:?}", program.build_info(device, ocl::enums::ProgramBuildInfo::BuildStatus).unwrap());
            println!("Binary type: {:?}", program.build_info(device, ocl::enums::ProgramBuildInfo::BinaryType).unwrap());
            println!("Build log: {:?}", program.build_info(device, ocl::enums::ProgramBuildInfo::BuildLog).unwrap());
            println!("Build options: {:?}", program.build_info(device, ocl::enums::ProgramBuildInfo::BuildOptions).unwrap());
            println!("Kernel names: {:?}", program.info(ocl::enums::ProgramInfo::KernelNames).unwrap());
        }

        let core = OclCore {
            device,
            context,
            program,
            queue_write,
            queue_init,
            queue_kernel,
            queue_read_debug,
        };

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


        let quad_decimate = OclQuadDecimate::new(&core, config.quad_decimate_mode(), download_decimate);
        let quad_sigma = OclQuadSigma::new(&core, config.quad_sigma, download_sigma);
        let tile_minmax = OclTileMinmax::new(&core);
        let tile_blur = OclTileBlur::new(&core);
        let threshold = OclThreshold::new(&core, config);
        let unionfind_init = OclUnionFindInit::new(&core);
        let connected_components = OclConnectedComponents::new(&core);
        let uf_flatten = OclUfFlatten::new(&core);

        Ok(Self {
            core,
            download_src,
            quad_decimate,
            quad_sigma,
            tile_minmax,
            tile_blur,
            threshold,
            unionfind_init,
            connected_components,
            uf_flatten,
        })
    }

    #[cfg(feature="debug")]
    fn debug_ocl_image(&self, file: &mut std::fs::File, image: &OclBufferState) -> Result<(), std::io::Error> {
        let host_img = self.core.download_image(image)?;
        
        host_img.write_pnm(file)
    }

    fn make_kernels(&self, config: &DetectorConfig, tp: &mut TimeProfile, img_src: &OclImageState) -> Result<Kernels, DetectError> {
        // Build kernels (relatively slow)
        let mut kernel_quad_decimate = None;
        let mut kernel_quad_sigma = None;
        let mut kernel_tile_minmax = None;
        let mut kernel_tile_blur = None;
        let mut kernel_threshold = None;
        let mut kernel_unionfind_init = None;
        let mut kernel_connected_components = None;
        let mut kernel_uf_flatten = None;
        
        fn map_kern<R: OclPrm>(kernel: Result<OclKernel, OclError>, buf: OclBufferState<R>) -> Result<KernelBuffers<R>, OclError> {
            Ok(KernelBuffers { buffer: buf, kernel: kernel?, p: PhantomData })
        }

        fn scope<'a, R: Send>(f: impl (FnOnce(Option<&Scope<'a>>) -> R) + Send) -> R {
            rayon::in_place_scope(|s| f(Some(s)))
        }

        scope(|s| -> Result<(), OclError> {
            let last_buf = img_src.buffer.clone();

            fn spawn<'a>(s: Option<&Scope<'a>>, f: impl (FnOnce()) + Send + 'a) {
                s.unwrap().spawn(|_| f());
            }

            // Quad decimate
            let last_buf = if let Some(quad_decimate) = self.quad_decimate.as_ref() {
                let qd_dims = quad_decimate.result_dims(&last_buf.dims);
                let qd_buf = self.core.temp_bufferstate::<u8>(&qd_dims, quad_decimate.download)?;
                let r = qd_buf.clone();
                spawn(s, || {
                    let last_buf = last_buf; // move
                    kernel_quad_decimate = Some(quad_decimate.make_kernel(&self.core, &last_buf, &qd_buf)
                        .map(|k| Some(KernelBuffers { buffer: qd_buf, kernel: k, p: PhantomData })));
                    tp.stamp("qd kernel");
                });
                r
            } else {
                kernel_quad_decimate = Some(Ok(None));
                last_buf
            };

            // Quad sigma
            let last_buf = if let Some(quad_sigma) = self.quad_sigma.as_ref() {
                let qs_dims = quad_sigma.result_dims(&last_buf.dims);
                let qs_buf = self.core.temp_bufferstate::<u8>(&qs_dims, quad_sigma.download)?;
                let r = qs_buf.clone();
                spawn(s, || {
                    let last_buf = last_buf; // move
                    kernel_quad_sigma = Some(quad_sigma.make_kernel(&self.core, &last_buf, &qs_buf)
                        .map(|k| Some(KernelBuffers { buffer: qs_buf, kernel: k, p: PhantomData })));
                });
                // tp.stamp("qs kernel");
                r
            } else {
                kernel_quad_sigma = Some(Ok(None));
                last_buf
            };

            // Tile minmax
            let tm_buf = {
                let tm_dims = self.tile_minmax.result_dims(&last_buf.dims);
                let tm_buf = self.core.temp_bufferstate::<Uchar2>(&tm_dims, config.generate_debug_image())?;
                let r = tm_buf.clone();
                let last_buf = last_buf.clone();
                spawn(s, || {
                    let last_buf = last_buf; // move
                    kernel_tile_minmax = Some(map_kern(self.tile_minmax.make_kernel(&self.core, &last_buf, &tm_buf), tm_buf));
                    // tp.stamp("tm kernel");
                });
                r
            };

            // Tile blur
            let tb_buf = {
                let tb_dims = self.tile_blur.result_dims(&tm_buf.dims);
                let tb_buf = self.core.temp_bufferstate::<Uchar2>(&tb_dims, config.generate_debug_image())?;
                let r = tb_buf.clone();
                spawn(s, || {
                    let tm_buf = tm_buf; // move
                    kernel_tile_blur = Some(map_kern(self.tile_blur.make_kernel(&self.core, &tm_buf, &tb_buf), tb_buf));
                    // tp.stamp("tb kernel");
                });
                r
            };

            // Threshold
            let th_buf = {
                let th_dims = self.threshold.result_dims(&last_buf.dims);
                let th_buf = self.core.temp_bufferstate::<u8>(&th_dims, true)?;
                let thb1 = th_buf.clone();
                let last_buf = last_buf.clone();
                spawn(s, || {
                    let last_buf = last_buf; // move
                    let tb_buf = tb_buf; // move
                    kernel_threshold = Some(map_kern(self.threshold.make_kernel(&self.core, &last_buf, &tb_buf, &th_buf), th_buf));
                    // tp.stamp("th kernel");
                });
                thb1
            };

            // UnionFind Init
            let uf_buf = {
                let uf_dims = self.unionfind_init.result_dims(&last_buf.dims);
                let uf_buf = self.core.mapped_buffer::<u32>(uf_dims)?;
                let last_buf = last_buf.clone();
                let r = uf_buf.clone();
                spawn(s, || {
                    let last_buf = last_buf; // move
                    kernel_unionfind_init = Some(self.unionfind_init.make_kernel(&self.core, &last_buf.dims, &uf_buf)
                        .map(|kernel| KernelBuffers { buffer: uf_buf, kernel, p: PhantomData }));
                    // tp.stamp("ufi kernel");
                });
                r
            };

            // Connected Components
            let debug = self.core.temp_bufferstate(&ImageDimensions { width: 2, height: 1, stride: 2 }, true)?;
            {
                let uf_buf = uf_buf.clone();
                spawn(s, || {
                    let th_buf = th_buf; // move
                    let uf_buf = uf_buf; // move
                    kernel_connected_components = Some(self.connected_components.make_kernel(&self.core, &th_buf, &uf_buf, &debug)
                        .map(|kernel| KernelBuffers { buffer: debug, kernel, p: PhantomData }));
                    // tp.stamp("cc kernel");
                });
            }

            spawn(s, || {
                let uf_buf = uf_buf; // move
                kernel_uf_flatten = Some(self.uf_flatten.make_kernel(&self.core, &uf_buf));
            });
            Ok(())
        }).expect("Error building kernels");//TODO: forward error

        println!("Made kernels");

        Ok(Kernels {
            quad_decimate: kernel_quad_decimate.unwrap()?,
            quad_sigma: kernel_quad_sigma.unwrap()?,
            tile_minmax: kernel_tile_minmax.unwrap()?,
            tile_blur: kernel_tile_blur.unwrap()?,
            threshold: kernel_threshold.unwrap()?,
            unionfind_init: kernel_unionfind_init.unwrap()?,
            connected_components: kernel_connected_components.unwrap()?,
            uf_flatten: kernel_uf_flatten.unwrap()?,
        })
    }

    pub(crate) fn cluster(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<(ImageY8, Clusters), DetectError> {
        // Upload source image (TODO: async?)
        let img_src = {
            let img_src = self.core.upload_image(self.download_src, tp, image)
                .expect("Uploading src");
            #[cfg(feature="debug")]
            config.debug_image("00_debug_src.pnm", |mut f| self.debug_ocl_image(&mut f, &img_src.buffer));
            img_src
        };

        let mut kernels = self.make_kernels(config, tp, &img_src)
            .expect("Creating kernels");

        tp.stamp("ocl kernels");
        

        // Enqueue kernels
        fn enqueue_kernel<R: OclPrm, B: OclBufferLike<R> + Clone, K: Borrow<OclKernel>>(kb: &mut KernelBuffers<R, B, K>, last_event: Option<&OclEvent>) -> Result<B, OclError> {
            let mut evt = OclEvent::empty();
            let kcmd = kb.kernel
                .borrow()
                .cmd()
                .enew(&mut evt)
                .ewait(last_event)
                ;

            unsafe { kcmd.enq() }.expect("Unabel to enqueue kernel");

            *kb.buffer.event_mut() = Some(evt.clone());
            Ok(kb.buffer.clone())
        }
        println!("Will enqueue");

        // let mut last_event = img_src.event();
        let last_buf = &img_src.buffer;
        // Quad Decimate
        let last_buf = if let Some(k_quad_decimate) = kernels.quad_decimate.as_mut() {
            let qd_buf = enqueue_kernel(k_quad_decimate, last_buf.event())
                .unwrap();
            tp.stamp("quad_decimate enq");

            #[cfg(feature="debug")]
            config.debug_image("00a_debug_decimate.pnm", |mut f| {
                let img = self.core.download_image(&qd_buf)?;
                img.write_pnm(&mut f)
            });
            qd_buf
        } else {
            last_buf.clone()
        };
        println!("Enqueued quad_decimate");
        let qd_buf = last_buf.clone();

        // Quad Sigma
        let last_buf = if let Some(k_quad_sigma) = kernels.quad_sigma.as_mut() {
            let buf = enqueue_kernel(k_quad_sigma, last_buf.event())?;
            tp.stamp("quad_sigma enq");
            buf
        } else {
            last_buf
        };
        let qs_buf = last_buf.clone();
        println!("Enqueued quad_sigma");

        #[cfg(feature="debug")]
        config.debug_image("01_debug_preprocess.pnm", |mut f| {
            let img = self.core.download_image(&last_buf)?;
            img.write_pnm(&mut f)
        });

        // Tile MinMax
        let tm_buf = enqueue_kernel(&mut kernels.tile_minmax, last_buf.event())?;
        tp.stamp("tile_minmax enq");

        #[cfg(feature="debug")]
        config.debug_image("02a_tile_minmax_min.pnm", |mut f| {
            println!("Enqueued tile_minmax");
            let (img_min, img_max) = split_minmax(&self.core, &tm_buf);
            config.debug_image("02b_tile_minmax_max.pnm", |mut f| img_max.write_pnm(&mut f));
            img_min.write_pnm(&mut f)
        });

        // Tile Blur
        let tb_buf = enqueue_kernel(&mut kernels.tile_blur, tm_buf.event())?;
        tp.stamp("tile_blur enq");
        #[cfg(feature="debug")]
        config.debug_image("02c_tile_minmax_blur_min.pnm", |mut f| {
            println!("Enqueued tile_blur");
            let (img_min, img_max) = split_minmax(&self.core, &tb_buf);
            config.debug_image("02d_tile_minmax_blur_max.pnm", |mut f| img_max.write_pnm(&mut f));
            img_min.write_pnm(&mut f)
        });

        // Threshold
        let th_buf = {
            let mut evt = OclEvent::empty();
            let ewait = combined_events(&[&last_buf, &tb_buf]);
            let kcmd = kernels.threshold.kernel
                .cmd()
                .enew(&mut evt)
                .ewait(&ewait)
                ;

            unsafe { kcmd.enq() }.expect("threshold enq");
            tp.stamp("threshold enq");

            kernels.threshold.buffer.event = Some(evt);
            &kernels.threshold.buffer
        };

        #[cfg(feature="debug")]
        config.debug_image("02_debug_threshold.pnm", |mut f| {
            println!("Enqueued threshold");
            let img = self.core.download_image(&th_buf).expect("Download threshold");
            img.write_pnm(&mut f)
        });

        // UnionFind Init
        let uf_buf = enqueue_kernel(&mut kernels.unionfind_init, None)?;
        tp.stamp("ufi enq");
        println!("Enqueued ufi");
        #[cfg(feature="debug")]
        let uf0_buf = uf_buf.clone();

        let (uf_buf, db_buf) = {
            let mut evt = OclEvent::empty();
            let ewait = combined_events(&[&uf_buf, th_buf]);
            let kcmd = kernels.connected_components.kernel
                .cmd()
                .enew(&mut evt)
                .ewait(&ewait)
                ;

            unsafe { kcmd.enq() }.expect("uf submit");
            tp.stamp("cc enq");
            
            (uf_buf.with_event(evt.clone()), kernels.connected_components.buffer.with_event(evt))
        };

        #[cfg(feature="debug")]
        let uf1_buf = uf_buf.clone();

        let uf_buf = {
            let mut evt = OclEvent::empty();
            let kcmd = kernels.uf_flatten
                .cmd()
                .enew(&mut evt)
                .ewait(uf_buf.event())
                ;

            unsafe { kcmd.enq() }.expect("uf submit");
            tp.stamp("cc enq");
            
            uf_buf.with_event(evt)
        };

        tp.stamp("kernel enqueue");

        uf_buf.event().unwrap().wait_for().unwrap();
        tp.stamp("await event");

        let (threshim, mut uf) = rayon::join(
            || self.core.download_image(th_buf).unwrap(),
            || {
                let uf_data = self.core.fetch_ocl_buffer(&uf_buf).unwrap();
                UnionFind2D::wrap(uf_buf.width() / 2, uf_buf.height(), WrappedUnionFind {
                    data: uf_data,
                })
            }
        );
        tp.stamp("Download uf");
        println!("Enqueued uf");

        // Debug images
        #[cfg(feature="debug")]
        if config.generate_debug_image() {
            println!("Debugging uf");
            let db = self.core.fetch_ocl_buffer(&db_buf).unwrap();
            debug_unionfind(config, tp, &th_buf.dims, &mut uf);
            println!("UF debug: {db:?}");
            println!("Debug uf");
        }

        let quad_im = self.core.download_image(&last_buf);
        tp.stamp("download quad_im");
        println!("Download quad_im");
        let clusters = gradient_clusters(config, &threshim.as_ref(), uf);
        println!("Grad clusters");
        
        // let (quad_im, clusters) = rayon::join(
        //     || self.core.download_image(&last_buf),
        //     || gradient_clusters(config, &threshim.as_ref(), uf),
        // );
        tp.stamp("grad_clusters");

        // let mut uf = {
        //     let uf_data = self.fetch_ocl_buffer(&uf_buf).unwrap();
        //         UnionFind2D::wrap(uf_buf.width() / 2, uf_buf.height(), WrappedUnionFind {
        //             data: uf_data,
        //         })
        // };
        // tp.stamp("Download uf");

        // let threshim = self.download_image(th_buf).unwrap();
        tp.stamp("Download threshim");

        #[cfg(feature="debug")]
        if config.debug {
            let mut tp_gpu = TimeProfile::default();
            let t0 = Instant::now();
            let get_time = |event: &OclEvent, info: ProfilingInfo| -> Instant {
                let time = event.profiling_info(info).unwrap().time().unwrap();
                t0.checked_add(Duration::from_nanos(time*1000)).unwrap()
            };
            
            tp_gpu.set_start(get_time(qd_buf.event.as_ref().unwrap(), ProfilingInfo::Queued));
            let mut record_evt = |name, evt| {
                tp_gpu.stamp_at(&format!("{name} queued"), get_time(evt, ProfilingInfo::Queued));
                tp_gpu.stamp_at(&format!("{name} submit"), get_time(evt, ProfilingInfo::Submit));
                tp_gpu.stamp_at(&format!("{name} start"), get_time(evt, ProfilingInfo::Start));
                tp_gpu.stamp_at(&format!("{name} end"), get_time(evt, ProfilingInfo::End));
            };
            if self.quad_decimate.is_some() {
                record_evt("qd", qd_buf.event().unwrap());
            }
            if self.quad_sigma.is_some() {
                record_evt("qs", qs_buf.event().unwrap());
            }
            record_evt("tm", tm_buf.event().unwrap());
            record_evt("tb", tb_buf.event().unwrap());
            record_evt("th", th_buf.event().unwrap());
            record_evt("ufi", uf0_buf.event().unwrap());
            record_evt("cc", uf1_buf.event().unwrap());
            record_evt("uf", uf_buf.event().unwrap());
            println!("{tp_gpu}");
        }

        let quad_im = quad_im.expect("Unable to download quad_im");

        Ok((quad_im, clusters))
    }
}
