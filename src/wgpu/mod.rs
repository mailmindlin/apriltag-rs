#![cfg(feature="opencl")]

mod quad_decimate;
mod quad_sigma;

use std::marker::PhantomData;
use std::mem::size_of;

use bytemuck::Pod;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use wgpu::{
    Instance, Device, Queue, ShaderStages, RequestAdapterOptions, PowerPreference, BufferUsages, BufferDescriptor, BindGroupDescriptor, BindGroupEntry, ComputePassDescriptor, CommandEncoder,
};

use crate::quad_thresh::threshold;
use crate::util::image::{ImageWritePNM, ImageRefY8};
use crate::util::multiple::lcm;
use crate::{OpenClMode, DetectorBuildError};
use crate::util::mem::{calloc, SafeZero};
use crate::{DetectorConfig, util::{ImageY8, image::ImageDimensions}, DetectError, TimeProfile};

use self::quad_decimate::WQuadDecimate;
use self::quad_sigma::WQuadSigma;


const PROG_QUAD_SIGMA: &str = include_str!("./02_quad_sigma.wgsl");
// const PROG_THRESHOLD: &str = include_str!("./03_threshold.cl");

pub(crate) struct WGPUDetector {
    pub(crate) mode: OpenClMode,
    device: wgpu::Device,
    queue: wgpu::Queue,
    quad_decimate: Option<WQuadDecimate>,
    quad_sigma: Option<WQuadSigma>,
}

struct WContext<'a> {
    tp: &'a mut TimeProfile,
    config: &'a DetectorConfig,
    device: &'a wgpu::Device,
    next_read: bool,
    next_align: usize,
    encoder: &'a mut CommandEncoder,
}

impl<'a> WContext<'a> {
    fn encoder(&self) -> &'a mut CommandEncoder {
        self.encoder
    }
    fn temp_buffer(&self, width: usize, height: usize, align: usize, read: bool) -> Result<WImage<u8>, WGPUError> {
        let usage  = if self.next_read | read {
            BufferUsages::MAP_WRITE | BufferUsages::MAP_READ
        } else {
            BufferUsages::MAP_WRITE
        };

        let alignment = lcm(self.next_align, align);
        debug_assert_eq!(alignment % size_of::<u32>(), 0);


        let dims = ImageDimensions {
            width,
            height,
            stride: width.next_multiple_of(alignment),
        };
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: ((dims.stride * height)) as u64,
            usage,
            mapped_at_creation: false,
        });
        Ok(WImage { buffer, dims, data: PhantomData })
    }
    fn temp_buffer_dims(&self, dims: &ImageDimensions) -> Result<WImage<u8>, WGPUError> {
        self.temp_buffer(dims.width, dims.height, 1, false)
    }
}

#[derive(Debug)]
pub enum WGPUError {

}

trait WStage {
    fn src_alignment(&self) -> usize;
    fn debug_name(&self) -> &'static str;

    fn apply(&self, ctx: &mut WContext<'_>, img_src: WImage) -> Result<WImage, WGPUError>;
}

struct WImage<E=u8> {
    buffer: wgpu::Buffer,
    dims: ImageDimensions,
    data: PhantomData<E>,
}

async fn find_device(mode: &OpenClMode) -> Result<(Device, Queue), DetectorBuildError> {
    let inst = wgpu::Instance::default();
    let mut opts = RequestAdapterOptions::default();
    opts.power_preference = PowerPreference::HighPerformance;
    let adapter = inst.request_adapter(&opts).await
        .unwrap();
    println!("{adapter:?} {:?} {:?}", adapter.features(), adapter.get_info());
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();
    println!("{device:?}");
    Ok((device, queue))
}

impl WGPUDetector {
    async fn new_async(config: &DetectorConfig, mode: OpenClMode) -> Result<Self, DetectorBuildError> {
        let (device, queue) = find_device(&mode).await?;

        let quad_decimate = config.quad_decimate_mode().map(|m| WQuadDecimate::new(&device, m));
        let quad_sigma = WQuadSigma::new(&device, config.quad_sigma);

        // let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //     label: Some("apriltag_bgl"),
        //     entries: &[
        //         wgpu::BindGroupLayoutEntry {
        //             binding: 0,
        //             visibility: ShaderStages::COMPUTE,
        //             ty: wgpu::BindingType::Buffer {
        //                 ty: wgpu::BufferBindingType::Storage { read_only: true },
        //                 has_dynamic_offset: false,
        //                 min_binding_size: None,
        //             },
        //             count: None,
        //         },
        //         wgpu::BindGroupLayoutEntry {
        //             binding: 1,
        //             visibility: ShaderStages::COMPUTE,
        //             ty: wgpu::BindingType::Buffer {
        //                 ty: wgpu::BufferBindingType::Storage { read_only: false },
        //                 has_dynamic_offset: false,
        //                 min_binding_size: None,
        //             },
        //             count: None,
        //         },
        //         wgpu::BindGroupLayoutEntry {
        //             binding: 2,
        //             visibility: ShaderStages::COMPUTE,
        //             ty: wgpu::BindingType::Buffer {
        //                 ty: wgpu::BufferBindingType::Storage { read_only: false },
        //                 has_dynamic_offset: false,
        //                 min_binding_size: None,
        //             },
        //             count: None,
        //         },
        //     ],
        // });

        // let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //     label: Some("apriltag_detect"),
        //     bind_group_layouts: &[
        //         &bg_layout,
        //     ],
        //     push_constant_ranges: &[],
        // });
        
        // let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        //     label: Some("apriltag_detect"),
        //     layout: Some(&pipeline_layout),
        //     module: &module_quad_decimate,
        //     entry_point: "main",
        // });

        Ok(Self {
            mode,
            device,
            queue,
            quad_decimate,
            quad_sigma,
        })
    }

    pub(super) fn new(config: &DetectorConfig, mode: OpenClMode) -> Result<Self, DetectorBuildError> {
        todo!()
    }

    fn fetch_ocl_buffer<E: SafeZero + Pod>(&self, image: &WImage<E>) -> Result<Box<[E]>, std::io::Error> {
        let mut data = calloc::<E>(image.buffer.size() as usize);
        let buf_view = image.buffer.slice(..).get_mapped_range();
        let buf_slice = bytemuck::cast_slice::<u8, E>(&buf_view);
        data.copy_from_slice(buf_slice);
        image.buffer.unmap();

        Ok(data)
    }

    fn download_image(&self, image: &WImage) -> Result<ImageY8, std::io::Error> {
        let host_buf = self.fetch_ocl_buffer(image)?;
        
        let host_img = ImageY8::wrap(host_buf, image.dims.width, image.dims.height, image.dims.stride);
        Ok(host_img)
    }

    #[cfg(feature="debug")]
    fn debug_image(&self, file: &mut std::fs::File, image: &WImage) -> Result<(), std::io::Error> {
        use crate::util::image::ImageWritePNM;
        let host_img = self.download_image(image)?;
        
        host_img.write_pnm(file)
    }

    /// Upload source image to OpenCL device
    fn upload_image(&self, downloadable: bool, stride_alignment: usize, image: &ImageRefY8) -> WImage {
        let usage  = if downloadable {
            BufferUsages::MAP_WRITE | BufferUsages::MAP_READ
        } else {
            BufferUsages::MAP_WRITE
        };
        let label = None;
        //TODO: make async?
        let (buffer, dims) = if (image.stride() % stride_alignment == 0) && image.width() > (image.stride() * 3 / 2) {
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
                stride: image.dimensions().width.next_multiple_of(stride_alignment),
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

        WImage {
            buffer,
            dims,
            data: PhantomData
        }
    }

    fn tile_minmax(&self, tp: &mut TimeProfile, prev: &OclImageState) -> Result<OclImageState<Uchar2>, OclError> {
        // Create output buffer
        let tile_dims = {
            // Two bytes/
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

            builder.global_work_size((tile_dims.width / 2, tile_dims.height));
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

        Ok(OclImageState {
            buf: buf_minmax,
            dims: tile_dims,
            event: Some(evt_minmax),
        })
    }

    fn tile_blur(&self, tp: &mut TimeProfile, prev: OclImageState<Uchar2>) -> Result<OclImageState<Uchar2>, OclError> {
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
        Ok(OclImageState { buf: buf_minmax_blur, dims: prev.dims, event: Some(evt_blur) })
    }

    fn threshim(&self, config: &DetectorConfig, tp: &mut TimeProfile, prev: &OclImageState) -> Result<OclImageState, OclError> {
        let tiled_raw = self.tile_minmax(tp, &prev)?;

        fn split_image(me: &OpenCLDetector, src: &OclImageState<Uchar2>) -> Result<(ImageY8, ImageY8), std::io::Error> {
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

        let res = self.temp_buffer(&prev.dims, true)?;
        tp.stamp("threshim buffer");

        // Make threshold kernel
        let kernel = {
            let mut builder = OclKernel::builder();
            builder.program(&self.program);
            builder.queue(self.queue_kernel.clone());

            builder.name("k03_build_threshim");
            builder.arg(&prev.buf);
            builder.arg(prev.dims.stride as u32);
            builder.arg(prev.dims.width as u32);
            builder.arg(prev.dims.height as u32);
            builder.arg(tiled.buf);
            builder.arg(tiled.dims.stride);
            builder.arg(config.qtp.min_white_black_diff);
            builder.arg(&res);
            builder.arg(prev.dims.stride);

            builder.global_work_size((prev.dims.width, prev.dims.height));
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
            if let Some(prev_event) = &prev.event {
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

        if config.qtp.deglitch {
            tp.stamp("build_threshim");
            panic!("Deglitch not supported in OpenCL");
        }
        tp.stamp("threshim");

        Ok(OclImageState {
            buf: res,
            dims: prev.dims,
            event: Some(evt),
        })
    }

    pub(crate) fn preproces1s(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8<'_>) {
        
        
        let quad_im = self.quad_sigma(config, download_sigma, tp, decimated)
            .expect("Error in quad_sigma");
        #[cfg(feature="debug")]
        config.debug_image("debug_preprocess.pnm", |mut f| self.debug_image(&mut f, &quad_im));

        let threshim = self.threshim(config, tp, &quad_im)
            .expect("Error in threshold");
        #[cfg(feature="debug")]
        config.debug_image("debug_threshold.pnm", |mut f| self.debug_image(&mut f, &threshim));

        let quad_im_host = quad_im.read_image(&self.queue_read_debug)
            .expect("Unable to read quad_im");
        tp.stamp("read quad_im");
        let threshim_host = threshim.read_image(&self.queue_read_debug)
            .expect("Unable to read threshim");

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_src.as_entire_binding(),
                }
            ],
        });
        self.device.create_texture(desc)

        let mut command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.push_debug_group("Upload src");
        {
            // compute pass
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.particle_bind_groups[self.frame_num % 2], &[]);
            cpass.dispatch_workgroups(self.work_group_count, 1, 1);
        }
        command_encoder.pop_debug_group();
        let idx = self.queue.submit(command_encoder.finish());
        self.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx));
    }

    fn stages(&self) -> Vec<&dyn WStage> {
        let mut result = Vec::new();
        if let Some(qd) = self.quad_decimate.as_ref() {
            result.push(qd);
        }
        if let Some(qs) = self.quad_sigma.as_ref() {
            result.push(qs);
        }
        
        result
    }
    fn run_stage<S: WStage>(&self, current: &Option<S>, next: Option<&dyn WStage>, ctx: &mut WContext, image: WImage) -> WImage {
        
    }
    pub(crate) fn preprocess(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: &ImageRefY8) -> Result<(ImageY8, ImageY8), DetectError> {
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
        } else if config.quad_decimate_mode().is_enabled() {
            download_decimate = true;
        } else {
            download_src = true;
        }

        let gpu_src = self.upload_image(download_src, src_stride_alignment, &image);
        #[cfg(feature="debug")]
        config.debug_image("debug_src.pnm", |mut f| self.debug_image(&mut f, &gpu_src));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut ctx = WContext {
            tp,
            config,
            device: &self.device,
            encoder: &mut encoder,
            next_align: 0,
            next_read: true,
        };

        let stages = vec![self.quad_decimate, &mut self.quad_sigma];

        for stage in 

        ctx.next_read = download_decimate;
        let gpu_decimated = match self.quad_decimate {
            None => gpu_src,
            Some(qd) => qd.apply(&mut ctx, gpu_src)
                .expect("Error in quad_decimate"),
        };
        #[cfg(feature="debug")]
        config.debug_image("debug_decimate.pnm", |mut f| self.debug_image(&mut f, &gpu_decimated));

        let gpu_quad = match self.quad_sigma {
            None => gpu_decimated,
            Some(qd) => qd.apply(&mut ctx, download_decimate, gpu_decimated),
        };
        #[cfg(feature="debug")]
        config.debug_image("debug_preprocess.pnm", |mut f| self.debug_image(&mut f, &gpu_quad));

        let threshim = self.threshim(config, tp, &gpu_quad)
            .expect("Error in threshold");
        #[cfg(feature="debug")]
        config.debug_image("debug_threshold.pnm", |mut f| self.debug_image(&mut f, &threshim));

        let quad_im_host = gpu_quad.read_image(&self.queue_read_debug)
            .expect("Unable to read quad_im");
        tp.stamp("read quad_im");
        let threshim_host = threshim.read_image(&self.queue_read_debug)
            .expect("Unable to read threshim");
        tp.stamp("read threshim");
        Ok((quad_im_host, threshim_host))
    }
}

#[cfg(test)]
mod test {

}