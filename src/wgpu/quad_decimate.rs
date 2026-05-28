use std::{borrow::Cow, mem::size_of};

use wgpu::{ComputePipelineDescriptor, BindGroupEntry};

use crate::{detector::config::QuadDecimateMode, util::multiple::lcm};

use super::{WImage, WContext, WStage};

const PROG_QUAD_DECIMATE: &str = include_str!("./01_quad_decimate.wgsl");

pub(super) struct WQuadDecimate {
    mode: QuadDecimateMode,
    bg_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
}

const SRC_ALIGN_32: usize = size_of::<u32>() * 3;
const SRC_ALIGN_SC: usize = size_of::<u32>();
const DST_ALIGN_32: usize = size_of::<u32>() * 2;
const DST_ALIGN_SC: usize = size_of::<u32>();

impl WQuadDecimate {
    pub(super) fn new(device: &wgpu::Device, mode: QuadDecimateMode) -> Self {
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("01_quad_decimate"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROG_QUAD_DECIMATE)),
        });

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl:quad_decimate"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let entry_point = match mode {
            QuadDecimateMode::Scaled(_) => "k01_filter_quad_decimate",
            QuadDecimateMode::ThreeHalves => "k01_filter_quad_decimate_32",
        };

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("cp:quad_decimate"),
            layout: None,
            module: &cs_module,
            entry_point,
        });

        Self {
            mode,
            compute_pipeline,
            bg_layout,
        }
    }
}

impl WStage for WQuadDecimate {
    fn src_alignment(&self) -> usize {
        match self.mode {
            QuadDecimateMode::ThreeHalves => SRC_ALIGN_32,
            QuadDecimateMode::Scaled(factor) => size_of::<u32>() * (factor.get() as usize),
        }
    }

    fn debug_name() -> &'static str {
        "quad_decimate"
    }

    fn apply(&self, ctx: &mut WContext<'_>, src: WImage) -> Result<WImage, super::WGPUError> {
        let (swidth, sheight, factor) = match self.mode {
            QuadDecimateMode::ThreeHalves => {
                let swidth = src.dims.width / 3 * 2;
                let sheight = src.dims.height / 3 * 2;
                assert_eq!(swidth % 2, 0, "Input dimension must be multiple of two");
                assert_eq!(sheight % 2, 0, "Input dimension must be multiple of two");
                (swidth, sheight, 1)
            },
            QuadDecimateMode::Scaled(factor) => {
                let factor = factor.get() as usize;
                let swidth = 1 + (src.dims.width - 1) / factor;
                let sheight = 1 + (src.dims.height - 1) / factor;
                // println!("dims = {}x{}, sdims ={swidth}x{sheight} factor={factor}", prev.dims.width, prev.dims.height);
                assert!((swidth - 1) * factor <= src.dims.width);
                assert!((sheight - 1) * factor <= src.dims.height);
                (swidth, sheight, factor)
            },
        };

        let sstride = swidth.next_multiple_of(16);
        
        #[cfg(feature="debug")]
        ctx.encoder.push_debug_group("quad_decimate");
        let dst = ctx.temp_buffer(swidth, sheight, DST_ALIGN_32, false)?;
        ctx.tp.stamp("quad_decimate buffer");

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quad_decimate bindgroup"),
            layout: &self.bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: src.buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dst.buffer.as_entire_binding(),
                }
            ],
        });

        let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("quad_decimate:cp") });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_push_constants(0, bytemuck::cast_slice(&[
            // Constant 0: stride_src
            (src.dims.stride / size_of::<u32>()) as u32,
            // Constant 1: stride_dst
            (dst.dims.stride / size_of::<u32>()) as u32,
        ]));

        cpass.set_bind_group(0, &bind_group, &[]);
        match self.mode {
            QuadDecimateMode::ThreeHalves => {
                cpass.dispatch_workgroups((swidth / DST_ALIGN_32) as u32, (sheight / DST_ALIGN_32) as u32, 1);
            },
            QuadDecimateMode::Scaled(_) => {
                cpass.dispatch_workgroups((swidth / size_of::<u32>()) as u32, (sheight / size_of::<u32>()) as u32, 1);
            },
        }
        #[cfg(feature="debug")]
        if ctx.config.debug() {
            cpass.insert_debug_marker("quad_decimate");
            ctx.encoder.pop_debug_group();
        }
        ctx.tp.stamp("quad_decimate");

        Ok(dst)
    }
}