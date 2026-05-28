use std::{hint::black_box, mem::size_of, num::NonZeroU32};

use futures::{executor::block_on, future::{join, try_join}};
use rand::{thread_rng, Rng};
use wgpu::CommandEncoderDescriptor;

use crate::{TimeProfile, DetectorConfig, util::{ImageY8, image::ImageRefY8}, detector::{config::{QuadDecimateMode, AccelerationRequest}, quad_sigma_cpu}, wgpu::{GpuStageContext, GpuQuadDecimate, util::{GpuStage, GpuImageDownload}}, quad_thresh::threshold::threshold};

use super::WGPUDetector;

/// Generate image with random data
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

/// Generate sequential image
/// 
/// The value of each pixel is equivalent to its row
#[allow(unused)]
fn seq_image(width: usize, height: usize) -> ImageY8 {
    let mut result = ImageY8::zeroed_packed(width, height);
    for y in 0..height {
        for x in 0..width {
            // result[(x, y)] = (y * width + x) as _;
            result[(x, y)] = y as _;
        }
    }
    result
}

fn make_env() -> (WGPUDetector, TimeProfile) {
    let mut config = DetectorConfig::default();
    config.acceleration = AccelerationRequest::Required;
    let ocl = WGPUDetector::new(&config).unwrap();
    let tp = TimeProfile::default();
    (ocl, tp)
}

#[test]
fn test_create() {
    let mut config = DetectorConfig::default();
    config.acceleration = AccelerationRequest::Required;
    let detector = WGPUDetector::new(&config).unwrap();
    black_box(detector);
}

/// Test that we can upload and download an image without error
#[test]
fn test_upload_download() {
    let (gpu, _) = make_env();

    let img_cpu = random_image(3, 128);
    let img_gpu = gpu.upload_image(true, size_of::<u32>(), &img_cpu.as_ref());

    let img_cpu2 = block_on(img_gpu.download_image(&gpu.context)).unwrap();
    assert_eq!(img_cpu, img_cpu2);
}

async fn gpu_quad_decimate(quad_decimate: QuadDecimateMode, src_cpu: &ImageY8) -> ImageY8 {
    let mut config = DetectorConfig::default();
    config.debug = false;
    config.acceleration = AccelerationRequest::Required;
    config.quad_decimate = match quad_decimate {
        QuadDecimateMode::ThreeHalves => 1.5,
        QuadDecimateMode::Scaled(n) => n.get() as _,
    };
    let mut gpu = WGPUDetector::new_async(&config).await.unwrap();
    if gpu.quad_decimate.is_none() && quad_decimate == QuadDecimateMode::Scaled(NonZeroU32::new(1).unwrap()) {
        gpu.quad_decimate = Some(GpuQuadDecimate::new_factor(&gpu.context, NonZeroU32::new(1).unwrap()).await.unwrap());
    }

    let mut tp = TimeProfile::default();
    tp.stamp("Start");
    println!("A");

    let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref())
        .unwrap();
    tp.stamp("Upload");
    println!("B");

    // Decimate on GPU
    let quad_decimate = gpu.quad_decimate.as_ref().unwrap();
    // let dims = quad_decimate.result_dims(src_cpu.dimensions());
    let (cmd_buf, mut dst_gpu) = {
        let mut queries = gpu.context.make_queries(2);
        let mut encoder = gpu.context.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("gpu_quad_decimate") });
        let cpass = encoder.begin_compute_pass(&queries.make_cpd("test_quad_decimate"));

        let mut quad_decimate_data = Default::default();

        let mut ctx = GpuStageContext {
            tp: &mut tp,
            config: &config,
            context: &gpu.context,
            next_read: true,
            next_align: size_of::<u32>(),
            queries: &mut queries,
            cpass: cpass,
            stage_name: Some("quad_decimate"),
        };
        ctx.tp.stamp("Kernel0");
        let dst_gpu = quad_decimate.apply(&mut ctx, &src_gpu, &mut quad_decimate_data)
            .unwrap();
        ctx.tp.stamp("Kernel1");
        drop(ctx);
        drop(quad_decimate_data);
        queries.resolve(&mut encoder);
        (encoder.finish(), dst_gpu)
    };
    tp.stamp("Build Kernel");
    
    dst_gpu.index = Some(gpu.context.submit([cmd_buf]));
    
    tp.stamp("Execute");

    // gpu.context.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(dst_gpu.index.clone().unwrap()));
    tp.stamp("Poll");

    let img_cpu2 = dst_gpu.download_image(&gpu.context)
        .await
        .unwrap();
    tp.stamp("Done");
    println!("{tp}");
    img_cpu2
}

/// Test GPU quad_decimate with ffactor=1.5 (special case)
#[test]
fn test_quad_decimate_identity() {
    let src_cpu = random_image(8, 8);

    // Decimate on GPU
    let dst_gpu = gpu_quad_decimate(QuadDecimateMode::Scaled(NonZeroU32::new(1).unwrap()), &src_cpu);
    let dst_gpu = block_on(dst_gpu);

    // Decimate on CPU
    let dst_cpu = src_cpu;

    assert_eq!(dst_cpu, dst_gpu);
}

/// Test GPU quad_decimate with ffactor=1.5 (special case)
#[test]
fn test_quad_decimate_32() {
    let src_cpu = random_image(16, 16);

    // Decimate on GPU
    let dst_gpu = gpu_quad_decimate(QuadDecimateMode::ThreeHalves, &src_cpu);
    let dst_gpu = block_on(dst_gpu);

    // Decimate on CPU
    let dst_cpu = src_cpu.decimate_three_halves();

    println!("{dst_gpu:?}");
    assert_eq!(dst_cpu, dst_gpu);
}

/// Test GPU quad_decimate with ffactor=2.0
#[test]
fn test_quad_decimate_f2() {
    let src_cpu = random_image(128, 128);

    // Decimate on GPU
    let dst_gpu = gpu_quad_decimate(QuadDecimateMode::Scaled(NonZeroU32::new(2).unwrap()), &src_cpu);
    let dst_gpu = block_on(dst_gpu);

    // Decimate on CPU
    let dst_cpu = src_cpu.decimate(2);

    assert_eq!(dst_cpu, dst_gpu);
}

/// Test GPU quad_decimate with ffactor=3.0
#[test]
fn test_quad_decimate_f3() {
    let src_cpu = random_image(128, 128);

    // Decimate on GPU
    let dst_gpu = gpu_quad_decimate(QuadDecimateMode::Scaled(NonZeroU32::new(3).unwrap()), &src_cpu);
    let dst_gpu = block_on(dst_gpu);

    // Decimate on CPU
    let dst_cpu = src_cpu.decimate(3);

    assert_eq!(dst_cpu, dst_gpu);
}

fn gpu_quad_sigma(quad_sigma: f32, src_cpu: ImageRefY8) -> ImageY8 {
    let mut config = DetectorConfig::default();
    config.debug = false;
    config.quad_sigma = quad_sigma;
    config.acceleration = AccelerationRequest::Required;
    assert!(config.do_quad_sigma());

    let gpu = WGPUDetector::new(&config).unwrap();

    let mut tp = TimeProfile::default();
    // let src_gpu_align = gpu.quad_sigma.as_ref().unwrap().src_alignment();
    let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref())
        .unwrap();

    // Decimate on GPU
    let quad_sigma = gpu.quad_sigma.as_ref().unwrap();
    // let dims = quad_decimate.result_dims(src_cpu.dimensions());
    let (cmd_buf, mut dst_gpu, queries) = {
        let mut encoder = gpu.context.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("gpu_quad_sigma") });
        let mut quad_sigma_data = Default::default();

        let mut queries = gpu.context.make_queries(2);
        let cpass = encoder.begin_compute_pass(&queries.make_cpd("test_quad_sigma"));
        let mut ctx = GpuStageContext {
            tp: &mut tp,
            config: &config,
            context: &gpu.context,
            next_read: true,
            next_align: size_of::<u32>(),
            cpass,
            queries: &mut queries,
            stage_name: Some("quad_sigma"),
        };
        ctx.tp.stamp("Kernel0");
        let dst_gpu = quad_sigma.apply(&mut ctx, &src_gpu, &mut quad_sigma_data)
            .unwrap();
        ctx.tp.stamp("Kernel1");
        drop(ctx);
        tp.stamp("Kernel2");
        queries.resolve(&mut encoder);
        tp.stamp("Kernel3");
        let cmd_buf = encoder.finish();
        (cmd_buf, dst_gpu, queries)
    };
    tp.stamp("Build Kernel");
    
    dst_gpu.index = Some(gpu.context.submit([cmd_buf]));
    
    tp.stamp("Execute");

    // gpu.context.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(dst_gpu.index.clone().unwrap()));
    tp.stamp("Poll");

    let (img_cpu2, tp_gpu) = block_on(
        try_join(
            dst_gpu.download_image(&gpu.context),
            queries.wait_for_results(&gpu.context)
        )
    ).unwrap();
    tp.stamp("Done");
    if let Some(tp_gpu) = tp_gpu {
        println!("tp_gpu:\n{tp_gpu}");
    }
    println!("{tp}");
    img_cpu2
}

#[test]
fn test_quad_sigma_blur() {
    let img_cpu = seq_image(16, 16);

    // Blur on GPU
    let qs_gpu = gpu_quad_sigma(0.8, img_cpu.as_ref());

    // Blur on CPU
    let qs_cpu = {
        let mut img_cpu1 = img_cpu.clone();
        // img_cpu1.gaussian_blur(config.quad_sigma as _, ksz);
        quad_sigma_cpu(&mut img_cpu1, 0.8);
        img_cpu1
    };

    assert_eq!(qs_cpu, qs_gpu);
}

#[test]
#[ignore = "Who even uses sharpening?"]
fn test_quad_sigma_sharp() {
    let img_cpu = random_image(16, 16);

    // Sharp on GPU
    let qs_gpu = gpu_quad_sigma(-0.8, img_cpu.as_ref());

    // Sharp on CPU
    let qs_cpu = {
        let mut img_cpu1 = img_cpu.clone();
        // img_cpu1.gaussian_blur(config.quad_sigma as _, ksz);
        quad_sigma_cpu(&mut img_cpu1, -0.8);
        img_cpu1
    };

    assert_eq!(qs_cpu, qs_gpu);
}

#[test]
#[ignore = "reason"]
fn bench_both() {
    // let src_cpu = random_image(1280, 720);
    // let src_cpu = random_image(1900, 1080);
    // let src_cpu = random_image(1900, 1200);
    let src_cpu = random_image(2048, 2048);

    let mut config = DetectorConfig::default();
    config.acceleration = AccelerationRequest::Required;
    config.debug = false;
    config.quad_decimate = 1.5;
    config.quad_sigma = 0.8;
    let gpu = WGPUDetector::new(&config).unwrap();

    
    let mut tp = TimeProfile::default();
    for _ in 0..100 {
        tp = TimeProfile::default();
        tp.stamp("Start");

        let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref())
            .unwrap();
        tp.stamp("Upload");

        // Decimate on GPU
        let quad_decimate = gpu.quad_decimate.as_ref().unwrap();
        let quad_sigma = gpu.quad_sigma.as_ref().unwrap();
        // let dims = quad_decimate.result_dims(src_cpu.dimensions());
        let (cmd_buf, mut dst_gpu) = {
            let mut queries = gpu.context.make_queries(32);
            let mut encoder = gpu.context.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("gpu_quad_decimate") });
            let mut quad_decimate_data = Default::default();
            let mut quad_sigma_data = Default::default();
            let mut threshim_data = Default::default();

            let cpass = encoder.begin_compute_pass(&queries.make_cpd("test_both"));
            let mut ctx = GpuStageContext {
                tp: &mut tp,
                config: &config,
                context: &gpu.context,
                next_read: true,
                next_align: size_of::<u32>(),
                queries: &mut queries,
                cpass,
                stage_name: Some("quad_decimate"),
            };
            let dst_gpu1 = quad_decimate.apply(&mut ctx, &src_gpu, &mut quad_decimate_data)
                .unwrap();
            ctx.stage_name = Some("quad_decimate");
            let dst_gpu2 = quad_sigma.apply(&mut ctx, &dst_gpu1, &mut quad_sigma_data)
                .unwrap();
            ctx.stage_name = Some("threshold");
            let dst_gpu3 = gpu.threshim.apply(&mut ctx, &dst_gpu2, &mut threshim_data)
                .unwrap()
                .threshim;
            drop(ctx);
            drop(threshim_data);
            drop(quad_decimate_data);
            drop(quad_sigma_data);

            queries.resolve(&mut encoder);
            (encoder.finish(), dst_gpu3)
        };
        tp.stamp("Build Kernel");
        
        dst_gpu.index = Some(gpu.context.submit([cmd_buf]));
        
        tp.stamp("Execute");

        gpu.context.device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(dst_gpu.index.clone().unwrap()));
        tp.stamp("Poll");

        let img_cpu2 = block_on(dst_gpu.download_image(&gpu.context))
            .unwrap();
        tp.stamp("Done");
        black_box(img_cpu2);
    }

    let mut tp_cpu = TimeProfile::default();

    for _ in 0..100 {
        tp_cpu = TimeProfile::default();
        let img_cpu3 = {
            tp_cpu.stamp("Start");
            let src = src_cpu.clone();
            tp_cpu.stamp("Clone");
            let mut decimated = match config.quad_decimate_mode() {
                Some(QuadDecimateMode::Scaled(factor)) => src.decimate(factor.get() as _),
                Some(QuadDecimateMode::ThreeHalves) => src.decimate_three_halves(),
                None => src,
            };
            tp_cpu.stamp("quad_decimate");
            let sigma = if config.do_quad_sigma() {
                quad_sigma_cpu(&mut decimated, config.quad_sigma);
                decimated
            } else {
                decimated
            };
            tp_cpu.stamp("quad_sigma");
            let threshim = threshold(&config, &mut tp_cpu, sigma.as_ref()).unwrap();
            tp_cpu.stamp("Done");
            threshim
        };
        black_box(img_cpu3);
    }
    println!("GPU:\n{tp}");
    println!("CPU:\n{tp_cpu}");
}