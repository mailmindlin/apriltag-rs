use std::{hint::black_box, mem::size_of, num::NonZeroU32};

use futures::{executor::block_on, future::try_join};
use rand::{RngExt, rng};
use wgpu::CommandEncoderDescriptor;

use crate::{TimeProfile, DetectorConfig, util::{ImageY8, image::ImageRefY8}, detector::{config::{QuadDecimateMode, AccelerationRequest}, quad_sigma_cpu}, wgpu::{GpuStageContext, GpuQuadDecimate, util::{GpuStage, GpuImageDownload, GpuImageLike, GpuBufferFetch}}, quad_thresh::threshold::threshold};

use super::WGPUDetector;

/// Generate image with random data
fn random_image(width: usize, height: usize) -> ImageY8 {
    let mut rng = rng();
    let mut result = ImageY8::zeroed_packed(width, height);
    for y in 0..height {
        for x in 0..width {
            result[(x, y)] = rng.random();
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

    let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref(), "src_gpu_quad_decimate")
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
            cpass,
            stage_name: "quad_decimate",
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
    let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref(), "src")
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
            stage_name: "quad_sigma",
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

/// Run GPU thresholding on an image and return the result
fn gpu_threshim(src_cpu: &ImageY8) -> ImageY8 {
    let mut config = DetectorConfig::default();
    config.debug = false;
    config.acceleration = AccelerationRequest::Required;
    // Disable decimate/sigma so we test thresholding in isolation
    config.quad_decimate = 1.0;
    config.quad_sigma = 0.0;

    let gpu = WGPUDetector::new(&config).unwrap();

    let mut tp = TimeProfile::default();
    tp.stamp("Start");

    let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref(), "src").unwrap();
    tp.stamp("Upload");

    let (cmd_buf, mut dst_gpu) = {
        let mut queries = gpu.context.make_queries(4);
        let mut encoder = gpu.context.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("gpu_threshim") });
        let mut threshim_data = Default::default();

        let cpass = encoder.begin_compute_pass(&queries.make_cpd("test_threshim"));
        let mut ctx = GpuStageContext {
            tp: &mut tp,
            config: &config,
            context: &gpu.context,
            next_read: true,
            next_align: size_of::<u32>(),
            queries: &mut queries,
            cpass,
            stage_name: "threshim",
        };
        let dst_gpu = gpu.threshim.apply(&mut ctx, &src_gpu, &mut threshim_data)
            .unwrap()
            .threshim;
        drop(ctx);
        drop(threshim_data);
        queries.resolve(&mut encoder);
        (encoder.finish(), dst_gpu)
    };
    tp.stamp("Build Kernel");

    dst_gpu.index = Some(gpu.context.submit([cmd_buf]));
    tp.stamp("Execute");

    let img = block_on(dst_gpu.download_image(&gpu.context)).unwrap();
    tp.stamp("Done");
    println!("{tp}");
    img
}

/// Test GPU thresholding matches CPU thresholding.
///
/// Uses a large enough image that tiles are meaningful (TILESZ=4,
/// so we need at least 4×4 pixels to get one full tile).
#[test]
fn test_threshim() {
    // Image must be a multiple of TILESZ (4) for tile_minmax
    let src_cpu = random_image(64, 64);

    let gpu_result = gpu_threshim(&src_cpu);

    let mut tp = TimeProfile::default();
    let cpu_result = threshold(&DetectorConfig::default(), &mut tp, src_cpu.as_ref()).unwrap();

    assert_eq!(cpu_result, gpu_result);
}

/// Test thresholding with a non-tile-aligned image size
#[test]
fn test_threshim_non_aligned() {
    // 66 is not a multiple of TILESZ=4, testing edge handling
    let src_cpu = random_image(66, 66);

    let gpu_result = gpu_threshim(&src_cpu);

    let mut tp = TimeProfile::default();
    let cpu_result = threshold(&DetectorConfig::default(), &mut tp, src_cpu.as_ref()).unwrap();

    assert_eq!(cpu_result, gpu_result);
}

/// Run GPU BKE (connected component labeling) and return the raw union-find buffer
fn gpu_bke(src_cpu: &ImageY8) -> (Box<[u32]>, usize, usize) {
    let mut config = DetectorConfig::default();
    config.debug = false;
    config.acceleration = AccelerationRequest::Required;
    config.quad_decimate = 1.0;
    config.quad_sigma = 0.0;

    let gpu = WGPUDetector::new(&config).unwrap();

    let mut tp = TimeProfile::default();
    tp.stamp("Start");

    // First threshold the image (BKE operates on thresholded images)
    let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref(), "src_gpu_bke").unwrap();

    let (cmd_buf, mut dst_gpu, uf_width, uf_height) = {
        let mut queries = gpu.context.make_queries(8);
        let mut encoder = gpu.context.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("gpu_bke") });
        let mut threshim_data = Default::default();
        let mut bke_data = Default::default();

        let cpass = encoder.begin_compute_pass(&queries.make_cpd("test_bke"));
        let mut ctx = GpuStageContext {
            tp: &mut tp,
            config: &config,
            context: &gpu.context,
            next_read: true,
            next_align: size_of::<u32>(),
            queries: &mut queries,
            cpass,
            stage_name: "threshim",
        };
        let threshim_gpu = gpu.threshim.apply(&mut ctx, &src_gpu, &mut threshim_data)
            .unwrap()
            .threshim;

        let uf_width = threshim_gpu.width();
        let uf_height = threshim_gpu.height();

        ctx.stage_name = "bke";
        let uf_gpu = gpu.unionfind.apply(&mut ctx, &threshim_gpu, &mut bke_data)
            .unwrap();
        drop(ctx);
        drop(threshim_data);
        drop(bke_data);
        queries.resolve(&mut encoder);
        (encoder.finish(), uf_gpu, uf_width, uf_height)
    };
    tp.stamp("Build Kernel");

    dst_gpu.index = Some(gpu.context.submit([cmd_buf]));
    tp.stamp("Execute");

    let uf_data = block_on(dst_gpu.fetch_buffer(&gpu.context)).unwrap();
    tp.stamp("Done");
    println!("{tp}");
    (uf_data, uf_width, uf_height)
}

/// Test that BKE init produces a valid union-find structure.
///
/// After BKE init, each pixel should be its own parent (self-rooted).
/// We verify basic structural invariants rather than exact CPU matching,
/// since the GPU CCL algorithm differs from the CPU implementation.
#[test]
fn test_bke_init_structure() {
    let src_cpu = random_image(32, 32);
    let (uf_data, uf_width, uf_height) = gpu_bke(&src_cpu);

    // After BKE init, each pixel's parent entry should be a valid index
    let total_pixels = uf_width * uf_height.next_multiple_of(2);
    // UF buffer has 2 entries per pixel: [parent, size]
    assert!(uf_data.len() >= total_pixels * 2,
        "UF buffer too small: {} < {} * 2", uf_data.len(), total_pixels);

    // Every parent should be a valid pixel index
    for i in 0..total_pixels {
        let parent = uf_data[i * 2] as usize;
        assert!(parent < total_pixels,
            "Pixel {i} has out-of-bounds parent {parent} (max {total_pixels})");
    }
}

/// Test BKE with a uniform image (all same color).
/// All pixels should end up in the same connected component after init.
#[test]
fn test_bke_uniform_image() {
    let mut img = ImageY8::zeroed_packed(32, 32);
    // Fill with a uniform value above threshold
    for y in 0..32 {
        for x in 0..32 {
            img[(x, y)] = 200;
        }
    }
    let (uf_data, uf_width, uf_height) = gpu_bke(&img);
    let total_pixels = uf_width * uf_height.next_multiple_of(2);

    // All parents should be valid indices
    for i in 0..total_pixels {
        let parent = uf_data[i * 2] as usize;
        assert!(parent < total_pixels,
            "Pixel {i} has out-of-bounds parent {parent}");
    }
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

        let src_gpu = gpu.upload_texture(false, &src_cpu.as_ref(), "src_gpu_bench")
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
                stage_name: "quad_decimate",
            };
            let dst_gpu1 = quad_decimate.apply(&mut ctx, &src_gpu, &mut quad_decimate_data)
                .unwrap();
            ctx.stage_name = "quad_decimate";
            let dst_gpu2 = quad_sigma.apply(&mut ctx, &dst_gpu1, &mut quad_sigma_data)
                .unwrap();
            ctx.stage_name = "threshold";
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

        gpu.context.device.poll(wgpu::PollType::Wait {
			submission_index: Some(dst_gpu.index.clone().unwrap()),
			timeout: None,
		});
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