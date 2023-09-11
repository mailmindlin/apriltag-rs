#![cfg(test)]
use std::{hint::black_box, num::NonZeroU32};

use ocl::{Event, prm::Uchar2};
use rand::{thread_rng, Rng};

use crate::{util::{ImageY8, ImageBuffer}, DetectorConfig, TimeProfile, detector::{config::QuadDecimateMode, quad_sigma_cpu}, quad_thresh::threshold::{tile_minmax_cpu, TILESZ, tile_blur_cpu}, ocl::{OclStage, buffer::OclAwaitable}};

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

fn make_env() -> (OpenCLDetector, TimeProfile) {
    let config = DetectorConfig::default();
    let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
    let tp = TimeProfile::default();
    (ocl, tp)
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
    let (ocl, mut tp) = make_env();

    let img_cpu = random_image(128, 128);
    let img_gpu = ocl.core.upload_image(true, &mut tp, img_cpu.as_ref())
        .unwrap();

    let img_cpu2 = ocl.core.download_image(&img_gpu.buffer)
        .unwrap();
    assert_eq!(img_cpu, img_cpu2);
}

fn gpu_quad_decimate(quad_decimate: QuadDecimateMode, src_cpu: &ImageY8) -> ImageY8 {
    let mut config = DetectorConfig::default();
    config.debug = false;
    config.quad_decimate = match quad_decimate {
        QuadDecimateMode::ThreeHalves => 1.5,
        QuadDecimateMode::Scaled(n) => n.get() as _,
    };
    assert_eq!(config.quad_decimate_mode(), Some(quad_decimate));
    let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
    let mut tp = TimeProfile::default();

    let src_gpu = ocl.core.upload_image(false, &mut tp, src_cpu.as_ref())
        .unwrap();

    // Decimate on GPU
    let quad_decimate = ocl.quad_decimate.as_ref().unwrap();
    let dims = quad_decimate.result_dims(src_cpu.dimensions());
    let mut dst_gpu = ocl.core.temp_bufferstate(&dims, true)
        .unwrap();
    let kernel = quad_decimate.make_kernel(&ocl.core, &src_gpu.buffer, &dst_gpu)
        .unwrap();

    dst_gpu.event = Some(Event::empty());
    let kcmd = kernel
        .cmd()
        .ewait(src_gpu.event())
        .enew(dst_gpu.event.as_mut());
    unsafe { kcmd.enq() }.unwrap();
    drop(kernel);
    drop(src_gpu);

    let img_cpu2 = ocl.core.download_image(&dst_gpu)
        .unwrap();
    img_cpu2
}

/// Test GPU quad_decimate with ffactor=1.5 (special case)
#[test]
fn test_quad_decimate_32() {
    let src_cpu = random_image(128, 128);

    // Decimate on GPU
    let dst_gpu = gpu_quad_decimate(QuadDecimateMode::ThreeHalves, &src_cpu);

    // Decimate on CPU
    let dst_cpu = src_cpu.decimate_three_halves();

    assert_eq!(dst_cpu, dst_gpu);
}

/// Test GPU quad_decimate with ffactor=3.0
#[test]
fn test_quad_decimate_3() {
    let src_cpu = random_image(128, 128);

    // Decimate on GPU
    let dst_gpu = gpu_quad_decimate(QuadDecimateMode::Scaled(NonZeroU32::new(3).unwrap()), &src_cpu);

    // Decimate on CPU
    let dst_cpu = src_cpu.decimate(3);

    assert_eq!(dst_cpu, dst_gpu);
}

// #[test]
// fn test_quad_sigma_blur() {
//     let mut config = DetectorConfig::default();
//     config.debug = false;
//     config.quad_sigma = 0.8;
//     assert!(config.do_quad_sigma());

//     let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
//     assert!(ocl.quad_sigma_filter.is_some());
//     let mut tp = TimeProfile::default();
//     // let img_cpu = random_image(4, 4);
//     let img_cpu = random_image(128, 128);
//     let img_gpu = ocl.core.upload_image(false, &mut tp, img_cpu.as_ref())
//         .unwrap();

//     // Blur on GPU
//     let img_gpu = ocl.quad_sigma(&config, true, &mut tp, img_gpu)
//         .unwrap();

//     // Blur on CPU
//     let img_cpu1 = {
//         let mut img_cpu1 = img_cpu.clone();
//         // img_cpu1.gaussian_blur(config.quad_sigma as _, ksz);
//         quad_sigma_cpu(&mut img_cpu1, config.quad_sigma);
//         img_cpu1
//     };

//     // Download GPU image
//     let img_cpu2 = ocl.core.download_image(&img_gpu)
//         .unwrap();
//     assert_eq!(img_cpu1, img_cpu2);
// }

// #[test]
// #[ignore=""]
// fn test_quad_sigma_sharp() {
//     let mut config = DetectorConfig::default();
//     config.debug = false;
//     config.quad_sigma = -0.8;
//     assert!(config.do_quad_sigma());

//     let ocl = OpenCLDetector::new(&config, crate::OpenClMode::Required).unwrap();
//     assert!(ocl.quad_sigma_filter.is_some());
//     let mut tp = TimeProfile::default();
//     let img_cpu = random_image(128, 128);
//     let img_gpu = ocl.core.upload_image(false, &mut tp, img_cpu.as_ref())
//         .unwrap();

//     // Sharpen on GPU
//     let img_gpu = ocl.quad_sigma(&config, true, &mut tp, img_gpu)
//         .unwrap();

//     // Sharpen on CPU
//     let img_cpu1 = {
//         let mut img_cpu1 = img_cpu.clone();
//         // img_cpu1.gaussian_blur(config.quad_sigma as _, ksz);
//         quad_sigma_cpu(&mut img_cpu1, config.quad_sigma);
//         img_cpu1
//     };

//     // Download GPU image
//     let img_cpu2 = ocl.core.download_image(&img_gpu)
//         .unwrap();
//     assert_eq!(img_cpu1, img_cpu2);
// }

#[test]
fn test_tile_minmax() {
    let (ocl, mut tp) = make_env();

    let src_cpu = random_image(128, 128);
    let src_gpu = ocl.core.upload_image(false, &mut tp, src_cpu.as_ref())
        .unwrap();

    // Tile on GPU
    let dims = ocl.tile_minmax.result_dims(src_cpu.dimensions());
    let mut dst_gpu = ocl.core.temp_bufferstate::<Uchar2>(&dims, true)
        .unwrap();
    {
        let kernel = ocl.tile_minmax.make_kernel(&ocl.core, &src_gpu.buffer, &dst_gpu)
            .unwrap();
        dst_gpu.event = Some(Event::empty());
        let kcmd = kernel.cmd()
            .ewait(src_gpu.event())
            .enew(dst_gpu.event.as_mut());
        unsafe { kcmd.enq() }.unwrap();
    }
    
    // Tile on CPU
    let img_cpu1 = tile_minmax_cpu::<TILESZ>(src_cpu.as_ref());

    // Download GPU image
    let dst_gpu = {
        let buf_cpu2 = ocl.core.fetch_ocl_buffer(&dst_gpu)
            .unwrap();
        let mut dst = ImageBuffer::<[u8; 2]>::zeroed_with_stride(dst_gpu.width(), dst_gpu.height(), dst_gpu.stride());
        for i in 0..buf_cpu2.len() {
            dst.data[i * 2 + 0] = buf_cpu2[i][0];
            dst.data[i * 2 + 1] = buf_cpu2[i][1];
        }
        dst
    };
    for (i, row) in src_cpu.rows() {
        // let mut rd = Vec::new();
        for pix in row.as_slice().chunks(1) {
            print!("{:>02x} ", pix[0]);
        }
        println!();
        if i % 4 == 3 {
            println!();
        }
    }
    println!();
    for (_, row) in dst_gpu.rows() {
        // let mut rd = Vec::new();
        for pix in row.as_slice().chunks(2) {
            print!("{:>02x} {:>02x}  ", pix[0], pix[1]);
        }
        println!();
    }
    println!();
    for (_, row) in img_cpu1.rows() {
        // let mut rd = Vec::new();
        for pix in row.as_slice().chunks(2) {
            print!("{:>02x} {:>02x}  ", pix[0], pix[1]);
        }
        println!();
    }
    assert_eq!(img_cpu1, dst_gpu);
}

#[test]
fn test_tile_blur() {
    let (ocl, mut tp) = make_env();

    let src_cpu = random_image(128, 128);
    let src_gpu = ocl.core.upload_image(false, &mut tp, src_cpu.as_ref())
        .unwrap();

    // Tile on GPU
    let tm_dims = ocl.tile_minmax.result_dims(src_cpu.dimensions());
    let mut tm_gpu = ocl.core.temp_bufferstate::<Uchar2>(&tm_dims, true)
        .unwrap();
    {
        let kernel = ocl.tile_minmax.make_kernel(&ocl.core, &src_gpu.buffer, &tm_gpu)
            .unwrap();
        tm_gpu.event = Some(Event::empty());
        let kcmd = kernel.cmd()
            .ewait(src_gpu.event())
            .enew(tm_gpu.event.as_mut());
        unsafe { kcmd.enq() }.unwrap();
    }

    let tb_dims = ocl.tile_blur.result_dims(&tm_dims);
    let mut dst_gpu = ocl.core.temp_bufferstate::<Uchar2>(&tb_dims, true)
        .unwrap();
    {
        let kernel = ocl.tile_blur.make_kernel(&ocl.core, &tm_gpu, &dst_gpu)
            .unwrap();
        dst_gpu.event = Some(Event::empty());
        let kcmd = kernel.cmd()
            .ewait(tm_gpu.event())
            .enew(dst_gpu.event.as_mut());
        unsafe { kcmd.enq() }.unwrap();
    }

    // Tile on CPU
    let tile_cpu = tile_minmax_cpu::<TILESZ>(src_cpu.as_ref());
    let blur_cpu = tile_blur_cpu(tile_cpu);
    assert_eq!(blur_cpu.dimensions(), &dst_gpu.dims);

    // Download GPU image
    let img_cpu2 = {
        let buf_cpu2 = ocl.core.fetch_ocl_buffer(&dst_gpu)
            .unwrap();
        let mut dst = ImageBuffer::<[u8; 2]>::zeroed_with_stride(dst_gpu.width(), dst_gpu.height(), dst_gpu.stride());
        for i in 0..buf_cpu2.len() {
            dst.data[i * 2 + 0] = buf_cpu2[i][0];
            dst.data[i * 2 + 1] = buf_cpu2[i][1];
        }
        dst
    };
    assert_eq!(blur_cpu, img_cpu2);
}
/*
#[test]
fn test_uf_init() {
    let (ocl, mut tp) = make_env();

    let uf = ocl.unionfind_init(&mut tp, &crate::util::image::ImageDimensions { width: 4, height: 4, stride: 4 }).unwrap();
    let uf = ocl.unionfind_test(&mut tp, uf).unwrap();
    let uf_cpu = ocl.fetch_ocl_buffer(&uf).unwrap();
    for row in uf_cpu.chunks(uf.width()) {
        // let mut rd = Vec::new();
        for elem in row.chunks(2) {
            print!("({:>2}, {:>2}) ", elem[0][0], elem[1][0]);
        }
        
        println!();
    }
    panic!();
}*/