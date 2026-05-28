use apriltag_rs::{AprilTagDetector, AprilTagFamily, AccelerationRequest, ImageY8};

pub fn make_detector(threads: usize) -> AprilTagDetector {
    let mut builder = AprilTagDetector::builder();
    builder.set_gpu_mode(AccelerationRequest::Disabled);

    let family = AprilTagFamily::for_name("tag36h11")
        .expect("tag36h11 family not available (enable 'tag36h11' feature)");
    builder.add_family_bits(family, 1).unwrap();

    builder.config.quad_decimate = 2.0;
    builder.config.quad_sigma = 0.0;
    builder.config.nthreads = threads;
    builder.config.refine_edges = true;
    builder.config.debug = false;

    builder.build().expect("Error building detector")
}

#[cfg(feature = "wgpu")]
pub fn make_gpu_detector() -> AprilTagDetector {
    let mut builder = AprilTagDetector::builder();
    builder.set_gpu_mode(AccelerationRequest::Required);

    let family = AprilTagFamily::for_name("tag36h11")
        .expect("tag36h11 family not available");
    builder.add_family_bits(family, 1).unwrap();

    builder.config.quad_decimate = 2.0;
    builder.config.quad_sigma = 0.0;
    builder.config.nthreads = 1;
    builder.config.refine_edges = true;
    builder.config.debug = false;

    builder.build().expect("Error building GPU detector")
}

#[cfg(feature = "opencl")]
pub fn make_opencl_detector() -> AprilTagDetector {
    let mut builder = AprilTagDetector::builder();
    builder.set_gpu_mode(AccelerationRequest::Required);

    let family = AprilTagFamily::for_name("tag36h11")
        .expect("tag36h11 family not available");
    builder.add_family_bits(family, 1).unwrap();

    builder.config.quad_decimate = 2.0;
    builder.config.quad_sigma = 0.0;
    builder.config.nthreads = 1;
    builder.config.refine_edges = true;
    builder.config.debug = false;

    builder.build().expect("Error building OpenCL detector")
}

pub fn load_image(path: &str) -> ImageY8 {
    use image::ImageReader;
    let reader = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {path}: {e}"));
    let image = reader.decode()
        .unwrap_or_else(|e| panic!("Failed to decode image {path}: {e}"))
        .into_luma8();

    let mut result = ImageY8::zeroed(image.width() as usize, image.height() as usize);
    for (x, y, value) in image.enumerate_pixels() {
        result[(x as usize, y as usize)] = value.0[0];
    }
    result
}

pub fn checkerboard_image(width: usize, height: usize, cell_size: usize) -> ImageY8 {
    let mut img = ImageY8::zeroed(width, height);
    for y in 0..height {
        for x in 0..width {
            let is_white = ((x / cell_size) + (y / cell_size)) % 2 == 0;
            img[(x, y)] = if is_white { 255 } else { 0 };
        }
    }
    img
}
