mod common;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};

use common::{make_detector, load_image, checkerboard_image};

// --- Image Processing Benchmarks (public APIs) ---

fn bench_image_processing(c: &mut Criterion) {
    let tags_img = load_image("examples/tags.png");
    let pixels = (tags_img.width() * tags_img.height()) as u64;

    let mut group = c.benchmark_group("image_processing");
    group.throughput(Throughput::Elements(pixels));

    group.bench_function("decimate_2x", |b| {
        b.iter(|| black_box(&tags_img).decimate(2))
    });

    group.bench_function("decimate_three_halves", |b| {
        b.iter(|| black_box(&tags_img).decimate_three_halves())
    });

    group.bench_function("gaussian_blur", |b| {
        b.iter_batched(
            || tags_img.clone(),
            |mut img| img.gaussian_blur(1.0, 3),
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// --- Pipeline Stage Benchmarks (via TimeProfile) ---

/// CPU pipeline stage names in execution order.
const CPU_STAGES: &[&str] = &[
    "decimate",
    "blur/sharp",
    "tile_minmax",
    "blur",
	"build_threshim",
    "threshold",
    "unionfind",
    "make clusters",
    "fit quads to clusters",
    "quads",
    "decode+refinement",
    "reconcile",
];

fn bench_pipeline_stages(c: &mut Criterion) {
    let detector = make_detector(1);
    let tags_img = load_image("examples/tags.png");

    let mut group = c.benchmark_group("pipeline_stages");

    // First, do a test run to discover which stages are present
    let test_result = detector.detect(&tags_img).unwrap();
    let available_stages = test_result.tp.stage_names();
    eprintln!("Available stages: {:?}", available_stages);

    for &stage in CPU_STAGES {
        // Only benchmark stages that actually appear in the profile
        if !available_stages.iter().any(|s| *s == stage) {
            continue;
        }

        group.bench_function(stage, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let result = detector.detect(black_box(&tags_img)).unwrap();
                    total += result.tp.stage_duration(stage).unwrap_or_default();
                }
                total
            });
        });
    }

    group.finish();
}

fn bench_pipeline_stages_multithreaded(c: &mut Criterion) {
    let detector = make_detector(4);
    let tags_img = load_image("examples/tags.png");

    let test_result = detector.detect(&tags_img).unwrap();
    let available_stages = test_result.tp.stage_names();

    let mut group = c.benchmark_group("pipeline_stages_4threads");

    for &stage in CPU_STAGES {
        if !available_stages.iter().any(|s| *s == stage) {
            continue;
        }

        group.bench_function(stage, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let result = detector.detect(black_box(&tags_img)).unwrap();
                    total += result.tp.stage_duration(stage).unwrap_or_default();
                }
                total
            });
        });
    }

    group.finish();
}

// --- Full Pipeline Benchmarks ---

fn bench_full_pipeline(c: &mut Criterion) {
    let tags_img = load_image("examples/tags.png");
    let airport_img = load_image("examples/airport.png");
    let checkerboard = checkerboard_image(640, 480, 16);

    let mut group = c.benchmark_group("full_pipeline");

    for threads in [1, 4] {
        let detector = make_detector(threads);

        group.bench_with_input(
            BenchmarkId::new("tags_png", format!("{threads}t")),
            &tags_img,
            |b, img| {
                b.iter(|| detector.detect(black_box(img)).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("airport_png", format!("{threads}t")),
            &airport_img,
            |b, img| {
                b.iter(|| detector.detect(black_box(img)).unwrap())
            },
        );
    }

    // Synthetic stress test (many edges, no real tags)
    let detector_1t = make_detector(1);
    group.bench_function("checkerboard_640x480", |b| {
        b.iter(|| detector_1t.detect(black_box(&checkerboard)).unwrap())
    });

    group.finish();
}

// --- GPU Benchmarks (WGPU) ---

#[cfg(feature = "wgpu")]
fn bench_gpu_pipeline(c: &mut Criterion) {
    let detector = match std::panic::catch_unwind(|| common::make_gpu_detector()) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping GPU benchmarks: failed to create GPU detector");
            return;
        }
    };
    let tags_img = load_image("examples/tags.png");

    let mut group = c.benchmark_group("full_pipeline_gpu");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("tags_png", |b| {
        b.iter(|| detector.detect(black_box(&tags_img)).unwrap())
    });

    group.finish();
}

#[cfg(feature = "wgpu")]
fn bench_gpu_stages(c: &mut Criterion) {
    let detector = match std::panic::catch_unwind(|| common::make_gpu_detector()) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping GPU stage benchmarks: failed to create GPU detector");
            return;
        }
    };
    let tags_img = load_image("examples/tags.png");

    let test_result = detector.detect(&tags_img).unwrap();
    let available_stages = test_result.tp.stage_names();
    eprintln!("GPU stages: {:?}", available_stages);

    let mut group = c.benchmark_group("pipeline_stages_gpu");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(10));

    for stage in &available_stages {
        let stage = stage.to_string();
        group.bench_function(&stage, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let result = detector.detect(black_box(&tags_img)).unwrap();
                    total += result.tp.stage_duration(&stage).unwrap_or_default();
                }
                total
            });
        });
    }

    group.finish();
}

// --- GPU Benchmarks (OpenCL) ---

#[cfg(feature = "opencl")]
fn bench_opencl_pipeline(c: &mut Criterion) {
    let detector = match std::panic::catch_unwind(|| common::make_opencl_detector()) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping OpenCL benchmarks: failed to create OpenCL detector");
            return;
        }
    };
    let tags_img = load_image("examples/tags.png");

    let mut group = c.benchmark_group("full_pipeline_opencl");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("tags_png", |b| {
        b.iter(|| detector.detect(black_box(&tags_img)).unwrap())
    });

    group.finish();
}

#[cfg(feature = "opencl")]
fn bench_opencl_stages(c: &mut Criterion) {
    let detector = match std::panic::catch_unwind(|| common::make_opencl_detector()) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping OpenCL stage benchmarks: failed to create OpenCL detector");
            return;
        }
    };
    let tags_img = load_image("examples/tags.png");

    let test_result = detector.detect(&tags_img).unwrap();
    let available_stages = test_result.tp.stage_names();
    eprintln!("OpenCL stages: {:?}", available_stages);

    let mut group = c.benchmark_group("pipeline_stages_opencl");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(10));

    for stage in &available_stages {
        let stage = stage.to_string();
        group.bench_function(&stage, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let result = detector.detect(black_box(&tags_img)).unwrap();
                    total += result.tp.stage_duration(&stage).unwrap_or_default();
                }
                total
            });
        });
    }

    group.finish();
}

// --- Criterion Group Setup ---

criterion_group!(
    image_benches,
    bench_image_processing,
);

criterion_group!(
    stage_benches,
    bench_pipeline_stages,
    bench_pipeline_stages_multithreaded,
);

criterion_group!(
    pipeline_benches,
    bench_full_pipeline,
);

#[cfg(feature = "wgpu")]
criterion_group!(
    gpu_benches,
    bench_gpu_pipeline,
    bench_gpu_stages,
);

#[cfg(feature = "opencl")]
criterion_group!(
    opencl_benches,
    bench_opencl_pipeline,
    bench_opencl_stages,
);

#[cfg(all(not(feature = "wgpu"), not(feature = "opencl")))]
criterion_main!(image_benches, stage_benches, pipeline_benches);

#[cfg(all(feature = "wgpu", not(feature = "opencl")))]
criterion_main!(image_benches, stage_benches, pipeline_benches, gpu_benches);

#[cfg(all(not(feature = "wgpu"), feature = "opencl"))]
criterion_main!(image_benches, stage_benches, pipeline_benches, opencl_benches);

#[cfg(all(feature = "wgpu", feature = "opencl"))]
criterion_main!(image_benches, stage_benches, pipeline_benches, gpu_benches, opencl_benches);
