#![feature(duration_millis_float)]
use std::{
    fs::{self, OpenOptions},
    hint::black_box,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::SystemTime,
};

use indexmap::IndexMap;

use anyhow::{Context, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};

use apriltag_rs::{
    AccelerationRequest, AprilTagDetector, AprilTagFamily,
    AprilTagQuadThreshParams, DetectorBuilder, DetectorConfig, GpuDeviceInfo as LibGpuDeviceInfo,
    ImageY8, Luma, SourceDimensions,
};

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "apriltag-bench", about = "AprilTag detection benchmark runner")]
struct Cli {
    /// Path(s) to TOML config file(s). Multiple configs are run in sequence.
    #[arg(short, long, required = true, num_args = 1..)]
    config: Vec<PathBuf>,

    /// Override output file path (applies to all configs)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// ── TOML config types ────────────────────────────────────────────────

#[derive(Deserialize)]
struct BenchmarkToml {
    benchmark: BenchmarkParams,
    detector: DetectorToml,
    families: Vec<FamilyConfig>,
}

#[derive(Deserialize, Serialize, Clone)]
struct BenchmarkParams {
    iterations: usize,
    warmup_iterations: usize,
    images_folder: String,
    #[serde(default = "default_output_file")]
    output_file: String,
    #[serde(default)]
    label: Option<String>,
}

fn default_output_file() -> String {
    "benchmark_results.jsonl".into()
}

#[derive(Deserialize, Serialize, Clone)]
struct DetectorToml {
    #[serde(default = "default_nthreads")]
    nthreads: usize,
    #[serde(default = "default_quad_decimate")]
    quad_decimate: f32,
    #[serde(default)]
    quad_sigma: f32,
    #[serde(default = "default_true")]
    refine_edges: bool,
    #[serde(default = "default_decode_sharpening")]
    decode_sharpening: f64,
    #[serde(default = "default_acceleration")]
    acceleration: String,
    #[serde(default = "default_true")]
    allow_concurrency: bool,
    /// Source dimensions hint: "dynamic", "WxH" (e.g. "1920x1080")
    #[serde(default = "default_source_dimensions")]
    source_dimensions: String,
    #[serde(default)]
    qtp: QtpToml,
}

fn default_nthreads() -> usize { 1 }
fn default_quad_decimate() -> f32 { 2.0 }
fn default_true() -> bool { true }
fn default_decode_sharpening() -> f64 { 0.25 }
fn default_acceleration() -> String { "disabled".into() }
fn default_source_dimensions() -> String { "dynamic".into() }

#[derive(Deserialize, Serialize, Clone)]
struct QtpToml {
    #[serde(default = "default_min_cluster_pixels")]
    min_cluster_pixels: u32,
    #[serde(default = "default_max_nmaxima")]
    max_nmaxima: u8,
    #[serde(default = "default_cos_critical_rad")]
    cos_critical_rad: f32,
    #[serde(default = "default_max_line_fit_mse")]
    max_line_fit_mse: f32,
    #[serde(default = "default_min_white_black_diff")]
    min_white_black_diff: u8,
    #[serde(default)]
    deglitch: bool,
}

fn default_min_cluster_pixels() -> u32 { 5 }
fn default_max_nmaxima() -> u8 { 10 }
fn default_cos_critical_rad() -> f32 { (10f32).to_radians().cos() }
fn default_max_line_fit_mse() -> f32 { 10.0 }
fn default_min_white_black_diff() -> u8 { 5 }

impl Default for QtpToml {
    fn default() -> Self {
        Self {
            min_cluster_pixels: default_min_cluster_pixels(),
            max_nmaxima: default_max_nmaxima(),
            cos_critical_rad: default_cos_critical_rad(),
            max_line_fit_mse: default_max_line_fit_mse(),
            min_white_black_diff: default_min_white_black_diff(),
            deglitch: false,
        }
    }
}

#[derive(Deserialize, Serialize, Clone)]
struct FamilyConfig {
    name: String,
    #[serde(default = "default_hamming")]
    hamming: usize,
}

fn default_hamming() -> usize { 1 }

// ── JSON output types ────────────────────────────────────────────────

#[derive(Serialize)]
struct BenchmarkRun {
    metadata: RunMetadata,
    config: ConfigOutput,
    benchmark_params: BenchmarkParamsOutput,
    images: Vec<ImageResults>,
}

#[derive(Serialize)]
struct RunMetadata {
    timestamp: String,
    git_commit: Option<String>,
    git_dirty: Option<bool>,
    label: Option<String>,
    hostname: Option<String>,
    gpu_device: Option<GpuDeviceInfo>,
}

#[derive(Serialize)]
struct GpuDeviceInfo {
    /// Which apriltag-rs accelerator: "wgpu" or "opencl"
    accelerator: String,
    /// Underlying GPU API (e.g. "Metal", "Vulkan", "OpenCL")
    backend: String,
    name: String,
    device_type: String,
    vendor: String,
    driver: String,
}

impl From<LibGpuDeviceInfo> for GpuDeviceInfo {
    fn from(info: LibGpuDeviceInfo) -> Self {
        Self {
            accelerator: info.accelerator,
            backend: info.backend,
            name: info.name,
            device_type: info.device_type,
            vendor: info.vendor,
            driver: info.driver,
        }
    }
}

#[derive(Serialize)]
struct ConfigOutput {
    #[serde(flatten)]
    detector: DetectorToml,
    families: Vec<FamilyConfig>,
}

#[derive(Serialize)]
struct BenchmarkParamsOutput {
    iterations: usize,
    warmup_iterations: usize,
    images_folder: String,
}

#[derive(Serialize)]
struct ImageResults {
    path: String,
    width: u32,
    height: u32,
    iterations: Vec<IterationResult>,
    summary: ImageSummary,
}

#[derive(Serialize)]
struct IterationResult {
    wall_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_time_ms: Option<f64>,
    nquads: u32,
    stages: IndexMap<String, f64>,
    /// CPU (process) time per stage
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_stages: Option<IndexMap<String, f64>>,
    /// GPU-side stage timings (if hardware acceleration with timestamp queries)
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_stages: Option<IndexMap<String, f64>>,
    detections: Vec<DetectionRecord>,
}

#[derive(Serialize)]
struct DetectionRecord {
    family: String,
    id: usize,
    hamming: u16,
    decision_margin: f32,
    center: [f64; 2],
    corners: [[f64; 2]; 4],
}

#[derive(Serialize)]
struct ImageSummary {
    wall_time: Stats,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_time: Option<Stats>,
    stage_means_ms: IndexMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_stage_means_ms: Option<IndexMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_stage_means_ms: Option<IndexMap<String, f64>>,
    detection_count: usize,
}

#[derive(Serialize)]
struct Stats {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stddev_ms: f64,
}

// ── Conversion helpers ───────────────────────────────────────────────

fn parse_acceleration(s: &str) -> anyhow::Result<AccelerationRequest> {
    let s = s.trim().to_lowercase();
    if let Some(idx) = s.strip_prefix("prefer_device:") {
        return Ok(AccelerationRequest::PreferDeviceIdx(idx.parse().context("invalid device index")?));
    }
    if let Some(idx) = s.strip_prefix("required_device:") {
        return Ok(AccelerationRequest::RequiredDeviceIdx(idx.parse().context("invalid device index")?));
    }
    match s.as_str() {
        "disabled" => Ok(AccelerationRequest::Disabled),
        "prefer" => Ok(AccelerationRequest::Prefer),
        "prefer_gpu" => Ok(AccelerationRequest::PreferGpu),
        "required" => Ok(AccelerationRequest::Required),
        "required_gpu" => Ok(AccelerationRequest::RequiredGpu),
        other => anyhow::bail!("unknown acceleration mode: {other}"),
    }
}

fn parse_source_dimensions(s: &str) -> anyhow::Result<SourceDimensions> {
    let s = s.trim().to_lowercase();
    if s == "dynamic" || s.is_empty() {
        return Ok(SourceDimensions::Dynamic);
    }
    if let Some((w, h)) = s.split_once('x') {
        let width: usize = w.parse().context("invalid width in source_dimensions")?;
        let height: usize = h.parse().context("invalid height in source_dimensions")?;
        return Ok(SourceDimensions::Exactly { width, height });
    }
    anyhow::bail!("invalid source_dimensions: {s} (expected \"dynamic\" or \"WxH\")")
}

fn build_detector(toml: &BenchmarkToml) -> anyhow::Result<AprilTagDetector> {
    let DetectorToml {
		nthreads,
		quad_decimate,
		quad_sigma,
		refine_edges,
		decode_sharpening,
		ref acceleration,
		allow_concurrency,
		ref source_dimensions,
		qtp: QtpToml {
			min_cluster_pixels,
			max_nmaxima,
			cos_critical_rad,
			max_line_fit_mse,
			min_white_black_diff,
			deglitch,
		},
		..
	} = toml.detector;
    let config = DetectorConfig {
		nthreads,
		quad_decimate,
		quad_sigma,
		refine_edges,
		decode_sharpening,
		debug: false,
		acceleration: parse_acceleration(acceleration)?,
		allow_concurrency,
		source_dimensions: parse_source_dimensions(source_dimensions)?,
		qtp: AprilTagQuadThreshParams {
			min_cluster_pixels,
			max_nmaxima,
			cos_critical_rad,
			max_line_fit_mse,
			min_white_black_diff,
			deglitch,
		},
		..Default::default()
	};

    let mut builder = DetectorBuilder::new(config);
    for fam_cfg in &toml.families {
        let family = AprilTagFamily::for_name(&fam_cfg.name)
            .with_context(|| format!("unknown tag family: {}", fam_cfg.name))?;
        builder
            .add_family_bits(family, fam_cfg.hamming)
            .with_context(|| format!("failed to add family {}", fam_cfg.name))?;
    }

    builder.build().context("failed to build detector")
}

fn load_images(folder: &Path) -> anyhow::Result<Vec<(PathBuf, ImageY8)>> {
    let mut entries: Vec<PathBuf> = fs::read_dir(folder)
		.with_context(|| format!("cannot read images folder {}", folder.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    let ext = ext.to_lowercase();
                    ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "pnm"
                })
                .unwrap_or(false)
        })
        .collect();
    entries.sort();

	ensure!(!entries.is_empty(), "no images found in {}", folder.display());

    entries
        .into_iter()
        .map(|path| {
            let img = load_image(&path)?;
            Ok((path, img))
        })
        .collect()
}

fn load_image(path: &Path) -> anyhow::Result<ImageY8> {
    use image::ImageReader;
    let reader = ImageReader::open(path)
        .with_context(|| format!("cannot open image {}", path.display()))?;
    let decoded = reader
        .decode()
        .with_context(|| format!("cannot decode image {}", path.display()))?;
    let gray = decoded.into_luma8();
    let (w, h) = gray.dimensions();
    let mut out = apriltag_rs::ImageBuffer::<Luma<u8>, _>::zeroed(w as usize, h as usize);
    for (x, y, pixel) in gray.enumerate_pixels() {
        out[(x as usize, y as usize)] = pixel.0[0];
    }
    Ok(out)
}

fn extract_iteration(detections: apriltag_rs::Detections) -> IterationResult {
    let tp = &detections.tp;
    let wall_time_ms = tp.total_duration().as_millis_f64();
    let cpu_time_ms = tp.total_cpu_duration().map(|d| d.as_millis_f64());

    let mut stages = IndexMap::new();
    let mut cpu_stages = IndexMap::new();
    for name in tp.stage_names() {
        if let Some(dur) = tp.stage_duration(name) {
            stages.insert(name.to_string(), dur.as_millis_f64());
        }
        if let Some(dur) = tp.stage_cpu_duration(name) {
            cpu_stages.insert(name.to_string(), dur.as_millis_f64());
        }
    }
    let cpu_stages = if cpu_stages.is_empty() { None } else { Some(cpu_stages) };

    let gpu_stages = detections.gpu_tp.map(|gpu_tp| {
        let mut gs = IndexMap::new();
        for name in gpu_tp.stage_names() {
            if let Some(dur) = gpu_tp.stage_duration(name) {
                gs.insert(name.to_string(), dur.as_millis_f64());
            }
        }
        gs
    });

    let det_records: Vec<DetectionRecord> = detections
        .detections
        .into_iter()
        .map(|d| DetectionRecord {
            family: d.family.name.to_string(),
            id: d.id,
            hamming: d.hamming,
            decision_margin: d.decision_margin,
            center: d.center.as_array(),
            corners: d.corners.as_array(),
        })
        .collect();

    IterationResult {
        wall_time_ms,
        cpu_time_ms,
        nquads: detections.nquads,
        stages,
        cpu_stages,
        gpu_stages,
        detections: det_records,
    }
}

fn compute_summary(iterations: &[IterationResult]) -> ImageSummary {
    let wall_times: Vec<f64> = iterations.iter().map(|i| i.wall_time_ms).collect();
    let wall_time = compute_stats(&wall_times);

    // Compute CPU time stats if available
    let cpu_time = {
        let cpu_times: Vec<f64> = iterations.iter().filter_map(|i| i.cpu_time_ms).collect();
        if cpu_times.is_empty() { None } else { Some(compute_stats(&cpu_times)) }
    };

    // Compute mean for each stage
    let mut stage_sums: IndexMap<String, Vec<f64>> = IndexMap::new();
    for iter in iterations {
        for (name, &ms) in &iter.stages {
            stage_sums.entry(name.clone()).or_default().push(ms);
        }
    }
    let stage_means_ms: IndexMap<String, f64> = stage_sums
        .into_iter()
        .map(|(name, vals)| {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            (name, mean)
        })
        .collect();

    // Compute CPU stage means if any iteration has CPU timing
    let cpu_stage_means_ms = compute_optional_stage_means(iterations, |i| i.cpu_stages.as_ref());

    // Compute GPU stage means if any iteration has GPU timing
    let gpu_stage_means_ms = compute_optional_stage_means(iterations, |i| i.gpu_stages.as_ref());

    let detection_count = iterations
        .last()
        .map(|i| i.detections.len())
        .unwrap_or(0);

    ImageSummary {
        wall_time,
        cpu_time,
        stage_means_ms,
        cpu_stage_means_ms,
        gpu_stage_means_ms,
        detection_count,
    }
}

fn compute_optional_stage_means(
    iterations: &[IterationResult],
    get_stages: impl Fn(&IterationResult) -> Option<&IndexMap<String, f64>>,
) -> Option<IndexMap<String, f64>> {
    if !iterations.iter().any(|i| get_stages(i).is_some()) {
        return None;
    }
    let mut sums: IndexMap<String, Vec<f64>> = IndexMap::new();
    for iter in iterations {
        if let Some(stages) = get_stages(iter) {
            for (name, &ms) in stages {
                sums.entry(name.clone()).or_default().push(ms);
            }
        }
    }
    Some(
        sums.into_iter()
            .map(|(name, vals)| {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                (name, mean)
            })
            .collect(),
    )
}

fn compute_stats(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            mean_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            stddev_ms: 0.0,
        };
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();
    Stats {
        mean_ms: mean,
        min_ms: min,
        max_ms: max,
        stddev_ms: stddev,
    }
}

fn collect_metadata(label: Option<String>, gpu_device: Option<GpuDeviceInfo>) -> RunMetadata {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| {
            let secs = d.as_secs();
            // Format as ISO 8601 manually
            let s = secs;
            let days = s / 86400;
            let time = s % 86400;
            let hours = time / 3600;
            let minutes = (time % 3600) / 60;
            let seconds = time % 60;

            // Days since epoch to date (simplified)
            let mut y = 1970i64;
            let mut remaining_days = days as i64;
            loop {
                let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                    366
                } else {
                    365
                };
                if remaining_days < days_in_year {
                    break;
                }
                remaining_days -= days_in_year;
                y += 1;
            }
            let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
            let month_days = [
                31,
                if leap { 29 } else { 28 },
                31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
            ];
            let mut m = 0;
            for &md in &month_days {
                if remaining_days < md {
                    break;
                }
                remaining_days -= md;
                m += 1;
            }
            format!(
                "{y:04}-{:02}-{:02}T{hours:02}:{minutes:02}:{seconds:02}Z",
                m + 1,
                remaining_days + 1,
            )
        })
        .unwrap_or_else(|_| "unknown".into());

    let git_commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        });

    let git_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty());

    let hostname = Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        });

    RunMetadata {
        timestamp,
        git_commit,
        git_dirty,
        label,
        hostname,
        gpu_device,
    }
}

// ── Main ─────────────────────────────────────────────────────────────

fn run_config(config_path: &Path, output_path: &Path) -> anyhow::Result<()> {
    let config_text = fs::read_to_string(config_path)
        .with_context(|| format!("cannot read config file {}", config_path.display()))?;
    let toml_config: BenchmarkToml =
        toml::from_str(&config_text).context("failed to parse config TOML")?;

    let label = toml_config.benchmark.label.as_deref()
        .unwrap_or_else(|| config_path.file_stem().and_then(|s| s.to_str()).unwrap_or("?"));
    eprintln!(
        "\n[{}] {} iterations ({} warmup), images from {}",
        label,
        toml_config.benchmark.iterations,
        toml_config.benchmark.warmup_iterations,
        toml_config.benchmark.images_folder,
    );

    // Build detector
    let detector = build_detector(&toml_config)?;
    eprintln!("[{label}] Detector built (acceleration: {})", toml_config.detector.acceleration);

    // Load images
    let images_folder = Path::new(&toml_config.benchmark.images_folder);
    let images = load_images(images_folder).context("load images")?;
    eprintln!("[{label}] Loaded {} images", images.len());

    // Query GPU info and collect metadata
    let gpu_device = detector.gpu_device_info().map(GpuDeviceInfo::from);
    let metadata = collect_metadata(toml_config.benchmark.label.clone(), gpu_device);

    // Run benchmarks
    let total_iters_per_image =
        toml_config.benchmark.warmup_iterations + toml_config.benchmark.iterations;
    let total_work = images.len() as u64 * total_iters_per_image as u64;

    let pb = indicatif::ProgressBar::new(total_work);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{msg}\n  [{elapsed_precise}] [{bar:40.cyan/dim}] {pos}/{len} ({per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let mut image_results = Vec::with_capacity(images.len());
    for (img_idx, (path, img)) in images.iter().enumerate() {
        let (w, h) = (img.width() as u32, img.height() as u32);
        let rel_path = path.to_string_lossy().to_string();
        pb.set_message(format!(
            "[{label}] [{}/{}] {rel_path} ({w}x{h})",
            img_idx + 1,
            images.len(),
        ));

        // Warmup — use black_box to prevent the compiler from optimizing away the work
        for _ in 0..toml_config.benchmark.warmup_iterations {
            let result = detector.detect(img);
            black_box(&result);
            pb.inc(1);
        }

        // Measurement
        let mut iterations = Vec::with_capacity(toml_config.benchmark.iterations);
        for _ in 0..toml_config.benchmark.iterations {
            let result = detector.detect(img).context("detection failed")?;
            iterations.push(extract_iteration(result));
            pb.inc(1);
        }

        let summary = compute_summary(&iterations);
        image_results.push(ImageResults {
            path: rel_path,
            width: w,
            height: h,
            iterations,
            summary,
        });
    }
    pb.finish_and_clear();

    // Print summary
    for res in &image_results {
        eprintln!(
            "  {} ({}x{}): {:.2}ms ± {:.2}ms ({} detections)",
            res.path, res.width, res.height,
            res.summary.wall_time.mean_ms, res.summary.wall_time.stddev_ms,
            res.summary.detection_count,
        );
    }

    // Build output
    let run = BenchmarkRun {
        metadata,
        config: ConfigOutput {
            detector: toml_config.detector.clone(),
            families: toml_config.families.clone(),
        },
        benchmark_params: BenchmarkParamsOutput {
            iterations: toml_config.benchmark.iterations,
            warmup_iterations: toml_config.benchmark.warmup_iterations,
            images_folder: toml_config.benchmark.images_folder.clone(),
        },
        images: image_results,
    };

    // Append to output file
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .with_context(|| format!("cannot open output file {}", output_path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &run).context("failed to write JSON")?;
    writeln!(writer).context("failed to write newline")?;
    writer.flush().context("failed to flush output")?;

    eprintln!("[{label}] Results appended to {}", output_path.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Determine output path: CLI override, or from first config, or default
    let output_path = if let Some(ref path) = cli.output {
        path.clone()
    } else {
        // Read first config to get its output_file setting
        let first_text = fs::read_to_string(&cli.config[0])
            .with_context(|| format!("cannot read config file {}", cli.config[0].display()))?;
        let first: BenchmarkToml =
            toml::from_str(&first_text).context("failed to parse first config TOML")?;
        PathBuf::from(&first.benchmark.output_file)
    };

    eprintln!("Running {} config(s), output → {}", cli.config.len(), output_path.display());

    for (i, config_path) in cli.config.iter().enumerate() {
        eprintln!("━━━ Config {}/{}: {} ━━━", i + 1, cli.config.len(), config_path.display());
        run_config(config_path, &output_path)?;
    }

    eprintln!("\nAll {} config(s) complete.", cli.config.len());
    Ok(())
}
