use std::{path::{PathBuf, Path}, time::Duration, io, panic::AssertUnwindSafe};

use apriltag_rs::{AprilTagDetector, AprilTagFamily, util::{ImageY8, ImageRGB8, image::Pixel}, TimeProfileStatistics, Detections};
use image::{ImageBuffer as IImageBuffer, Rgb};
use rayon::prelude::*;
use clap::{Parser, arg, command};

const HAMM_HIST_MAX: usize = 10;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Enable debugging output (slow)
    #[arg(short, long)]
    debug: bool,
    /// Reduce output
    #[arg(short, long, default_value_t=false)]
    quiet: bool,
    /// Tag family to use
    #[arg(short, long)]
    family: Vec<String>,
    /// Repeat processing on input set this many times
    #[arg(short, long, default_value_t=1)]
    iters: usize,
    /// Use this many CPU threads
    #[arg(short, long, default_value_t=1)]
    threads: usize,
    /// Detect tags with up to this many bit errors.
    #[arg(short='a', long, default_value_t=1)]
    hamming: usize,
    /// Decimate input image by this factor
    #[arg(short='x', long, default_value_t=2.0)]
    decimate: f32,
    /// Apply low-pass blur to input; negative sharpens
    #[arg(short, long, default_value_t=0.0)]
    blur: f32,
    /// Spend more time trying to align edges of tags
    #[arg(short, long, default_value_t=true)]
    refine_edges: bool,

    #[arg(long)]
    debug_path: Option<PathBuf>,
    input_files: Vec<PathBuf>,
}

fn build_detector(args: &Args, gpu: bool) -> AprilTagDetector {
    let mut builder = AprilTagDetector::builder();
    if gpu {
        builder.set_gpu_mode(apriltag_rs::AccelerationRequest::Required)
    } else {
        builder.set_gpu_mode(apriltag_rs::AccelerationRequest::Disabled)
    }

    if args.family.len() == 0 {
        panic!("No AprilTag families to detect. Use --family [family name]");
    }
    for family_name in args.family.iter() {
        let family = if let Some(family) = AprilTagFamily::for_name(&family_name) {
            family
        } else {
            println!("Error: Unknown family name: {}", family_name);
            println!("Valid family names:");
            for name in AprilTagFamily::names() {
                println!(" - {name}");
            }
            panic!();
        };

        builder.add_family_bits(family, args.hamming).unwrap();
    }

    builder.config.quad_decimate = args.decimate;
    builder.config.quad_sigma = args.blur;
    builder.config.nthreads = args.threads;
    builder.config.debug = args.debug;
    builder.config.refine_edges = args.refine_edges;

    {
        let mut debug_path = match &args.debug_path {
            Some(p) => p.to_owned(),
            None => PathBuf::from("./debug"),
        };
        if gpu {
            debug_path.push("gpu");
        } else {
            debug_path.push("cpu");
        };
        if !debug_path.exists() {
            std::fs::create_dir_all(&debug_path)
                .expect("Unable to create debug dir");
        }
        builder.config.debug_path = Some(debug_path);
    }

    builder.build()
        .expect("Error building detector")
}

fn load_image(path: &Path) -> io::Result<ImageY8> {
    if let Some(extension) = path.extension() {
        if extension.eq_ignore_ascii_case("pnm") {
            return ImageY8::create_from_pnm(path);
        }
    }

    use image::io::Reader as ImageReader;
    let reader = ImageReader::open(path)?;
    let image = reader.decode().unwrap().into_luma8();

    let mut result = ImageY8::zeroed(image.width() as usize, image.height() as usize);
    for (x, y, value) in image.enumerate_pixels() {
        let value = value.0[0];
        result[(x as usize, y as usize)] = value;
    }
    Ok(result)
}

fn detect_with(detector: &AprilTagDetector, image: &ImageY8, quiet: bool) -> Detections {
    match std::panic::catch_unwind(AssertUnwindSafe(|| detector.detect(image))) {
        Ok(Ok(detections)) => {
            if !quiet {
                println!("Found {} tags", detections.detections.len());
                for (i, det) in detections.detections.iter().enumerate() {
                    println!("detection {:2}: id ({:>9})-{:4}, hamming {}, margin {:8.3}",
                        i,
                        det.family.name,
                        det.id,
                        det.hamming,
                        det.decision_margin
                    );
                }
            }
            detections
        },
        Ok(Err(e)) => {
            eprintln!("Error (converting images)");
            convert_images(detector);
            panic!("Error detecting AprilTags: {e:?}");
        }
        Err(e) => {
            eprintln!("Error (converting images)");
            convert_images(detector);
            std::panic::resume_unwind(e);
        },
    }
}

fn convert_images(det: &AprilTagDetector) {
    let debug_path = match &det.params.debug_path {
        Some(dbp) => dbp.as_path(),
        None => return,
    };
    let rd = match std::fs::read_dir(debug_path) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!("Image conversion: Unable to read dir {}: {e:?}", debug_path.display());
            return;
        }
    };

    rd
        .par_bridge()
        .for_each(|path| {
            let path = path.unwrap();
            if !path.file_type().unwrap().is_file() {
                return;
            }
            if !path.file_name().to_str().unwrap().ends_with(".pnm") {
                return;
            }
            let img = ImageRGB8::create_from_pnm(&path.path()).unwrap();
            let mut buf = IImageBuffer::<Rgb<u8>, _>::new(img.width() as u32, img.height() as u32);
            for ((x, y), value) in img.enumerate_pixels() {
                *buf.get_pixel_mut(x as u32, y as u32) = Rgb(value.0);
            }
            buf.save_with_format(path.path().with_extension("png"), image::ImageFormat::Png)
                .unwrap();
        });
}

fn compare_images(path_cpu: &PathBuf, path_gpu: PathBuf) {
    std::fs::read_dir(path_cpu).unwrap()
        .par_bridge()
        .for_each(|entry_cpu| {
            let entry_cpu = entry_cpu.unwrap();
            if !entry_cpu.file_type().unwrap().is_file() {
                return;
            }
            if !entry_cpu.file_name().to_str().unwrap().ends_with(".pnm") {
                return;
            }
            let entry_cpu = entry_cpu.path();
            let entry_name = entry_cpu.file_name().unwrap();
            let mut entry_gpu = path_gpu.clone();
            entry_gpu.push(entry_name);
            if !entry_gpu.exists() {
                return;
            }

            let img_cpu = ImageRGB8::create_from_pnm(&entry_cpu).unwrap();
            let img_gpu = ImageRGB8::create_from_pnm(&entry_gpu).unwrap();
            if img_cpu.width() != img_gpu.width() || img_cpu.height() != img_gpu.height() {
                println!("Debug image {} has different dimensions (cpu={:?}, gpu={:?})", entry_name.to_string_lossy(), img_cpu.dimensions(), img_gpu.dimensions());
                return;
            }
            let mut delta = ImageRGB8::zeroed_packed(img_cpu.width(), img_cpu.height());
            let mut different = false;
            for ((x, y), px2) in img_gpu.enumerate_pixels() {
                let [r1, g1, b1] = img_cpu[(x, y)].to_value();
                let [r2, g2, b2] = px2.to_value();
                let rgb = [r1.abs_diff(r2), g1.abs_diff(g2), b1.abs_diff(b2)];
                different |= rgb != [0,0,0];
                let rgb = if rgb != [0,0,0] {
                    // rgb[0] = 255;
                    let s = ((r2 as i16) - (r1 as i16)) + ((g2 as i16) - (g1 as i16)) + ((b2 as i16) - (b1 as i16));
                    if s > 0 {
                        [r1.abs_diff(r2).max(127), 0, 0]
                    } else {
                        [0, g2.abs_diff(g2).max(127), 0]
                    }
                } else {
                    // [r1, g2, 0]
                    [0,0,0]
                };
                delta[(x, y)] = rgb;
            }
            println!("{} {}", if different { "Different" } else { "     Same" }, entry_name.to_string_lossy());
            let h = delta.height() as u32;
            let mut buf = IImageBuffer::<Rgb<u8>, _>::new(delta.width() as u32, h * 3);
            for ((x, y), value) in img_cpu.enumerate_pixels() {
                *buf.get_pixel_mut(x as u32, y as u32) = Rgb(value.0);
            }
            for ((x, y), value) in delta.enumerate_pixels() {
                *buf.get_pixel_mut(x as u32, y as u32 + h) = Rgb(value.0);
            }
            for ((x, y), value) in img_gpu.enumerate_pixels() {
                *buf.get_pixel_mut(x as u32, y as u32 + 2 * h) = Rgb(value.0);
            }
            let dst = PathBuf::from(format!("./debug/{}", entry_name.to_string_lossy()));
            let dst = dst.with_extension("png");
            buf.save_with_format(dst, image::ImageFormat::Png).unwrap();
        });
}

const WARMUP_ITERS: usize = 0;

fn main() {
    let args = Args::parse();
    if args.debug {
        println!("Arguments:");
        println!(" - threads: {}", args.threads);
    }
    let detector = build_detector(&args, false);
    let detector_gpu = build_detector(&args, true);

    let quiet = args.quiet;
    let mut acc = TimeProfileStatistics::default();

    for iter in 0..args.iters {
        let mut total_quads = 0;
        let mut total_hamm_hist = [0usize; HAMM_HIST_MAX];
        let mut total_time = Duration::ZERO;

        if args.iters > 1 {
            println!("iter {} / {}", iter + 1, args.iters);
        }

        for input in args.input_files.iter() {
            let mut hamm_hist = [0u32; HAMM_HIST_MAX];

            if quiet {
                print!("{:20}", input.display());
            } else {
                println!("loading {}", input.display());
            }

            let im = match load_image(&input) {
                Ok(image) => image,
                Err(e) => {
                    println!("Error: couldn't load {}", input.display());
                    println!("Cause: {}", e);
                    continue;
                }
            };

            println!("image: {} {}x{}", input.display(), im.width(), im.height());

            println!("==== GPU ====");
            for _ in 0..WARMUP_ITERS {
                detect_with(&detector_gpu, &im, true);
            }
            let detections_gpu = detect_with(&detector_gpu, &im, quiet);
            if !quiet {
                print!("{:#}", detections_gpu.tp);
            }

            println!("==== CPU ====");
            for _ in 0..WARMUP_ITERS {
                detect_with(&detector, &im, true);
            }
            let detections = detect_with(&detector, &im, quiet);
            

            for det in detections.detections.iter() {
                hamm_hist[det.hamming as usize] += 1;
                total_hamm_hist[det.hamming as usize] += 1;
            }

            if !quiet {
                print!("{:#}", detections.tp);
                acc.add(&detections.tp);
            }

            total_quads += detections.nquads;

            if !quiet {
                print!("hamm ");
            }

            for i in 0..HAMM_HIST_MAX {
                print!("{:5} ", hamm_hist[i]);
            }

            let t = detections.tp.total_duration();
            total_time += t;
            print!("{:12.3}s", t.as_secs_f32());
            print!("{:4}", detections.nquads);

            println!();
        }


        println!("Summary");

        print!("hamm ");
        for v in total_hamm_hist {
            print!("{:5} ", v);
        }
        print!("{:12.3} ", total_time.as_secs_f32());
        print!("{:5}", total_quads);
        println!();
    }

    println!("==== Done ====");

    if args.iters > 1 {
        println!("{acc}");
    }

    [&detector, &detector_gpu]
        .into_par_iter()
        .for_each(convert_images);

    if let Some(path_cpu) = detector.params.debug_path.as_ref() {
        let path_gpu = detector_gpu.params.debug_path.unwrap();
        compare_images(path_cpu, path_gpu);
    }
}