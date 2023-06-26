use std::{path::{PathBuf, Path}, time::Duration, io};

use apriltag_rs::{AprilTagDetector, AprilTagFamily, util::ImageY8, TimeProfileStatistics};
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
    input_files: Vec<PathBuf>,
}

fn build_detector(args: &Args) -> AprilTagDetector {
    println!("refine_edges: {}", args.refine_edges);
    let mut detector = AprilTagDetector::default();
    for family_name in args.family.iter() {
        let family = if let Some(family) = AprilTagFamily::for_name(&family_name) {
            family
        } else {
            println!("Error: Unknown family name: {}", family_name);
            println!("Valid family names:");
            println!(" - tag16h5");
            println!(" - tag25h9");
            println!(" - tag36h10");
            println!(" - tag36h11");
            panic!();
        };

        detector.add_family_bits(family, args.hamming).unwrap();
    }

    detector.params.quad_decimate = args.decimate;
    detector.params.quad_sigma = args.blur;
    detector.params.nthreads = args.threads;
    detector.params.debug = args.debug;
    detector.params.refine_edges = args.refine_edges;

    detector
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

fn main() {
    let args = Args::parse();
    if args.debug {
        println!("Arguments:");
        println!(" - threads: {}", args.threads);
    }
    let detector = build_detector(&args);

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

            let detections = detector.detect(&im);

            println!("Found {} tags", detections.detections.len());

            for (i, det) in detections.detections.iter().enumerate() {
                if !quiet {
                    println!("detection {:3}: id ({:2}x{:2})-{:4}, hamming {}, margin {:8.3}",
                           i,
                           det.family.bits.len(),
                           det.family.min_hamming,
                           det.id,
                           det.hamming,
                           det.decision_margin
                    );
                }

                hamm_hist[det.hamming as usize] += 1;
                total_hamm_hist[det.hamming as usize] += 1;
            }

            if !quiet {
                print!("{}", detections.tp);
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

    if args.iters > 1 {
        acc.display();
    }
}