use std::path::PathBuf;

use apriltag_rs::{ApriltagDetector, AprilTagFamily, Image};
use clap::{Parser, command};
use opencv::core::{TickMeter, Point, Scalar};
use opencv::videoio::{VideoCapture, VideoCaptureAPIs, VideoCaptureProperties};
use opencv::prelude::*;
use opencv::imgproc::*;
use opencv::highgui::{imshow, wait_key};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Enable debugging output (slow)
    #[arg(short, long)]
    debug: bool,
    /// Camera ID
    #[arg(short, long)]
    camera: i32,
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

fn build_detector(args: &Args) -> ApriltagDetector {
    let mut detector = ApriltagDetector::default();
    if args.family.len() == 0 {
        panic!("No AprilTag families to detect");
    }
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

fn main() {
    println!("Parsing args");
    let args = Args::parse();

    println!("Enabling video capture");

    let mut meter = TickMeter::default().unwrap();
    meter.start().unwrap();

    // Initialize camera
    let mut cap = VideoCapture::new(args.camera, VideoCaptureAPIs::CAP_ANY as i32).unwrap();
    if !cap.is_opened().unwrap() {
        panic!("Couldn't open video capture device");
    }

    // Initialize tag detector with options
    let detector = build_detector(&args);

    meter.stop().unwrap();
    let m = String::from("multiple");
    println!("Detector {} initialized in {:.3} seconds", args.family.first().unwrap_or(&m), meter.get_time_sec().unwrap());
    println!("  {}x{} @{}FPS",
        cap.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH as i32).unwrap(),
        cap.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT as i32).unwrap(),
        cap.get(VideoCaptureProperties::CAP_PROP_FPS as i32).unwrap());
    meter.reset().unwrap();

    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame).unwrap();

        let mut gray = Mat::default();
        cvt_color(&mut frame, &mut gray, ColorConversionCodes::COLOR_BGR2GRAY as i32, 0).unwrap();
        // Make an image_u8_t header for the Mat data
        let mut im = Image::<u8>::create(gray.cols() as usize, gray.rows() as usize);
        for y in 0..im.height {
            for x in 0..im.width {
                im[(x, y)] = *gray.at_2d::<u8>(y as i32, x as i32).unwrap();
            }
        }

        let detections = detector.detect(&im);

        // Draw detection outlines
        for det in detections.detections {
            line(&mut frame,
                Point{ x: det.corners[0].x() as i32, y: det.corners[0].y() as i32 },
                Point{ x: det.corners[1].x() as i32, y: det.corners[1].y() as i32 },
                Scalar::new(0 as f64, 0xff as f64, 0 as f64, 0 as f64),
                2,
                LineTypes::LINE_AA as i32,
                0).unwrap();
            line(&mut frame,
                Point{ x: det.corners[0].x() as i32, y: det.corners[0].y() as i32 },
                Point{ x: det.corners[3].x() as i32, y: det.corners[3].y() as i32 },
                Scalar::new(0 as f64, 0 as f64, 0xff as f64, 0 as f64),
                2,
                LineTypes::LINE_AA as i32,
                0).unwrap();
            line(&mut frame,
                Point{ x: det.corners[1].x() as i32, y: det.corners[1].y() as i32 },
                Point{ x: det.corners[2].x() as i32, y: det.corners[2].y() as i32 },
                Scalar::new(0xff as f64, 0 as f64, 0 as f64, 0 as f64),
                2,
                LineTypes::LINE_AA as i32,
                0).unwrap();
            line(&mut frame,
                Point{ x: det.corners[2].x() as i32, y: det.corners[2].y() as i32 },
                Point{ x: det.corners[3].x() as i32, y: det.corners[3].y() as i32 },
                Scalar::new(0xff as f64, 0 as f64, 0 as f64, 0 as f64),
                2,
                LineTypes::LINE_AA as i32,
                0).unwrap();

            let text = format!("{}", det.id);
            let fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
            let fontscale = 1.0;
            let mut baseline = 0;
            let textsize = get_text_size(&text, fontface, fontscale, 2, &mut baseline).unwrap();
            put_text(&mut frame,
                &text,
                Point {
                    x: det.center.x() as i32 - textsize.width/2,
                    y: det.center.y() as i32 + textsize.height/2
                },
                fontface, fontscale, Scalar::new(0xff as f64, 0x99 as f64, 0., 0.),
                2,
                LineTypes::LINE_AA as i32,
                false
            ).unwrap();
        }

        imshow(&"Tag Detections", &frame).unwrap();
        if wait_key(30).unwrap() >= 0 {
            break;
        }
    }
}
