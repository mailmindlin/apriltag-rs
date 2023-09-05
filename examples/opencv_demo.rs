use std::path::PathBuf;
use std::time::Instant;

use apriltag_rs::util::ImageY8;
use apriltag_rs::{AprilTagDetector, AprilTagFamily, TimeProfile};
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
    #[arg(long)]
    debug_path: Option<PathBuf>,
    #[arg(long)]
    max_frames: Option<usize>,
}

fn build_detector(args: &Args, path_override: Option<&str>) -> AprilTagDetector {
    let mut builder = AprilTagDetector::builder();
    if args.family.len() == 0 {
        panic!("No AprilTag families to detect");
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

        builder.add_family_bits(family, args.hamming)
            .expect("Error adding AprilTag family");
    }

    builder.config.quad_decimate = args.decimate;
    builder.config.quad_sigma = args.blur;
    builder.config.nthreads = args.threads;
    builder.config.debug = args.debug;
    builder.config.refine_edges = args.refine_edges;
    if let Some(path) = &args.debug_path {
        builder.config.debug_path = Some(path.to_str().unwrap().to_owned());
    }

    builder.build()
        .unwrap()
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

    cap.set(VideoCaptureProperties::CAP_PROP_FPS as i32, 60.).unwrap();
    cap.set(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH as i32, 10000.).unwrap();
    cap.set(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT as i32, 10000.).unwrap();

    // Initialize tag detector with options
    // args.opencl = false;
    let detector = build_detector(&args, Some("cpu"));

    // args.opencl = true;
    // let detector1 = build_detector(&args, Some("gpu"));

    meter.stop().unwrap();
    let m = String::from("multiple");
    println!("Detector {} initialized in {:.3} seconds", args.family.first().unwrap_or(&m), meter.get_time_sec().unwrap());
    println!("  {}x{} @{}FPS",
        cap.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH as i32).unwrap(),
        cap.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT as i32).unwrap(),
        cap.get(VideoCaptureProperties::CAP_PROP_FPS as i32).unwrap());
    meter.reset().unwrap();

    let mut nframes = 0;

    loop {
        nframes += 1;
        if let Some(max_frames) = args.max_frames {
            if nframes >= max_frames {
                break;
            }
        }
        let mut tp = TimeProfile::default();
        let mut frame = Mat::default();
        cap.read(&mut frame).unwrap();
        tp.stamp("imread");

        let mut gray = Mat::default();
        cvt_color(&mut frame, &mut gray, ColorConversionCodes::COLOR_BGR2GRAY as i32, 0).unwrap();
        tp.stamp("cvt_color");
        // Make an image_u8_t header for the Mat data
        let mut im = ImageY8::zeroed(gray.cols() as usize, gray.rows() as usize);
        for (y, mut dst) in im.rows_mut() {
            let src = gray.at_row::<u8>(y as i32).unwrap();
            dst.as_slice_mut().copy_from_slice(src);
        }
        // for ((x, y), dst) in im.enumerate_pixels_mut() {
        //     let v = *gray.at_2d::<u8>(y as i32, x as i32).unwrap();

        //     *dst = v.into();
        // }

        tp.stamp("bufer_copy");

        let detections = match detector.detect(&im) {
            Ok(dets) => dets,
            Err(e) => {
                eprintln!("Detection error: {e:?}");
                continue;
            }
        };

        tp.stamp("detect");

        // Draw detection outlines
        for det in detections.detections {
            // println!("dm={}", det.decision_margin);
            if det.decision_margin < 32. {
                continue;
            }
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

        tp.stamp("draw");

        imshow(&"Tag Detections", &frame).unwrap();
        tp.stamp("imshow");
        if wait_key(1).unwrap() >= 0 {
            break;
        }
        println!("{tp}");

        println!("{}", detections.tp);
    }
}
