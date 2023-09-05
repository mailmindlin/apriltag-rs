use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use apriltag_rs::util::image::ImageWritePNM;
use apriltag_rs::util::image::Pixel;
use clap::Parser;
use clap::command;
use image::Rgb;
use image::{io::Reader as ImageReader, ImageBuffer as IImageBuffer};
use apriltag_rs::util::ImageRGB8;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    input_file: PathBuf,
    compare_file: Option<PathBuf>,
}

fn load_image(path: &Path) -> io::Result<ImageRGB8> {
    if let Some(extension) = path.extension() {
        if extension.eq_ignore_ascii_case("pnm") {
            return ImageRGB8::create_from_pnm(path);
        }
    }
    
    let reader = ImageReader::open(path)?;
    let image = reader.decode().unwrap().into_rgb8();

    let mut result = ImageRGB8::new(image.width() as usize, image.height() as usize);
    for (x, y, value) in image.enumerate_pixels() {
        result[(x as usize, y as usize)] = value.0;
    }
    Ok(result)
}

#[cfg(feature="dep:opencv")]
fn show_opencv(img: ImageRGB8) {
    use opencv::prelude::*;
    use opencv::core::*;
    use opencv::highgui::{imshow, wait_key};
    let mut mat = Mat::zeros(img.height() as i32, img.width() as i32, CV_8UC3 as i32).unwrap()
    .to_mat()
    .unwrap();

    for ((x, y), value) in img.enumerate_pixels() {
        let dst = mat.at_2d_mut::<Vec3b>(y as i32, x as i32).unwrap();
        dst.0 = value.to_value().into();
    }


    imshow(&args.input_file.display().to_string(), &mat).unwrap();
    wait_key(0).unwrap();
}

fn main() {
    let args = Args::parse();

    let mut img = load_image(&args.input_file).unwrap();
    println!("Loaded image size: {}x{}", img.width(), img.height());

    if let Some(path2) = args.compare_file {
        // Compare file
        let img2 = load_image(&path2).unwrap();
        assert_eq!(img.width(), img2.width());
        assert_eq!(img.height(), img2.height());
        let mut different = false;
        for ((x, y), px2) in img2.enumerate_pixels() {
            let [r1, g1, b1] = img[(x, y)].to_value();
            let [r2, g2, b2] = px2.to_value();
            let rgb = [r1.abs_diff(r2), g1.abs_diff(g2), b1.abs_diff(b2)];
            different |= rgb != [0,0,0];
            let rgb = if rgb != [0,0,0] {
                println!("Different at {:?}", (x, y));
                // rgb[0] = 255;
                let s = ((r2 as i16) - (r1 as i16)) + ((g2 as i16) - (g1 as i16)) + ((b2 as i16) - (b1 as i16));
                if s > 0 {
                    [r1, 0, 0]
                } else {
                    [0, g2, 0]
                }
            } else {
                // [r1, g2, 0]
                [0,0,0]
            };
            img[(x, y)] = rgb;
        }
        println!("Different: {different}");
    } else if let Some(extension) = args.input_file.extension() {
        // Save converted file
        if extension.eq_ignore_ascii_case(&"pnm") {
            let mut buf = IImageBuffer::<Rgb<u8>, _>::new(img.width() as u32, img.height() as u32);
            for ((x, y), value) in img.enumerate_pixels() {
                *buf.get_pixel_mut(x as u32, y as u32) =  Rgb(value.0);
            }
            buf.save_with_format(args.input_file.with_extension("png"), image::ImageFormat::Png).unwrap();
        } else {
            let dst = args.input_file.with_extension("pnm");
            let mut f = File::create(dst).unwrap();
            img.write_pnm(&mut f).unwrap();
        }
    }

    #[cfg(feature="dep:opencv")]
    show_opencv(img);
}