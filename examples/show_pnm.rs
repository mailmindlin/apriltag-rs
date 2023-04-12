use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use opencv::prelude::*;
use opencv::core::*;
use opencv::highgui::{imshow, wait_key};
use apriltag_rs::Image;
use image::io::Reader as ImageReader;
use apriltag_rs::util::ImageWritePNM;

fn load_image(path: &Path) -> io::Result<Image<[u8; 3]>> {
    if let Some(extension) = path.extension() {
        if extension.eq_ignore_ascii_case("pnm") {
            return Image::<[u8; 3]>::create_from_pnm(path);
        }
    }

    
    let reader = ImageReader::open(path)?;
    let image = reader.decode().unwrap().into_rgb8();

    let mut result = Image::<[u8; 3]>::create(image.width() as usize, image.height() as usize);
    for (x, y, value) in image.enumerate_pixels() {
        result[(x as usize, y as usize)] = value.0;
    }
    Ok(result)
}

fn main() {
    let path = PathBuf::from(std::env::args().skip(1).next().unwrap());

    let img = load_image(&path).unwrap();

    let mut mat = Mat::zeros(img.height as i32, img.width as i32, CV_8UC3 as i32).unwrap()
        .to_mat()
        .unwrap();

    for y in 0..img.height {
        for x in 0..img.width {
            let value = img[(x, y)];
            let dst = mat.at_2d_mut::<Vec3b>(y as i32, x as i32).unwrap();
            *dst = value.into();
        }
    }

    if let Some(extension) = path.extension() {
        if extension.eq_ignore_ascii_case(&"pnm") {
            //TODO
        } else {
            let dst = path.with_extension("pnm");
            let mut f = File::create(dst).unwrap();
            img.write_pnm(&mut f).unwrap();
        }
    }
    

    imshow(&"image", &mat).unwrap();
    wait_key(0).unwrap();
}