use std::ops::Deref;
use std::{path::Path, io};

use crate::util::mem::SafeZero;

use super::pixel::PixelConvert;
use super::{ImageBuffer, SubpixelArray};

use super::{PNM, pnm::PNMFormat, ImageWritePNM, Luma, pixel::{Primitive, Pixel}};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Rgb<T>(pub [T; 3]);

impl<T: Primitive> Pixel for Rgb<T> {
    type Subpixel = T;
    type Value = [T; 3];

    const CHANNEL_COUNT: usize = 3;

    fn channels(&self) -> &[Self::Subpixel] {
        &self.0
    }

    fn channels_mut(&mut self) -> &mut [Self::Subpixel] {
        &mut self.0
    }

    fn to_value(self) -> Self::Value {
        self.0
    }

    fn from_slice<'a>(slice: &'a [Self::Subpixel]) -> &'a Self {
        assert_eq!(slice.len(), Self::CHANNEL_COUNT);
        unsafe { &*(slice.as_ptr() as *const Rgb<T>) }
    }

    fn slice_to_value<'a>(slice: &'a [Self::Subpixel]) -> &'a Self::Value {
        slice.try_into().unwrap()
    }

    fn from_slice_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self {
        assert_eq!(slice.len(), Self::CHANNEL_COUNT);
        unsafe { &mut *(slice.as_mut_ptr() as *mut Rgb<T>) }
    }

    fn slice_to_value_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self::Value {
        slice.try_into().unwrap()
    }
}

impl<T: Primitive> PixelConvert for Rgb<T> {
    fn to_rgb(&self) -> Rgb<Self::Subpixel> {
        *self
    }

    fn to_luma(&self) -> Luma<Self::Subpixel> {
        todo!()
    }
}

impl<T: SafeZero> SafeZero for Rgb<T> {}

/*impl<P: Pixel> From<Image<P>> for Image<Rgb<P::Subpixel>> {
    fn from(src: Image<P>) -> Self {
        let mut dst = Image::<[u8; 3]>::create(src.width, src.height);
        for y in 0..src.height {
            for x in 0..src.width {
                let value = src[(x, y)];
                dst[(x, y)] = value.to_rgb();
            }
        }
        dst
    }
}*/

impl<T: Primitive> ImageBuffer<Rgb<T>> {
    /// Least common multiple of 64 (sandy bridge cache line) and 48 (stride needed
    /// for 16byte-wide RGB processing). (It's possible that 48 would be enough).
    const DEFAULT_ALIGNMENT: usize = 192;
    pub fn create(width: usize, height: usize) -> Self where T: SafeZero {
        Self::zeroed_with_alignment(width, height, Self::DEFAULT_ALIGNMENT)
    }
}

impl ImageBuffer<Rgb<u8>, Box<[u8]>> {
    // Create an RGB image from PNM
    pub fn create_from_pnm(path: &Path) -> io::Result<Self> {
        let pnm = PNM::create_from_file(path)?;
        match pnm.format {
            PNMFormat::Binary => todo!("Support binary files"),
            PNMFormat::Gray => {
                let mut im = Self::create(pnm.width, pnm.height);

                let mut max_x = 0;
                let mut max_y = 0;
                for ((x, y), dst) in im.enumerate_pixels_mut() {
                    let gray = pnm.buf[y*pnm.width + x];
                    *dst = Rgb([gray; 3]);
                    max_x = std::cmp::max(max_x, x);
                    max_y = std::cmp::max(max_y, y);
                }
                dbg!(max_x);
                dbg!(max_y);
                Ok(im)
            },
            PNMFormat::RGB => {
                let mut im = Self::create(pnm.width, pnm.height);
                let width = pnm.width;

                for ((x, y), dst) in im.enumerate_pixels_mut() {
                    let r = pnm.buf[y*width*3 + 3*x];
                    let g = pnm.buf[y*width*3 + 3*x+1];
                    let b = pnm.buf[y*width*3 + 3*x+2];

                    *dst = Rgb([r, g, b]);
                }
                Ok(im)
            },
        }
    }
}

impl<Container: Deref<Target=SubpixelArray<Rgb<u8>>>> ImageWritePNM for ImageBuffer<Rgb<u8>, Container> {
    fn write_pnm(&self, f: &mut impl io::Write) -> io::Result<()> {
        // Only outputs to RGB
        writeln!(f, "P6")?;
        writeln!(f, "{} {}", self.width(), self.height())?;
        writeln!(f, "255")?;
        for (_, row) in self.rows() {
            f.write_all(row.as_slice())?;
        }

        Ok(())
    }
}