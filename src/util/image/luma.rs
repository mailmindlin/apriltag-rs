use std::{io::{self, Write}, path::Path, ops::{DerefMut, Deref}};

use crate::util::mem::{calloc, SafeZero};

use super::{ImageBuffer, pnm::PNM, pixel::{Primitive, Pixel, PixelConvert, DefaultAlignment}, Rgb, ImageWritePNM, ImageWritePostscript, SubpixelArray};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Luma<T>(pub [T; 1]);

impl From<u8> for Luma<u8> {
    fn from(value: u8) -> Self {
        Self([value])
    }
}

impl<T: Primitive> Pixel for Luma<T> {
    type Subpixel = T;
    type Value = T;

    const CHANNEL_COUNT: usize = 1;

    fn channels(&self) -> &[Self::Subpixel] {
        &self.0
    }

    fn channels_mut(&mut self) -> &mut [Self::Subpixel] {
        &mut self.0
    }

    fn to_value(self) -> Self::Value {
        self.0[0]
    }

    #[inline(always)]
    fn from_slice<'a>(slice: &'a [Self::Subpixel]) -> &'a Self {
        assert_eq!(slice.len(), Self::CHANNEL_COUNT);
        unsafe { &*(slice.as_ptr() as *const Luma<T>) }
    }

    #[inline(always)]
    fn slice_to_value<'a>(slice: &'a [Self::Subpixel]) -> &'a Self::Value {
        &slice[0]
    }

    #[inline(always)]
    fn from_slice_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self {
        assert_eq!(slice.len(), Self::CHANNEL_COUNT);
        unsafe { &mut *(slice.as_mut_ptr() as *mut Luma<T>) }
    }

    #[inline(always)]
    fn slice_to_value_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self::Value {
        &mut slice[0]
    }
}

impl<T: Primitive> PixelConvert for Luma<T> {
    fn to_rgb(&self) -> Rgb<Self::Subpixel> {
        let v = self.0[0];
        Rgb([v; 3])
    }

    #[inline(always)]
    fn to_luma(&self) -> Luma<Self::Subpixel> {
        *self
    }
}

impl DefaultAlignment for Luma<u8> {
    /// least common multiple of 64 (sandy bridge cache line) and 24 (stride
    /// needed for RGB in 8-wide vector processing)
    const DEFAULT_ALIGNMENT: usize = 96;
}

impl<T: SafeZero> SafeZero for Luma<T> {}

/// 1-d convolution
fn convolve(x: &[u8], y: &mut [u8], k: &[u8], need_copy: bool) {
    assert_eq!(k.len() % 1, 1, "Kernel size must be odd");
    assert_eq!(x.len(), y.len());

    // Copy left
    if need_copy {
        let left_end = std::cmp::min(k.len() / 2, x.len());
        y[..left_end].copy_from_slice(&x[..left_end]);
    }

    // Convolve middle
    for i in 0..(x.len() - k.len()) {
        let mut acc = 0u32;

        for j in 0..k.len() {
            acc += k[j] as u32 * x[i + j] as u32;
        }

        y[k.len()/2 + i] = (acc >> 8).clamp(0, 255) as u8;
    }

    // Copy right
    if need_copy {
        let right_start = x.len() - k.len() + k.len()/2;
        y[right_start..].copy_from_slice(&x[right_start..])
    }
}

/// Grayscale image
impl ImageBuffer<Luma<u8>, Box<SubpixelArray<Luma<u8>>>> {
    pub fn create_from_pnm(path: &Path) -> io::Result<Self> {
        Self::create_from_pnm_alignment(path, Luma::<u8>::DEFAULT_ALIGNMENT)
    }

    pub fn create_from_pnm_alignment(path: &Path, alignment: usize) -> io::Result<Self> {
        let pnm = PNM::create_from_file(path)?;
        let width = pnm.width;
        let mut im = Self::zeroed_with_alignment(pnm.width, pnm.height, alignment);

        match pnm.format {
            super::pnm::PNMFormat::Gray => {
                match pnm.max {
                    255 => {
                        for (y, mut row) in im.rows_mut() {
                            let src = &pnm.buf[y*width..(y+1)*width];
                            row.as_slice_mut().copy_from_slice(src);
                        }
                    },
                    65535 => {
                        for ((x, y), dst) in im.enumerate_pixels_mut() {
                            *dst = pnm.buf[2*(y*width + x)].into();
                        }
                    },
                    //TODO: return error
                    _ => panic!(),
                }
            }
            super::pnm::PNMFormat::RGB => {
                match pnm.max {
                    255 => {
                        // Gray conversion for RGB is gray = (r + g + g + b)/4
                        for ((x, y), dst) in im.enumerate_pixels_mut() {
                            let r = pnm.buf[y*width*3 + 3*x+0] as u16;
                            let g = pnm.buf[y*width*3 + 3*x+1] as u16;
                            let b = pnm.buf[y*width*3 + 3*x+2] as u16;

                            let gray = (r + g + g + b) / 4;
                            *dst = (gray.clamp(0, 255) as u8).into();
                        }
                    },
                    65535 => {
                        for ((x, y), dst) in im.enumerate_pixels_mut() {
                            let r = pnm.buf[6*(y*width + x) + 0];
                            let g = pnm.buf[6*(y*width + x) + 2];
                            let b = pnm.buf[6*(y*width + x) + 4];

                            let gray = (r + g + g + b) / 4;
                            *dst = gray.into();
                        }
                    },
                    //TODO: return error
                    _ => panic!(),
                }
            }
            super::pnm::PNMFormat::Binary => {
                // image is padded to be whole bytes on each row.

                // how many bytes per row on the input?
                let pbmstride = (im.width() + 7) / 8;

                for ((x, y), dst) in im.enumerate_pixels_mut() {
                    let byteidx = y * pbmstride + x / 8;
                    let bitidx = 7 - (x & 7);

                    // ack, black is one according to pbm docs!
                    let value = if ((pnm.buf[byteidx] >> bitidx) & 1) != 0 { 0 } else { 255 };
                    *dst = value.into();
                }
            }
        }
        Ok(im)
    }

    /// Downsample the image by a factor of exactly 1.5
    pub fn decimate_three_halves(&self) -> Self {
        let swidth = self.width() / 3 * 2;
        let sheight = self.height() / 3 * 2;

        let mut dst = Self::zeroed(swidth, sheight);

        let mut y = 0;
        for sy in (0..sheight).step_by(2) {
            let mut x = 0;
            for sx in (0..swidth).step_by(2) {
                // a b c
                // d e f
                // g h i
                let a = self[(x+0, y+0)];
                let b = self[(x+1, y+0)];
                let c = self[(x+2, y+0)];

                let d = self[(x+0, y+1)];
                let e = self[(x+1, y+1)];
                let f = self[(x+2, y+1)];

                let g = self[(x+0, y+2)];
                let h = self[(x+1, y+2)];
                let i = self[(x+2, y+2)];

                dst[(sx+0, sy+0)] = (4*a+2*b+2*d+e)/9;
                dst[(sx+1, sy+0)] = (4*c+2*b+2*f+e)/9;
                dst[(sx+0, sy+1)] = (4*g+2*d+2*h+e)/9;
                dst[(sx+1, sy+1)] = (4*i+2*f+2*h+e)/9;
                    
                x += 3;
            }
            y += 3;
        }
        dst
    }

    pub fn decimate(&self, ffactor: f32) -> Self {
        if ffactor == 1.5 {
            return self.decimate_three_halves();
        }

        let width = self.width();
        let height = self.height();
        
        let factor = ffactor.round() as usize;

        let swidth = 1 + (width - 1) / factor;
        let sheight = 1 + (height - 1) / factor;

        let mut decim = Self::zeroed(swidth, sheight);
        let mut dy = 0;
        for y in (0..height).step_by(factor) {
            let mut dr = decim.row_mut(dy);
            let sr = self.row(y);
            let mut dx = 0;
            for x in (0..width).step_by(factor) {
                dr[dx] = sr[x];
                dx += 1;
            }
            dy += 1;
        }
        decim
    }
}

impl<Container: DerefMut<Target=SubpixelArray<Luma<u8>>>> ImageBuffer<Luma<u8>, Container> {
    pub fn darken(&mut self) {
        self.apply(|pixel| {
            pixel.0[0] /= 2;
        });
    }

    pub fn convolve2d_mut(&mut self, kernel: &[u8]) {
        assert_eq!(kernel.len() % 1, 1, "Kernel size must be odd");

        // Convolve horizontally
        {
            // Allocate this buffer once
            let mut row_buf = calloc::<u8>(self.width());
    
            for (_, mut row) in self.rows_mut() {
                row_buf.copy_from_slice(row.as_slice());
                convolve(&row_buf, row.as_slice_mut(), kernel, false);
            }
        }

        // Convolve vertically
        {
            let mut xb = calloc::<u8>(self.height());
            let mut yb = calloc::<u8>(self.height());

            for x in 0..self.width() {
                //TODO: we can optimize this loop
                for y in 0..self.height() {
                    xb[y] = self[(x, y)];
                }

                convolve(&xb, &mut yb, kernel, true);

                //TODO: we can optimize this loop
                for y in 0..self.height() {
                    self[(x, y)] = yb[y];
                }
            }
        }
    }

    pub fn gaussian_blur(&mut self, sigma: f64, kernel_size: usize) {
        if sigma == 0. {
            return;
        }

        assert_eq!(kernel_size % 1, 1, "kernel_size must be odd");

        // build the kernel.
        let kernel = {
            let mut dk = vec![0f64; kernel_size];

            // for kernel of length 5:
            // dk[0] = f(-2), dk[1] = f(-1), dk[2] = f(0), dk[3] = f(1), dk[4] = f(2)
            for i in 0..kernel_size {
                let x = i as isize - (kernel_size as isize / 2);
                let x_sig = x as f64 / sigma;
                let v = f64::exp(-0.5*(x_sig * x_sig));
                dk[i] = v;
            }

            // normalize
            let acc = dk.iter().sum::<f64>();

            dk.iter()
                .map(|x| {
                    let x_norm = x / acc;
                    (x_norm * 255.) as u8 //TODO: round?
                })
                .collect::<Vec<_>>()
        };

        /*if false {
            for (int i = 0; i < ksz; i++)
                printf("%d %15f %5d\n", i, dk[i], k[i]);
        } */

        self.convolve2d_mut(&kernel);
    }
}

impl<C: Deref<Target=[u8]>> ImageWritePNM for ImageBuffer<Luma<u8>, C> {
    fn write_pnm(&self, f: &mut impl io::Write) -> io::Result<()> {
        // Only outputs to RGB
        writeln!(f, "P5")?;
        let dims = self.dimensions();
        writeln!(f, "{} {}", dims.width, dims.height)?;
        writeln!(f, "255")?;
        for (_, row) in self.rows() {
            f.write_all(row.as_slice())?;
        }

        Ok(())
    }
}

impl ImageWritePostscript for ImageBuffer<Luma<u8>> {
    fn write_postscript(&self, f: &mut super::PostScriptWriter<impl io::Write>) -> io::Result<()> {
        writeln!(f, "/picstr {} string def", self.width())?;

        writeln!(f, "{} {} 8 [1 0 0 1 0 0]", self.width(), self.height())?;
        writeln!(f, "{{currentfile picstr readhexstring pop}}")?;
        writeln!(f, "image")?;

        for ((x, _y), v) in self.enumerate_pixels() {
            write!(f, "{:02x}", v.to_value())?;
            if (x % 32) == 31 {
                writeln!(f)?;
            }
        }

        writeln!(f)?;
        Ok(())
    }
}