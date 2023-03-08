use std::{io::{self, Write}};

use crate::util::mem::calloc;

use super::{Image, ImageWritePNM, ImageWritePostscript};

/// 1-d convolution
fn convolve(x: &[u8], y: &mut [u8], k: &[u8]) {
    assert_eq!(k.len() % 1, 1, "Kernel size must be odd");
    assert_eq!(x.len(), y.len());

    // Copy left
    for i in 0..std::cmp::min(k.len() / 2, x.len()) {
        y[i] = x[i];
    }

    // Convolve middle
    for i in 0..(x.len() - k.len()) {
        let mut acc = 0u32;

        for j in 0..k.len() {
            acc += k[j] as u32 * x[i + j] as u32;
        }

        y[k.len()/2 + i] = (acc >> 8) as u8;
    }

    // Copy right
    {
        let right_start = x.len() - k.len() + k.len()/2;
        y[right_start..].copy_from_slice(&x[right_start..])
    }
}

/// Grayscale image
impl Image<u8> {
    /// least common multiple of 64 (sandy bridge cache line) and 24 (stride
    /// needed for RGB in 8-wide vector processing)
    const DEFAULT_ALIGNMENT: usize = 96;

    /// Create new grayscale image of dimensions
    pub fn create(width: usize, height: usize) -> Self {
        Self::create_alignment(width, height, Self::DEFAULT_ALIGNMENT)
    }

    pub fn darken(&mut self) {
        if self.stride == self.width {
            for pixel in self.buf.iter_mut() {
                *pixel /= 2;
            }
        } else {
            //TODO: efficient iteration
            for y in 0..self.height {
                for x in 0..self.width {
                    self[(x, y)] /= 2;
                }
            }
        }
    }

    pub fn decimate(&self, ffactor: f32) -> Self {
        let width = self.width;
        let height = self.height;

        if ffactor == 1.5 {
            let swidth = width / 3 * 2;
            let sheight = height / 3 * 2;

            let mut dst = Self::create(swidth, sheight);

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
        } else {
            let factor = ffactor as usize;

            let swidth = 1 + (width - 1) / factor;
            let sheight = 1 + (height - 1) / factor;

            let mut decim = Self::create(swidth, sheight);
            let mut sy = 0;
            for y in (0..height).step_by(factor) {
                let mut sx = 0;
                for x in (0..width).step_by(factor) {
                    decim[(sx, sy)] = self[(x, y)];
                    sx += 1;
                }
                sy += 1;
            }
            decim
        }
    }

    pub fn convolve2d_mut(&mut self, kernel: &[u8]) {
        assert_eq!(kernel.len() % 1, 1, "Kernel size must be odd");

        // Convolve horizontally
        {
            // Allocate this buffer once
            let mut row_buf = calloc::<u8>(self.width);
    
            for y in 0..self.height {
                let row = &mut self[(.., y)];
                row_buf.copy_from_slice(row);
                convolve(&row_buf, row, kernel);
            }
        }

        // Convolve vertically
        {
            let mut xb = calloc::<u8>(self.height);
            let mut yb = calloc::<u8>(self.height);

            for x in 0..self.width {
                //TODO: we can optimize this loop
                for y in 0..self.height {
                    xb[y] = self[(x, y)];
                }

                convolve(&xb, &mut yb, kernel);

                //TODO: we can optimize this loop
                for y in 0..self.height {
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

impl ImageWritePNM for Image<u8> {
    fn write_pnm(&self, f: &mut impl io::Write) -> io::Result<()> {
        // Only outputs to RGB
        writeln!(f, "P5")?;
        writeln!(f, "{} {}", self.width, self.height)?;
        writeln!(f, "255")?;
        for y in 0..self.height {
            f.write_all(&self[(.., y)])?;
        }

        Ok(())
    }
}

impl ImageWritePostscript for Image<u8> {
    fn write_postscript(&self, f: &mut super::PostScriptWriter<impl io::Write>) -> io::Result<()> {
        writeln!(f, "/picstr {} string def", self.width)?;

        writeln!(f, "{} {} 8 [1 0 0 1 0 0]", self.width, self.height)?;
        writeln!(f, "{{currentfile picstr readhexstring pop}}")?;
        writeln!(f, "image")?;

        for y in 0..self.height {
            for x in 0..self.width {
                let v = self[(x, y)];
                write!(f, "{:02x}", v)?;
                if (x % 32) == 31 {
                    writeln!(f)?;
                }
            }
        }

        writeln!(f)?;
        Ok(())
    }
}

impl From<Image<f32>> for Image<u8> {
    fn from(value: Image<f32>) -> Self {
        let mut res = Self::create(value.width, value.height);
        for y in 0..value.height {
            for x in 0..value.width {
                res[(x, y)] = (255. * value[(x, y)]).round() as u8;
            }
        }
        res
    }
}