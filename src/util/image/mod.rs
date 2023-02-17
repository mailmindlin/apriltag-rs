mod pnm;

use std::{io::{self, Write}, fs::OpenOptions, ops::{IndexMut, Index}, path::Path};

use self::pnm::PNM;

use super::{mem::{calloc, SafeZero}, geom::Point2D};

#[derive(Clone)]
pub struct Image<T = u8> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub buf: Box<[T]>,
}

impl<T> Image<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.width * self.height
    }
    pub fn darken(&mut self) {
        todo!();
    }

    #[inline(always)]
    pub fn cell(&self, x: usize, y: usize) -> &T {
        return &self.buf[x + y * self.stride];
    }

    #[inline(always)]
    pub fn cell_mut(&mut self, x: usize, y: usize) -> &mut T {
        return &mut self.buf[x + y * self.stride];
    }

    pub fn draw_line(&mut self, p0: Point2D, p1: Point2D, color: &T, width: usize) {
        let dist = p0.distance_to(&p1);
        let delta = 0.5 / dist;
        let num_steps = f64::ceil(dist * 2.) as usize;
    
        // terrible line drawing code
        for i in 0..num_steps {
            let f = (i as f64) * delta;
            let c = &p1 + &((&p0 - &p1) * f);
            let x = c.x() as isize;
            let y = c.y() as isize;
    
            if x < 0 || y < 0 {
                continue;
            }

            let x = x as usize;
            let y = y as usize;
            if x >= self.width || y >= self.height {
                continue;
            }
    
            let idx = (y as usize)*self.stride + (x as usize);
            self.buf[idx + i] = *color;
        }
    }
}

impl<T: SafeZero> Image<T> {
    pub fn create_alignment(width: usize, height: usize, alignment: usize) -> Self {
        let mut stride = width;

        if (stride % alignment) != 0 {
            stride += alignment - (stride % alignment);
        }

        Self::create_stride(width, height, stride)
    }

    pub fn create_stride(width: usize, height: usize, stride: usize) -> Self {
        let buf = calloc::<T>(height*stride);

        Self {
            width,
            height,
            stride,
            buf,
        }
    }
}

impl<T> Index<(usize, usize)> for Image<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (x, y) = index;

        assert!(x < self.width);
        assert!(y < self.height);

        &self.buf[x + (y * self.width)]
    }
}

impl<T> IndexMut<(usize, usize)> for Image<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (x, y) = index;

        assert!(x < self.width);
        assert!(y < self.height);

        &mut self.buf[x + (y * self.width)]
    }
}

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
        let mut acc = 0;

        for j in 0..k.len() {
            acc += k[j] * x[i + j];
        }

        y[k.len()/2 + i] = acc >> 8;
    }

    // Copy right
    {
        let right_start = x.len() - k.len() + k.len()/2;
        y[right_start..].copy_from_slice(&x[right_start..])
    }
}

impl Image<u8> {
    /// least common multiple of 64 (sandy bridge cache line) and 24 (stride
    /// needed for RGB in 8-wide vector processing)
    const DEFAULT_ALIGNMENT: usize = 96;
    pub fn create(width: usize, height: usize) -> Self {
        Self::create_alignment(width, height, Self::DEFAULT_ALIGNMENT)
    }

    pub fn decimate(&self, ffactor: f32) -> Self {
        let width = self.width;
        let height = self.height;

        if ffactor == 1.5 {
            let swidth = width / 3 * 2;
            let sheight = height / 3 * 2;

            let dst = Self::create(swidth, sheight);

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

            let decim = Self::create(swidth, sheight);
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
                let start = y * self.stride;
                let row = &mut self.buf[start..(start + self.width)];
    
                row_buf.copy_from_slice(row);
    
                convolve(&row_buf, &mut row, kernel);
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

    pub fn gaussian_blur_mut(&mut self, sigma: f64, kernel_size: usize) {
        if sigma == 0. {
            return;
        }

        assert_eq!(kernel_size % 1, 1, "kernel_size must be odd");

        // build the kernel.
        let kernel = {
            let dk = vec![0f64; kernel_size];

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

    pub fn write_pnm(&self, outfile: &Path) -> io::Result<()> {
        let f = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(outfile)?;
        
        // Only outputs to RGB
        write!(f, "P5\n{} {}\n255\n", self.width, self.height);
        for line in self.buf.chunks(self.stride) {
            f.write_all(&line[0..self.width])?;
        }
        
        Ok(())
    }

    pub fn write_postscript(&self, f: &mut impl Write) -> io::Result<()> {
        writeln!(f, "/picstr {} string def", self.width)?;

        writeln!(f, "{} {} 8 [1 0 0 1 0 0]", self.width, self.height)?;

        f.write("{currentfile picstr readhexstring pop}\nimage\n".as_bytes())?;

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

impl Image<[u8; 3]> {
    /// Least common multiple of 64 (sandy bridge cache line) and 48 (stride needed
    /// for 16byte-wide RGB processing). (It's possible that 48 would be enough).
    const DEFAULT_ALIGNMENT: usize = 192;
    pub fn create(width: usize, height: usize) -> Self {
        Self::create_alignment(width, height, Self::DEFAULT_ALIGNMENT)
    }

    // Create an RGB image from PNM
    pub fn create_from_pnm(path: &Path) -> io::Result<Self> {
        let pnm = PNM::create_from_file(path)?;
        match pnm.format {
            pnm::PNMFormat::Binary => todo!("Support binary files"),
            pnm::PNMFormat::Gray => {
                let im = Self::create(pnm.width, pnm.height);

                for y in 0..im.height {
                    for x in 0..im.width {
                        let gray = pnm.buf[y*im.width + x];
                        im[(x, y)] = [gray; 3];
                    }
                }
                Ok(im)
            },
            pnm::PNMFormat::RGB => {
                let im = Self::create(pnm.width, pnm.height);

                for y in 0..im.height {
                    for x in 0..im.width {
                        let r = pnm.buf[y*im.width*3 + 3*x];
                        let g = pnm.buf[y*im.width*3 + 3*x+1];
                        let b = pnm.buf[y*im.width*3 + 3*x+2];

                        im[(x, y)] = [r, g, b];
                    }
                }
                Ok(im)
            },
        }
    }

    pub fn write_pnm(&self, outfile: &Path) -> io::Result<()> {
        let mut f = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(outfile)?;
        
        // Only outputs to RGB
        write!(f, "P6\n{} {}\n255\n", self.width, self.height);
        let linesz = self.width * 3;
        for y in 0..self.height {
            todo!();
            /*if (linesz != fwrite(&self.buf[y*self.stride], 1, linesz, f)) {
                res = -1;
                
            }*/
        }

        Ok(())
    }
}