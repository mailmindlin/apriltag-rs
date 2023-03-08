mod pnm;
pub mod gray;
pub mod rgb;
mod ps;

use std::{ops::{IndexMut, Index, RangeFull}, mem::MaybeUninit, path::Path, io, fs::OpenOptions};
pub use ps::PostScriptWriter;
use self::pnm::PNM;

use super::{mem::{calloc, SafeZero}, geom::Point2D};

#[derive(Clone)]
pub struct Image<T = u8> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub buf: Box<[T]>,
}

impl<T> Image<MaybeUninit<T>> {
    #[inline]
    pub unsafe fn assume_init(self) -> Image<T> {
        Image {
            width: self.width,
            height: self.height,
            stride: self.stride,
            buf: self.buf.assume_init()
        }
    }
}

impl<T> Image<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    #[inline(always)]
    pub fn cell(&self, x: usize, y: usize) -> &T {
        return &self.buf[x + y * self.stride];
    }

    #[inline(always)]
    pub fn cell_mut(&mut self, x: usize, y: usize) -> &mut T {
        return &mut self.buf[x + y * self.stride];
    }

    pub fn draw_line(&mut self, p0: Point2D, p1: Point2D, color: &T, width: usize) where T: Copy {
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

            self[(x, y)] = *color;
            if width > 1 {
                if x + 1 < self.width {
                    self[(x + 1, y)] = *color;
                }
                if y + 1 < self.height {
                    self[(x, y + 1)] = *color;
                }
                if x + 1 < self.width && y + 1 < self.height {
                    self[(x + 1, y + 1)] = *color;
                }
            }
        }
    }
}

/// calloc-based constructors
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

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);

        &self.buf[x + (y * self.stride)]
    }
}

impl<T> IndexMut<(usize, usize)> for Image<T> {
    fn index_mut(&mut self, (x,y): (usize, usize)) -> &mut Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);

        &mut self.buf[x + (y * self.stride)]
    }
}

impl<T> Index<(RangeFull, usize)> for Image<T> {
    type Output = [T];
    /// Get row
    fn index(&self, (_, y): (RangeFull, usize)) -> &Self::Output {
        let start_idx = 0 + (y * self.stride);
        let end_idx = self.width + start_idx;
        &self.buf[start_idx..end_idx]
    }
}

impl<T> IndexMut<(RangeFull, usize)> for Image<T> {
    /// Get row
    fn index_mut(&mut self, (_, y): (RangeFull, usize)) -> &mut Self::Output {
        let start_idx = 0 + (y * self.stride);
        let end_idx = self.width + start_idx;
        &mut self.buf[start_idx..end_idx]
    }
}

pub trait ImageWritePNM {
    /// Save to PNM file
    fn save_to_pnm(&self, outfile: impl AsRef<Path>) -> io::Result<()> {
        let mut f = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(outfile)?;
        
        self.write_pnm(&mut f)
    }

    /// Write PNM data
    fn write_pnm(&self, f: &mut impl io::Write) -> io::Result<()>;
}

pub trait ImageWritePostscript {
    /// Write PostScript data
    fn write_postscript(&self, f: &mut PostScriptWriter<impl io::Write>) -> io::Result<()>;
}