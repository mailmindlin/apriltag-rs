use std::{sync::Arc, borrow::Cow};

use crate::util::image::{ImageBuffer, Luma, ImageY8};


mod tag16h5;
mod tag25h9;
mod tag36h10;
mod tag36h11;
mod util;
pub use util::{rotate90, rotations};

pub use tag16h5::tag16h5_create;
pub use tag25h9::tag25h9_create;
pub use tag36h10::tag36h10_create;
pub use tag36h11::tag36h11_create;

pub(crate) type Code = u64;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
	Identity,
	Deg90,
	Deg180,
	Deg270,
}

impl Rotation {
    pub const fn values() -> [Rotation; 4] {
        [Self::Identity, Self::Deg90, Self::Deg180, Self::Deg270]
    }

    #[inline(always)]
    pub const fn count(&self) -> usize {
        match self {
            Rotation::Identity => 0,
            Rotation::Deg90 => 1,
            Rotation::Deg180 => 2,
            Rotation::Deg270 => 3,
        }
    }

    #[inline]
    pub fn theta(&self) -> f64 {
        self.count() as f64 * core::f64::consts::FRAC_PI_2
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub struct AprilTagFamily {
	/// The codes in the family.
	pub codes: Vec<u64>,

    pub bits: Vec<(u32, u32)>,

    pub width_at_border: u32,

    pub total_width: u32,

    pub reversed_border: bool,

	/// minimum hamming distance between any two codes. (e.g. 36h11 => 11)
	pub min_hamming: u32,

	// a human-readable name, e.g., "tag36h11"
	pub name: Cow<'static, str>,

	// some detector implementations may preprocess codes in order to
	// accelerate decoding.  They put their data here. (Do not use the
	// same apriltag_family instance in more than one implementation)
	// _impl: u32,
}

impl AprilTagFamily {
    pub fn for_name(name: &str) -> Option<Arc<AprilTagFamily>> {
        let res = match name {
            "tag16h5" => tag16h5_create(),
            "tag25h9" => tag25h9_create(),
            "tag36h10" => tag36h10_create(),
            "tag36h11" => tag36h11_create(),
            _ => return None,
        };
        //TODO: cache references
        Some(Arc::new(res))
    }

    pub fn names() -> impl IntoIterator<Item = &'static str> {
        vec!["tag16h5", "tag25h9", "tag36h10", "tag36h11"]
    }

    pub(crate) fn split_bits(&self) -> (Vec<u32>, Vec<u32>) {
        let mut bit_x = Vec::with_capacity(self.bits.len());
        let mut bit_y = Vec::with_capacity(self.bits.len());
        for (x, y) in self.bits.iter() {
            bit_x.push(*x);
            bit_y.push(*y);
        }
        (bit_x, bit_y)
    }
    fn similar_to(&self, other: &Self) -> bool {
        self.bits == other.bits
            && (self.width_at_border == other.width_at_border)
            && (self.total_width == other.total_width)
            && (self.reversed_border == other.reversed_border)
            && (self.min_hamming == other.min_hamming) //TODO: we might not care
    }

    #[cfg(all(test, feature="compare_reference"))]
    fn assert_similar(&self, ts: &apriltag_sys::apriltag_family) {
        assert_eq!(self.width_at_border, ts.width_at_border as u32);
        assert_eq!(self.total_width, ts.total_width as u32);
        assert_eq!(self.reversed_border, ts.reversed_border);
        assert_eq!(self.min_hamming, ts.h);
        assert_eq!(self.codes.len(), ts.ncodes as usize);
        assert_eq!(&self.codes, unsafe { std::slice::from_raw_parts(ts.codes, ts.ncodes as usize) });
        let (bit_x, bit_y) = self.split_bits();
        assert_eq!(&bit_x, unsafe { std::slice::from_raw_parts(ts.bit_x, ts.nbits as usize) });
        assert_eq!(&bit_y, unsafe { std::slice::from_raw_parts(ts.bit_y, ts.nbits as usize) });
    }

    pub fn to_image(&self, idx: usize) -> ImageBuffer<Luma<u8>> {
        // assert!(idx >= 0 && idx < self.codes.len());
    
        // let code = self.codes[idx];
        // let border = self.black_border + 1;
        // let dim = (self.dimensions + 2*border) as usize;
        // let im = Image::create(dim, dim);
    
        // // Make 1px white border
        // for i in 0..dim {
        //     im.buf[i] = 255;
        //     im.buf[(dim-1)*im.stride + i] = 255;
        //     im.buf[i*im.stride] = 255;
        //     im.buf[i*im.stride + (dim-1)] = 255;
        // }

        // for y in 0..self.dimensions {
        //     let base = ((y + border) as usize) * im.stride + (border as usize);
        //     for x in 0..self.dimensions {
        //         let pos = (self.dimensions-1 - y) * self.dimensions + (self.dimensions-1 - x);
        //         if (code >> pos) & 0x1 != 0 {
        //             let i = base + (x as usize);
        //             im.buf[i] = 255;
        //         }
        //     }
        // }
    
        // return im;

        assert!(idx < self.codes.len());

        let code = self.codes[idx];

        let mut im = ImageY8::zeroed(self.total_width as usize, self.total_width as usize);

        let white_border_width = self.width_at_border as usize + if self.reversed_border { 0 } else { 2 };
        let white_border_start = (self.total_width as usize - white_border_width)/2;
        // Make 1px white border
        for i in 0..(white_border_width - 1) {
            im[(white_border_start, white_border_start + i)] = 255;
            im[(white_border_start + i, self.total_width as usize - 1 - white_border_start)] = 255;
            im[(self.total_width as usize - 1 - white_border_start, white_border_start + i + 1)] = 255;
            im[(white_border_start + 1 + i, white_border_start)] = 255;
        }

        let border_start = (self.total_width - self.width_at_border) as usize/2;
        for i in 0..self.bits.len() {
            if code & (1u64 << (self.bits.len() - i - 1)) != 0 {
                let (bit_x, bit_y) = self.bits[i];
                im[(bit_y as usize + border_start, bit_x as usize + border_start)] = 255;
            }
        }
        im
    }
}

#[derive(Debug, Clone)]
pub struct AprilTagId {
    pub family: Arc<AprilTagFamily>,
    pub id: usize,
}

impl AprilTagId {
    pub fn code(&self) -> u64 {
        self.family.codes[self.id]
    }
}

impl PartialEq for AprilTagId {
    fn eq(&self, other: &Self) -> bool {
        if self.id != other.id {
            return false;
        }
        if Arc::ptr_eq(&self.family, &other.family) {
            return true;
        }
        // Check if the families is substantially similar, and specifically at the index
        if !self.family.similar_to(&other.family) {
            return false;
        }
        let code1 = match self.family.codes.get(self.id) {
            Some(code) => code,
            None => return false,
        };
        let code2 = match other.family.codes.get(self.id) {
            Some(code) => code,
            None => return false,
        };
        code1 == code2
    }
}

impl Eq for AprilTagId {}