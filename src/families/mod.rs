use crate::util::Image;


mod tag16h5;
mod tag25h9;
mod tag36h10;
mod tag36h11;

pub use tag16h5::tag16h5_create;
pub use tag25h9::tag25h9_create;
pub use tag36h10::tag36h10_create;
pub use tag36h11::tag36h11_create;

#[derive(Debug, PartialEq)]
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
    pub fn to_image(&self, idx: usize) -> Image {
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

        assert!(idx >= 0);
        assert!(idx < self.codes.len());

        let code = self.codes[idx];

        let mut im = Image::<u8>::create(self.total_width as usize, self.total_width as usize);

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
