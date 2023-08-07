use std::ops::{Range, Add, Sub};

use rand::{Rng, distributions::uniform::{SampleRange, SampleUniform}};

use super::image::{Rgb, Luma, Pixel, Primitive};

pub(crate) trait RandomColor {
    /// Generates a random RGB color
    fn gen_color_rgb<T>(&mut self, bias: T) -> Rgb<T>
        where
            Range<T>: SampleRange<T>,
            T: Primitive + Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform;
    
    fn gen_color_gray<T>(&mut self, bias: T) -> Luma<T>
        where
            Range<T>: SampleRange<T>,
            T: Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform;
}

impl<R: Rng> RandomColor for R {
    fn gen_color_rgb<T>(&mut self, bias: T) -> Rgb<T>
        where
            Range<T>: SampleRange<T>,
            T: Primitive + Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform {
        
        Rgb([
            self.gen_color_gray(bias).to_value(),
            self.gen_color_gray(bias).to_value(),
            self.gen_color_gray(bias).to_value(),
        ])
    }

    fn gen_color_gray<T>(&mut self, bias: T) -> Luma<T>
        where
            Range<T>: SampleRange<T>,
            T: Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform {
        // RNG range
        let range = bias..T::from(255);
        Luma([self.gen_range(range)])
    }
}