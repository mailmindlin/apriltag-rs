use std::ops::{Range, Add, Sub};

use rand::{thread_rng, Rng, distributions::uniform::{SampleRange, SampleUniform}};

pub(crate) trait RandomColor {
    /// Generates a random RGB color
    fn gen_color_rgb<T>(&mut self, bias: T) -> [T; 3]
        where
            Range<T>: SampleRange<T>,
            T: Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform;
    
    fn gen_color_gray<T>(&mut self, bias: T) -> T
        where
            Range<T>: SampleRange<T>,
            T: Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform;
}

impl<R: Rng> RandomColor for R {
    fn gen_color_rgb<T>(&mut self, bias: T) -> [T; 3]
        where
            Range<T>: SampleRange<T>,
            T: Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform {
        
        [
            self.gen_color_gray(bias),
            self.gen_color_gray(bias),
            self.gen_color_gray(bias),
        ]
    }

    fn gen_color_gray<T>(&mut self, bias: T) -> T
        where
            Range<T>: SampleRange<T>,
            T: Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform {
        // RNG range
        let range = T::from(0)..(T::from(255) - bias);
        bias + self.gen_range(range)
    }
}

#[deprecated]
pub(crate) fn random_color<T>(bias: T) -> [T; 3]
    where
        Range<T>: SampleRange<T>,
        T: Copy + Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform
{
    let mut rng = thread_rng();

    rng.gen_color_rgb(bias)
}