use std::ops::{Range, Add, Sub};

use rand::{thread_rng, Rng, distributions::uniform::{SampleRange, SampleUniform}};

pub(crate) fn random_color<T>(bias: T) -> [T; 3]
    where
        Range<T>: SampleRange<T>,
        T: Sub<T, Output = T> + Add<T, Output = T> + From<u8> + SampleUniform
{
    let mut rng = thread_rng();

    let rgb: [T; 3];
    for i in 0..3 {
        let range = T::from(0)..(T::from(255) - bias);
        let random_part = rng.gen_range(range);
        rgb[i] = bias + random_part;
    }
    rgb
}