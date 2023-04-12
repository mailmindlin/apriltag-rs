use std::ops::Index;

use super::{Pixel, ImageDimensions, HasDimensions, common::{BufferedImage, BufferedImageMut}, index};

pub struct ImageSlice<'a, P: Pixel> {
    dims: ImageDimensions,
    data: &'a [P::Subpixel],
}

impl<'a, P: Pixel> HasDimensions for ImageSlice<'a, P> {
    fn dimensions(&self) -> &ImageDimensions {
        &self.dims
    }
}

impl<'p, P: Pixel> Index<(usize, usize)> for ImageSlice<'p, P> {
    type Output = P::Value;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let slice = &self.data[index::pixel_idxs::<P>(&self.dims, x, y)];
		<P as Pixel>::slice_to_value(slice)
    }
}

impl<'p, P: Pixel + 'p> BufferedImage<'p, P> for ImageSlice<'p, P> {
    fn buffer(&self) -> &'p [<P as Pixel>::Subpixel] {
        &self.data
    }
}


pub struct ImageSliceMut<'a, P: Pixel> {
    dims: ImageDimensions,
    data: &'a mut [P::Subpixel],
}

impl<'a, P: Pixel> HasDimensions for ImageSliceMut<'a, P> {
    fn dimensions(&self) -> &ImageDimensions {
        &self.dims
    }
}

impl<'p, P: Pixel> Index<(usize, usize)> for ImageSliceMut<'p, P> {
    type Output = P::Value;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let slice = &self.data[index::pixel_idxs::<P>(&self.dims, x, y)];
		<P as Pixel>::slice_to_value(slice)
    }
}

impl<'p, P: Pixel + 'p> BufferedImage<'p, P> for ImageSliceMut<'p, P> {
    fn buffer(&self) -> &[<P as Pixel>::Subpixel] {
        &self.data
    }
}

impl<'p, P: Pixel + 'p> BufferedImageMut<'p, P> for ImageSliceMut<'p, P> {
    fn buffer_dims_mut<'b: 'p>(&'b mut self) -> (&'b ImageDimensions, &'b mut [<P as Pixel>::Subpixel]) {
        (&self.dims, &mut self.data)
    }

    fn buffer_mut<'b: 'p>(&'b mut self) -> &'b mut [<P as Pixel>::Subpixel] {
        &mut self.data
    }
}

#[cfg(test)]
mod test {
    use crate::util::{mem::calloc, image::ImageDimensions};

    use super::ImageSlice;

    #[test]
    fn wrap_buffer() {
        let mut data = calloc::<u8>(16);
        data[0] = 5;
        data[5] = 10;

        {
            let img: ImageSlice<[u8; 1]> = ImageSlice {
                dims: ImageDimensions {
                    width: 3,
                    height: 4,
                    stride: 4,
                },
                data: &data,
            };

            assert_eq!(img[(0, 0)], [5]);
            assert_eq!(img[(0, 1)], [10]);
            assert_eq!(img[(1, 0)], [0]);
        }
    }
}