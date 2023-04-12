use std::{mem::MaybeUninit, ops::{Range, Index, IndexMut}};

use crate::util::{mem::{SafeZero, calloc}};

use super::{pixel::Pixel, ImageDimensions, ImageBuffer, HasDimensions, common::{BufferedImage, BufferedImageMut}, index};

impl<T: Pixel> ImageBuffer<MaybeUninit<T>> where <T as Pixel>::Subpixel: SafeZero {
	#[inline]
	pub unsafe fn assume_init(self) -> ImageBuffer<T> {
		ImageBuffer {
			dims: self.dims,
			buf: self.buf.assume_init()
		}
	}
}

/// calloc-based constructors
impl<P: Pixel> ImageBuffer<P> where P::Subpixel: SafeZero {
	pub fn new_like<R: Pixel>(src: &dyn HasDimensions) -> Self {
		Self::with_stride(src.width(), src.height(), src.stride())
	}

	pub fn new_packed(width: usize, height: usize) -> Self {
		Self::with_stride(width, height, width)
	}

	pub fn with_alignment(width: usize, height: usize, alignment: usize) -> Self {
		let mut stride = width;

		if (stride % alignment) != 0 {
			stride += alignment - (stride % alignment);
		}

		Self::with_stride(width, height, stride)
	}

	pub fn with_stride(width: usize, height: usize, stride: usize) -> Self {
		let buf = calloc::<P::Subpixel>(height*stride*<P as Pixel>::CHANNEL_COUNT);

		Self {
			dims: ImageDimensions { width, height, stride },
			buf,
		}
	}
}

impl<P: Pixel> ImageBuffer<P> {
	#[inline]
	fn row_idxs_unchecked(&self, y: usize) -> Range<usize> {
		index::row_idxs_unchecked::<P>(&self.dims, y)
	}

	#[inline(always)]
	fn pixel_idxs(&self, x: usize, y: usize) -> Range<usize> {
		index::pixel_idxs::<P>(&self.dims, x, y)
	}

	#[inline(always)]
	fn pixel_idxs_checked(&self, x: usize, y: usize) -> Option<Range<usize>> {
		index::pixel_idxs_checked::<P>(&self.dims, x, y)
	}

	#[inline(always)]
	fn pixel_idxs_unchecked(&self, x: usize, y: usize) -> Range<usize> {
		index::pixel_idxs_unchecked::<P>(&self.dims, x, y)
	}
}

impl<P: Pixel> HasDimensions for ImageBuffer<P> {
	#[inline(always)]
    fn dimensions(&self) -> &ImageDimensions {
        &self.dims
    }
}
/*
impl<P: Pixel> Image<P> for ImageBuffer<P> {
	type Pixels<'a> = SlicePixels<'a, P> where P: 'a;
	type Rows<'a> = Rows<'a, P> where P: 'a;

	#[inline]
	fn pixel<'a>(&'a self, x: usize, y: usize) -> &'a P {
		let idx = self.pixel_idxs(x, y);
		<P as Pixel>::from_slice(&self.buf[idx])
	}

	#[inline(always)]
	fn pixel_checked<'a>(&'a self, x: usize, y: usize) -> Option<&'a P> {
		let idx = self.pixel_idxs_checked(x, y)?;
		Some(<P as Pixel>::from_slice(&mut self.buf[idx]))
	}

	#[inline(always)]
	unsafe fn pixel_unchecked<'a>(&'a self, x: usize, y: usize) -> &'a P {
		let idx = self.pixel_idxs_unchecked(x, y);
		<P as Pixel>::from_slice(&mut self.buf[idx])
	}

	fn row<'a>(&'a self, row: usize) -> <Self::Rows<'a> as ImageRows<'a, P>>::Row {
        todo!()
    }

	fn rows<'a>(&'a self) -> Self::Rows<'a> {
        Rows(self)
    }

	fn pixels<'a>(&'a self) -> Self::Pixels<'a> {
        SlicePixels {
			dims: &self.dims,
			buf: &self.buf,
		}
    }

	fn enumerate_pixels<'a>(&'a self) -> <Self::Pixels<'a> as ImagePixels<'a, P>>::Enumerated {
		SliceEnumeratePixels {
			image: self,
			offset: 0,
		}
	}

	fn map<Pr: Pixel>(&self, update: impl Fn(&P) -> Pr) -> ImageBuffer<Pr> where Pr::Subpixel: SafeZero {
		let mut result = ImageBuffer::<MaybeUninit<Pr>>::with_stride(self.dims.width, self.dims.height, self.dims.stride);
		if self.dims.width == self.dims.stride {
			let src_iter = self.buf.chunks_exact(<P as Pixel>::CHANNEL_COUNT);
			let dst_iter = result.buf.chunks_exact_mut(<Pr as Pixel>::CHANNEL_COUNT);
			for (src, dst) in src_iter.zip(dst_iter) {
				let src = <P as Pixel>::from_slice(src);
				let dst = <MaybeUninit<Pr> as Pixel>::from_slice_mut(dst);
				dst.write(update(src));
			}
		} else {

		}
		unsafe { result.assume_init() }
	}
}

impl<P: Pixel> ImageMut<P> for ImageBuffer<P> {
	type PixelsMut<'a> = SlicePixelsMut<'a, P> where P: 'a;
	type RowsMut<'a> = RowsMut<'a, P> where P: 'a;

	#[inline]
	fn pixel_mut<'a>(&'a mut self, x: usize, y: usize) -> &'a mut P {
		let idx = self.pixel_idxs(x, y);
		<P as Pixel>::from_slice_mut(&mut self.buf[idx])
	}

	#[inline(always)]
	fn pixel_mut_checked<'a>(&'a mut self, x: usize, y: usize) -> Option<&'a mut P> {
		let idx = self.pixel_idxs_checked(x, y)?;
		Some(<P as Pixel>::from_slice_mut(&mut self.buf[idx]))
	}

	#[inline(always)]
	unsafe fn pixel_mut_unchecked<'a>(&'a mut self, x: usize, y: usize) -> &'a mut P {
		let idx = self.pixel_idxs_unchecked(x, y);
		<P as Pixel>::from_slice_mut(&mut self.buf[idx])
	}

	fn row_mut<'a>(&'a mut self, row: usize) -> <Self::RowsMut<'a> as ImageRowsMut<'a, P>>::RowMut {
        todo!()
    }

	fn rows_mut<'a>(&'a mut self) -> Self::RowsMut<'a> {
        RowsMut(self)
    }

	fn pixels_mut<'a>(&'a mut self) -> Self::PixelsMut<'a> {
        SlicePixelsMut {
			dims: &self.dims,
			buf: &self.buf,
		}
    }

	fn enumerate_pixels_mut<'a>(&'a mut self) -> <Self::PixelsMut<'a> as ImagePixelsMut<'a, P>>::Enumerated {
		EnumeratePixelsMut {
			image: self,
			offset: 0,
		}
	}

	fn apply(&mut self, update: impl FnMut(&mut P) -> ()) {
		if self.dims.width == self.dims.stride {
			for slice in self.buf.chunks_exact_mut(<P as Pixel>::CHANNEL_COUNT) {
				update(<P as Pixel>::from_slice_mut(slice));
			}
		} else {
			for pixel in self.pixels_mut() {
				update(pixel);
			}
		}
	}
}
*/
impl<P: Pixel> Index<(usize, usize)> for ImageBuffer<P> {
	type Output = P::Value;

	fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
		let slice = &self.buf[self.pixel_idxs(x, y)];
		<P as Pixel>::slice_to_value(slice)
	}
}

/// Get a pixel
impl<P: Pixel> IndexMut<(usize, usize)> for ImageBuffer<P> {
	fn index_mut(&mut self, (x,y): (usize, usize)) -> &mut Self::Output {
		let idxs = self.pixel_idxs(x, y);
		let slice = &mut self.buf[idxs];
		<P as Pixel>::slice_to_value_mut(slice)
	}
}

impl<'a, P: Pixel + 'a> BufferedImage<'a, P> for ImageBuffer<P> {
	#[inline(always)]
    fn buffer(&self) -> &[<P as Pixel>::Subpixel] {
        &self.buf
    }
}

impl<'a, P: Pixel + 'a> BufferedImageMut<'a, P> for ImageBuffer<P> {
    fn buffer_dims_mut<'b: 'a>(&'b mut self) -> (&'b ImageDimensions, &'b mut [<P as Pixel>::Subpixel]) {
        (&self.dims, &mut self.buf)
    }

    fn buffer_mut<'b: 'a>(&'b mut self) -> &'b mut [<P as Pixel>::Subpixel] {
        &mut self.buf
    }
}