mod pnm;
pub mod luma;
pub mod rgb;
mod ps;
pub mod pixel;
mod buffer;
mod slice;
pub(super) mod common;
mod index;

use std::{ops::{IndexMut, Index, RangeBounds}, io};
pub use self::pixel::{Pixel, Primitive};
pub use rgb::Rgb;
pub use luma::Luma;
pub use ps::PostScriptWriter;
pub use pnm::ImageWritePNM;
pub use index::ImageDimensions;
use self::{pnm::PNM, slice::{ImageSlice, ImageSliceMut}};

use super::{mem::SafeZero, geom::Point2D, dims::Dimensions2D};

pub trait HasDimensions {
	fn dimensions(&self) -> &ImageDimensions;

	fn width(&self) -> usize {
		self.dimensions().width()
	}

	fn height(&self) -> usize {
		self.dimensions().height()
	}

	fn stride(&self) -> usize {
		self.dimensions().stride
	}

	fn is_packed(&self) -> bool {
		self.width() == self.stride()
	}

	#[inline]
	fn len(&self) -> usize {
		self.width() * self.height()
	}
}

pub trait Image<'p, P: Pixel + 'p>: Index<(usize, usize), Output = P::Value> + HasDimensions {
	type Pixels<'a: 'p>: ImagePixels<'a, P> where Self: 'a, P: 'a;
	type Rows<'a: 'p>: ImageRows<'a, P> + 'a where Self: 'a, P: 'a;

	fn row<'a: 'p>(&'a self, row: usize) -> <Self::Rows<'a> as ImageRows<'a, P>>::Row where P: 'a;
	fn rows<'a: 'p>(&'a self) -> Self::Rows<'a> where P: 'a;

	fn window1<'a: 'p>(&'a self, x: usize, y: usize, rx: usize, ry: usize) -> ImageSlice<'a, P> {
		let left = x.saturating_sub(rx);
		let right = std::cmp::min(x.saturating_add(rx), self.width());

		let top = y.saturating_sub(ry);
		let bottom = std::cmp::min(y.saturating_add(ry), self.height());

		self.slice(left..=right, top..=bottom)
	}

	fn slice<'a: 'p>(&'a self, x: impl RangeBounds<usize>, y: impl RangeBounds<usize>) -> ImageSlice<'a, P>;

	fn pixels<'a: 'p>(&'a self) -> Self::Pixels<'a> where P: 'a;
	fn enumerate_pixels<'a: 'p>(&'a self) -> <Self::Pixels<'a> as ImagePixels<'a, P>>::Enumerated where P: 'a;

	fn pixel<'a: 'p>(&'a self, x: usize, y: usize) -> &'a P where P: 'a;

	fn pixel_checked<'a: 'p>(&'a self, x: usize, y: usize) -> Option<&'a P> where P: 'a;

	unsafe fn pixel_unchecked<'a: 'p>(&'a self, x: usize, y: usize) -> &'a P where P: 'a;

	fn map<Pr: Pixel>(&'p self, update: impl Fn(&P) -> Pr) -> ImageBuffer<Pr> where Pr::Subpixel: SafeZero {
		let mut result = ImageBuffer::<Pr>::new_packed(self.width(), self.height());
		for ((x, y), v) in self.enumerate_pixels() {
			result[(x,y)] = update(v).to_value();
		}
		result
	}

	fn map_indexed<'a: 'p, Pr: Pixel>(&'a self, mut update: impl for<'b> FnMut(&'b Self, usize, usize) -> Pr) -> ImageBuffer<Pr> where Pr::Subpixel: SafeZero {
		let mut result = ImageBuffer::<Pr>::new_packed(self.width(), self.height());
		for ((x, y), v) in result.enumerate_pixels_mut() {
			*v = update(self, x, y);
		}
		result
	}
}

pub trait ImageMut<'p, P: Pixel + 'p>: Image<'p, P> + IndexMut<(usize, usize), Output = P::Value> {
	type PixelsMut<'a: 'p>: ImagePixelsMut<'a, P> where Self: 'a, P: 'a;
	type RowsMut<'a: 'p>: ImageRowsMut<'a, P> where Self: 'a, P: 'a;
	fn row_mut<'a: 'p>(&'a mut self, row: usize) -> <Self::RowsMut<'a> as ImageRowsMut<'a, P>>::RowMut
		where P: 'a;
	fn rows_mut<'a: 'p>(&'a mut self) -> Self::RowsMut<'a>
		where P: 'a;
	
	fn pixels_mut<'a: 'p>(&'a mut self) -> Self::PixelsMut<'a>
		where P: 'a;
	fn enumerate_pixels_mut<'a: 'p>(&'a mut self) -> <Self::PixelsMut<'a> as ImagePixelsMut<'a, P>>::Enumerated
		where P: 'a;

	fn window_mut<'a: 'p>(&'a mut self, x: usize, y: usize, width: usize, height: usize) -> ImageSliceMut<'a, P>
		where P: 'a;

	fn pixel_mut<'a: 'p>(&'a mut self, x: usize, y: usize) -> &'a mut P
		where P: 'a;
	fn pixel_mut_checked<'a: 'p>(&'a mut self, x: usize, y: usize) -> Option<&'a mut P>
		where P: 'a;
	unsafe fn pixel_mut_unchecked<'a: 'p>(&'a mut self, x: usize, y: usize) -> &'a mut P
		where P: 'a;

	fn apply(&'p mut self, mut update: impl FnMut(&mut P) -> ()) {
		for pixel in self.pixels_mut() {
			update(pixel);
		}
	}

	fn draw_line(&mut self, p0: Point2D, p1: Point2D, color: &P, width: usize) where P::Value: Copy {
		let dist = p0.distance_to(&p1);
		let delta = 0.5 / dist;
		let num_steps = f64::ceil(dist * 2.) as usize;

		let color = color.to_value();
	
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
			if x >= self.width() || y >= self.height() {
				continue;
			}

			self[(x, y)] = color;
			if width > 1 {
				if x + 1 < self.width() {
					self[(x + 1, y)] = color;
				}
				if y + 1 < self.height() {
					self[(x, y + 1)] = color;
				}
				if x + 1 < self.width() && y + 1 < self.height() {
					self[(x + 1, y + 1)] = color;
				}
			}
		}
	}
}


#[derive(Clone)]
pub struct ImageBuffer<P: Pixel = Luma<u8>> {
	dims: ImageDimensions,
	/// Image data
	pub(crate) buf: Box<[P::Subpixel]>,
}

pub type ImageY8 = ImageBuffer<Luma<u8>>;
pub type ImageRGB8 = ImageBuffer<Rgb<u8>>;

pub trait ImagePixels<'a, P: Pixel + 'a>: Index<(usize, usize), Output=P> {
	type Enumerated: Iterator<Item = ((usize, usize), &'a P)>;
	// type Windows: ImageWindows<'a, P>;
	// type ArrayWindows<const KW: usize, const KH: usize>: ImageArrayWindows<'a, P, KW, KH>;

	// fn windows2d(&self, width: usize, height: usize) -> Self::Windows;
	// fn array_windows2d<const KW: usize, const KH: usize>(&self) -> Self::ArrayWindows<KW, KH>;

	// fn chunks2d(&self, width: usize, height: usize) -> Self::Chunks;
	// fn array_chunks2d<const KW: usize, const KH: usize>(&self) -> Self::ArrayChunks<KW, KH>;
}

pub trait ImagePixelsMut<'a, P: Pixel + 'a>: IndexMut<(usize, usize), Output=P> + IntoIterator<Item=&'a mut P> {
	type Enumerated: Iterator<Item = ((usize, usize), &'a mut P)>;
}

pub trait ImageRow<'a, P: Pixel> {
	fn into_slice(self) -> &'a [P::Subpixel];
	fn as_slice(&self) -> &[P::Subpixel];
}

pub trait ImageRowMut<'a, P: Pixel>: ImageRow<'a, P> {
	fn into_slice_mut(self) -> &'a mut [<P as Pixel>::Subpixel];
	fn as_slice_mut(&mut self) -> &mut [P::Subpixel];
}

pub trait ImageRows<'a, P: Pixel>: IntoIterator<Item = (usize, Self::Row)> {
	// type Enumerated: IntoIterator<Item = (usize, Self::Row)>;
	type Row: ImageRow<'a, P>;
}

pub trait ImageRowsMut<'a, P: Pixel> /*: IntoIterator<Item = (usize, Self::RowMut)> */ {
	// type Enumerated: IntoIterator<Item = (usize, Self::RowMut)>;
	type RowMut: ImageRowMut<'a, P>;
}

pub trait ImageWritePostscript {
	/// Write PostScript data
	fn write_postscript(&self, f: &mut PostScriptWriter<impl io::Write>) -> io::Result<()>;
}