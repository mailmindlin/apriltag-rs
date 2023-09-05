mod pnm;
pub mod luma;
pub mod rgb;
mod ps;
pub mod pixel;
mod index;
mod rows;
mod pixels;
mod svg;

use std::{ops::{RangeBounds, Deref, Range, DerefMut}, mem::MaybeUninit, marker::PhantomData};
pub use self::pixel::{Pixel, Primitive};
pub use rgb::Rgb;
pub use luma::Luma;
pub use ps::{PostScriptWriter, ImageWritePostscript};
pub(crate) use ps::{VectorPathWriter};
pub use pnm::ImageWritePNM;
pub use index::ImageDimensions;
use self::{pnm::PNM, rows::{Row, Rows, RowMut, RowsMut}, pixels::{Pixels, EnumeratePixels, PixelsMut, EnumeratePixelsMut}, pixel::DefaultAlignment};

use super::{mem::{SafeZero, calloc}, geom::Point2D, dims::Dimensions2D};

pub type Image<P, Container = DC<P>> = ImageBuffer<P, Container>;

/// Default image with 8-bit greyscale pixels
pub type ImageY8 = Image<Luma<u8>, DC<Luma<u8>>>;
/// Default image with 3x 8-bit RGB pixels
pub type ImageRGB8 = Image<Rgb<u8>, DC<Rgb<u8>>>;
pub type ImageRefY8<'a> = Image<Luma<u8>, &'a SubpixelArray<Luma<u8>>>;

/// Array of subpixel elements for Pixel type `P`
type SubpixelArray<P> = [<P as Pixel>::Subpixel];

/// Default container for Pixel type `P`
type DC<P> = Box<SubpixelArray<P>>;

/// Buffer for 2D image
#[derive(Debug, Clone)]
pub struct ImageBuffer<P: Pixel, Container = DC<P>> {
	pub(crate) data: Container,
	dims: ImageDimensions,
	pix: PhantomData<P>,
}

impl<P: Pixel, Container: Deref<Target = [P::Subpixel]>> PartialEq for ImageBuffer<P, Container> where <P as Pixel>::Subpixel: PartialEq {
    fn eq(&self, other: &Self) -> bool {
		if self.dims.width != other.dims.width {
			return false;
		}
		if self.dims.height != other.dims.height {
			return false;
		}
		for (y, row) in self.rows() {
			let row2 = other.row(y);
			if row.as_slice() != row2.as_slice() {
				return false;
			}
		}
		true
    }
}

impl<P: Pixel, Container: AsRef<[<P as Pixel>::Subpixel]>> ImageBuffer<P, Container> {
	pub fn as_ref<'a>(&'a self) -> ImageBuffer<P,&'a [<P as Pixel>::Subpixel]> {
		ImageBuffer { dims: self.dims, data: self.data.as_ref(), pix: self.pix }
	}
}

impl<P: Pixel, Container> ImageBuffer<P, Container> {
	/// Extract raw data
	pub fn into_raw(self) -> Container {
		self.data
	}

	/// Get raw data container
	pub fn as_raw(&self) -> &Container {
		&self.data
	}

	/// Get image dimensions
	#[inline(always)]
    pub const fn dimensions(&self) -> &ImageDimensions {
        &self.dims
    }

	/// Get image width (in pixels)
	#[inline]
	pub fn width(&self) -> usize {
		self.dimensions().width()
	}

	/// Get image height (in pixels)
	#[inline]
	pub fn height(&self) -> usize {
		self.dimensions().height()
	}

	#[inline]
	pub fn stride(&self) -> usize {
		self.dimensions().stride
	}

	#[inline]
	pub fn is_packed(&self) -> bool {
		self.width() == self.stride()
	}

	/// Get number of pixels
	#[inline]
	pub fn pixel_count(&self) -> usize {
		self.width() * self.height()
	}

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


/// calloc-based constructors
impl<P: Pixel> ImageBuffer<P, Box<[<P as Pixel>::Subpixel]>> where P::Subpixel: SafeZero {
	pub fn zeroed(width: usize, height: usize) -> Self where P: DefaultAlignment {
		Self::zeroed_with_alignment(width, height, <P as DefaultAlignment>::DEFAULT_ALIGNMENT)
	}

	pub fn zeroed_packed(width: usize, height: usize) -> Self {
		Self::zeroed_with_stride(width, height, width)
	}

	pub fn zeroed_with_alignment(width: usize, height: usize, alignment: usize) -> Self {
		let stride = width.next_multiple_of(alignment);

		Self::zeroed_with_stride(width, height, stride)
	}

	pub fn zeroed_with_stride(width: usize, height: usize, stride: usize) -> Self {
		assert!(stride >= width);
		let data = calloc::<P::Subpixel>(height*stride*<P as Pixel>::CHANNEL_COUNT);

		Self {
			dims: ImageDimensions { width, height, stride },
			data,
			pix: PhantomData,
		}
	}

	pub fn clone_like<C: Deref<Target = [<P as Pixel>::Subpixel]>>(src: &ImageBuffer<P, C>) -> Self {
		if src.stride() == src.width() {
			let data = src.data.deref().to_vec().into_boxed_slice();
			Self {
				dims: src.dims,
				data,
				pix: PhantomData,
			}
		} else {
			let mut res = Self::zeroed_with_stride(src.width(), src.height(), src.stride());
			for ((_, mut dst), (_, src)) in res.rows_mut().into_iter().zip(src.rows().into_iter()) {
				dst.as_slice_mut().copy_from_slice(src.as_slice());
			}
			res
		}
	}
}

impl<P: Pixel, Container: Deref<Target = [P::Subpixel]>> ImageBuffer<P, Container> {
	pub(crate) fn wrap(data: Container, width: usize, height: usize, stride: usize) -> Self {
		Self {
			dims: ImageDimensions { width, height, stride },
			data,
			pix: PhantomData,
		}
	}
	#[inline]
	pub fn pixel(&self, x: usize, y: usize) -> &P {
		let dims = self.dimensions();
		let x = x;
		let y = y;
		let idx = match index::pixel_idxs_checked::<P>(dims, x, y) {
			Some(value) => value,
			None => {
				panic!("Image index {:?} out of bounds {:?}", (x, y), dims);
			}
		};

		<P as Pixel>::from_slice(&self.data[idx])
	}

	#[inline(always)]
	pub fn pixel_checked(&self, x: usize, y: usize) -> Option<&P> {
		let idx = index::pixel_idxs_checked::<P>(self.dimensions(), x, y)?;
		Some(<P as Pixel>::from_slice(&self.data[idx]))
	}

	#[inline(always)]
	unsafe fn pixel_unchecked(&self, x: usize, y: usize) -> &P {
		let idx = index::pixel_idxs_unchecked::<P>(self.dimensions(), x, y);
		<P as Pixel>::from_slice(&self.data[idx])
	}

	pub fn row(&self, row: usize) -> Row<P> {
        let idxs = index::row_idxs::<P>(self.dimensions(), row);
        Row(&self.data[idxs])
    }

	pub fn window(&self, x: usize, y: usize, rx: usize, ry: usize) -> ImageBuffer<P, &[P::Subpixel]> {
		let left = x.saturating_sub(rx);
		let right = std::cmp::min(x.saturating_add(rx), self.dims.width - 1);

		let top = y.saturating_sub(ry);
		let bottom = std::cmp::min(y.saturating_add(ry), self.dims.height - 1);

		self.slice(left..=right, top..=bottom)
	}

    pub fn slice(&self, x: impl RangeBounds<usize>, y: impl RangeBounds<usize>) -> ImageBuffer<P, &[P::Subpixel]> {
        let (dims, idxs) = index::slice_idxs::<P>(&self.dims, x, y);
        ImageBuffer {
			dims,
            data: &self.data[idxs],
			pix: PhantomData,
        }
    }

	pub fn rows(&self) -> Rows<P> {
        Rows {
            buf: &self.data,
            dims: &self.dims,
        }
    }

	pub fn pixels(&self) -> Pixels<P> {
        Pixels {
			buf: &self.data,
            dims: &self.dims,
		}
    }

	pub fn enumerate_pixels(&self) -> EnumeratePixels<P> {
		EnumeratePixels::new(&self.data, &self.dims)
	}

	pub fn map<Pr: Pixel>(&self, update: impl Fn(&P) -> Pr) -> ImageBuffer<Pr, Box<[Pr::Subpixel]>> where Pr::Subpixel: SafeZero {
		let mut result = ImageBuffer::<MaybeUninit<Pr>, Box<SubpixelArray<MaybeUninit<Pr>>>>::zeroed_with_stride(self.dims.width, self.dims.height, self.dims.stride);
		if self.is_packed() {
			let src_iter = self.data.chunks_exact(<P as Pixel>::CHANNEL_COUNT);
			let dst_iter = result.data.chunks_exact_mut(<Pr as Pixel>::CHANNEL_COUNT);
			for (src, dst) in src_iter.zip(dst_iter) {
				let src = <P as Pixel>::from_slice(src);
				let dst = <MaybeUninit<Pr> as Pixel>::from_slice_mut(dst);
				dst.write(update(src));
			}
		} else {
            for ((x, y), v) in self.enumerate_pixels() {
				result[(x,y)].write(update(v).to_value());
			}
		}
		unsafe { result.assume_init() }
	}

	pub fn map_indexed<Pr: Pixel>(&self, mut update: impl for<'b> FnMut(&'b Self, usize, usize) -> Pr) -> ImageBuffer<Pr> where Pr::Subpixel: SafeZero {
		let mut result = ImageBuffer::<Pr>::zeroed_packed(self.width(), self.height());
		for ((x, y), v) in result.enumerate_pixels_mut() {
			*v = update(self, x, y);
		}
		result
	}
}

impl<P: Pixel, Container: DerefMut<Target = [P::Subpixel]>> ImageBuffer<P, Container> {
	#[inline]
	pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut P {
		let idx = index::pixel_idxs::<P>(self.dimensions(), x, y);
		<P as Pixel>::from_slice_mut(&mut self.data[idx])
	}

	#[inline(always)]
	pub fn pixel_mut_checked(&mut self, x: usize, y: usize) -> Option<&mut P> {
		let idx = index::pixel_idxs_checked::<P>(self.dimensions(), x, y)?;
		Some(<P as Pixel>::from_slice_mut(&mut self.data[idx]))
	}

	#[inline(always)]
	unsafe fn pixel_mut_unchecked(&mut self, x: usize, y: usize) -> &mut P {
		let idx = index::pixel_idxs_unchecked::<P>(self.dimensions(), x, y);
		<P as Pixel>::from_slice_mut(&mut self.data[idx])
	}

	pub fn row_mut(&mut self, row: usize) -> RowMut<P> {
        let row_idxs = index::row_idxs::<P>(self.dimensions(), row);
        RowMut(&mut self.data[row_idxs])
    }

	pub fn rows2_mut(&mut self, y0: usize, y1: usize) -> (RowMut<P>, RowMut<P>) {
		assert_ne!(y0, y1);
		let idxs0 = index::row_idxs::<P>(self.dimensions(), y0);
		let idxs1 = index::row_idxs::<P>(self.dimensions(), y1);
		if idxs0.end < idxs1.start {
			// y0 < y1
			let (left, right) = self.data.split_at_mut(idxs0.end);
			let idxs1 = (idxs1.start - idxs0.end)..(idxs1.end - idxs0.end);
			let r0 = RowMut(&mut left[idxs0]);
			let r1 = RowMut(&mut right[idxs1]);
			(r0, r1)
		} else {
			// y1 < y0
			debug_assert!(idxs1.end < idxs0.start);
			let (left, right) = self.data.split_at_mut(idxs1.end);
			let idxs0 = (idxs0.start - idxs1.end)..(idxs0.end - idxs1.end);
			let r0 = RowMut(&mut right[idxs0]);
			let r1 = RowMut(&mut left[idxs1]);
			(r0, r1)
		}
	}

	pub fn rows_mut(&mut self) -> RowsMut<P> {
        RowsMut {
			dims: &self.dims,
			buf: &mut self.data,
		}
    }

	pub fn pixels_mut(&mut self) -> PixelsMut<P> {
        PixelsMut {
			dims: &self.dims,
			buf: &mut self.data,
		}
    }

	pub fn enumerate_pixels_mut(&mut self) -> EnumeratePixelsMut<P> {
		EnumeratePixelsMut::new(&mut self.data, &self.dims)
	}

	pub fn apply(&mut self, mut update: impl FnMut(&mut P) -> ()) {
		if self.is_packed() {
			for slice in self.data.chunks_exact_mut(<P as Pixel>::CHANNEL_COUNT) {
				update(<P as Pixel>::from_slice_mut(slice));
			}
		} else {
			for pixel in self.pixels_mut() {
				update(pixel);
			}
		}
	}

	pub fn draw_line(&mut self, p0: Point2D, p1: Point2D, color: &P, width: usize) where P::Value: Copy {
		let dist = p0.distance_to(&p1);
		let delta = 0.5 / dist;
		let num_steps = f64::ceil(dist * 2.) as usize;

		let color = color.to_value();

		let step = &p0 - &p1;

		let im_width = self.width();
		let im_height = self.height();
	
		// terrible line drawing code
		for i in 0..num_steps {
			let f = (i as f64) * delta;
			let c = &p1 + (step * f);
			let x = c.x() as isize;
			let y = c.y() as isize;
	
			if x < 0 || y < 0 {
				continue;
			}

			let x = x as usize;
			let y = y as usize;
			if x >= im_width || y >= im_height {
				continue;
			}

			self[(x, y)] = color;
			if width > 1 {
				if x + 1 < im_width {
					self[(x + 1, y)] = color;
				}
				if y + 1 < im_height {
					self[(x, y + 1)] = color;
				}
				if x + 1 < im_width && y + 1 < self.height() {
					self[(x + 1, y + 1)] = color;
				}
			}
		}
	}
}

impl<P: Pixel> ImageBuffer<MaybeUninit<P>, Box<SubpixelArray<MaybeUninit<P>>>> where <P as Pixel>::Subpixel: SafeZero {
	#[inline]
	pub unsafe fn assume_init(self) -> ImageBuffer<P, Box<SubpixelArray<P>>> {
		ImageBuffer {
			dims: self.dims,
			data: self.data.assume_init(),
			pix: PhantomData
		}
	}
}

#[cfg(test)]
mod test {
    use super::ImageY8;

	#[test]
	fn slice_simple() {
		let image = ImageY8::zeroed_packed(5, 5);
		let slice = image.slice(0..1, 0..1);
		assert_eq!(slice.width(), 1);
		assert_eq!(slice.height(), 1);
	}
	#[test]
	fn window() {
		let image = ImageY8::zeroed_packed(5, 5);
		let window = image.window(3, 3, 1, 1);
		assert_eq!(window.width(), 3);
		assert_eq!(window.height(), 3);
		assert_eq!(window.stride(), 5);
	}

	#[test]
	fn slice() {
		let image = ImageY8::zeroed_with_stride(5, 5, 10);
		{
			let slice_full = image.slice(.., ..);
			assert_eq!(slice_full.width(), 5);
			assert_eq!(slice_full.height(), 5);
			assert_eq!(slice_full.stride(), 10);
			assert_eq!(slice_full.pixels().into_iter().count(), 25);
		}
		{
			let slice = image.slice(.., 1..=2);
			assert_eq!(slice.width(), 5);
			assert_eq!(slice.height(), 2);
			assert_eq!(slice.stride(), 10);
			assert_eq!(slice.pixels().into_iter().count(), 10);
		}
		{
			let slice = image.slice(1..=2, ..);
			assert_eq!(slice.width(), 2);
			assert_eq!(slice.height(), 5);
			assert_eq!(slice.stride(), 10);
			assert_eq!(slice.pixels().into_iter().count(), 10);
		}
		{
			let slice = image.slice(1..=2, 1..=2);
			assert_eq!(slice.width(), 2);
			assert_eq!(slice.height(), 2);
			assert_eq!(slice.stride(), 10);
			assert_eq!(slice.pixels().into_iter().count(), 4);
		}
		{
			let slice = image.slice(0..=1, ..);
			assert_eq!(slice.width(), 2);
			assert_eq!(slice.height(), 5);
			assert_eq!(slice.stride(), 10);
			assert_eq!(slice.pixels().into_iter().count(), 10);
		}
	}
}