use std::{ops::{Index, IndexMut}, slice::{ChunksExactMut, ChunksExact}};

use super::{ImageDimensions, Pixel, index, rows::{RowsMutIter, RowsIter}, SubpixelArray};

pub struct Pixels<'a, P: Pixel + 'a> {
    pub(super) buf: &'a SubpixelArray<P>,
	pub(super) dims: &'a ImageDimensions,
}

impl<'a, P: Pixel> Index<(usize, usize)> for Pixels<'a, P> {
    type Output = P;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let idxs = index::pixel_idxs::<P>(self.dims, x, y);
        let raw = &self.buf[idxs];
        P::from_slice(raw)
    }
}

impl<'a, P: Pixel> IntoIterator for Pixels<'a, P> {
    type Item = &'a P;

    type IntoIter = PixelsIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
		PixelsIter {
			rows: RowsIter::new(self.buf, self.dims),
			row: None,
		}
    }
}


pub struct PixelsIter<'a, P: Pixel + 'a> {
	pub(super) rows: RowsIter<'a, P>,
	row: Option<ChunksExact<'a, P::Subpixel>>,
}

impl<'a, P: Pixel + 'a> Iterator for PixelsIter<'a, P> {
	type Item = &'a P;

	#[inline(always)]
	fn next(&mut self) -> Option<Self::Item> {
		loop {
			if let Some(row) = &mut self.row {
				if let Some(slice) = row.next() {
					return Some(<P as Pixel>::from_slice(slice));
				}
			}

			let (_, row) = self.rows.next()?;
			let num_channels = <P as Pixel>::CHANNEL_COUNT;
            let row_iter = row
                .into_slice()
                .chunks_exact(num_channels);
			self.row = Some(row_iter);
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let len = self.len();
		(len, Some(len))
	}
}

impl<'a, P: Pixel + 'a> ExactSizeIterator for PixelsIter<'a, P> {
    fn len(&self) -> usize {
		let main_len = self.rows.len() * self.rows.dims.width;
		// Count items in current row
		if let Some(row) = &self.row {
			main_len + row.len()
		} else {
			main_len
		}
    }
}

pub struct PixelsMutIter<'a, P: Pixel + 'a> {
	rows: RowsMutIter<'a, P>,
	row: Option<ChunksExactMut<'a, P::Subpixel>>,
}

impl<'a, P: Pixel + 'a> Iterator for PixelsMutIter<'a, P> {
	type Item = &'a mut P;

	#[inline(always)]
	fn next(&mut self) -> Option<Self::Item> {
		loop {
			if let Some(row) = &mut self.row {
				if let Some(slice) = row.next() {
					return Some(<P as Pixel>::from_slice_mut(slice));
				}
			}

			let (_, row) = self.rows.next()?;
            let row_iter = row
                .into_slice_mut()
                .chunks_exact_mut(<P as Pixel>::CHANNEL_COUNT);
			self.row = Some(row_iter);
		}
	}
}

pub struct PixelsMut<'a, P: Pixel> {
	pub(super) dims: &'a ImageDimensions,
	pub(super) buf: &'a mut [P::Subpixel],
}

impl<'a, P: Pixel> Index<(usize, usize)> for PixelsMut<'a, P> {
    type Output = P;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let idxs = index::pixel_idxs::<P>(self.dims, x, y);
        let slice = &self.buf[idxs];
        <P as Pixel>::from_slice(slice)
    }
}

impl<'a, P: Pixel> IndexMut<(usize, usize)> for PixelsMut<'a, P> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        let idxs = index::pixel_idxs::<P>(self.dims, x, y);
        let slice = &mut self.buf[idxs];
        <P as Pixel>::from_slice_mut(slice)
    }
}

impl<'a, P: Pixel + 'a> IntoIterator for PixelsMut<'a, P> {
    type Item = &'a mut P;

    type IntoIter = PixelsMutIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        PixelsMutIter {
			rows: RowsMutIter::new(self.buf, self.dims),
			row: None,
		}
    }
}

pub struct EnumeratePixels<'a, P: Pixel> {
    pub(super) dims: &'a ImageDimensions,
	pub(super) data: &'a SubpixelArray<P>,
	x: usize,
	y: usize,
}

impl<'a, P: Pixel + 'a> EnumeratePixels<'a, P> {
	pub(super) fn new(data: &'a SubpixelArray<P>, dims: &'a ImageDimensions) -> Self {
		Self {
			data,
			dims,
			x: 0,
			y: 0,
		}
	}
}

impl<'a, P: Pixel + 'a> Iterator for EnumeratePixels<'a, P> {
    type Item = ((usize, usize), &'a P);

    fn next(&mut self) -> Option<Self::Item> {
		while self.x >= self.dims.width {
			self.y += 1;
			self.x -= self.dims.width;
		}
		if self.y >= self.dims.height {
			return None;
		}
		let x = self.x;
		let y = self.y;
		self.x += 1;

		let idxs = index::pixel_idxs::<P>(self.dims, x, y);
		let slice = &self.data[idxs];
		let value = P::from_slice(slice);
		Some(((x, y), value))
    }
}

pub struct EnumeratePixelsMut<'a, P: Pixel> {
	pub(super) dims: &'a ImageDimensions,
	pub(super) data: &'a mut SubpixelArray<P>,
	off: usize,
}

impl<'a, P: Pixel + 'a> EnumeratePixelsMut<'a, P> {
	pub(super) fn new(data: &'a mut SubpixelArray<P>, dims: &'a ImageDimensions) -> Self {
		Self {
			data,
			dims,
			off: 0
		}
	}
}

impl<'a, P: Pixel + 'a> Iterator for EnumeratePixelsMut<'a, P> {
    type Item = ((usize, usize), &'a mut P);

    fn next(&mut self) -> Option<Self::Item> {
		let x = self.off % self.dims.stride;
		let x = if x >= self.dims.width {
			self.off += self.dims.stride - self.dims.width;
			0
		} else {
			x
		};
		let y = self.off / self.dims.stride;
		if y >= self.dims.height {
			return None;
		}
		let start = self.off * P::CHANNEL_COUNT;
		self.off += 1;

		let idxs = start..start + P::CHANNEL_COUNT;
		let slice = &mut self.data[idxs];
		let value = unsafe { std::mem::transmute(P::from_slice_mut(slice)) };
		Some(((x, y), value))
    }
}

#[cfg(test)]
mod test {
    use crate::util::{ImageY8, image::Luma};

	#[test]
	fn enumerate_pixels() {
		let img = ImageY8::zeroed(10, 10);
		let num_pixels = img.enumerate_pixels()
			.into_iter()
			.count();
		assert_eq!(num_pixels, 100);
	}

	#[test]
	fn enumerate_pixels_mut() {
		let mut img = ImageY8::zeroed(10, 10);
		let num_pixels = img.enumerate_pixels_mut()
			.into_iter()
			.count();
		assert_eq!(num_pixels, 100);
	}

	#[test]
	fn enumerate_pixels_idxs() {
		let mut img = ImageY8::zeroed_with_stride(2, 2, 4);
		img[(0, 0)] = 1;
		img[(1, 1)] = 1;
		
		let mut it = img.enumerate_pixels().into_iter();
		assert_eq!(it.next().unwrap(), ((0, 0), &Luma([1])));
		assert_eq!(it.next().unwrap(), ((1, 0), &Luma([0])));
		assert_eq!(it.next().unwrap(), ((0, 1), &Luma([0])));
		assert_eq!(it.next().unwrap(), ((1, 1), &Luma([1])));
		assert!(it.next().is_none());
	}

	#[test]
	fn enumerate_pixels_mut_idxs() {
		let mut img = ImageY8::zeroed_with_stride(2, 2, 4);
		img[(0, 0)] = 1;
		img[(1, 1)] = 1;
		
		let mut it = img.enumerate_pixels_mut().into_iter();
		assert_eq!(it.next().unwrap(), ((0, 0), &mut Luma([1])));
		assert_eq!(it.next().unwrap(), ((1, 0), &mut Luma([0])));
		assert_eq!(it.next().unwrap(), ((0, 1), &mut Luma([0])));
		assert_eq!(it.next().unwrap(), ((1, 1), &mut Luma([1])));
		assert!(it.next().is_none());
	}

	#[test]
	fn enumerate_pixels_test() {
		let mut img = ImageY8::zeroed_with_stride(323, 150, 384);
		
		let mut it = img.enumerate_pixels_mut().into_iter();
		it.count();
	}
}