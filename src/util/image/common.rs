use std::{slice::{ChunksExact, ChunksExactMut, ChunksMut}, ops::{Index, IndexMut, RangeBounds}, mem::MaybeUninit};

use crate::util::{mem::SafeZero, dims::{Dimensions2D}, image::slice::ImageSlice};

use super::{index, Image, Pixel, ImageDimensions, ImagePixels, HasDimensions, ImageRows, ImageBuffer, ImageMut, ImagePixelsMut, ImageRow, ImageRowMut, ImageRowsMut, slice::ImageSliceMut};


pub(super) trait BufferedImage<'a, P: Pixel + 'a>: HasDimensions {
    fn buffer(&self) -> &[P::Subpixel];
}




impl<'p, P: Pixel + 'p, I: BufferedImage<'p, P> + Index<(usize, usize), Output=P::Value>> Image<'p, P> for I {
    type Pixels<'a: 'p> = Pixels<'a, P> where P: 'a, I: 'a;
	type Rows<'a: 'p> = SliceRows<'a, P> where P: 'a, I: 'a;

	#[inline]
	fn pixel<'a: 'p>(&'a self, x: usize, y: usize) -> &'a P where P: 'a {
		let idx = index::pixel_idxs::<P>(self.dimensions(), x, y);
		<P as Pixel>::from_slice(&self.buffer()[idx])
	}

	#[inline(always)]
	fn pixel_checked<'a: 'p>(&'a self, x: usize, y: usize) -> Option<&'a P> where P: 'a {
		let idx = index::pixel_idxs_checked::<P>(self.dimensions(), x, y)?;
		Some(<P as Pixel>::from_slice(&self.buffer()[idx]))
	}

	#[inline(always)]
	unsafe fn pixel_unchecked<'a: 'p>(&'a self, x: usize, y: usize) -> &'a P where P: 'a {
		let idx = index::pixel_idxs_unchecked::<P>(self.dimensions(), x, y);
		<P as Pixel>::from_slice(&self.buffer()[idx])
	}

	fn row<'a: 'p>(&'a self, row: usize) -> <Self::Rows<'a> as ImageRows<'a, P>>::Row where P: 'a {
        let idxs = index::row_idxs::<P>(self.dimensions(), row);
        SliceRow(&self.buffer()[idxs])
    }

    fn slice<'a: 'p>(&'a self, x: impl RangeBounds<usize>, y: impl RangeBounds<usize>) -> super::slice::ImageSlice<'a, P> {
        let (dims, idxs) = index::slice_idxs::<P>(self.dimensions(), x, y);
        ImageSlice {
            dims,
            data: &self.buffer()[idxs],
        }
    }

	fn rows<'a: 'p>(&'a self) -> Self::Rows<'a> where P: 'a {
        let r: SliceRows<'a, P> = SliceRows {
            buf: self.buffer(),
            dims: self.dimensions(),
        };
        r
    }

	fn pixels<'a: 'p>(&'a self) -> Self::Pixels<'a> where P: 'a {
        Pixels {
			dims: self.dimensions(),
			buf: self.buffer(),
		}
    }

	fn enumerate_pixels<'a: 'p>(&'a self) -> <Self::Pixels<'a> as ImagePixels<'a, P>>::Enumerated where P: 'a {
		SliceEnumeratePixels {
			dims: self.dimensions(),
            buf: self.buffer(),
			offset: 0,
		}
	}

	fn map<Pr: Pixel>(&'p self, update: impl Fn(&P) -> Pr) -> ImageBuffer<Pr> where Pr::Subpixel: SafeZero {
		let mut result = ImageBuffer::<MaybeUninit<Pr>>::with_stride(self.dimensions().width, self.dimensions().height, self.dimensions().stride);
		if self.is_packed() {
			let src_iter = self.buffer().chunks_exact(<P as Pixel>::CHANNEL_COUNT);
			let dst_iter = result.buf.chunks_exact_mut(<Pr as Pixel>::CHANNEL_COUNT);
			for (src, dst) in src_iter.zip(dst_iter) {
				let src = <P as Pixel>::from_slice(src);
				let dst = <MaybeUninit<Pr> as Pixel>::from_slice_mut(dst);
				dst.write(update(src));
			}
		} else {
            todo!()
		}
		unsafe { result.assume_init() }
	}
}

pub(super) trait BufferedImageMut<'a, P: Pixel + 'a>: BufferedImage<'a, P> {
    fn buffer_dims_mut<'b: 'a>(&'b mut self) -> (&'b ImageDimensions, &'b mut [P::Subpixel]);

    fn buffer_mut<'b: 'a>(&'b mut self) -> &'b mut [P::Subpixel] {
        let (_dims, buf) = self.buffer_dims_mut();
        buf
    }
}

impl<'p, P: Pixel + 'p, I: BufferedImageMut<'p, P> + IndexMut<(usize, usize), Output=P::Value> + 'p> ImageMut<'p, P> for I {
    type PixelsMut<'a: 'p> = SlicePixelsMut<'a, P> where P: 'a, I: 'a;
	type RowsMut<'a: 'p> = RowsMut<'a, P> where P: 'a, I: 'a;

	#[inline]
	fn pixel_mut<'a: 'p>(&'a mut self, x: usize, y: usize) -> &'a mut P where P: 'a {
		let idx = index::pixel_idxs::<P>(self.dimensions(), x, y);
		<P as Pixel>::from_slice_mut(&mut self.buffer_mut()[idx])
	}

	#[inline(always)]
	fn pixel_mut_checked<'a: 'p>(&'a mut self, x: usize, y: usize) -> Option<&'a mut P> where P: 'a {
		let idx = index::pixel_idxs_checked::<P>(self.dimensions(), x, y)?;
		Some(<P as Pixel>::from_slice_mut(&mut self.buffer_mut()[idx]))
	}

	#[inline(always)]
	unsafe fn pixel_mut_unchecked<'a: 'p>(&'a mut self, x: usize, y: usize) -> &'a mut P where P: 'a {
		let idx = index::pixel_idxs_unchecked::<P>(self.dimensions(), x, y);
		<P as Pixel>::from_slice_mut(&mut self.buffer_mut()[idx])
	}

	fn row_mut<'a: 'p>(&'a mut self, row: usize) -> <Self::RowsMut<'a> as ImageRowsMut<'a, P>>::RowMut where P: 'a {
        let row_idxs = index::row_idxs::<P>(self.dimensions(), row);
        SliceRowMut(&mut self.buffer_mut()[row_idxs])
    }

	fn rows_mut<'a: 'p>(&'a mut self) -> Self::RowsMut<'a> where P: 'a {
        let (dims, buf) = self.buffer_dims_mut();
        RowsMut { dims, buf }
    }

	fn pixels_mut<'a: 'p>(&'a mut self) -> Self::PixelsMut<'a> where P: 'a {
        let (dims, buf) = self.buffer_dims_mut();
        SlicePixelsMut { dims, buf }
    }

	fn enumerate_pixels_mut<'a: 'p>(&'a mut self) -> <Self::PixelsMut<'a> as ImagePixelsMut<'a, P>>::Enumerated where P: 'a {
        let (dims, buf) = self.buffer_dims_mut();
		EnumeratePixelsMut {
			dims,
            buf,
			offset: 0,
		}
	}

    fn window_mut<'a: 'p>(&'a mut self, x: usize, y: usize, width: usize, height: usize) -> ImageSliceMut<'a, P> where P: 'a {
        todo!()
    }

	fn apply(&'p mut self, mut update: impl FnMut(&mut P) -> ()) {
		if self.is_packed() {
			for slice in self.buffer_mut().chunks_exact_mut(<P as Pixel>::CHANNEL_COUNT) {
				update(<P as Pixel>::from_slice_mut(slice));
			}
		} else {
			for pixel in self.pixels_mut() {
				update(pixel);
			}
		}
	}
}


pub struct SliceRow<'a, P: Pixel>(&'a [P::Subpixel]);

impl<'a, P: Pixel> ImageRow<'a, P> for SliceRow<'a, P> {
    fn into_slice(self) -> &'a [<P as Pixel>::Subpixel] {
        self.0
    }

    fn as_slice(&self) -> &'a [<P as Pixel>::Subpixel] {
        self.0
    }
}


pub struct SliceRowMut<'a, P: Pixel>(&'a mut [P::Subpixel]);

impl<'a, P: Pixel> ImageRow<'a, P> for SliceRowMut<'a, P> {
    fn into_slice(self) -> &'a [<P as Pixel>::Subpixel] {
        self.0
    }
    
    fn as_slice(&self) -> &[<P as Pixel>::Subpixel] {
        self.0
    }
}

impl<'a, P: Pixel> ImageRowMut<'a, P> for SliceRowMut<'a, P> {
    fn into_slice_mut(self) -> &'a mut [<P as Pixel>::Subpixel] {
        self.0
    }

    fn as_slice_mut(&mut self) -> &mut [<P as Pixel>::Subpixel] {
        self.0
    }
}


#[derive(Clone, Copy)]
pub struct SliceRows<'a, P: Pixel> {
    buf: &'a [P::Subpixel],
    dims: &'a ImageDimensions,
}

impl<'a, P: Pixel + 'a> ImageRows<'a, P> for SliceRows<'a, P> {
    type Row = SliceRow<'a, P>;
}

impl<'a, P: Pixel + 'a> IntoIterator for SliceRows<'a, P> {
	type Item = (usize, SliceRow<'a, P>);

	type IntoIter = SliceRowsIter<'a, P>;

	fn into_iter(self) -> Self::IntoIter {
		SliceRowsIter {
			buf: self.buf,
            dims: self.dims,
			y: 0,
		}
	}
}

pub struct SliceRowsIter<'a, P: Pixel> {
    buf: &'a [P::Subpixel],
    dims: &'a ImageDimensions,
	y: usize,
}

impl<'a, P: Pixel + 'a> Iterator for SliceRowsIter<'a, P> {
	type Item = (usize, SliceRow<'a, P>);

	fn next(&mut self) -> Option<Self::Item> {
		let y = self.y;
        let idxs = index::row_idxs_checked::<P>(self.dims, y)?;
        self.y += 1;
        Some((y, SliceRow(&self.buf[idxs])))
	}

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, P: Pixel + 'a> ExactSizeIterator for SliceRowsIter<'a, P> {
    fn len(&self) -> usize {
        self.dims.height.saturating_sub(self.y) * self.dims.width
    }
}

pub struct RowsMut<'a, P: Pixel> {
    buf: &'a mut [P::Subpixel],
    dims: &'a ImageDimensions,
}

impl<'a, P: Pixel + 'a> IntoIterator for RowsMut<'a, P> {
    type Item = (usize, SliceRowMut<'a, P>);

    type IntoIter = RowsMutIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        RowsMutIter::new(self.buf, self.dims)
    }
}

impl<'a, P: Pixel + 'a> ImageRowsMut<'a, P> for RowsMut<'a, P> {
    type RowMut = SliceRowMut<'a, P>;
}

pub struct RowsMutIter<'a, P: Pixel> {
    chunks: ChunksMut<'a, P::Subpixel>,
    width: usize,
    y: usize,
}

impl<'a, P: Pixel + 'a> RowsMutIter<'a, P> {
    fn new(data: &'a mut [P::Subpixel], dims: &'a ImageDimensions) -> Self {
        // Truncate trailing stride if not a full row
        let end = if data.len() % dims.stride < dims.width {
            data.len() % dims.stride
        } else {
            (data.len() % dims.stride) + dims.width
        };

        let chunks = (&mut data[..end]).chunks_mut(dims.stride * <P as Pixel>::CHANNEL_COUNT);
        Self {
            chunks,
            width: dims.width,
            y: 0,
        }
    }
}

impl<'a, P: Pixel + 'a> Iterator for RowsMutIter<'a, P> {
    type Item = (usize, SliceRowMut<'a, P>);

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.chunks.next()?;
        let chunk = &mut chunk[..self.width];

        let y = self.y;
        self.y += 1;
        Some((y, SliceRowMut(chunk)))
    }
}


pub struct Pixels<'a, P: Pixel + 'a> {
    buf: &'a [P::Subpixel],
	dims: &'a ImageDimensions,
}

impl<'a, P: Pixel> ImagePixels<'a, P> for Pixels<'a, P> {
    type Enumerated = SliceEnumeratePixels<'a, P>;
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

    type IntoIter = SlicePixelsIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        let rows = self.buf
			.chunks_exact(self.dims.stride * <P as Pixel>::CHANNEL_COUNT);
		SlicePixelsIter {
			rows: SliceRowsIter {
                buf: self.buf,
                dims: self.dims,
                y: 0,
            },
			row: None,
		}
    }
}


pub struct SlicePixelsIter<'a, P: Pixel + 'a> {
	rows: SliceRowsIter<'a, P>,
	row: Option<ChunksExact<'a, P::Subpixel>>,
}

impl<'a, P: Pixel + 'a> Iterator for SlicePixelsIter<'a, P> {
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

impl<'a, P: Pixel + 'a> ExactSizeIterator for SlicePixelsIter<'a, P> {
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

pub struct SlicePixelsMutIter<'a, P: Pixel + 'a> {
	rows: RowsMutIter<'a, P>,
	row: Option<ChunksExactMut<'a, P::Subpixel>>,
}

impl<'a, P: Pixel + 'a> Iterator for SlicePixelsMutIter<'a, P> {
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
			let num_channels = <P as Pixel>::CHANNEL_COUNT;
            let row_iter = row.into_slice_mut()
                .chunks_exact_mut(num_channels);
			self.row = Some(row_iter);
		}
	}
}

pub struct SlicePixelsMut<'a, P: Pixel> {
	dims: &'a ImageDimensions,
	buf: &'a mut [P::Subpixel],
}

impl<'a, P: Pixel> Index<(usize, usize)> for SlicePixelsMut<'a, P> {
    type Output = P;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let idxs = index::pixel_idxs::<P>(self.dims, x, y);
        let slice = &self.buf[idxs];
        <P as Pixel>::from_slice(slice)
    }
}

impl<'a, P: Pixel> IndexMut<(usize, usize)> for SlicePixelsMut<'a, P> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        let idxs = index::pixel_idxs::<P>(self.dims, x, y);
        let slice = &mut self.buf[idxs];
        <P as Pixel>::from_slice_mut(slice)
    }
}

impl<'a, P: Pixel + 'a> IntoIterator for SlicePixelsMut<'a, P> {
    type Item = &'a mut P;

    type IntoIter = SlicePixelsMutIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        SlicePixelsMutIter {
			rows: RowsMutIter::new(self.buf, self.dims),
			row: None,
		}
    }
}

impl<'a, P: Pixel + 'a> ImagePixelsMut<'a, P> for SlicePixelsMut<'a, P> {
	type Enumerated = EnumeratePixelsMut<'a, P>;
}

pub struct SliceEnumeratePixels<'a, P: Pixel> {
    dims: &'a ImageDimensions,
    buf: &'a [P::Subpixel],
	offset: usize,
}

impl<'a, P: Pixel + 'a> Iterator for SliceEnumeratePixels<'a, P> {
    type Item = ((usize, usize), &'a P);

    fn next(&mut self) -> Option<Self::Item> {
		let dims = self.dims;

		let idx = dims.index_for_offset_unchecked(self.offset);
		let idx = if idx.x == dims.width {
			self.offset += dims.stride - dims.width;
			dims.index_for_offset_unchecked(self.offset)
		} else {
			idx
		};

		if idx.y > dims.height {
			return None;
		}

		let num_channels = <P as Pixel>::CHANNEL_COUNT;
		let subpixel_offset = self.offset * num_channels;

		self.offset += 1;

		let slice = &self.buf[subpixel_offset..subpixel_offset+num_channels];
		let pixel = <P as Pixel>::from_slice(slice);
		Some(((idx.x, idx.y), pixel))
    }
}

pub struct EnumeratePixelsMut<'a, P: Pixel> {
	dims: &'a ImageDimensions,
    buf: &'a mut [P::Subpixel],
	offset: usize,
}

impl<'a, P: Pixel + 'a> Iterator for EnumeratePixelsMut<'a, P> {
    type Item = ((usize, usize), &'a mut P);

    fn next(&mut self) -> Option<Self::Item> {
        let dims = self.dims;

		let idx = dims.index_for_offset_unchecked(self.offset);
		let idx = if idx.x == dims.width {
			self.offset += dims.stride - dims.width;
			dims.index_for_offset_unchecked(self.offset)
		} else {
			idx
		};

		if idx.y > dims.height {
			return None;
		}

		let num_channels = <P as Pixel>::CHANNEL_COUNT;
		let subpixel_offset = self.offset * num_channels;

		self.offset += 1;

		let slice: &'a mut [P::Subpixel] = &mut self.buf[subpixel_offset..subpixel_offset+num_channels];
		let pixel = <P as Pixel>::from_slice_mut(slice);
		Some(((idx.x, idx.y), pixel))
    }
}