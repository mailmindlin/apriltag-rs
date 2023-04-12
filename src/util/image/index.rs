use std::ops::{Range, RangeBounds};

use crate::util::dims::{Dimensions2D, Index2D};

use super::Pixel;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ImageDimensions {
	/// Image width (in pixels)
	pub width: usize,
	/// Image height (in pixels)
	pub height: usize,
	/// Image stride (in pixels)
	pub stride: usize,
}

impl Dimensions2D<usize> for ImageDimensions {
	fn width(&self) -> usize {
		self.width
	}

	fn height(&self) -> usize {
		self.height
	}

	fn contains(&self, index: &Index2D<usize>) -> bool {
		index.x < self.width && index.y < self.height
	}

	fn offset_unchecked(&self, index: &Index2D<usize>) -> usize {
		self.stride * index.y + index.x
	}

	fn index_for_offset_unchecked(&self, offset: usize) -> Index2D<usize> {
		let y = offset / self.stride;
		let x = offset % self.stride;
		Index2D { x, y }
	}
}

#[inline(always)]
pub(super) fn pixel_idxs<P: Pixel>(dims: &ImageDimensions, x: usize, y: usize) -> Range<usize> {
    match pixel_idxs_checked::<P>(dims, x, y) {
        Some(value) => value,
        None => {
            panic!("Image index {:?} out of bounds {:?}", (x, y), dims);
        }
    }
}

#[inline(always)]
pub(super) fn pixel_idxs_checked<P: Pixel>(dims: &ImageDimensions, x: usize, y: usize) -> Option<Range<usize>> {
    let cell_offset = dims.offset_checked(&Index2D { x, y })?;
    let num_channels = <P as Pixel>::CHANNEL_COUNT;
    let start_idx = cell_offset * num_channels;
    Some(start_idx..start_idx + num_channels)
}

#[inline(always)]
pub(super) fn pixel_idxs_unchecked<P: Pixel>(dims: &ImageDimensions, x: usize, y: usize) -> Range<usize> {
    let cell_offset = dims.offset_unchecked(&Index2D { x, y });
    let num_channels = <P as Pixel>::CHANNEL_COUNT;
    let start_idx = cell_offset * num_channels;
    start_idx..start_idx + num_channels
}

pub(super) fn row_idxs<P: Pixel>(dims: &ImageDimensions, y: usize) -> Range<usize> {
    match row_idxs_checked::<P>(dims, y) {
        Some(value) => value,
        None => {
            panic!("Image row {:?} out of bounds {:?}", y, dims);
        }
    }
}

pub(super) const fn row_idxs_unchecked<P: Pixel>(dims: &ImageDimensions, y: usize) -> Range<usize> {
    let row_start = y * dims.stride * P::CHANNEL_COUNT;
    let row_end = row_start + dims.width * P::CHANNEL_COUNT;
    row_start..row_end
}

pub(super) const fn row_idxs_checked<P: Pixel>(dims: &ImageDimensions, y: usize) -> Option<Range<usize>> {
    if y >= dims.height {
        None
    } else {
        Some(row_idxs_unchecked::<P>(dims, y))
    }
}

pub(super) fn slice_idxs<P: Pixel>(dims: &ImageDimensions, x: impl RangeBounds<usize>, y: impl RangeBounds<usize>) -> (ImageDimensions, Range<usize>) {
    fn resolve(range: impl RangeBounds<usize>, length: usize) -> Range<usize> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(v) => *v,
            std::ops::Bound::Excluded(v) => v + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(v) => *v,
            std::ops::Bound::Excluded(v) => v - 1,
            std::ops::Bound::Unbounded => length - 1,
        };

        Range { start, end }
    }

    let x = resolve(x, dims.width);
    assert!(x.start <= x.end);
    assert!(x.end < dims.width);
    let y = resolve(y, dims.height);
    assert!(y.start <= y.end);
    assert!(y.end < dims.height);

    if x.start == 0 && x.end == dims.width - 1 {
        if y.start == 0 && y.end == dims.height - 1 {
            return (*dims, 0..(dims.area() * P::CHANNEL_COUNT));
        }
    }

    let start_idx = dims.offset_unchecked(&Index2D { x: x.start, y: y.start });
    let end_idx = dims.offset_unchecked(&Index2D { x: x.end, y: y.end });
    let width = x.end - x.start;
    let height = y.end - y.start;
    let stride = dims.offset_unchecked(&Index2D { x: x.start, y: y.start + 1 });

    (
        ImageDimensions {
            width,
            height,
            stride,
        },
        (start_idx * P::CHANNEL_COUNT..end_idx * P::CHANNEL_COUNT),
    )
}