use std::ops::{Range, RangeBounds, Index, Deref, DerefMut, IndexMut};

use crate::util::dims::{Dimensions2D, Index2D};

use super::{Pixel, ImageBuffer};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ImageDimensions {
	/// Image width (in pixels)
	pub width: usize,
	/// Image height (in pixels)
	pub height: usize,
	/// Image stride (in subpixels)
	pub stride: usize,
}

impl ImageDimensions {
    pub(super) const fn width_subpixels<P: Pixel>(&self) -> usize {
        self.width * P::CHANNEL_COUNT
    }
}

impl Dimensions2D<usize> for ImageDimensions {
    #[inline]
	fn width(&self) -> usize {
		self.width
	}

    #[inline]
	fn height(&self) -> usize {
		self.height
	}

    #[inline]
	fn contains(&self, index: &Index2D<usize>) -> bool {
		index.x < self.width && index.y < self.height
	}

	// fn offset_unchecked(&self, index: &Index2D<usize>) -> usize {
	// 	self.stride * index.y + index.x
	// }

	// fn index_for_offset_unchecked(&self, offset: usize) -> Index2D<usize> {
	// 	let y = offset / self.stride;
	// 	let x = offset % self.stride;
	// 	Index2D { x, y }
	// }
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
    if dims.contains(&Index2D { x, y }) {
        Some(pixel_idxs_unchecked::<P>(dims, x, y))
    } else {
        None
    }
}

#[inline(always)]
pub(super) fn pixel_idxs_unchecked<P: Pixel>(dims: &ImageDimensions, x: usize, y: usize) -> Range<usize> {
    let num_channels = <P as Pixel>::CHANNEL_COUNT;
    let start_idx = (x * num_channels) + (y * dims.stride);
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
    let row_start = y * dims.stride;
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

fn resolve_range(range: impl RangeBounds<usize>, length: usize) -> Range<usize> {
    let start = match range.start_bound() {
        std::ops::Bound::Included(v) => *v,
        std::ops::Bound::Excluded(v) => v + 1,
        std::ops::Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        std::ops::Bound::Included(v) => *v + 1,
        std::ops::Bound::Excluded(v) => *v,
        std::ops::Bound::Unbounded => length,
    };

    Range { start, end }
}

pub(super) fn slice_idxs<P: Pixel>(dims: &ImageDimensions, x: impl RangeBounds<usize>, y: impl RangeBounds<usize>) -> (ImageDimensions, Range<usize>) {
    let x = resolve_range(x, dims.width);
    assert!(x.start <= x.end);
    assert!(x.end <= dims.width);
    let y = resolve_range(y, dims.height);
    assert!(y.start <= y.end);
    assert!(y.end <= dims.height);

    if x.start == 0 && x.end == dims.width - 1 {
        if y.start == 0 && y.end == dims.height - 1 {
            return (*dims, 0..(dims.area() * P::CHANNEL_COUNT));
        }
    }

    let num_channels = <P as Pixel>::CHANNEL_COUNT;
    let start_idx = (x.start * num_channels) + (y.start * dims.stride);
    let end_idx = ((x.end - 1) * num_channels) + ((y.end - 1) * dims.stride) + num_channels;

    let width = x.end - x.start;
    let height = y.end - y.start;
    // let stride = dims.offset_unchecked(&Index2D { x: x.start, y: y.start + 1 }) - start_idx;
    let stride = dims.stride;

    (
        ImageDimensions {
            width,
            height,
            stride,
        },
        (start_idx..end_idx),
    )
}

impl<P: Pixel, Container: Deref<Target = [<P as Pixel>::Subpixel]>> Index<(usize, usize)> for ImageBuffer<P, Container> {
	type Output = P::Value;

	fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        #[cfg(debug_assertions)]
        let idx = match self.pixel_idxs_checked(x, y) {
            Some(idx) => idx,
            None => panic!("Range index ({x}, {y}) out of range for image size ({}, {})", self.width(), self.height()),
        };
        #[cfg(not(debug_assertions))]
        let idx = self.pixel_idxs(x, y);
		let slice = &self.data[idx];
		<P as Pixel>::slice_to_value(slice)
	}
}

/// Get a pixel
impl<P: Pixel, Container: DerefMut<Target = [<P as Pixel>::Subpixel]>> IndexMut<(usize, usize)> for ImageBuffer<P, Container> {
	fn index_mut(&mut self, (x,y): (usize, usize)) -> &mut Self::Output {
		let idxs = self.pixel_idxs(x, y);
		let slice = &mut self.data[idxs];
		<P as Pixel>::slice_to_value_mut(slice)
	}
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use super::resolve_range;

    #[test]
    fn resolve_ranges() {
        assert_eq!(resolve_range(0..1, 10), Range { start: 0, end: 1 });
        assert_eq!(resolve_range(0..=1, 10), Range { start: 0, end: 2 });
        assert_eq!(resolve_range(0.., 10), Range { start: 0, end: 10 });
        assert_eq!(resolve_range(..1, 10), Range { start: 0, end: 1 });
        assert_eq!(resolve_range(.., 10), Range { start: 0, end: 10 });
    }
}