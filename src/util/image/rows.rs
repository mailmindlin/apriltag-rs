use std::{slice::ChunksMut, ops::{Index, IndexMut}};

use super::{Pixel, ImageDimensions, index, SubpixelArray};
pub struct Row<'a, P: Pixel>(pub(super) &'a [P::Subpixel]);

impl<'a, P: Pixel> Row<'a, P> {
    pub fn into_slice(self) -> &'a [<P as Pixel>::Subpixel] {
        self.0
    }

    pub fn as_slice(&self) -> &'a [<P as Pixel>::Subpixel] {
        self.0
    }
}

impl<'a, P: Pixel> Index<usize> for Row<'a, P> {
    type Output = P::Value;

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * P::CHANNEL_COUNT;
        P::slice_to_value(&self.0[start..start+P::CHANNEL_COUNT])
    }
}

pub struct RowMut<'a, P: Pixel>(pub(super) &'a mut [P::Subpixel]);

impl<'a, P: Pixel> RowMut<'a, P> {
    pub fn into_slice(self) -> &'a [<P as Pixel>::Subpixel] {
        self.0
    }
    
    pub fn as_slice(&self) -> &[<P as Pixel>::Subpixel] {
        self.0
    }
}

impl<'a, P: Pixel> RowMut<'a, P> {
    pub fn into_slice_mut(self) -> &'a mut [<P as Pixel>::Subpixel] {
        self.0
    }

    pub fn as_slice_mut(&mut self) -> &mut [<P as Pixel>::Subpixel] {
        self.0
    }
}

impl<'a, P: Pixel> Index<usize> for RowMut<'a, P> {
    type Output = P::Value;

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * P::CHANNEL_COUNT;
        P::slice_to_value(&self.0[start..start+P::CHANNEL_COUNT])
    }
}

impl<'a, P: Pixel> IndexMut<usize> for RowMut<'a, P> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * P::CHANNEL_COUNT;
        let a = &mut self.0[start..start+P::CHANNEL_COUNT];
        P::slice_to_value_mut(a)
    }
}


#[derive(Clone, Copy)]
pub struct Rows<'a, P: Pixel> {
    pub(super) buf: &'a SubpixelArray<P>,
    pub(super) dims: &'a ImageDimensions,
}

impl<'a, P: Pixel + 'a> IntoIterator for Rows<'a, P> {
	type Item = (usize, Row<'a, P>);

	type IntoIter = RowsIter<'a, P>;

	fn into_iter(self) -> Self::IntoIter {
		RowsIter {
			buf: self.buf,
            dims: self.dims,
			y: 0,
		}
	}
}

pub struct RowsIter<'a, P: Pixel> {
    buf: &'a [P::Subpixel],
    pub(super) dims: &'a ImageDimensions,
	pub(super) y: usize,
}

impl<'a, P: Pixel> RowsIter<'a, P> {
    pub(super) fn new(data: &'a SubpixelArray<P>, dims: &'a ImageDimensions) -> Self {
        Self {
            buf: data,
            dims,
            y: 0,
        }
    }
}
impl<'a, P: Pixel + 'a> Iterator for RowsIter<'a, P> {
	type Item = (usize, Row<'a, P>);

	fn next(&mut self) -> Option<Self::Item> {
		let y = self.y;
        let idxs = index::row_idxs_checked::<P>(self.dims, y)?;
        self.y += 1;
        Some((y, Row(&self.buf[idxs])))
	}

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, P: Pixel + 'a> ExactSizeIterator for RowsIter<'a, P> {
    fn len(&self) -> usize {
        self.dims.height.saturating_sub(self.y) * self.dims.width
    }
}

pub struct RowsMut<'a, P: Pixel> {
    pub(super) buf: &'a mut [P::Subpixel],
    pub(super) dims: &'a ImageDimensions,
}

impl<'a, P: Pixel + 'a> IntoIterator for RowsMut<'a, P> {
    type Item = (usize, RowMut<'a, P>);

    type IntoIter = RowsMutIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        RowsMutIter::new(self.buf, self.dims)
    }
}

pub struct RowsMutIter<'a, P: Pixel> {
    chunks: ChunksMut<'a, P::Subpixel>,
    width: usize,
    pub(super) y: usize,
}

impl<'a, P: Pixel + 'a> RowsMutIter<'a, P> {
    pub(super) fn new(data: &'a mut [P::Subpixel], dims: &'a ImageDimensions) -> Self {
        // Truncate trailing stride if not a full row
        let end = if data.len() % dims.stride < dims.width {
            data.len() - (data.len() % dims.stride)
        } else {
            data.len() - (data.len() % dims.stride) + dims.width
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
    type Item = (usize, RowMut<'a, P>);

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.chunks.next()?;
        let chunk = &mut chunk[..self.width];

        let y = self.y;
        self.y += 1;
        Some((y, RowMut(chunk)))
    }
}

#[cfg(test)]
mod test {
    use crate::util::ImageY8;

    #[test]
    fn count_rows() {
        let img = ImageY8::zeroed_with_stride(10, 15, 20);
        let num_rows = img.rows()
            .into_iter()
            .count();
        assert_eq!(15, num_rows);
    }

    #[test]
    fn count_rows_mut() {
        let mut img = ImageY8::zeroed_with_stride(10, 15, 20);
        let num_rows = img.rows_mut()
            .into_iter()
            .count();
        assert_eq!(15, num_rows);
    }
}