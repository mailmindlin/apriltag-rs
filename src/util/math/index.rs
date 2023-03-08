#[derive(Debug, Clone, Copy)]
pub struct OutOfBoundsError<D: Dimensions2D> {
    /// Dimensions (that the index was out-of-bounds of)
	pub dims: D,
    /// Out-of-bounds index
	pub index: Index2D,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Index2D {
    pub x: usize,
    pub y: usize,
}

impl From<(usize, usize)> for Index2D {
    fn from(value: (usize, usize)) -> Self {
        let (x, y) = value;
        Self {
            x,
            y
        }
    }
}

trait Dimensions2D {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
}

impl<T: Dimensions2D> From<T> for (usize, usize) {
    fn from(value: T) -> Self {
        let width = value.width();
        let height = value.height();
        (width, height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SimpleDimensions2D {
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StrideDimensions2D {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl From<SimpleDimensions2D> for StrideDimensions2D {
    fn from(value: SimpleDimensions2D) -> Self {
        let SimpleDimensions2D { width, height } = value;
        Self {
            width,
            height,
            stride: width,
        }
    }
}

impl StrideDimensions2D {
    /// Check that index represents a point inside the area defined by these dimensions
	#[inline]
	pub const fn contains(&self, index: &Index2D) -> bool {
        (index.x < self.width) && (index.y < self.height)
	}

	#[inline]
	fn assert_contains(&self, index: &Index2D) -> Result<(), OutOfBoundsError<Self>> {
		if self.contains(index) {
			Ok(())
		} else {
			Err(OutOfBoundsError {
				dims: *self,
				index: *index,
			})
		}
	}

	#[inline(always)]
	pub fn compute_offset_unchecked(&self, index: &Index2D) -> usize {
        (self.width * index.y) + index.x
	}

	#[inline]
	pub fn compute_offset(&self, index: &Index2D) -> Result<usize, OutOfBoundsError<Self>> {
		self.assert_contains(&index)?;
		Ok(self.compute_offset_unchecked(index))
	}

	#[inline]
	pub fn index_for_offset_unchecked(&self, offset: usize) -> Index2D {
		Index2D {
			y: offset.div_floor(self.cols),
			x: offset % self.cols,
		}
	}

	#[inline]
	pub fn index_for_offset(&self, offset: usize) -> Result<Index2D, OutOfBoundsError<Self>> {
		let index = self.index_for_offset_unchecked(offset);
		self.assert_contains(&index)?;
		Ok(index)
	}

	/// Get number of elements in a matrix with these dimensions
	#[inline]
	pub fn len(&self) -> usize {
		self.width * self.height
	}

	#[inline]
	pub fn is_square(&self) -> bool {
		self.rows == self.cols
	}
}