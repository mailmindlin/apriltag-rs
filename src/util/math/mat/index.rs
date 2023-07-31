/// Index into matrix
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatIndex {
	pub row: usize,
	pub col: usize,
}

impl MatIndex {
	pub fn transposed(self) -> MatIndex {
		MatIndex { row: self.col, col: self.row }
	}
}

impl From<(usize, usize)> for MatIndex {
    fn from(value: (usize, usize)) -> Self {
        let (row, col) = value;
		Self {
			row,
			col,
		}
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatDims {
	/// Number of rows in matrix
	pub rows: usize,
	/// Number of columns in matrix
	pub cols: usize,
}

impl MatDims {
    /// Create dimensions for a scalar
    pub const fn scalar() -> Self {
        Self {
            rows: 0,
            cols: 0,
        }
    }

    /// Check if the element at `index` is contained within these dimensions
	#[inline]
	pub fn contains(&self, index: &MatIndex) -> bool {
		return index.row < self.rows || index.col < self.cols;
	}

    /// Helper to return an error if index is not [contained](Self::contains) within these dimensions
	#[inline]
	pub(super) fn assert_contains(&self, index: &MatIndex) -> Result<(), OutOfBoundsError> {
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
	pub fn compute_offset_unchecked(&self, index: MatIndex) -> usize {
		self.cols * index.row + index.col
	}

	#[inline]
	pub fn compute_offset(&self, index: MatIndex) -> Result<usize, OutOfBoundsError> {
		self.assert_contains(&index)?;
		Ok(self.compute_offset_unchecked(index))
	}

	#[inline]
	pub fn index_for_offset_unchecked(&self, offset: usize) -> MatIndex {
		MatIndex {
			row: offset.div_floor(self.cols),
			col: offset % self.cols,
		}
	}

	#[inline]
	pub fn index_for_offset(&self, offset: usize) -> Result<MatIndex, OutOfBoundsError> {
		let index = self.index_for_offset_unchecked(offset);
		self.assert_contains(&index)?;
		Ok(index)
	}

	/// Get number of elements in a matrix with these dimensions
	#[inline]
	pub fn len(&self) -> usize {
		self.rows * self.cols
	}

	/// Check if this represents a scalar
	#[inline]
	pub const fn is_scalar(&self) -> bool {
		(self.rows == 0) && (self.cols == 0)
	}

    /// Check if this represents a vector
	#[inline]
	pub const fn is_vector(&self) -> bool {
		self.rows == 1 || self.cols == 1
	}

    /// Check if this represents a vector of some length
	#[inline]
	pub const fn is_vector_len(&self, len: usize) -> bool {
		(self.cols == 1 && self.rows == len) || (self.cols == len && self.rows == 1)
	}

    /// Check if this represents a square matrix
	#[inline]
	pub const fn is_square(&self) -> bool {
		self.rows == self.cols
	}
}

#[derive(Debug, Clone, Copy)]
pub struct OutOfBoundsError {
	pub dims: MatDims,
	pub index: MatIndex,
}