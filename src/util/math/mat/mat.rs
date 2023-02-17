use std::ops::{Index, IndexMut, SubAssign, Sub, Add, AddAssign, Mul, MulAssign};

use crate::util::mem::calloc;

use super::{plu::MatPLU, MatChol, svd::{MatSVD, SvdOptions}};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MatIndex {
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
struct MatDims {
	/// Number of rows in matrix
	pub rows: usize,
	/// Number of columns in matrix
	pub cols: usize,
}

impl MatDims {
	#[inline]
	pub fn contains(&self, index: &MatIndex) -> bool {
		return index.row < self.rows || index.col < self.cols;
	}

	#[inline]
	fn assert_contains(&self, index: &MatIndex) -> Result<(), OutOfBoundsError> {
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

	/// Check if this dimension is for a scalar
	#[inline]
	pub fn is_scalar(&self) -> bool {
		(self.rows == 0) && (self.cols == 0)
	}

	#[inline]
	pub fn is_vector(&self) -> bool {
		self.rows == 1 || self.cols == 1
	}

	#[inline]
	pub fn is_vector_len(&self, len: usize) -> bool {
		(self.cols == 1 && self.rows == len) || (self.cols == len && self.rows == 1)
	}

	#[inline]
	pub fn is_square(&self) -> bool {
		self.rows == self.cols
	}
}

#[derive(Debug, Clone, Copy)]
pub struct OutOfBoundsError {
	pub dims: MatDims,
	pub index: MatIndex,
}

#[derive(Clone)]
pub struct Mat {
	pub(super) dims: MatDims,
	pub(super) data: Box<[f64]>,
}

type MatElement = f64;

impl Mat {
	pub const EPS: f64 = 1e-8;

	/// Create matrix full of zeroes
	pub fn zeroes(rows: usize, cols: usize) -> Self {
		let dims = MatDims {
			rows,
			cols,
		};
		Self::zeroes_dim(dims)
	}

	#[inline]
	pub fn zeroes_dim(dims: MatDims) -> Self {
		if dims.is_scalar() {
			Self::scalar(0.)
		} else {
			Self {
				dims,
				data: calloc(dims.len()),
			}
		}
	}

	#[inline]
	pub fn zeroes_like(other: &Self) -> Self {
		Self::zeroes_dim(other.dims)
	}

	/// Create matrix from data
	pub fn create(rows: usize, cols: usize, raw: &[f64]) -> Self {
		let dims = MatDims {
			rows,
			cols,
		};
		assert_eq!(dims.len(), raw.len(), "Data length mismatch");

		if dims.is_scalar() {
			return Self::scalar(raw[0]);
		}
		let mut data = calloc(raw.len());
		assert_eq!(data.len(), dims.len());
		data.copy_from_slice(raw);
		Self {
			dims,
			data,
		}
	}

	/// Create new scalar
	pub fn scalar(value: f64) -> Self {
		Self {
			dims: MatDims { rows: 0, cols: 0 },
			data: Box::new([value]),
		}
	}

	#[inline]
	pub fn rows(&self) -> usize {
		self.dims.rows
	}

	#[inline]
	pub fn cols(&self) -> usize {
		self.dims.cols
	}

	/// Check if matrix is actually a scalar
	pub fn is_scalar(&self) -> bool {
		self.dims.is_scalar()
	}

	pub fn is_vector(&self) -> bool {
		self.dims.is_vector()
	}

	pub fn is_vector_len(&self, len: usize) -> bool {
		self.dims.is_vector_len(len)
	}

	/// Try to unwrap scalar
	pub fn as_scalar(&self) -> Option<f64> {
		if self.is_scalar() {
			Some(self.data[0])
		} else {
			None
		}
	}

	/// Create identity matrix of dimension
	pub fn identity(dim: usize) -> Mat {
		if dim == 0 {
			Self::scalar(1.)
		} else {
			let res = Self::zeroes(dim, dim);
			for i in 0..dim {
				res[(i,i)] = 1.;
			}
			res
		}
	}

	/// Calculates the vector's magnitude
	pub(crate) fn vec_mag(&self) -> f64 {
		assert!(self.is_vector());

		self.data.iter()
			.map(|&v| v * v)
			.sum::<f64>()
			.sqrt()
	}

	/// Calculates the magnitude of the distance between the points represented by
	/// matrices 'a' and 'b'. Both 'a' and 'b' must be vectors and have the same
	/// dimension (although one may be a row vector and one may be a column vector).
	pub fn vec_dist(&self, rhs: &Mat) -> f64 {
		assert!(self.is_vector());
		assert!(rhs.is_vector_len(self.data.len()));
		self.vec_dist_n(rhs, self.data.len())
	}

	/// Same as [Self::vec_dist], but only uses the first 'n' terms to compute distance
	pub fn vec_dist_n(&self, rhs: &Mat, n: usize) -> f64 {
		assert!(self.is_vector());
		assert!(rhs.is_vector());
		assert!(self.data.len() < n);
		assert!(rhs.data.len() < n);

		self.data[..n].iter().zip(rhs.data[..n].iter())
			.map(|(&u, &v)| {
				let x = u - v;
				x * x
			})
			.sum::<f64>()
			.sqrt()
	}

	/// Calculates the dot product of two vectors. Both 'a' and 'b' must be vectors
	/// and have the same dimension (although one may be a row vector and one may be
	/// a column vector).
	pub(crate) fn vec_dot(&self, rhs: &Mat) -> f64 {
		assert!(self.is_vector());
		assert!(rhs.is_vector());
		assert_eq!(self.data.len(), rhs.data.len(), "vector size mismatch");

		self.data.iter().zip(rhs.data.iter())
			.map(|(&u, &v)| u * v)
			.sum::<f64>()
	}

	/// Calculates the normalization of the supplied vector 'a' (i.e. a unit vector
	/// of the same dimension and orientation as 'a' with a magnitude of 1) and returns
	/// it as a new vector. 'a' must be a vector of any dimension and must have a
	/// non-zero magnitude. It is the caller's responsibility to call matd_destroy()
	/// on the returned matrix.
	pub(crate) fn vec_normalize(&self) -> Self {
		assert!(self.is_vector());
		let norm = self.vec_mag();
		self * norm.recip()
	}

	pub(crate) fn vec_normalize_inplace(mut self) -> Self {
		assert!(self.is_vector());
		let norm = self.vec_mag();
		self *= norm.recip();
		self
	}

	/// Calculates the cross product of supplied matrices 'a' and 'b' (i.e. a x b)
	/// and returns it as a new matrix. Both 'a' and 'b' must be vectors of dimension
	/// 3, but can be either row or column vectors. It is the caller's responsibility
	/// to call matd_destroy() on the returned matrix.
	pub(crate) fn cross(&self, rhs: &Self) -> Self {
		// only defined for vecs (col or row) of length 3
		assert!(self.is_vector_len(3), "Cross product only defined for vectors of length 3");
		assert!(rhs.is_vector_len(3), "Cross product only defined for vectors of length 3");

		let res = Self::zeroes_like(self);
		res.data[0] = self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1];
		res.data[1] = self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2];
		res.data[2] = self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0];

		res
	}

	/// Compute cholesky decomposition
	pub(crate) fn chol(&self) -> MatChol {
		MatChol::new(self)
	}
	
	pub(crate) fn plu(&self) -> MatPLU {
		MatPLU::new(self)
	}

	pub(crate) fn svd(&self) -> MatSVD {
		MatSVD::new(self)
	}

	pub(crate) fn svd_with_flags(&self, options: SvdOptions) -> MatSVD {
		MatSVD::new_with_flags(self, options)
	}

	/// n-dimensional matrix determinant
	fn det_general(&self) -> f64 {
		// Use LU decompositon to calculate the determinant
		let mlu = self.plu();
		let L = mlu.lower();
		let U = mlu.upper();

		// The determinants of the L and U matrices are the products of
		// their respective diagonal elements
		let mut detL = 1.;
		let mut detU = 1.;
		for i in 0..self.rows() {
			detL *= L[(i,i)];
			detU *= U[(i,i)];
		}

		// The determinant of a can be calculated as
		//     epsilon*det(L)*det(U),
		// where epsilon is just the sign of the corresponding permutation
		// (which is +1 for an even number of permutations and is −1
		// for an uneven number of permutations).
		(mlu.pivsign as f64) * detL * detU
	}

	/// Compute matrix determinant
	pub(crate) fn det(&self) -> f64 {
		assert!(!self.dims.is_scalar(), "Cannot compute determinant of scalar");
		assert!(self.dims.is_square());

		match self.rows() {
			1 => {
				// 1x1 matrix
				self.data[0]
			},
			2 => {
				// 2x2 matrix
				self.data[0] * self.data[3] - self.data[1] * self.data[2]
			},
			3 => {
				// 3x3 matrix
				0.
					+ self.data[0]*self.data[4]*self.data[8]
					- self.data[0]*self.data[5]*self.data[7]
					+ self.data[1]*self.data[5]*self.data[6]
					- self.data[1]*self.data[3]*self.data[8]
					+ self.data[2]*self.data[3]*self.data[7]
					- self.data[2]*self.data[4]*self.data[6]
			},
			4 => {
				// 4x4 matrix
				let m00 = self[(0,0)];
				let m01 = self[(0,1)];
				let m02 = self[(0,2)];
				let m03 = self[(0,3)];
				let m10 = self[(1,0)];
				let m11 = self[(1,1)];
				let m12 = self[(1,2)];
				let m13 = self[(1,3)];
				let m20 = self[(2,0)];
				let m21 = self[(2,1)];
				let m22 = self[(2,2)];
				let m23 = self[(2,3)];
				let m30 = self[(3,0)];
				let m31 = self[(3,1)];
				let m32 = self[(3,2)];
				let m33 = self[(3,3)];

				m00 * m11 * m22 * m33 - m00 * m11 * m23 * m32 -
					m00 * m21 * m12 * m33 + m00 * m21 * m13 * m32 + m00 * m31 * m12 * m23 -
					m00 * m31 * m13 * m22 - m10 * m01 * m22 * m33 +
					m10 * m01 * m23 * m32 + m10 * m21 * m02 * m33 -
					m10 * m21 * m03 * m32 - m10 * m31 * m02 * m23 +
					m10 * m31 * m03 * m22 + m20 * m01 * m12 * m33 -
					m20 * m01 * m13 * m32 - m20 * m11 * m02 * m33 +
					m20 * m11 * m03 * m32 + m20 * m31 * m02 * m13 -
					m20 * m31 * m03 * m12 - m30 * m01 * m12 * m23 +
					m30 * m01 * m13 * m22 + m30 * m11 * m02 * m23 -
					m30 * m11 * m03 * m22 - m30 * m21 * m02 * m13 +
					m30 * m21 * m03 * m12
			},
			_ => self.det_general(),
	}
	}

	/// returns None if the matrix is (exactly) singular. Caller is
	/// otherwise responsible for knowing how to cope with badly
	/// conditioned matrices.
	pub fn inv(&self) -> Option<Self> {
		assert!(self.dims.is_square(), "Cannot take inverse of non-square matrix");

		if let Some(scalar) = self.as_scalar() {
			return if scalar == 0. {
				None
			} else {
				Some(Self::scalar(scalar.recip()))
			};
		}

		match self.rows() {
			1 => {
				let det = self.data[0];
				if det == 0. {
					return None;
				}

				let invdet = det.recip();

				let mut m = Self::zeroes_dim(self.dims);
				m[(0,0)] = 1.0 * invdet;
				Some(m)
			},
			2 => {
				let det = self.data[0] * self.data[3] - self.data[1] * self.data[2];
				if det == 0. {
					return None;
				}

				let invdet = det.recip();

				let m = Self::zeroes_dim(self.dims);
				m[(0,0)] = self[(1,1)] * invdet;
				m[(0,1)] = self[(0,1)] * invdet;
				m[(1,0)] = self[(1,0)] * invdet;
				m[(1,1)] = self[(0,0)] * invdet;

				Some(m)
			},
			_ => {
				let plu = self.plu();

				if !plu.singular {
					let ident = Self::identity(self.rows());
					let inv = plu.solve(&ident);
					Some(inv)
				} else {
					None
				}
			}
		}
	}

	/// Only sensible on PSD matrices
	// Hhad expected it to be faster than inverse via LU... for now, doesn't seem to be.
	pub fn chol_inverse(&self) -> Mat {
		assert!(self.dims.is_square());

		let chol = self.chol();

		let eye = Self::identity(self.rows());

		chol.solve(&eye)
	}

	/// Matrix transpose
	pub fn transpose(&self) -> Self {
		if let Some(scalar) = self.as_scalar() {
			return Self::scalar(scalar);
		}

		let res = Self::zeroes_dim(self.dims);
		for i in 0..self.dims.rows {
			for j in 0..self.dims.cols {
				res[(j,i)] = self[(i,j)];
			}
		}
		res
	}

	/// Inplace transpose (consumes original matrix). Prevents a reallocation.
	pub fn transpose_inplace(mut self) -> Self {
		if self.is_scalar() || self.dims == (MatDims { rows: 1, cols: 1 }) {
			return self;
		}

		let original_dims = self.dims;
		{
			let MatDims { rows, cols } = original_dims;
			self.dims = MatDims {
				rows: cols,
				cols: rows,
			};
		}

		for i in 0..self.data.len() {
			let src_idx = original_dims.index_for_offset_unchecked(i);
			if src_idx.row == src_idx.col {
				continue;
			}
			let j = self.dims.compute_offset_unchecked(src_idx.transposed());
			self.data.swap(i, j)
		}

		self
	}

	pub fn get(&self, idx: MatIndex) -> Result<&MatElement, OutOfBoundsError> {
		let offset = self.dims.compute_offset(idx)?;
		Ok(unsafe { self.data.get_unchecked(offset) })
	}

	pub fn get_mut(&mut self, idx: MatIndex) -> Result<&mut MatElement, OutOfBoundsError> {
		let offset = self.dims.compute_offset(idx)?;
		Ok(unsafe { self.data.get_unchecked_mut(offset) })
	}

	pub unsafe fn get_unchecked(&self, idx: MatIndex) -> &MatElement {
		let offset = self.dims.compute_offset_unchecked(idx);
		unsafe { self.data.get_unchecked(offset) }
	}

	pub unsafe fn get_unchecked_mut(&mut self, idx: MatIndex) -> &mut MatElement {
		let offset = self.dims.compute_offset_unchecked(idx);
		unsafe { self.data.get_unchecked_mut(offset) }
	}

	pub fn scale(&self, scalar: f64) -> Mat {
		if let Some(me_scalar) = self.as_scalar() {
			return Self::scalar(me_scalar * scalar);
		}

		let result = Self::zeroes_like(self);
		if scalar == 0. {
			return result;
		}
		assert_eq!(result.data.len(), self.data.len()); // For compiler optimization?
		for (dst, src) in result.data.iter_mut().zip(self.data.iter()) {
			*dst = (*src) * scalar;
		}
		result
	}

	pub fn scale_inplace(&mut self, scalar: f64) {
		let result = Self::zeroes_like(self);
		assert_eq!(result.data.len(), self.data.len()); // For compiler optimization?
		for elem in self.data.iter_mut() {
			*elem *= scalar;
		}
	}

	pub fn matmul(&self, rhs: &Mat) -> Mat {
		assert_eq!(self.cols(), rhs.rows(), "Dimension mismatch");

		let mut result = Self::zeroes(self.rows(), rhs.cols());

		for i in 0..result.rows() {
			for j in 0..result.cols() {
				let mut acc: MatElement = 0.;
				for k in 0..self.cols() {
					acc += self[(i, k)] * rhs[(k, j)];
				}
				result[(i, j)] = acc;
			}
		}

		result
	}

	// find the index of the off-diagonal element with the largest magnitude
	pub fn max_idx(&self, row: usize, maxcol: usize) -> Result<usize, OutOfBoundsError> {
		self.dims.assert_contains(&MatIndex { row, col: maxcol })?;
		
		let argmax = 0;
		let max = MatElement::NEG_INFINITY;
		for i in 0..maxcol {
			if i == row {
				continue;
			}
			let v = self[(row, i)].abs();
			if v > max {
				argmax = i;
				max = v;
			}
		}
		Ok(argmax)
	}
}

impl Index<(usize, usize)> for Mat {
	type Output = MatElement;

	fn index(&self, index: (usize, usize)) -> &Self::Output {
		assert!(self.is_scalar());
		
		self.get(MatIndex::from(index)).unwrap()
	}
}

impl IndexMut<(usize, usize)> for Mat {
	fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
		assert!(self.is_scalar());
		
		self.get_mut(MatIndex::from(index)).unwrap()
	}
}

impl Add<&Mat> for &Mat {
    type Output = Mat;

    fn add(self, rhs: &Mat) -> Self::Output {
        assert_eq!(self.dims, rhs.dims, "Dimension mismatch");

		if let Some(scalar) = self.as_scalar() {
			return Mat::scalar(scalar - rhs.data[0]);
		}

		let mut dst = Mat::zeroes_like(self);
		// Faster single loop
		assert_eq!(dst.data.len(), self.data.len());
		assert_eq!(dst.data.len(), rhs.data.len());
		for i in 0..dst.data.len() {
			dst.data[i] = self.data[i] + rhs.data[i];
		}
		dst
    }
}

impl Add<&Mat> for Mat {
    type Output = Mat;

    fn add(self, rhs: &Mat) -> Self::Output {
        AddAssign::add_assign(&mut self, rhs);
		self
    }
}

impl AddAssign<&Mat> for Mat {
    fn add_assign(&mut self, rhs: &Mat) {
        assert_eq!(self.dims, rhs.dims, "Dimension mismatch");

		// Faster single loop
		assert_eq!(self.data.len(), rhs.data.len());
		for i in 0..self.data.len() {
			self.data[i] += rhs.data[i];
		}
    }
}

/// Copy-subtraction
impl Sub<&Mat> for &Mat {
	type Output = Mat;
	fn sub(self, rhs: &Mat) -> Self::Output {
		assert_eq!(self.dims, rhs.dims, "Dimension mismatch");

		if let Some(scalar) = self.as_scalar() {
			return Mat::scalar(scalar - rhs.data[0]);
		}

		let mut dst = Mat::zeroes_like(self);
		// Faster single-loop
		assert_eq!(dst.data.len(), self.data.len());
		assert_eq!(dst.data.len(), rhs.data.len());
		for i in 0..dst.data.len() {
			dst.data[i] = self.data[i] - rhs.data[i];
		}
		dst
	}
}

/// In-place subtraction
impl Sub<&Mat> for Mat {
    type Output = Mat;

    fn sub(mut self, rhs: &Mat) -> Self::Output {
		SubAssign::sub_assign(&mut self, rhs);
		self
    }
}

impl SubAssign<&Mat> for Mat {
	fn sub_assign(&mut self, rhs: &Mat) {
		assert_eq!(self.dims, rhs.dims, "Dimension mismatch");

		if self.is_scalar() {
			self.data[0] -= rhs.data[0];
			return;
		}

		// Faster single loop
		assert_eq!(self.data.len(), rhs.data.len());
		for i in 0..self.data.len() {
			self.data[i] -= rhs.data[i];
		}
	}
}

impl Mul<f64> for &Mat {
    type Output = Mat;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<f64> for Mat {
    type Output = Mat;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale_inplace(rhs);
		self
    }
}

impl MulAssign<f64> for Mat {
    fn mul_assign(&mut self, rhs: f64) {
        self.scale_inplace(rhs);
    }
}

#[cfg(test)]
mod test {
	use super::Mat;
	#[test]
	fn scalar() {
		let s = Mat::scalar(1.0);

		assert!(s.is_scalar());

		if let Some(sc) = s.as_scalar() {
			assert_eq!(sc, 1.0);
		} else {
			assert!(false);
		}
	}

	#[test]
	fn eye_3x3() {
		let m = Mat::identity(3);
		assert_eq!(m.rows(), 3);
		assert_eq!(m.cols(), 3);

		for i in 0..3 {
			for j in 0..3 {
				assert_eq!(m[(i,j)], if i == j { 1. } else { 0. });
			}
		}

		assert!(!m.is_scalar());
	}
}