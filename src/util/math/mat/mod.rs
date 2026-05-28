mod mat;
/// Choelsky Decomposition
mod chol;
/// PLU Decomposition
mod plu;
/// SVD decomposition
mod svd;
/// matd_op
// mod op;
/// Matrix dimensions and indices
mod index;
mod mat22;
mod mat33;
mod mat99;
mod opm;

use std::ops::IndexMut;

pub use mat::Mat;
pub(crate) use mat22::Mat22;
pub(crate) use opm::{matrix_op, Matmul as Matmul2, MatmulTranspose, TransposedMatmul};
pub use mat33::Mat33;
pub(crate) use mat99::Mat99;

pub(crate) use index::{MatDims, MatIndex, OutOfBoundsError};
pub(crate) use chol::MatChol;
#[allow(unused_imports)]
pub(crate) use plu::MatPLU;
#[allow(unused_imports)]
pub(crate) use svd::{SvdOptions, MatSVD};

pub(crate) trait MatLike: IndexMut<(usize, usize), Output = f64> + Clone {
	fn rows(&self) -> usize;
	fn cols(&self) -> usize;
	fn is_square(&self) -> bool {
		self.rows() == self.cols()
	}
	fn data(&self) -> &[f64];
	fn data_mut(&mut self) -> &mut [f64];
	fn transpose(&self) -> Self;
	fn max_idx(&self, row: usize, maxcol: usize) -> Result<usize, OutOfBoundsError> {
		let mut argmax = 0;
		let mut max = f64::NEG_INFINITY;
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

trait Matmul<Rhs: MatLike = Self> {
	type Output: MatLike;
	fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

trait IdentityLike {
	type Output: MatLike;
	fn identity_like(&self) -> Self::Output;
}