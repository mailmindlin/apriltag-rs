use std::ops::{Index, IndexMut};

use super::{Mat, MatChol, MatDims, MatLike, MatPLU, MatSVD, OutOfBoundsError, SvdOptions};


/// 9x9 matrix
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat99(pub [f64; 81]);


/// Try to convert from a dynamic 9x9 matrix
impl TryFrom<super::Mat> for Mat99 {
    type Error = OutOfBoundsError;

    fn try_from(value: super::Mat) -> Result<Self, Self::Error> {
        if value.dims != (MatDims { rows: 9, cols: 9 }) {
            return Err(OutOfBoundsError { dims: value.dims, index: super::MatIndex { row: 8, col: 8 } });
        }
        let a: &[f64] = &value.data;
        let b: Result<[f64; 81], _> = a.try_into();
        match b {
            Ok(elems) => Ok(Self(elems)),
            Err(_) => Err(OutOfBoundsError { dims: value.dims, index: super::MatIndex { row: 8, col: 8 } })
        }
    }
}

/// Dynamic-sized matrix with same data
impl From<Mat99> for Mat {
    fn from(value: Mat99) -> Self {
        Self::create(9, 9, &value.0)
    }
}

#[cfg(feature="compare_reference")]
impl float_cmp::ApproxEq for Mat99 {
    type Margin = float_cmp::F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        <&[f64] as float_cmp::ApproxEq>::approx_eq(&self.0, &other.0, margin)
    }
}

pub(crate) struct Mat99SVD {
    pub U: Mat99,
    #[allow(unused)]
    pub S: Mat99,
    pub V: Mat99,
}

impl Mat99 {
    /// Create matrix with all zeroes
    pub(crate) const fn zeroes() -> Self {
        Self([0.; 81])
    }

    /// Create from array
    pub(crate) const fn of(v: [f64; 81]) -> Self {
        Self(v)
    }

    pub(crate) const fn data(&self) -> &[f64] {
        &self.0
    }

    /// Create identity matrix
    pub(crate) fn identity() -> Self {
        let mut result = Self::zeroes();
        let mut i = 0;
        while i < 9 {
            result.0[i * 9 + i] = 1.0;
            i += 1;
        }
        result
    }

    /// Determinant via PLU decomposition
    pub(crate) fn det(&self) -> f64 {
        self.plu().det()
    }

    /// Matrix inverse via Gauss-Jordan elimination with partial pivoting.
    ///
    /// Returns None if this matrix is singular (not invertible).
    pub(crate) fn inv(&self) -> Option<Mat99> {
        // Work matrix: [A | I] stored as two separate 9x9 arrays on the stack
        let mut a = self.0;
        let mut inv = [0.0f64; 81];
        // Initialize inv to identity
        let mut i = 0;
        while i < 9 {
            inv[i * 9 + i] = 1.0;
            i += 1;
        }

        // Forward elimination with partial pivoting
        let mut col = 0;
        while col < 9 {
            // Find pivot row
            let mut max_val = a[col * 9 + col].abs();
            let mut max_row = col;
            let mut row = col + 1;
            while row < 9 {
                let val = a[row * 9 + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
                row += 1;
            }

            if max_val < 1e-15 {
                return None;
            }

            // Swap rows if needed
            if max_row != col {
                let mut j = 0;
                while j < 9 {
                    a.swap(col * 9 + j, max_row * 9 + j);
                    inv.swap(col * 9 + j, max_row * 9 + j);
                    j += 1;
                }
            }

            // Scale pivot row
            let pivot_inv = 1.0 / a[col * 9 + col];
            let mut j = 0;
            while j < 9 {
                a[col * 9 + j] *= pivot_inv;
                inv[col * 9 + j] *= pivot_inv;
                j += 1;
            }

            // Eliminate column in all other rows
            let mut row = 0;
            while row < 9 {
                if row != col {
                    let factor = a[row * 9 + col];
                    if factor != 0.0 {
                        let mut j = 0;
                        while j < 9 {
                            a[row * 9 + j] -= factor * a[col * 9 + j];
                            inv[row * 9 + j] -= factor * inv[col * 9 + j];
                            j += 1;
                        }
                    }
                }
                row += 1;
            }
            col += 1;
        }

        Some(Mat99(inv))
    }

    pub(crate) fn plu(&self) -> MatPLU<Self> {
        MatPLU::new(self)
    }

    pub(crate) fn chol(&self) -> MatChol<Self> {
        MatChol::new(self)
    }

    #[inline]
    fn map_elements(&self, map_fn: impl Fn(f64) -> f64) -> Self {
        Self::of(self.0.map(map_fn))
    }

    pub(crate) fn scale(&self, scalar: f64) -> Self {
        self.map_elements(|e| e * scalar)
    }

    pub(crate) fn scale_inplace(&mut self, scalar: f64) {
        for e in self.0.iter_mut() {
            *e *= scalar;
        }
    }

    pub(crate) fn svd_with_flags(&self, options: SvdOptions) -> MatSVD<Self> {
        MatSVD::new_with_flags(self, options)
    }
}

impl MatLike for Mat99 {
    fn rows(&self) -> usize {
        9
    }

    fn cols(&self) -> usize {
        9
    }

    fn data(&self) -> &[f64] {
        &self.0
    }

    fn data_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }

    fn transpose(&self) -> Self {
        let mut result = Mat99::zeroes();
        for i in 0..9 {
            for j in 0..9 {
                result.0[j * 9 + i] = self.0[i * 9 + j];
            }
        }
        result
    }

    // max_idx uses the default provided by MatLike
}

impl Index<(usize, usize)> for Mat99 {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(row < 9);
        debug_assert!(col < 9);
        let idx = row * 9 + col;
        &self.0[idx]
    }
}

impl IndexMut<(usize, usize)> for Mat99 {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(row < 9);
        debug_assert!(col < 9);
        let idx = row * 9 + col;
        &mut self.0[idx]
    }
}

#[cfg(test)]
mod test {
    use super::Mat99;
    use crate::util::math::mat::{Mat, MatLike, SvdOptions};

    const EPS: f64 = 1e-7;

    macro_rules! assert_close {
        ($a:expr, $b:expr) => {{
            let a = $a;
            let b = $b;
            assert!(
                (a - b).abs() < EPS,
                "expected {a} ≈ {b}, diff = {}",
                (a - b).abs()
            );
        }};
    }

    fn assert_mat_close(a: &Mat, b: &Mat) {
        assert_eq!(a.rows(), b.rows());
        assert_eq!(a.cols(), b.cols());
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                assert_close!(a[(i, j)], b[(i, j)]);
            }
        }
    }

    // Diagonally-dominant symmetric matrix → invertible and SPD
    fn spd_data() -> [f64; 81] {
        let mut data = [0.0f64; 81];
        for i in 0..9usize {
            for j in 0..9usize {
                data[i * 9 + j] = if i == j {
                    (i + 1) as f64 * 2.0  // diagonal: 2, 4, ..., 18
                } else {
                    0.1
                };
            }
        }
        data
    }

    // Non-symmetric invertible matrix
    fn invertible_data() -> [f64; 81] {
        let mut data = [0.0f64; 81];
        for i in 0..9usize {
            for j in 0..9usize {
                data[i * 9 + j] = if i == j {
                    (i + 1) as f64 * 3.0
                } else {
                    ((i as f64 + 1.) * 0.13 + j as f64 * 0.07).sin() * 0.5
                };
            }
        }
        data
    }

    // --- transpose ---

    #[test]
    fn transpose_roundtrip() {
        let m = Mat99::of(invertible_data());
        let tt = m.transpose().transpose();
        for i in 0..81 {
            assert_close!(m.0[i], tt.0[i]);
        }
    }

    #[test]
    fn transpose_conformance() {
        let data = invertible_data();
        let m99 = Mat99::of(data);
        let m_dyn = Mat::create(9, 9, &data);

        let t99: Mat = m99.transpose().into();
        let t_dyn = m_dyn.transpose();
        assert_mat_close(&t99, &t_dyn);
    }

    // --- inv ---

    #[test]
    fn inv_identity() {
        // A * A^-1 ≈ I
        let data = invertible_data();
        let m99 = Mat99::of(data);
        let m_dyn: Mat = m99.into();

        let inv99: Mat = m99.inv().unwrap().into();
        let product = m_dyn.matmul_dyn(&inv99);
        assert_mat_close(&product, &Mat::identity(9));
    }

    #[test]
    fn inv_conformance() {
        let data = invertible_data();
        let m99 = Mat99::of(data);
        let m_dyn = Mat::create(9, 9, &data);

        let inv99: Mat = m99.inv().unwrap().into();
        let inv_dyn = m_dyn.inv().unwrap();
        assert_mat_close(&inv99, &inv_dyn);
    }

    #[test]
    fn inv_singular_returns_none() {
        // All-zero matrix is singular
        let m99 = Mat99::zeroes();
        assert!(m99.inv().is_none());
    }

    // --- PLU / det ---

    #[test]
    fn plu_det_conformance() {
        let data = invertible_data();
        let m99 = Mat99::of(data);
        let m_dyn = Mat::create(9, 9, &data);

        assert_close!(m99.det(), m_dyn.det());
    }

    // --- Cholesky ---

    #[test]
    fn chol_solve_correctness() {
        let data = spd_data();
        let m99 = Mat99::of(data);
        let m_dyn: Mat = m99.into();

        let b = Mat::create(9, 1, &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let x = m99.chol().solve(&b);

        // A * x ≈ b
        let ax = m_dyn.matmul_dyn(&x);
        assert_mat_close(&ax, &b);
    }

    #[test]
    fn chol_solve_conformance() {
        let data = spd_data();
        let m99 = Mat99::of(data);
        let m_dyn = Mat::create(9, 9, &data);

        let b = Mat::create(9, 1, &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let x99 = m99.chol().solve(&b);
        let x_dyn = m_dyn.chol().solve(&b);
        assert_mat_close(&x99, &x_dyn);
    }

    // --- SVD ---

    #[test]
    fn svd_reconstruct() {
        // U * S * V^T ≈ A
        let data = invertible_data();
        let m99 = Mat99::of(data);
        let m_dyn = Mat::create(9, 9, &data);

        let svd = m99.svd_with_flags(SvdOptions { suppress_warnings: true });
        let reconstructed = svd.U.matmul_dyn(&svd.S).matmul_dyn(&svd.V.transpose());
        assert_mat_close(&reconstructed, &m_dyn);
    }

    #[test]
    fn svd_conformance() {
        // SVD of Mat99 and Mat produce the same reconstruction
        let data = invertible_data();
        let m99 = Mat99::of(data);
        let m_dyn = Mat::create(9, 9, &data);

        let svd99 = m99.svd_with_flags(SvdOptions { suppress_warnings: true });
        let svd_dyn = m_dyn.svd_with_flags(SvdOptions { suppress_warnings: true });

        // Compare reconstructions rather than U/S/V directly (sign ambiguity)
        let r99 = svd99.U.matmul_dyn(&svd99.S).matmul_dyn(&svd99.V.transpose());
        let r_dyn = svd_dyn.U.matmul_dyn(&svd_dyn.S).matmul_dyn(&svd_dyn.V.transpose());
        assert_mat_close(&r99, &r_dyn);
    }
}
