use std::{alloc::{AllocError, Layout, alloc_zeroed}, iter, mem::MaybeUninit, panic::UnwindSafe};

use libc::{c_uint, c_double, c_int, c_float, c_char};

use crate::{Mat33, Vec3, util::math::mat::Mat};

use super::{FFIConvertError, shim::{CFFIError, InOutPtr, InPtr, ReadPtr, cffi_wrapper, param}};

/// C-compatible matd_t struct, matching the C header layout:
/// ```c
/// typedef struct {
///     unsigned int nrows, ncols;
///     double *data;
/// } matd_t;
/// ```
///
/// A matd_t with nrows=0 and ncols=0 is a SCALAR (not the same as 1x1).
/// Scalars still have one element allocated in `data`.
#[repr(C)]
pub struct matd_t {
    pub nrows: c_uint,
    pub ncols: c_uint,
    pub data: *mut c_double,
}

impl matd_t {
	fn data_count(nrows: c_uint, ncols: c_uint) -> usize {
        let n = (nrows as usize) * (ncols as usize);
        if n == 0 { 1 } else { n }
    }

    fn data_layout(nrows: c_uint, ncols: c_uint) -> Layout {
        Layout::array::<c_double>(Self::data_count(nrows, ncols)).unwrap()
    }

    fn new(nrows: c_uint, ncols: c_uint) -> Result<Self, AllocError> {
        let layout = Self::data_layout(nrows, ncols);
		let data = unsafe { alloc_zeroed(layout) }.cast::<f64>();
		if data.is_null() {
			return Err(AllocError);
		}
		Ok(Self {
			nrows,
			ncols,
			data,
		})
    }

    pub(super) fn nrows(&self) -> c_uint {
        self.nrows
    }

    pub(super) fn ncols(&self) -> c_uint {
        self.ncols
    }

	fn index(&self, row: usize, col: usize) -> usize {
		row * (self.ncols() as usize) + col
	}

	pub(super) fn get(&self, row: usize, col: usize) -> Option<&c_double> {
		self.data().get(self.index(row, col))
    }

    pub(super) fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut c_double> {
		let idx = self.index(row, col);
		self.data_mut().get_mut(idx)
    }

    pub(super) fn get_uninit_mut(&mut self, row: usize, col: usize) -> Option<&mut MaybeUninit<c_double>> {
		let idx = self.index(row, col);
        self.data_uninit_mut().get_mut(idx)
    }

    pub(super) fn data(&self) -> &[c_double] {
        let count = Self::data_count(self.nrows(), self.ncols());
        unsafe { std::slice::from_raw_parts(self.data, count) }
    }

    pub(super) fn data_mut(&mut self) -> &mut [c_double] {
        let count = Self::data_count(self.nrows(), self.ncols());
        unsafe { std::slice::from_raw_parts_mut(self.data, count) }
    }

    pub(super) fn data_uninit_mut(&mut self) -> &mut [MaybeUninit<c_double>] {
        let count = Self::data_count(self.nrows(), self.ncols());
        unsafe { std::slice::from_raw_parts_mut(self.data as *mut MaybeUninit<c_double>, count) }
    }

    fn is_scalar(&self) -> bool {
        self.nrows() <= 1 && self.ncols() <= 1
    }
}

impl TryFrom<&matd_t> for Mat {
	type Error = CFFIError;

	fn try_from(value: &matd_t) -> Result<Self, Self::Error> {
		if value.data.is_null() {
			return Err(CFFIError::NullArgument("matd_t->data"));
		}
		Ok(if value.is_scalar() {
			let scalar = unsafe { *value.data };
			//TODO: fallible alloc
			Self::scalar(scalar)
		} else {
			let nrows = value.nrows().try_into().map_err(|_| FFIConvertError::FieldOverflow)?;
			let ncols = value.ncols().try_into().map_err(|_| FFIConvertError::FieldOverflow)?;
			Self::try_create(nrows, ncols, value.data())?
		})
	}
}

impl TryFrom<Mat> for Option<Box<matd_t>> {
	type Error = CFFIError;

	fn try_from(value: Mat) -> Result<Self, Self::Error> {
		//TODO: try to reuse the allocation
		let nrows = value.rows().try_into().map_err(|_| FFIConvertError::FieldOverflow)?;
		let ncols = value.cols().try_into().map_err(|_| FFIConvertError::FieldOverflow)?;
		let mut result = matd_t::new(nrows, ncols)?;
		result.data_mut().copy_from_slice(&value.data);
		Ok(Some(Box::try_new(result)?))
	}
}

impl TryFrom<Mat33> for Box<matd_t> {
	type Error = AllocError;

	fn try_from(value: Mat33) -> Result<Self, Self::Error> {
		//TODO: try to reuse the allocation
		let mut result = matd_t::new(3, 3)?;
		result.data_mut().copy_from_slice(&value.0);
		Ok(Box::try_new(result)?)
	}
}

impl TryFrom<Vec3> for Box<matd_t> {
	type Error = AllocError;

	fn try_from(value: Vec3) -> Result<Self, Self::Error> {
		//TODO: try to reuse the allocation
		let mut result = matd_t::new(1, 3)?;
		result.data_mut().copy_from_slice(&[value.0, value.1, value.2]);
		Ok(Box::try_new(result)?)
	}
}
/*/// Owned pointer to a `matd_t`. Manages both the struct and data allocations.
/// Returned as `Option<matd_ptr>` from FFI functions (`None` = NULL).
#[repr(transparent)]
pub struct matd_ptr(NonNull<matd_t>);

impl matd_ptr {
    /// Number of data elements to allocate (at least 1, for scalars).
    fn data_count(nrows: c_uint, ncols: c_uint) -> usize {
        let n = (nrows as usize) * (ncols as usize);
        if n == 0 { 1 } else { n }
    }

    fn data_layout(nrows: c_uint, ncols: c_uint) -> Layout {
        Layout::array::<c_double>(Self::data_count(nrows, ncols)).unwrap()
    }

    pub(super) fn nrows(&self) -> c_uint {
        unsafe { self.0.as_ref().nrows }
    }

    pub(super) fn ncols(&self) -> c_uint {
        unsafe { self.0.as_ref().ncols }
    }

	pub(super) fn get(&self, row: usize, col: usize) -> Option<&c_double> {
		let idx = row * (self.ncols() as usize) + col;
		self.data().get(idx)
    }

    pub(super) fn data(&self) -> &[c_double] {
        let count = Self::data_count(self.nrows(), self.ncols());
        unsafe { std::slice::from_raw_parts(self.0.as_ref().data, count) }
    }

    pub(super) fn data_mut(&mut self) -> &mut [c_double] {
        let count = Self::data_count(self.nrows(), self.ncols());
        unsafe { std::slice::from_raw_parts_mut(self.0.as_ref().data, count) }
    }

    fn is_scalar(&self) -> bool {
        self.nrows() <= 1 && self.ncols() <= 1
    }

    /// Allocate a new matd_ptr with zeroed data.
    fn alloc(nrows: c_uint, ncols: c_uint) -> Result<Self, AllocError> {
        unsafe {
            let struct_layout = Layout::new::<matd_t>();
            let m = alloc_zeroed(struct_layout) as *mut matd_t;
            if m.is_null() {
                return Err(AllocError);
            }

            let data_layout = Self::data_layout(nrows, ncols);
            let data_ptr = alloc_zeroed(data_layout) as *mut c_double;
            if data_ptr.is_null() {
                dealloc(m as *mut u8, struct_layout);
                return Err(AllocError);
            }

            (*m).nrows = nrows;
            (*m).ncols = ncols;
            (*m).data = data_ptr;

            Ok(Self(NonNull::new_unchecked(m)))
        }
    }

    /// Create a new matd_ptr from dimensions and a data slice.
    /// `data.len()` must equal `nrows*ncols` (or 1 for scalars).
    pub fn new(nrows: c_uint, ncols: c_uint, data: &[c_double]) -> Result<Self, AllocError> {
        let count = Self::data_count(nrows, ncols);
        assert_eq!(data.len(), count, "Data length mismatch: expected {count}, got {}", data.len());

        let mut result = Self::alloc(nrows, ncols)?;
        result.data_mut().copy_from_slice(data);
        Ok(result)
    }

    /// Convert a Rust `Mat` into a `matd_ptr`. Tries to reuse allocations
    pub fn from_mat(mat: Mat) -> Result<Self, AllocError> {
		// Let's try to eliminate some allocations
        let nrows = mat.rows() as c_uint;
        let ncols = mat.cols() as c_uint;
        let mut result = Self::alloc(nrows, ncols)?;
        result.data_mut().copy_from_slice(&mat.data);
        Ok(result)
    }

	/// Convert a Rust `Mat` into a `matd_ptr`.
    pub fn copy_mat(mat: &Mat) -> Result<Self, AllocError> {
        let nrows = mat.rows() as c_uint;
        let ncols = mat.cols() as c_uint;
        let mut result = Self::alloc(nrows, ncols)?;
        result.data_mut().copy_from_slice(&mat.data);
        Ok(result)
    }

    /// Convert back to a Rust `Mat`.
    pub fn to_mat(&self) -> Mat {
        let nrows = self.nrows() as usize;
        let ncols = self.ncols() as usize;
        if nrows == 0 || ncols == 0 {
            Mat::scalar(self.data()[0])
        } else {
            Mat::create(nrows, ncols, self.data())
        }
    }
}

impl Drop for matd_ptr {
    fn drop(&mut self) {
        unsafe {
            let m = self.0.as_ref();
            let nrows = m.nrows;
            let ncols = m.ncols;
            let data_ptr = m.data;

            if !data_ptr.is_null() {
                dealloc(data_ptr as *mut u8, Self::data_layout(nrows, ncols));
            }
            dealloc(self.0.as_ptr() as *mut u8, Layout::new::<matd_t>());
        }
    }
}*/

pub type matd_ptr = Box<matd_t>;

// ============================================================
// Construction
// ============================================================

/// Creates a double matrix with the given number of rows and columns (or a scalar
/// in the case where rows=0 and/or cols=0). All data elements will be initialized
/// to zero.
#[no_mangle]
pub unsafe extern "C" fn matd_create(rows: c_int, cols: c_int) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let rows: usize = param::try_read(rows, "rows")?;
        let cols: usize = param::try_read(cols, "cols")?;
        Mat::zeroes(rows, cols).try_into()
    })
}

/// Creates a double matrix with the given number of rows and columns (or a scalar
/// in the case where rows=0 and/or cols=0). All data elements will be initialized
/// using the supplied array of data, which must contain at least rows*cols elements,
/// arranged in row-major order (i.e. index = row*ncols + col).
#[no_mangle]
pub unsafe extern "C" fn matd_create_data<'a>(rows: c_int, cols: c_int, data: InPtr<'a, c_double>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let rows: usize = param::try_read(rows, "rows")?;
        let cols: usize = param::try_read(cols, "cols")?;
        if rows == 0 || cols == 0 {
            let scalar: f64 = *data.try_ref("data")?;
            return Mat::scalar(scalar).try_into();
        }
        let data = data.try_array(rows * cols, "data")?;
        Mat::create(rows, cols, data)
			.try_into()
    })
}

/// Creates a scalar with the supplied value 'v'.
///
/// NOTE: Scalars are different than 1x1 matrices (implementation note:
/// they are encoded as 0x0 matrices). For example: for matrices A*B, A
/// and B must both have specific dimensions. However, if A is a
/// scalar, there are no restrictions on the size of B.
#[no_mangle]
pub unsafe extern "C" fn matd_create_scalar(v: c_double) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        Mat::scalar(v).try_into()
    })
}

/// Creates a double matrix with the given number of rows and columns (or a scalar
/// in the case where rows=0 and/or cols=0). All data elements will be initialized
/// using the supplied array of float data, which must contain at least rows*cols elements,
/// arranged in row-major order (i.e. index = row*ncols + col).
#[no_mangle]
pub unsafe extern "C" fn matd_create_dataf<'a>(rows: c_int, cols: c_int, data: InPtr<'a, c_float>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let rows: usize = param::try_read(rows, "rows")?;
        let cols: usize = param::try_read(cols, "cols")?;
        if rows == 0 || cols == 0 {
            let scalar = *data.try_ref("data")? as f64;
            return Mat::scalar(scalar).try_into();
        }
        let fdata = data.try_array(rows * cols, "data")?;
        let ddata: Vec<f64> = fdata.iter().map(|&f| f as f64).collect();
        Mat::create(rows, cols, &ddata).try_into()
    })
}

/// Creates a square identity matrix with the given number of rows (and
/// therefore columns), or a scalar with value 1 in the case where dim=0.
#[no_mangle]
pub unsafe extern "C" fn matd_identity(dim: c_int) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let dim: usize = param::try_read(dim, "dim")?;
        Mat::identity(dim).try_into()
    })
}

// ============================================================
// Accessors
// ============================================================

/// Retrieves the cell value for matrix 'm' at the given zero-based row and column index.
#[no_mangle]
pub unsafe extern "C" fn matd_get<'a>(m: InPtr<'a, matd_t>, row: c_uint, col: c_uint) -> c_double {
    cffi_wrapper(|| {
		let m = m.try_ref("m")?;
        assert!(!m.is_scalar(), "matd_get cannot be used on scalars");
        assert!(row < m.nrows(), "row out of bounds");
        assert!(col < m.ncols(), "col out of bounds");
		Ok(*m.get(row as usize, col as usize).unwrap())
    })
}

/// Assigns the given value to the matrix cell at the given zero-based row and
/// column index.
#[no_mangle]
pub unsafe extern "C" fn matd_put<'a>(mut m: InOutPtr<'a, matd_t>, row: c_uint, col: c_uint, value: c_double) {
    cffi_wrapper(move || {
        let m = m.try_mut("m")?;
        if m.is_scalar() {
            m.data_mut()[0] = value;
            return Ok(());
        }
        assert!(row < m.nrows(), "row out of bounds");
        assert!(col < m.ncols(), "col out of bounds");
		*m.get_mut(row as _, col as _).unwrap() = value;
        Ok(())
    });
}

/// Retrieves the scalar value of the given element ('m' must be a scalar).
#[no_mangle]
pub unsafe extern "C" fn matd_get_scalar<'a>(m: InPtr<'a, matd_t>) -> c_double {
    cffi_wrapper(|| {
        let m = m.try_ref("m")?;
        assert!(m.is_scalar(), "matd_get_scalar: not a scalar");
        Ok(m.data()[0])
    })
}

/// Assigns the given value to the supplied scalar element ('m' must be a scalar).
#[no_mangle]
pub unsafe extern "C" fn matd_put_scalar<'a>(mut m: InOutPtr<'a, matd_t>, value: c_double) {
    cffi_wrapper(move || {
        let m = m.try_mut("m")?;
        assert!(m.is_scalar(), "not a scalar");
        m.data_mut()[0] = value;
        Ok(())
    })
}

// ============================================================
// Copy / select / print / destroy
// ============================================================

/// Creates an exact copy of the supplied matrix 'm'.
#[no_mangle]
pub unsafe extern "C" fn matd_copy<'a>(m: InPtr<'a, matd_t>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let m: Mat = m.try_ref("m")?
			// Clones
			.try_into()?;
		m.try_into()
    })
}

/// Creates a copy of a subset of the supplied matrix 'a'. The subset will include
/// rows 'r0' through 'r1', inclusive ('r1' >= 'r0'), and columns 'c0' through 'c1',
/// inclusive ('c1' >= 'c0').
#[no_mangle]
pub unsafe extern "C" fn matd_select<'a>(a: InPtr<'a, matd_t>, r0: c_uint, r1: c_int, c0: c_uint, c1: c_int) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let a = a.try_ref("a")?;
        assert!(!a.is_scalar(), "matd_select: cannot be used on scalars");
        assert!(r0 < a.nrows());
        assert!(c0 < a.ncols());

        let r0 = r0 as usize;
        let r1 = r1 as usize;
        let c0 = c0 as usize;
        let c1 = c1 as usize;
        let ncols_a = a.ncols() as usize;

        let nrows = r1 - r0 + 1;
        let ncols = c1 - c0 + 1;

        let mut buf = vec![0.0f64; nrows * ncols];
        for row in r0..=r1 {
            for col in c0..=c1 {
                buf[(row - r0) * ncols + (col - c0)] = a.data()[row * ncols_a + col];
            }
        }
		Mat::create(nrows, ncols, &buf)
			.try_into()
    })
}

/// Prints the supplied matrix 'm' to standard output by applying the supplied
/// printf format specifier 'fmt' for each individual element.
#[no_mangle]
pub unsafe extern "C" fn matd_print<'a>(m: InPtr<'a, matd_t>, _fmt: *const c_char) {
    cffi_wrapper(|| {
        let m = m.try_ref("m")?;
        if m.is_scalar() {
            println!("{}", m.data()[0]);
        } else {
            let ncols = m.ncols() as usize;
            for i in 0..m.nrows() as usize {
                for j in 0..ncols {
                    print!("{:12.6} ", m.data()[i * ncols + j]);
                }
                println!();
            }
        }
        Ok(())
    })
}

/// Prints the transpose of the supplied matrix 'm' to standard output.
#[no_mangle]
pub unsafe extern "C" fn matd_print_transpose<'a>(m: InPtr<'a, matd_t>, _fmt: *const c_char) {
    cffi_wrapper(|| {
        let m = m.try_ref("m")?;
        if m.is_scalar() {
            println!("{}", m.data()[0]);
        } else {
            let ncols = m.ncols() as usize;
            for j in 0..ncols {
                for i in 0..m.nrows() as usize {
                    print!("{:12.6} ", m.data()[i * ncols + j]);
                }
                println!();
            }
        }
        Ok(())
    })
}

/// Frees the memory associated with matrix 'm'.
#[no_mangle]
pub unsafe extern "C" fn matd_destroy(m: Option<matd_ptr>) {
    drop(m);
}

// ============================================================
// Arithmetic
// ============================================================

#[track_caller]
unsafe fn matd_binop<'a, F>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>, f: F) -> Option<matd_ptr>
where 
	F: Fn(c_double, c_double) -> c_double,
	F: UnwindSafe
{
	cffi_wrapper(move || {
        let a = a.try_ref("a")?;
        let b = b.try_ref("b")?;
		assert_eq!(a.nrows(), b.nrows());
		assert_eq!(a.ncols(), b.ncols());

		let mut res = matd_t::new(a.nrows(), a.ncols())?;
		for (dst, (a, b)) in iter::zip(res.data_mut(), iter::zip(a.data(), b.data())) {
			*dst = f(*a, *b);
		}
        Ok(Some(Box::try_new(res)?))
    })
}

/// Applies a binary operation in place on `a`. Handles a/b aliasing.
#[track_caller]
unsafe fn matd_binop_inplace<'a, F>(mut a: InOutPtr<'a, matd_t>, b: InPtr<'a, matd_t>, f: F)
where 
	F: Fn(&mut c_double, c_double),
	F: UnwindSafe
{
	cffi_wrapper(move || {
		if a.aliases(&b) {
			let a = a.try_mut("a")?;
			for v in a.data_mut() {
				f(v, *v);
			}
		} else {
			//TODO: make sure the arrays don't overlap
			let a = a.try_mut("a")?;
			let b = b.try_ref("b")?;
			for (u, v) in std::iter::zip(a.data_mut(), b.data()) {
				f(u, *v);
			}
		}

		Ok(())
    })
}

/// Adds the two supplied matrices together, cell-by-cell.
#[no_mangle]
pub unsafe extern "C" fn matd_add<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>) -> Option<matd_ptr> {
	matd_binop(a, b, |a, b| &a + &b)
}

/// Adds the values of 'b' to matrix 'a', cell-by-cell, in place.
#[no_mangle]
pub unsafe extern "C" fn matd_add_inplace<'a>(a: InOutPtr<'a, matd_t>, b: InPtr<'a, matd_t>) {
	matd_binop_inplace(a, b, |a, b| {
		*a += b;
	})
    // cffi_wrapper(move || {
    //     let b = b.try_ref("b")?;
    //     assert_eq!(a.ptr() as usize & 0, 0); // ensure a is read after b
    //     let b_data: Vec<f64> = b.data().to_vec();
    //     let a = a.try_mut("a")?;
    //     for (d, s) in a.data_mut().iter_mut().zip(b_data.iter()) {
    //         *d += s;
    //     }
    //     Ok(())
    // })
}

/// Subtracts matrix 'b' from matrix 'a', cell-by-cell.
#[no_mangle]
pub unsafe extern "C" fn matd_subtract<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>) -> Option<matd_ptr> {
	matd_binop(a, b, |a, b| a - b)
}

/// Subtracts the values of 'b' from matrix 'a', cell-by-cell, in place.
#[no_mangle]
pub unsafe extern "C" fn matd_subtract_inplace<'a>(a: InOutPtr<'a, matd_t>, b: InPtr<'a, matd_t>) {
    matd_binop_inplace(a, b, |a, b| {
		*a -= b;
	})
}

/// Scales all cell values of matrix 'a' by the given scale factor 's'.
#[no_mangle]
pub unsafe extern "C" fn matd_scale<'a>(a: InPtr<'a, matd_t>, s: c_double) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let mut a: Mat = a.try_ref("a")?.try_into()?;
		a *= s;
		a.try_into()
    })
}

/// Scales all cell values of matrix 'a' by the given scale factor 's', in place.
#[no_mangle]
pub unsafe extern "C" fn matd_scale_inplace<'a>(mut a: InOutPtr<'a, matd_t>, s: c_double) {
    cffi_wrapper(move || {
        let a = a.try_mut("a")?;
        for d in a.data_mut() {
            *d *= s;
        }
        Ok(())
    })
}

/// Multiplies the two supplied matrices together (matrix product).
/// Scalars can multiply any matrix.
#[no_mangle]
pub unsafe extern "C" fn matd_multiply<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        let b: Mat = b.try_ref("b")?.try_into()?;
        if a.is_scalar() {
            return (&b * a.data[0]).try_into();
        }
        if b.is_scalar() {
            return (&a * b.data[0]).try_into();
        }
        a.matmul(&b).try_into()
    })
}

/// Creates a matrix which is the transpose of the supplied matrix 'a'.
#[no_mangle]
pub unsafe extern "C" fn matd_transpose<'a>(a: InPtr<'a, matd_t>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        a.transpose().try_into()
    })
}

// ============================================================
// Linear algebra
// ============================================================

/// Calculates the determinant of the supplied matrix 'a'.
#[no_mangle]
pub unsafe extern "C" fn matd_det<'a>(a: InPtr<'a, matd_t>) -> c_double {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        Ok(a.det())
    })
}

/// Attempts to compute an inverse of the supplied matrix 'a'.
/// Returns NULL if the matrix is singular.
#[no_mangle]
pub unsafe extern "C" fn matd_inverse<'a>(a: InPtr<'a, matd_t>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        match a.inv() {
            Some(inv) => inv.try_into(),
            None => Ok(None),
        }
    })
}

// ============================================================
// Vector operations
// ============================================================

/// Calculates the magnitude of the supplied vector 'a'.
#[no_mangle]
pub unsafe extern "C" fn matd_vec_mag<'a>(a: InPtr<'a, matd_t>) -> c_double {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        Ok(a.vec_mag())
    })
}

/// Calculates the distance between vectors 'a' and 'b'.
#[no_mangle]
pub unsafe extern "C" fn matd_vec_dist<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>) -> c_double {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        let b: Mat = b.try_ref("b")?.try_into()?;
        Ok(a.vec_dist(&b))
    })
}

/// Same as matd_vec_dist, but only uses the first 'n' terms.
#[no_mangle]
pub unsafe extern "C" fn matd_vec_dist_n<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>, n: c_int) -> c_double {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        let b: Mat = b.try_ref("b")?.try_into()?;
        let n: usize = param::try_read(n, "n")?;
        Ok(a.vec_dist_n(&b, n))
    })
}

/// Calculates the dot product of two vectors.
#[no_mangle]
pub unsafe extern "C" fn matd_vec_dot_product<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>) -> c_double {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        let b: Mat = b.try_ref("b")?.try_into()?;
        Ok(a.vec_dot(&b))
    })
}

/// Calculates the normalization of the supplied vector 'a'.
#[no_mangle]
pub unsafe extern "C" fn matd_vec_normalize<'a>(a: InPtr<'a, matd_t>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        a.vec_normalize().try_into()
    })
}

/// Calculates the cross product of vectors 'a' and 'b' (both must be length-3).
#[no_mangle]
pub unsafe extern "C" fn matd_crossproduct<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let a: Mat = a.try_ref("a")?.try_into()?;
        let b: Mat = b.try_ref("b")?.try_into()?;
        a.cross(&b).try_into()
    })
}

/// Computes the infinity-norm error between two matrices: max |a_ij - b_ij|.
#[no_mangle]
pub unsafe extern "C" fn matd_err_inf<'a>(a: InPtr<'a, matd_t>, b: InPtr<'a, matd_t>) -> c_double {
    cffi_wrapper(|| {
        let a = a.try_ref("a")?;
        let b = b.try_ref("b")?;
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), b.ncols());
        let maxf = a.data().iter().zip(b.data().iter())
            .map(|(av, bv)| (av - bv).abs())
            .fold(0.0f64, f64::max);
        Ok(maxf)
    })
}

/// Returns the maximum value in the matrix.
#[no_mangle]
pub unsafe extern "C" fn matd_max<'a>(m: InPtr<'a, matd_t>) -> c_double {
    cffi_wrapper(|| {
        let m = m.try_ref("m")?;
        Ok(m.data().iter().copied().fold(f64::NEG_INFINITY, f64::max))
    })
}
