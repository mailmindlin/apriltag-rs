use std::{alloc::{alloc_zeroed, AllocError, dealloc, Layout}, ptr::NonNull};

use libc::{c_uint, c_double, c_void, c_int, c_float};

use crate::util::math::mat::Mat;

use super::shim::{IncompleteArrayField, InPtr, cffi_wrapper, param, ReadPtr};

#[repr(transparent)]
pub struct matd_ptr(NonNull<c_void>);

#[repr(C)]
pub struct matd_t {
    pub nrows: c_uint,
    pub ncols: c_uint,
    pub data: IncompleteArrayField<c_double>,
}

/// Creates a double matrix with the given number of rows and columns (or a scalar
/// in the case where rows=0 and/or cols=0). All data elements will be initialized
/// to zero. It is the caller's responsibility to call matd_destroy() on the
/// returned matrix.
#[no_mangle]
pub unsafe extern "C" fn matd_create(rows: c_int, cols: c_int) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let rows: usize = param::try_read(rows, "rows")?;
        let cols: usize = param::try_read(cols, "cols")?;
        let result = Mat::zeroes(rows, cols);
        todo!()
    })
}

/// Creates a double matrix with the given number of rows and columns (or a scalar
/// in the case where rows=0 and/or cols=0). All data elements will be initialized
/// using the supplied array of data, which must contain at least rows*cols elements,
/// arranged in row-major order (i.e. index = row*ncols + col). It is the caller's
/// responsibility to call matd_destroy() on the returned matrix.
#[no_mangle]
pub unsafe extern "C" fn matd_create_data<'a>(rows: c_int, cols: c_int, data: InPtr<'a, c_double>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let rows: usize = param::try_read(rows, "rows")?;
        let cols: usize = param::try_read(cols, "cols")?;
        let data = data.try_array(rows * cols, "data")?;
        let result = Mat::create(rows, cols, data);
        todo!()
    })
}

/// Creates a double matrix with the given number of rows and columns (or a scalar
/// in the case where rows=0 and/or cols=0). All data elements will be initialized
/// using the supplied array of float data, which must contain at least rows*cols elements,
/// arranged in row-major order (i.e. index = row*ncols + col). It is the caller's
/// responsibility to call matd_destroy() on the returned matrix.
#[no_mangle]
pub unsafe extern "C" fn matd_create_dataf<'a>(rows: c_int, cols: c_int, data: InPtr<'a, c_float>) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let rows: usize = param::try_read(rows, "rows")?;
        let cols: usize = param::try_read(cols, "cols")?;
        let data = data.try_array(rows * cols, "data")?;
        let mat = Mat::zeroes(rows, cols);
        
        todo!();
    })
}

/// Creates a square identity matrix with the given number of rows (and
/// therefore columns), or a scalar with value 1 in the case where dim=0.
/// It is the caller's responsibility to call matd_destroy() on the
/// returned matrix.
#[no_mangle]
pub unsafe extern "C" fn matd_identity(dim: c_int) -> Option<matd_ptr> {
    cffi_wrapper(|| {
        let dim: usize = param::try_read(dim, "dim")?;
        let _result = Mat::identity(dim);
        todo!()
    })
}

#[repr(C)]
struct matd_like<T: ?Sized> {
    nrows: c_uint,
    ncols: c_uint,
    data: T
}

type matd_sample = matd_like<[c_double; 1]>;
const MATD_EMPTY: matd_sample = matd_sample {
    nrows: 0,
    ncols: 0,
    data: [0.]
};

impl matd_ptr {
    fn layout(nrows: c_uint, ncols: c_uint) -> (Layout, usize) {
        let base_layout = Layout::for_value(&MATD_EMPTY);
        base_layout.extend(Layout::array::<c_double>((nrows * ncols) as usize).unwrap()).unwrap()
    }
    unsafe fn ref_field<T>(&self, offset: isize) -> &T {
        self.0
            .byte_offset(offset)
            .cast::<T>()
            .as_ref()
    }
    unsafe fn mut_field<T>(&self, offset: isize) -> &mut T {
        self.0
            .byte_offset(offset)
            .cast::<T>()
            .as_mut()
    }
    const fn rows_offset() -> isize {
        unsafe { (&MATD_EMPTY.nrows as *const u32).offset_from(&MATD_EMPTY as *const matd_sample as *const u32) }
    }
    pub(super) fn nrows(&self) -> c_uint {
        *unsafe { self.ref_field(Self::rows_offset()) }
    }
    pub(super) fn nrows_mut(&mut self) -> &mut c_uint {
        unsafe { self.mut_field(Self::rows_offset()) }
    }
    const fn cols_offset() -> isize {
        unsafe { (&MATD_EMPTY.ncols as *const u32).offset_from(&MATD_EMPTY as *const matd_sample as *const u32) }
    }
    pub(super) fn ncols(&self) -> c_uint {
        *unsafe { self.ref_field(Self::cols_offset()) }
    }
    pub(super) fn ncols_mut(&mut self) -> &mut c_uint {
        unsafe { self.mut_field(Self::cols_offset()) }
    }
    const fn data_offset() -> isize {
        unsafe { (&MATD_EMPTY.data as *const c_double).offset_from(&MATD_EMPTY as *const matd_sample as *const c_double) }
    }
    pub(super) fn data(&self) -> &[c_double] {
        // Some(unsafe { (self.0 as *mut c_double).offset(data_offset) })
        todo!()
    }
    pub(super) fn new(nrows: c_uint, ncols: c_uint, data: &[c_double]) -> Result<Self, AllocError> {
        //See: https://stackoverflow.com/questions/67171086/how-can-a-dynamically-sized-object-be-constructed-on-the-heap
        assert_eq!(data.len(), (nrows * ncols) as usize, "Data length mismatch");
        
        let (layout, arr_offset) = Self::layout(nrows, ncols);
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(AllocError);
        }

        // Write data
        unsafe {
            let data_ptr = ptr.byte_offset(arr_offset as isize) as *mut c_double;
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
        }

        let mut res = Self(ptr as *const c_void);
        unsafe {
            *res.ncols_mut() = ncols;
            *res.nrows_mut() = nrows;
        }
        Ok(res)
    }
}

impl Drop for matd_ptr {
    fn drop(&mut self) {
        let ptr = self.0.as_ptr() as *mut u8;
        unsafe {
            let (layout, _) = Self::layout(*self.nrows(), *self.ncols());
            dealloc(ptr, layout);
        }
    }
}