use std::{alloc::{alloc_zeroed, AllocError, dealloc, Layout}, ptr::NonNull};

use libc::{c_uint, c_double, c_void};

#[repr(transparent)]
pub struct matd_ptr(*const c_void);

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
    pub(super) fn nrows(&self) -> Option<&c_uint> {
        if self.0.is_null() {
            return None;
        }
        let rows_offset = unsafe { (&MATD_EMPTY.nrows as *const u32).offset_from(&MATD_EMPTY as *const matd_sample as *const u32) };
        unsafe { (self.0 as *mut u32).offset(rows_offset).as_ref() }
    }
    pub(super) fn nrows_mut(&mut self) -> Option<NonNull<c_uint>> {
        if self.0.is_null() {
            return None;
        }
        let rows_offset = unsafe { (&MATD_EMPTY.nrows as *const u32).offset_from(&MATD_EMPTY as *const matd_sample as *const u32) };
        NonNull::new(unsafe { (self.0 as *mut u32).offset(rows_offset) })
    }
    pub(super) fn ncols(&self) -> Option<&c_uint> {
        if self.0.is_null() {
            return None;
        }
        let cols_offset = unsafe { (&MATD_EMPTY.nrows as *const u32).offset_from(&MATD_EMPTY as *const matd_sample as *const u32) };
        unsafe { (self.0 as *mut u32).offset(cols_offset).as_ref() }
    }
    pub(super) fn ncols_mut(&mut self) -> Option<NonNull<c_uint>> {
        if self.0.is_null() {
            return None;
        }
        let cols_offset = unsafe { (&MATD_EMPTY.nrows as *const u32).offset_from(&MATD_EMPTY as *const matd_sample as *const u32) };
        NonNull::new(unsafe { (self.0 as *mut u32).offset(cols_offset) })
    }
    pub(super) fn data(&self) -> Option<*const c_double> {
        if self.0.is_null() {
            return None;
        }
        let data_offset = unsafe { (&MATD_EMPTY.data as *const c_double).offset_from(&MATD_EMPTY as *const matd_sample as *const c_double) };
        Some(unsafe { (self.0 as *mut c_double).offset(data_offset) })
    }
    pub(super) fn new(nrows: c_uint, ncols: c_uint, data: &[c_double]) -> Result<Self, AllocError> {
        //See: https://stackoverflow.com/questions/67171086/how-can-a-dynamically-sized-object-be-constructed-on-the-heap
        assert_eq!(data.len(), (nrows * ncols) as usize);
        
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
            *res.ncols_mut().unwrap().as_mut() = ncols;
            *res.nrows_mut().unwrap().as_mut() = nrows;
        }
        Ok(res)
    }
}

impl Drop for matd_ptr {
    fn drop(&mut self) {
        let ptr = self.0 as *mut u8;
        if ptr.is_null() {
            return;
        }
        unsafe {
            let (layout, _) = Self::layout(*self.nrows().unwrap(), *self.ncols().unwrap());
            dealloc(ptr, layout);
        }
    }
}