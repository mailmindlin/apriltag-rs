use libc::{size_t, c_int, c_char, c_uint, c_double};

use crate::util::math::mat::Mat;

#[repr(C)]
pub(super) struct zarray {
    /// size of each element
    el_sz: size_t,

    /// how many elements?
    size: c_int,
    /// we've allocated storage for how many elements?
    alloc: c_int,
    data: *mut c_char,
}

impl<T: Sized> From<Vec<T>> for zarray where T: ~const Drop {
    fn from(value: Vec<T>) -> Self {
        todo!()
    }
}

#[repr(C)]
pub struct matd_t {
    nrows: c_uint,
    ncols: c_uint,
    data: [c_double],
}

impl matd_t {
    pub(super) fn convert(src: &Mat) -> *mut matd_t {
        todo!()
    }
}