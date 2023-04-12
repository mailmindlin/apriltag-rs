use std::ffi::c_int;

use cpython::py_capsule;
use cpython::_detail::ffi::{PyObject, Py_intptr_t};
use libc::c_void;

#[repr(C)]
struct NumpyArrayInterface {
    /// contains the integer 2 -- simple sanity check
    two: c_int,
    /// number of dimensions
    nd: c_int,
    /// kind in array --- character code of typestr
    typekind: c_char,
    itemsize: c_int,         /* size of each element */
    flags: c_int,            /* flags indicating how the data should be interpreted */
                          /*   must set ARR_HAS_DESCR bit to validate descr */
    /// A length-nd array of shape information
    shape: *const Py_intptr_t,
    /// A length-nd array of stride information
    strides: *const Py_intptr_t,
    /// A pointer to the first element of the array
    data: *const c_void,
    /// NULL or data-description (same as descr key
    /// of __array_interface__) -- must set ARR_HAS_DESCR
    /// flag or this will be ignored.
    descr: *mut PyObject,
}

py_capsule!(from numpy import ucnhash_CAPI as capsmod for unicode_name_CAPI);
