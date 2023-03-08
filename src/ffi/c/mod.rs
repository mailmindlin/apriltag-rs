mod img;
mod pose;
mod detector;
mod util;

use std::ffi::CString;

pub use img::*;
use libc::c_char;
pub use pose::*;
pub use detector::*;
pub use util::*;

#[derive(Debug)]
enum FFIConvertError {
    FieldOverflow,
    NullPointer,
}

fn drop_boxed<T>(ptr: &mut *const T) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null()) as *mut T;
    unsafe { Box::from_raw(ptr) };
}

fn drop_boxed_mut<T>(ptr: &mut *mut T) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null_mut()) as *mut T;
    unsafe { Box::from_raw(ptr) };
}

fn drop_array<T>(ptr: &mut *const T, len: usize) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null()) as *mut T;
    unsafe { Vec::from_raw_parts(ptr, len, len) };
}

fn drop_array_mut<T>(ptr: &mut *mut T, len: usize) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null_mut()) as *mut T;
    unsafe { Vec::from_raw_parts(ptr, len, len) };
}

fn drop_str(ptr: &mut *const c_char) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null()) as *mut _;
    unsafe { CString::from_raw(ptr) };
}