#![allow(non_camel_case_types)]
mod img_u8_t;
mod pose;
mod detector;
mod zarray;
mod family;
mod matd;
mod extras;

use std::ffi::CString;

use libc::c_char;

pub use family::{
    apriltag_family_t,
    tag16h5_create, tag16h5_destroy,
    tag25h9_create, tag25h9_destroy,
    tag36h10_create, tag36h10_destroy,
    tag36h11_create, tag36h11_destroy,
};
pub use img_u8_t::{
    image_u8_t,
    image_u8_create,
    image_u8_destroy
};
pub use detector::{
    apriltag_detector_t,
    apriltag_detection_t,
    apriltag_detector_create,
    apriltag_detector_add_family_bits,
    apriltag_detector_add_family,
    apriltag_detector_remove_family,
    apriltag_detector_clear_families,
    apriltag_detector_destroy,
    apriltag_detector_detect,
    apriltag_detection_destroy,
    apriltag_detections_destroy,
    apriltag_to_image,
};
pub use matd::{
    matd_ptr,
};
pub use zarray::{
    ZArray,
};
pub use extras::{
    timeprofile_t,
    timeprofile_display,
    timeprofile_total_utime,
};


#[derive(Debug)]
pub enum FFIConvertError {
    FieldOverflow,
    NullPointer,
}


fn drop_str(ptr: &mut *const c_char) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null()) as *mut _;
    let str = unsafe { CString::from_raw(ptr) };
    drop(str);
}