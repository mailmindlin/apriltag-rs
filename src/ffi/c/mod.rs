#![allow(non_camel_case_types)]
mod img_u8_t;
mod pose;
mod detector;
mod zarray;
mod family;
mod matd;
mod debug;
mod shim;

use std::{ffi::CString, num::TryFromIntError};

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
    matd_create,
    matd_create_data,
    matd_create_dataf,
    matd_identity,
};
pub use zarray::ZArray;
pub use debug::{
    timeprofile_t,
    timeprofile_display,
    timeprofile_total_utime,
};


#[derive(Debug)]
pub enum FFIConvertError {
    NullPointer,
    FieldOverflow,
    Other(String),
}

impl From<TryFromIntError> for FFIConvertError {
    fn from(_value: TryFromIntError) -> Self {
        Self::FieldOverflow
    }
}

fn drop_str(ptr: &mut *const c_char) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null()) as *mut _;
    let str = unsafe { CString::from_raw(ptr) };
    drop(str);
}

#[cfg(test)]
mod test {
    use crate::ffi::c::*;

    #[test]
    fn test_construct_tag() {
        let atf = unsafe { tag16h5_create() };
        assert!(!atf.is_null());
        
        unsafe {
            let a = atf.as_ref().unwrap();
            assert_eq!(a.h, 5);
            tag16h5_destroy(atf);
        }
    }

    #[test]
    fn test_construct_detector() {
        let mut det = unsafe { apriltag_detector_create() };
        unsafe {
            let det_ptr = det.as_ptr().as_mut().unwrap();
        }
        unsafe { apriltag_detector_destroy(det); }
    }
}