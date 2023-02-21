#![allow(non_camel_case_types)]
use std::ffi::c_float;

use crate::{ApriltagDetector, families::AprilTagFamily, ApriltagDetection};
use libc::{c_int, c_double};

use super::{util::zarray, img::image_u8_t, matd_t};

type apriltag_detector_t = ApriltagDetector;
type apriltag_family_t = AprilTagFamily;

#[repr(C)]
pub struct apriltag_detection_t {
    /// a pointer for convenience. not freed by apriltag_detection_destroy.
    family: *const apriltag_family_t,

    /// The decoded ID of the tag
    id: c_int,

    /// How many error bits were corrected? Note: accepting large numbers of
    /// corrected errors leads to greatly increased false positive rates.
    /// NOTE: As of this implementation, the detector cannot detect tags with
    /// a hamming distance greater than 2.
    hamming: c_int,

    /// A measure of the quality of the binary decoding process: the
    /// average difference between the intensity of a data bit versus
    /// the decision threshold. Higher numbers roughly indicate better
    /// decodes. This is a reasonable measure of detection accuracy
    /// only for very small tags-- not effective for larger tags (where
    /// we could have sampled anywhere within a bit cell and still
    /// gotten a good detection.)
    decision_margin: c_float,

    /// The 3x3 homography matrix describing the projection from an
    /// "ideal" tag (with corners at (-1,1), (1,1), (1,-1), and (-1,
    /// -1)) to pixels in the image. This matrix will be freed by
    /// apriltag_detection_destroy.
    H: *const matd_t,

    // The center of the detection in image pixel coordinates.
    c: [c_double; 2],

    /// The corners of the tag in image pixel coordinates. These always
    /// wrap counter-clock wise around the tag.
    p: [[c_double; 2]; 4],
}

impl From<ApriltagDetection> for apriltag_detection_t {
    fn from(value: ApriltagDetection) -> Self {
        Self {
            family: todo!(),
            id: value.id as c_int,
            hamming: value.hamming as c_int,
            decision_margin: todo!(),
            H: matd_t::convert(&value.H),
            c: value.center.into(),
            p: [
                value.corners[0].into(),
                value.corners[1].into(),
                value.corners[2].into(),
                value.corners[3].into(),
            ],
        }
    }
}

/// don't forget to add a family!
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_create() -> *mut apriltag_detector_t {
    let detector = Box::new(ApriltagDetector::default());

    Box::into_raw(detector)
}

/// add a family to the apriltag detector. caller still "owns" the family.
/// a single instance should only be provided to one apriltag detector instance.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_add_family_bits(td: *mut apriltag_detector_t, fam: *const apriltag_family_t, bits_corrected: c_int) {
    let detector = td.as_mut().unwrap();
    let fam = fam.as_ref().unwrap();
    let bits_corrected = bits_corrected.try_into().unwrap();
    detector.add_family_bits(*fam, bits_corrected)
}

/// Tunable, but really, 2 is a good choice. Values of >=3
/// consume prohibitively large amounts of memory, and otherwise
/// you want the largest value possible.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_add_family(td: *mut apriltag_detector_t, fam: *const apriltag_family_t) {
    apriltag_detector_add_family_bits(td, fam, 2)
}

/// does not deallocate the family.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_remove_family(td: *mut apriltag_detector_t, fam: *const apriltag_family_t) {
    let detector = td.as_mut().unwrap();
    let fam = fam.as_ref().unwrap();
    detector.remove_family(fam);
}

/// unregister all families, but does not deallocate the underlying tag family objects.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_clear_families(td: *mut apriltag_detector_t) {
    let detector = td.as_mut().unwrap();
    detector.clear_families();
}

/// Destroy the april tag detector (but not the underlying
/// apriltag_family_t used to initialize it.)
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_destroy(td: *mut apriltag_detector_t) {
    if td.is_null() {
        return;
    }
    Box::from_raw(td);
}

/// Detect tags from an image and return an array of
/// apriltag_detection_t*. You can use apriltag_detections_destroy to
/// free the array and the detections it contains, or call
/// _detection_destroy and zarray_destroy yourself.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_detect(td: *mut apriltag_detector_t, im_orig: *const image_u8_t) -> *mut zarray {
    let detector = td.as_mut().unwrap();
    let im_orig = im_orig.as_ref().unwrap();
    let results = detector.detect(&im_orig.pretend_ref());
    todo!()
}

/// Call this method on each of the tags returned by apriltag_detector_detect
//TODO
/*#[no_mangle]
pub unsafe extern "C" fn apriltag_detection_destroy(det: *mut apriltag_detection_t) {

}*/

// destroys the array AND the detections within it.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detections_destroy(detections: *mut zarray) {

}

// Renders the apriltag.
// Caller is responsible for calling image_u8_destroy on the image
#[no_mangle]
pub unsafe extern "C" fn apriltag_to_image(fam: *const apriltag_family_t, idx: c_int) -> *mut image_u8_t {
    let fam = fam.as_ref().unwrap();
    todo!()
}