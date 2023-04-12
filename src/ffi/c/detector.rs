#![allow(non_camel_case_types)]
use std::{ffi::{c_float, CStr, CString}, sync::Arc, slice, ptr};

use crate::{ApriltagDetector, ApriltagDetection, families::AprilTagFamily, util::{geom::Point2D, image::ImageY8}, ffi::c::{drop_array, drop_str}};
use libc::{c_int, c_double, c_void, c_char};

use super::{util::zarray, img::image_u8_t, matd_t, FFIConvertError};

type apriltag_detector_t = ApriltagDetector;

#[repr(C)]
pub struct apriltag_family_t {
    // How many codes are there in this tag family?
    ncodes: u32,

    // The codes in the family.
    codes: *const u64,

    width_at_border: c_int,
    total_width: c_int,
    reversed_border: bool,

    // The bit locations.
    nbits: u32,
    bit_x: *const u32,
    bit_y: *const u32,

    // minimum hamming distance between any two codes. (e.g. 36h11 => 11)
    h: u32,

    // a human-readable name, e.g., "tag36h11"
    name: *const c_char,

    // some detector implementations may preprocess codes in order to
    // accelerate decoding.  They put their data here. (Do not use the
    // same apriltag_family instance in more than one implementation)
    ximpl: *const c_void,
}

impl Drop for apriltag_family_t {
    fn drop(&mut self) {
        drop_array(&mut self.codes, self.ncodes as usize);

        drop_array(&mut self.bit_x, self.nbits as usize);
        drop_array(&mut self.bit_y, self.nbits as usize);
        drop_str(&mut self.name);
        assert!(self.ximpl.is_null());
    }
}

impl From<AprilTagFamily> for apriltag_family_t {
    fn from(mut value: AprilTagFamily) -> Self {
        let mut bits_x = Vec::with_capacity(value.bits.len());
        let mut bits_y = Vec::with_capacity(value.bits.len());
        for (bit_x, bit_y) in value.bits.iter() {
            bits_x.push(bit_x);
            bits_y.push(bit_y);
        }
        bits_x.shrink_to_fit();
        bits_y.shrink_to_fit();

        value.codes.shrink_to_fit();
        let (codes, ncodes, codes_cap) = value.codes.into_raw_parts();
        assert_eq!(ncodes, codes_cap);

        Self {
            ncodes: ncodes as _,
            codes: codes,
            width_at_border: value.width_at_border as _,
            total_width: value.total_width as _,
            reversed_border: value.reversed_border,
            nbits: value.bits.len() as _,
            bit_x: bits_x.into_raw_parts().0 as *const _,
            bit_y: bits_y.into_raw_parts().0 as *const _,
            h: value.min_hamming as _,
            name: CString::new(value.name.as_ref()).unwrap().into_raw(),
            ximpl: std::ptr::null(),
        }
    }
}

impl TryFrom<&apriltag_family_t> for Arc<AprilTagFamily> {
    type Error = FFIConvertError;

    fn try_from(value: &apriltag_family_t) -> Result<Self, Self::Error> {
        let codes = {
            let mut codes = Vec::with_capacity(value.ncodes as usize);
            if value.ncodes == 0 {
                // Skip
            } else if value.codes.is_null() {
                return Err(FFIConvertError::NullPointer);
            } else {
                let codes_raw = unsafe { slice::from_raw_parts(value.codes, value.ncodes as usize) };
                codes.extend_from_slice(codes_raw);
            }
            codes
        };
        let bits = {
            let mut bits = Vec::with_capacity(value.nbits as usize);
            if value.nbits == 0 {
                // Skip
            } else if value.bit_x.is_null() || value.bit_y.is_null() {
                return Err(FFIConvertError::NullPointer);
            } else {
                let bits_x = unsafe { slice::from_raw_parts(value.bit_x, value.nbits as usize) };
                let bits_y = unsafe { slice::from_raw_parts(value.bit_y, value.nbits as usize) };
                for (bit_x, bit_y) in bits_x.iter().zip(bits_y.iter()) {
                    bits.push((*bit_x, *bit_y));
                }
            }
            bits
        };
        let name = unsafe { CStr::from_ptr(value.name as *const i8) }
            .to_string_lossy();
        
        Ok(Arc::new(AprilTagFamily {
            codes,
            bits,
            width_at_border: value.width_at_border as u32,
            total_width: value.total_width as u32,
            reversed_border: value.reversed_border,
            min_hamming: value.h.into(),
            name,
        }))
    }
}

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
            family: Box::into_raw(Box::new(apriltag_family_t::from(value.family.as_ref().clone()))),
            id: value.id as c_int,
            hamming: value.hamming as c_int,
            decision_margin: value.decision_margin,
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

impl TryFrom<*const apriltag_detection_t> for ApriltagDetection {
    
    type Error = FFIConvertError;

    fn try_from(value: *const apriltag_detection_t) -> Result<Self, Self::Error> {
        let value = unsafe { value.as_ref() }.ok_or(FFIConvertError::NullPointer)?;
        let family_raw = unsafe { value.family.as_ref() }.ok_or(FFIConvertError::NullPointer)?;

        Ok(Self {
            family: family_raw.try_into()?,
            id: value.id as usize,
            hamming: value.hamming as i16,
            decision_margin: value.decision_margin,
            H: value.H.try_into()?,
            center: Point2D::from(value.c),
            corners: [
                value.p[0].into(),
                value.p[1].into(),
                value.p[2].into(),
                value.p[3].into(),
            ],
        })
    }
}

/// don't forget to add a family!
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_create() -> *mut apriltag_detector_t {
    let detector = box ApriltagDetector::default();

    Box::into_raw(detector)
}

/// add a family to the apriltag detector. caller still "owns" the family.
/// a single instance should only be provided to one apriltag detector instance.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_add_family_bits(td: *mut apriltag_detector_t, fam: *const apriltag_family_t, bits_corrected: c_int) {
    let detector = td.as_mut().unwrap();
    let fam = fam.as_ref().unwrap().try_into().unwrap();
    let bits_corrected = bits_corrected.try_into().unwrap();
    
    detector.add_family_bits(fam, bits_corrected).unwrap();
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
    let fam: Arc<AprilTagFamily> = fam.as_ref().unwrap().try_into().unwrap();
    detector.remove_family(&fam);
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
    drop(Box::from_raw(td));
}

/// Detect tags from an image and return an array of
/// apriltag_detection_t*. You can use apriltag_detections_destroy to
/// free the array and the detections it contains, or call
/// _detection_destroy and zarray_destroy yourself.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_detect(td: *mut apriltag_detector_t, im_orig: *const image_u8_t) -> *mut zarray {
    let detector = if let Some(det) = td.as_mut() {
        det
    } else {
        return ptr::null_mut();
    };
    let im_orig = if let Some(im_orig) = im_orig.as_ref() {
        im_orig
    } else {
        return ptr::null_mut();
    };
    
    let res_vec = {
        let results = detector.detect(&im_orig.pretend_ref());
        //TODO: store time profile on apriltag_detector_t

        results.detections
            .into_iter()
            .map(|det| det.try_into().unwrap())
            .collect::<Vec<apriltag_detection_t>>()
    };

    Box::into_raw(box zarray::from(res_vec))
}

/// Call this method on each of the tags returned by apriltag_detector_detect
//TODO
/*#[no_mangle]
pub unsafe extern "C" fn apriltag_detection_destroy(det: *mut apriltag_detection_t) {

}*/

// destroys the array AND the detections within it.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detections_destroy(detections: *mut zarray) {
    if detections.is_null() {
        return;
    }
    let detections = unsafe { Box::from_raw(detections) };
    let detections: Vec<apriltag_detection_t> = Box::into_inner(detections).into();
    std::mem::drop(detections)
}

/// Renders the apriltag.
/// Caller is responsible for calling image_u8_destroy on the image
#[no_mangle]
pub unsafe extern "C" fn apriltag_to_image(fam: *const apriltag_family_t, idx: c_int) -> *mut image_u8_t {
    let fam = fam.as_ref().unwrap();
    
    assert!(idx >= 0 && (idx as u32) < fam.ncodes);
    let code = *fam.codes.offset(idx as isize);

    let mut im = ImageY8::zeroed(fam.total_width as usize, fam.total_width as usize);

    let white_border_width = fam.width_at_border as usize + (if fam.reversed_border { 0 } else { 2 });
    let white_border_start = (fam.total_width as usize - white_border_width)/2;
    // Make 1px white border
    for i in 0..(white_border_width-1) {
        im[(white_border_start + i, white_border_start)] = 255;
        im[(fam.total_width as usize - 1 - white_border_start, white_border_start + i)] = 255;
        im[(white_border_start + i + 1, fam.total_width as usize - 1 - white_border_start)] = 255;
        im[(white_border_start, white_border_start + 1 + i)] = 255;
    }

    let border_start = ((fam.total_width - fam.width_at_border)/2) as usize;
    for i in 0..fam.nbits {
        if (code & (1u64 << (fam.nbits - i - 1))) != 0 {
            let bit_y = *fam.bit_y.offset(i as isize) as usize;
            let bit_x = *fam.bit_x.offset(i as isize) as usize;
            im[(bit_x + border_start, bit_y + border_start)] = 255;
        }
    }
    let im = image_u8_t::from(im);
    Box::into_raw(Box::new(im))
}