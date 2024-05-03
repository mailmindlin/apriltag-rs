use core::slice;
use std::{sync::Arc, alloc::AllocError, ptr, ops::Deref, ffi::CStr};

use crate::AprilTagFamily;
use super::super::util::{drop_array, AtomicManagedPtr};
use super::{drop_str, FFIConvertError};

use libc::{c_int, c_void, c_char};

#[repr(C)]
pub struct apriltag_family_t {
    /// How many codes are there in this tag family?
    pub ncodes: u32,

    /// The codes in the family.
    /// Points to ximpl -> codes
    pub codes: *const u64,

    pub width_at_border: c_int,
    pub total_width: c_int,
    pub reversed_border: bool,

    /// The bit locations.
    pub nbits: u32,
    pub bit_x: *const u32,
    pub bit_y: *const u32,

    /// minimum hamming distance between any two codes. (e.g. 36h11 => 11)
    pub h: u32,

    /// a human-readable name, e.g., "tag36h11"
    /// (Points to ximpl -> name)
    pub name: *const c_char,

    // some detector implementations may preprocess codes in order to
    // accelerate decoding.  They put their data here. (Do not use the
    // same apriltag_family instance in more than one implementation)
    ximpl: AtomicManagedPtr<Arc<AprilTagFamily>, c_void>,
}

impl apriltag_family_t {
    pub(super) fn wrap(base: Arc<AprilTagFamily>) -> Result<Arc<Self>, AllocError> {
        let ncodes: u32 = base.codes.len().try_into().expect("ncodes overflow");
        let codes = base.codes.deref().as_ptr();

        let (bit_x, bit_y) = base.split_bits();
        debug_assert_eq!(bit_x.len(), bit_y.len());

        Arc::try_new(apriltag_family_t {
            ncodes,
            codes,
            width_at_border: base.width_at_border.try_into().expect("width_at_border overflow"),
            total_width: base.total_width.try_into().expect("total_width overflow"),
            reversed_border: base.reversed_border,

            nbits: base.bits.len().try_into().expect("nbits overflow"),
            bit_x: Box::into_raw(bit_x.into_boxed_slice()) as *const u32,
            bit_y: Box::into_raw(bit_y.into_boxed_slice()) as *const u32,

            h: base.min_hamming.try_into().expect("h overflow"),
            // name: base.name.as_ptr() as *const c_char,//TODO: fixme
            name: ptr::null(),
            ximpl: AtomicManagedPtr::from(base),
        })
    }

    pub(super) fn as_arc(&self) -> Arc<AprilTagFamily> {
        let res = Arc::<AprilTagFamily>::try_from(&*self)
            .unwrap();
        if self.ximpl.is_null() {
            self.ximpl.swap(res.clone());
        }
        res
    }
}

impl Drop for apriltag_family_t {
    fn drop(&mut self) {
        if self.ximpl.is_null() {
            drop_str(&mut self.name);
            drop_array(&mut self.codes, self.ncodes as usize);
        } else {
            self.ximpl.take();
        }
        drop_array(&mut self.bit_x, self.nbits as usize);
        drop_array(&mut self.bit_y, self.nbits as usize);
        assert!(self.ximpl.is_null());
    }
}

fn ffi_tag_create(raw: AprilTagFamily) -> *const apriltag_family_t {
    let arc = Arc::new(raw);
    let wrapped = apriltag_family_t::wrap(arc).unwrap();
    Arc::into_raw(wrapped)
}

unsafe fn ffi_tag_destroy(fam: *const apriltag_family_t) {
    if !fam.is_null() {
        Arc::from_raw(fam);
    }
}

#[no_mangle]
pub unsafe extern "C" fn tag16h5_create() -> *const apriltag_family_t {
    ffi_tag_create(crate::families::tag16h5_create())
}
#[no_mangle]
pub unsafe extern "C" fn tag16h5_destroy(fam: *const apriltag_family_t) {
    ffi_tag_destroy(fam)
}
#[no_mangle]
pub unsafe extern "C" fn tag25h9_create() -> *const apriltag_family_t {
    ffi_tag_create(crate::families::tag16h5_create())
}
#[no_mangle]
pub unsafe extern "C" fn tag25h9_destroy(fam: *const apriltag_family_t) {
    ffi_tag_destroy(fam)
}
#[no_mangle]
pub unsafe extern "C" fn tag36h10_create() -> *const apriltag_family_t {
    ffi_tag_create(crate::families::tag16h5_create())
}
#[no_mangle]
pub unsafe extern "C" fn tag36h10_destroy(fam: *const apriltag_family_t) {
    ffi_tag_destroy(fam)
}
#[no_mangle]
pub unsafe extern "C" fn tag36h11_create() -> *const apriltag_family_t {
    ffi_tag_create(crate::families::tag16h5_create())
}
#[no_mangle]
pub unsafe extern "C" fn tag36h11_destroy(fam: *const apriltag_family_t) {
    ffi_tag_destroy(fam)
}


impl TryFrom<&apriltag_family_t> for Arc<AprilTagFamily> {
    type Error = FFIConvertError;

    fn try_from(value: &apriltag_family_t) -> Result<Self, Self::Error> {
        if let Some(arc) = value.ximpl.borrow() {
            return Ok(arc);
        }

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
        
        let arc = Arc::new(AprilTagFamily {
            codes,
            bits,
            width_at_border: value.width_at_border as u32,
            total_width: value.total_width as u32,
            reversed_border: value.reversed_border,
            min_hamming: value.h.into(),
            name,
        });
        Ok(arc)
    }
}