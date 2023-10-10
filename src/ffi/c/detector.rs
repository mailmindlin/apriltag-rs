#![allow(non_camel_case_types)]
use core::slice;
use std::{ffi::c_float, sync::Arc, ops::Deref, collections::HashMap};
use crate::{AprilTagDetector, util::{image::ImageY8, geom::{Point2D, quad::Quadrilateral}, math::mat::Mat33}, detector::{DetectorBuilder, DetectorBuildError}, AprilTagQuadThreshParams, AprilTagFamily, ffi::util::AtomicManagedPtr, AprilTagDetection};
use errno::{set_errno, Errno};
use libc::{c_int, c_double};

use parking_lot::RwLock;
use super::{zarray::ZArray, img_u8_t::image_u8_t, family::apriltag_family_t, matd_ptr, timeprofile_t, FFIConvertError};
use super::super::util::{drop_boxed_mut, ManagedPtr};

enum LazyDetector {
    Building(DetectorBuilder),
    Detector(AprilTagDetector),
    Invalid,
}

impl LazyDetector {
    fn as_builder(&mut self) -> &mut DetectorBuilder {
        match self {
            LazyDetector::Building(builder) => builder,
            LazyDetector::Detector(_) => {
                let builder = if let LazyDetector::Detector(detector) = std::mem::replace(self, LazyDetector::Invalid) {
                    DetectorBuilder::from(detector)
                } else {
                    unreachable!()
                };
                *self = Self::Building(builder);
                if let LazyDetector::Building(builder) = self {
                    builder
                } else {
                    unreachable!()
                }
            },
            LazyDetector::Invalid => unreachable!(),
        }
    }

    fn as_detector(&mut self) -> Result<&AprilTagDetector, DetectorBuildError> {
        match self {
            LazyDetector::Building(builder) => {
                let detector = builder.clone().build()?;
                *self = Self::Detector(detector);

                if let Self::Detector(det) = self {
                    return Ok(det);
                } else {
                    unreachable!()
                }
            }
            LazyDetector::Detector(det) => Ok(det),
            LazyDetector::Invalid => unreachable!(),
        }
    }
}

struct ExtraData {
    detector: RwLock<LazyDetector>,
}

impl ExtraData {
    fn update(&self, callback: impl FnOnce(&mut DetectorBuilder) -> ()) {
        let mut detector_wlock = self.detector.write();

        let builder = detector_wlock.as_builder();
        callback(builder);
    }

    fn detector<'a, R>(&'a self, callback: impl FnOnce(&AprilTagDetector) -> R) -> Result<R, DetectorBuildError> {
        let mut rlock = self.detector.upgradable_read();
        loop {
            if let LazyDetector::Detector(det) = rlock.deref() {
                return Ok(callback(det));
            }
            // Hopefully we are guaraunteed to only loop once, but it's not clear if the lock goes from write->unlocked->read or write->read
            rlock.with_upgraded(|data| -> Result<(), DetectorBuildError> {
                data.as_detector()?;
                Ok(())
            })?;
        }
    }
}

#[repr(C)]
pub struct apriltag_detector_t {
    // User-configurable parameters.

    /// How many threads should be used?
    pub nthreads: c_int,

    /// detection of quads can be done on a lower-resolution image,
    /// improving speed at a cost of pose accuracy and a slight
    /// decrease in detection rate. Decoding the binary payload is
    /// still done at full resolution. .
    pub quad_decimate: c_float,

    /// What Gaussian blur should be applied to the segmented image
    /// (used for quad detection?)  Parameter is the standard deviation
    /// in pixels.  Very noisy images benefit from non-zero values
    /// (e.g. 0.8).
    pub quad_sigma: c_float,

    /// When true, the edges of the each quad are adjusted to "snap
    /// to" strong gradients nearby. This is useful when decimation is
    /// employed, as it can increase the quality of the initial quad
    /// estimate substantially. Generally recommended to be on (true).
    ///
    /// Very computationally inexpensive. Option is ignored if
    /// quad_decimate = 1.
    pub refine_edges: bool,

    /// How much sharpening should be done to decoded images? This
    /// can help decode small tags but may or may not help in odd
    /// lighting conditions or low light conditions.
    ///
    /// The default value is 0.25.
    pub decode_sharpening: c_double,

    /// When true, write a variety of debugging images to the
    /// current working directory at various stages through the
    /// detection process. (Somewhat slow).
    pub debug: bool,

    pub qtp: AprilTagQuadThreshParams,

    ///////////////////////////////////////////////////////////////
    // Statistics relating to last processed frame
    tp: AtomicManagedPtr<Box<timeprofile_t>>,

    nedges: u32,
    nsegments: u32,
    pub nquads: u32,

    ///////////////////////////////////////////////////////////////
    // Internal variables below
    wp: ExtraData,
}

impl apriltag_detector_t {
    fn new() -> Self {
        let builder = DetectorBuilder::default();
        Self {
            nthreads:      builder.config.nthreads as _,
            quad_decimate: builder.config.quad_decimate,
            quad_sigma:    builder.config.quad_sigma,
            refine_edges:  builder.config.refine_edges as _,
            decode_sharpening: builder.config.decode_sharpening,
            debug:         builder.config.debug,
            qtp:           builder.config.qtp,
            tp: AtomicManagedPtr::null(),
            nedges: 0,
            nsegments: 0,
            nquads: 0,
            wp: ExtraData {
                detector: RwLock::new(
                    LazyDetector::Building(DetectorBuilder::default())
                ),
            },
        }
    }
    fn update(&mut self, callback: impl FnOnce(&mut DetectorBuilder) -> ()) {
        self.wp.update(|builder| {
            builder.config.nthreads = self.nthreads as _;
            builder.config.quad_decimate = self.quad_decimate;
            builder.config.quad_sigma = self.quad_sigma;
            builder.config.refine_edges = self.refine_edges;
            builder.config.decode_sharpening = self.decode_sharpening;
            builder.config.debug = self.debug;
            builder.config.qtp = self.qtp;

            callback(builder);
        });
    }
}

#[repr(C)]
pub struct apriltag_detection_t {
    /// a pointer for convenience. not freed by apriltag_detection_destroy.
    family: ManagedPtr<Arc<apriltag_family_t>>,

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
    H: matd_ptr,

    // The center of the detection in image pixel coordinates.
    c: [c_double; 2],

    /// The corners of the tag in image pixel coordinates. These always
    /// wrap counter-clock wise around the tag.
    p: [[c_double; 2]; 4],
}

impl TryFrom<&apriltag_detection_t> for AprilTagDetection {
    type Error = FFIConvertError;

    fn try_from(value: &apriltag_detection_t) -> Result<Self, Self::Error> {
        let family = value.family.borrow()
            .ok_or(FFIConvertError::NullPointer)?
            .as_arc();

        if value.H.ncols().cloned() != Some(3) && value.H.nrows().cloned() != Some(3) {
            return Err(FFIConvertError::FieldOverflow);
        }
        let H = match value.H.data() {
            Some(ptr) => {
                let slice = unsafe { slice::from_raw_parts(ptr, 9) };
                let arr = <[f64; 9]>::try_from(slice)
                    .unwrap();
                Mat33::of(arr)
            },
            None => return Err(FFIConvertError::NullPointer),
        };
        Ok(Self {
            family,
            id: value.id as _,
            hamming: value.hamming as _,
            decision_margin: value.decision_margin,
            H,
            center: Point2D::of(value.c[0], value.c[1]),
            corners: Quadrilateral::from_array(&value.p),
        })
    }
}

/// don't forget to add a family!
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_create() -> ManagedPtr<Box<apriltag_detector_t>> {
    ManagedPtr::from(Box::new(apriltag_detector_t::new()))
}

/// add a family to the apriltag detector. caller still "owns" the family.
/// a single instance should only be provided to one apriltag detector instance.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_add_family_bits(td: *mut apriltag_detector_t, fam: ManagedPtr<Arc<apriltag_family_t>>, bits_corrected: c_int) {
    let detector = td.as_mut()
        .expect("Null parameter: td");
    let fam = fam.borrow()
        .expect("Null parameter: fam");
    let bits_corrected = bits_corrected
        .try_into()
        .expect("Invalid value for bits_corrected");

    detector.update(|b| b.add_family_bits(fam.as_arc(), bits_corrected).expect("Error adding family"));
}

/// Tunable, but really, 2 is a good choice. Values of >=3
/// consume prohibitively large amounts of memory, and otherwise
/// you want the largest value possible.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_add_family(td: *mut apriltag_detector_t, fam: ManagedPtr<Arc<apriltag_family_t>>) {
    apriltag_detector_add_family_bits(td, fam, 2)
}

/// does not deallocate the family.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_remove_family(td: *mut apriltag_detector_t, fam: ManagedPtr<Arc<apriltag_family_t>>) {
    let detector = td.as_mut().expect("Null parameter: td");
    let fam = fam.borrow()
        .expect("Null parameter: fam")
        .as_arc();
    
    detector.update(|b| b.remove_family(&fam));
}

/// unregister all families, but does not deallocate the underlying tag family objects.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_clear_families(td: *mut apriltag_detector_t) {
    let detector = td.as_mut().unwrap();
    detector.update(|b| b.clear_families());
}

/// Destroy the april tag detector (but not the underlying
/// apriltag_family_t used to initialize it.)
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_destroy(mut td: *mut apriltag_detector_t) {
    drop_boxed_mut(&mut td);
}

/// Detect tags from an image and return an array of
/// apriltag_detection_t*. You can use apriltag_detections_destroy to
/// free the array and the detections it contains, or call
/// _detection_destroy and zarray_destroy yourself.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detector_detect(td: *mut apriltag_detector_t, im_orig: *const image_u8_t) -> ManagedPtr<Box<ZArray<ManagedPtr<Box<apriltag_detection_t>>>>> {
    let td = match td.as_mut() {
        Some(td) => td,
        None => return ManagedPtr::null(),
    };
    let im_orig = match im_orig.as_ref() {
        Some(img) => img,
        None => return ManagedPtr::null(),
    };

    let detections = match td.wp.detector(|detector| detector.detect(&im_orig.as_ref())) {
        Ok(Ok(detections)) => detections,
        Ok(Err(e)) => {
            #[cfg(feature="debug")]
            eprintln!("apriltag_detector_detect error: {e:?}");
            return ManagedPtr::null();
        },
        Err(DetectorBuildError::Threadpool(t)) => {
            #[cfg(feature="debug")]
            eprintln!("apriltag_detector_detect error: {t:?}");
            set_errno(Errno(libc::EAGAIN));
            return ManagedPtr::null();
        },
        // Other build errors
        #[allow(unreachable_patterns)]
        Err(e) => {
            #[cfg(feature="debug")]
            eprintln!("apriltag_detector_detect error: {e:?}");
            return ManagedPtr::null();
        }
    };

    // Store timeprofile on detector
    #[cfg(feature="debug")]
    {
        let prev = match Box::try_new(detections.tp) {
            Ok(tp) => td.tp.swap(tp),
            Err(_) => td.tp.take(),
        };
        // Drop previous value
        drop(prev);

        td.nquads = detections.nquads;
    }

    fn alloc_error<T>() -> ManagedPtr<Box<T>> {
        set_errno(Errno(libc::ENOMEM));
        ManagedPtr::null()
    }

    let res_vec = {
        let mut res_vec: Vec<ManagedPtr<Box<apriltag_detection_t>>> = Vec::new();
        if let Err(_) = res_vec.try_reserve_exact(detections.detections.len()) {
            return alloc_error();
        }

        let mut families = HashMap::<Arc<AprilTagFamily>, Arc<apriltag_family_t>>::new();

        for detection in detections.detections.into_iter() {
            let family = match families.entry(detection.family) {
                std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let value = match apriltag_family_t::wrap(e.key().clone()) {
                        Ok(fam) => fam,
                        Err(_) => return alloc_error(),
                    };
                    e.insert(value).clone()
                }
            };
            let H = match matd_ptr::new(3, 3, detection.H.data()) {
                Ok(H) => H,
                Err(_) => return alloc_error(),
            };

            let native = Box::new(apriltag_detection_t {
                family: ManagedPtr::from(family),
                id: detection.id as _,
                hamming: detection.hamming as _,
                decision_margin: detection.decision_margin,
                H,
                c: detection.center.as_array(),
                p: [
                    detection.corners[0].as_array(),
                    detection.corners[1].as_array(),
                    detection.corners[2].as_array(),
                    detection.corners[3].as_array(),
                ],
            });

            res_vec.push(ManagedPtr::from(native));
        }
        res_vec
    };

    ManagedPtr::from(Box::new(ZArray::from(res_vec)))
}

/// Call this method on each of the tags returned by apriltag_detector_detect
//TODO
#[no_mangle]
pub unsafe extern "C" fn apriltag_detection_destroy(mut detection: ManagedPtr<Box<apriltag_detection_t>>) {
    let _ = detection.take();
}

// destroys the array AND the detections within it.
#[no_mangle]
pub unsafe extern "C" fn apriltag_detections_destroy(mut detections: ManagedPtr<Box<ZArray<apriltag_detection_t>>>) {
    let _ = detections.take();
}

/// Renders the apriltag.
/// Caller is responsible for calling image_u8_destroy on the image
#[no_mangle]
pub unsafe extern "C" fn apriltag_to_image(fam: *const apriltag_family_t, idx: c_int) -> ManagedPtr<Box<image_u8_t>> {
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
    ManagedPtr::from(Box::new(image_u8_t::from(im)))
}