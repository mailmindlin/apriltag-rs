use crate::ApriltagDetector;


pub extern "C" fn apriltag_detector_create() -> *mut ApriltagDetector {
    let detector = Box::new(ApriltagDetector::default());

    Box::into_raw(detector)
}