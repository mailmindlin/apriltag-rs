#![feature(box_syntax, int_roundings, new_uninit)]
mod apriltag_math;
mod families;
mod detector;
mod util;
// mod apriltag;
mod quickdecode;
pub(crate) mod quad_thresh;
pub(crate) mod ffi;
pub(crate) mod quad_decode;
mod pose;

pub use detector::{ApriltagDetector, ApriltagDetection};