#![feature(box_syntax, int_roundings, new_uninit, unwrap_infallible, vec_into_raw_parts, const_trait_impl, slice_flatten)]
mod apriltag_math;
mod families;
mod detector;
mod util;
mod quickdecode;
pub(crate) mod quad_thresh;
pub(crate) mod ffi;
pub(crate) mod quad_decode;
mod pose;

pub use detector::{ApriltagDetector, ApriltagDetection};