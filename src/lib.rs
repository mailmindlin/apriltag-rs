#![feature(box_syntax, int_roundings, new_uninit, unwrap_infallible, vec_into_raw_parts, const_trait_impl, slice_flatten)]
#![allow(non_snake_case)]
mod apriltag_math;
pub mod families;
mod detector;
mod util;
mod quickdecode;
pub(crate) mod quad_thresh;
// pub(crate) mod ffi;
pub(crate) mod quad_decode;
mod pose;

pub use detector::{ApriltagDetector, ApriltagDetection, DetectionResult};
pub use quad_thresh::ApriltagQuadThreshParams;

pub use pose::{estimate_tag_pose, estimate_pose_for_tag_homography, estimate_tag_pose_orthogonal_iteration, ApriltagDetectionInfo, ApriltagPose, OrthogonalIterationResult};