#![feature(
    int_roundings,
    const_trait_impl,
    allocator_api,
)]
#![allow(non_snake_case)]
#![allow(clippy::collapsible_if)]

pub mod families;
mod detector;
pub mod util;
mod quickdecode;
pub(crate) mod quad_thresh;
pub(crate) mod ffi;
pub(crate) mod quad_decode;
mod pose;
mod dbg;
mod detection;
mod ocl;
mod wgpu;

pub use util::Image;
pub use quickdecode::AddFamilyError;
pub use families::AprilTagFamily;
pub use detector::{AprilTagDetector, DetectorBuilder, DetectorBuildError, DetectError, DetectorConfig, AccelerationRequest};
pub use detection::{AprilTagDetection, Detections};
pub use quad_thresh::AprilTagQuadThreshParams;
pub use dbg::{TimeProfile, TimeProfileStatistics};

pub use pose::{
    estimate_tag_pose,
    estimate_pose_for_tag_homography,
    estimate_tag_pose_orthogonal_iteration,
    AprilTagDetectionInfo,
    AprilTagPose,
    OrthogonalIterationResult
};