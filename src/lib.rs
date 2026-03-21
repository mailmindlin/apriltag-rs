#![feature(
    int_roundings,
    const_trait_impl,
    allocator_api,
)]
#![cfg_attr(feature="cffi", feature(ptr_as_uninit))]
#![allow(non_snake_case)]
#![allow(clippy::collapsible_if)]

pub mod families;
mod detector;
pub(crate) mod util;
mod quickdecode;
mod quad_thresh;
pub mod ffi;
mod quad_decode;
mod pose;
mod dbg;
mod detection;
mod ocl;
mod wgpu;

// Core detection
pub use detector::{AprilTagDetector, DetectorBuilder, DetectorBuildError, DetectError, DetectorConfig, AccelerationRequest};
pub use detection::{AprilTagDetection, Detections};
pub use quad_thresh::AprilTagQuadThreshParams;

// Tag families
pub use families::AprilTagFamily;
pub use quickdecode::AddFamilyError;

// Pose estimation
pub use pose::{
    estimate_tag_pose,
    estimate_pose_for_tag_homography,
    estimate_tag_pose_orthogonal_iteration,
    AprilTagDetectionInfo,
    AprilTagPose,
    OrthogonalIterationResult,
};

// Image types
pub use util::Image;
pub use util::image::{ImageBuffer, ImageRGB8, ImageWritePNM, Luma, ImageY8, Pixel};
pub use util::math::{Vec2, Vec3, Mat33};
pub use util::geom::{Point2D, quad::Quadrilateral};

// Profiling
pub use dbg::{TimeProfile, TimeProfileStatistics};
