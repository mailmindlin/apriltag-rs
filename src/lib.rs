#![feature(
    int_roundings,
    const_trait_impl,
    allocator_api,
    ptr_as_uninit,
	maybe_uninit_array_assume_init,
	array_try_from_fn,
	array_try_map,
)]
#![cfg_attr(feature="cffi", feature(ptr_as_uninit))]
#![allow(non_snake_case)]
// #![allow(unused)]
#![allow(clippy::collapsible_if)]
// #![forbid(clippy::print_stdout)]

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
pub use detector::{AprilTagDetector, DetectorBuilder, DetectorBuildError, DetectError, DetectorConfig, AccelerationRequest, GpuDeviceInfo};
pub use detector::config::SourceDimensions;
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
pub(crate) use util::math::Vec3;
pub use util::math::Mat33;

// Profiling
pub use dbg::{TimeProfile, TimeProfileStatistics};
