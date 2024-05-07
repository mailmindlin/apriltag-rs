#![allow(non_local_definitions)] // Not sure why, but cpython throws a billion of these warnings
mod debug;
mod detection;
mod detector;
mod shim;
mod family;
mod pose;

use cpython::py_module_initializer;

pub use debug::{
    TimeProfile as PyTimeProfile,
    TimeProfileIter,
};
pub use detection::{
    Detections as PyDetections,
    Detection as PyDetection,
    DetectionsIter as PyDetectionsIter,
};
pub use detector::{
    DetectorConfig as PyDetectorConfig,
    QuadThresholdParams as PyQuadThesholdParams,
    DetectorBuilder as PyDetectorBuilder,
    Detector as PyDetector,
};
pub use pose::{
    PoseEstimator as PyPoseEstimator,
    OrthogonalIterationResult as PyOrthogonalIterationResult,
    AprilTagPoseWithError as PyAprilTagPoseWithError,
    AprilTagPose as PyAprilTagPose,
};
pub use family::AprilTagFamily as PyAprilTagFamily;


py_module_initializer!(apriltag_rs_native, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    // m.add(py, "sum_as_string", py_fn!(py, sum_as_string_py(a: i64, b:i64)))?;
    m.add_class::<PyTimeProfile>(py)?;
    m.add_class::<TimeProfileIter>(py)?;

    // Detections
    m.add_class::<PyDetections>(py)?;
    m.add_class::<PyDetectionsIter>(py)?;
    m.add_class::<PyDetection>(py)?;

    // Detector
    m.add_class::<PyDetectorConfig>(py)?;
    m.add_class::<PyQuadThesholdParams>(py)?;
    m.add_class::<PyDetectorBuilder>(py)?;
    m.add_class::<PyDetector>(py)?;

    m.add_class::<PyAprilTagFamily>(py)?;

    m.add_class::<PyPoseEstimator>(py)?;
    m.add_class::<PyOrthogonalIterationResult>(py)?;
    m.add_class::<PyAprilTagPoseWithError>(py)?;
    m.add_class::<PyAprilTagPose>(py)?;

    Ok(())
});