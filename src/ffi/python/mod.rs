#![allow(non_local_definitions)] // Not sure why, but cpython throws a billion of these warnings
mod debug;
mod detection;
mod detector;
mod shim;
mod family;
mod pose;
mod timeout;

use pyo3::{pymodule, PyResult};
use pyo3::prelude::*;

use debug::{
    TimeProfile as PyTimeProfile,
    TimeProfileIter,
	TimeProfileEntry,
};
use detection::{
    Detections as PyDetections,
    Detection as PyDetection,
    DetectionsIter as PyDetectionsIter,
};
use detector::{
    DetectorConfig as PyDetectorConfig,
    QuadThresholdParams as PyQuadThesholdParams,
    DetectorBuilder as PyDetectorBuilder,
    Detector as PyDetector,
};
use pose::{
    PoseEstimator as PyPoseEstimator,
    OrthogonalIterationResult as PyOrthogonalIterationResult,
    AprilTagPoseWithError as PyAprilTagPoseWithError,
    AprilTagPose as PyAprilTagPose,
};
use family::AprilTagFamily as PyAprilTagFamily;

#[pymodule]
#[pyo3(name = "_native")]
fn apriltag_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ? -https://github.com/PyO3/maturin/issues/475
	m.add("__doc__", "This module is implemented in Rust.")?;

	// TimeProfile
    m.add_class::<PyTimeProfile>()?;
	m.add_class::<TimeProfileEntry>()?;
	m.add_class::<TimeProfileIter>()?;

    // Detector
    m.add_class::<PyDetectorConfig>()?;
    m.add_class::<PyQuadThesholdParams>()?;
    m.add_class::<PyDetectorBuilder>()?;
    m.add_class::<PyDetector>()?;

	// Detections
    m.add_class::<PyDetections>()?;
    m.add_class::<PyDetectionsIter>()?;
    m.add_class::<PyDetection>()?;

	// Family
    m.add_class::<PyAprilTagFamily>()?;

	// Pose
    m.add_class::<PyPoseEstimator>()?;
    m.add_class::<PyOrthogonalIterationResult>()?;
    m.add_class::<PyAprilTagPoseWithError>()?;
    m.add_class::<PyAprilTagPose>()?;

    Ok(())
}
