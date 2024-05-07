use pyo3::{marker::Ungil, pyclass, pymethods, types::PyList, Bound, FromPyObject, IntoPy, Py, PyAny, PyResult};

use crate::{pose, AprilTagDetection};
pub(super) use crate::pose::{AprilTagPose, OrthogonalIterationResult, AprilTagPoseWithError};

#[pyclass(frozen, module="apriltag_rs")]
pub(super) struct PoseEstimator {
	config: pose::PoseParams,
}

#[derive(FromPyObject)]
enum DetectionOrDetections<'py> {
    Detection(Bound<'py, super::PyDetection>),
	Detections(Bound<'py, super::PyDetections>),
}

impl<'py> DetectionOrDetections<'py> {
	fn map<R: Ungil + Send, F>(&self, callback: F) -> PyResult<Py<PyAny>>
		where
			Vec<R>: Ungil,
			F: Fn(&AprilTagDetection) -> R,
			F: Send + Ungil,
			R: IntoPy<Py<PyAny>>,
	{
		let obj = match self {
			Self::Detection(det) => {
				let dr: &AprilTagDetection = &det.get().detection;
				let res = det.py().allow_threads(move || callback(dr));
				res.into_py(det.py())
			},
			Self::Detections(dets) => {
				let dets_list = &dets.get().0.detections;
				let py = dets.py();
				let res = py.allow_threads(move || {
					dets_list
						.iter()
						.map(|it| callback(it))
						.collect::<Vec<_>>()
				});
				let items = res.into_iter()
					.map(|it| it.into_py(py));
				PyList::new_bound(dets.py(), items).unbind().into_any()
			},
		};
		Ok(obj)
	}
}

#[pymethods]
impl PoseEstimator {
	#[new]
	fn __new__(cx: f64, cy: f64, fx: f64, fy: f64, tagsize: f64) -> PyResult<Self> {
		Ok(Self {
			config: pose::PoseParams {
				cx, cy, fx, fy, tagsize
			}
		})
	}

	fn estimate_pose_for_tag_homography(&self, detection: DetectionOrDetections) -> PyResult<Py<PyAny>> {
		let config = &self.config;
		detection.map(|detection| {
			// Estimate single pose
			pose::estimate_pose_for_tag_homography(detection, config)
		})
	}

	#[pyo3(signature=(detection, n_iters = 50))]
	fn estimate_tag_pose_orthogonal_iteration(&self, detection: DetectionOrDetections, n_iters: usize) -> PyResult<Py<PyAny>> {
		let config = &self.config;
		detection.map(|detection| {
			pose::estimate_tag_pose_orthogonal_iteration(detection, config, n_iters)
		})
	}

	/// Estimate tag pose, returning best option
	fn estimate_tag_pose(&self, detection: DetectionOrDetections) -> PyResult<Py<PyAny>> {
		let config = &self.config;
		detection.map(|detection| {
			pose::estimate_tag_pose(detection, config)
		})
	}
}

#[pymethods]
impl OrthogonalIterationResult {
	fn __len__(&self) -> usize {
		let len = if self.solution2.is_some() { 2 } else { 1 };
		len
	}

	//TODO: iter

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}
}

#[pymethods]
impl AprilTagPoseWithError {
	fn __repr__(&self) -> String {
		format!("{self:?}")
	}
}

#[pymethods]
impl AprilTagPose {
	fn __repr__(&self) -> String {
		format!("{self:?}")
	}
}