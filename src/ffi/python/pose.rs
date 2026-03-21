use pyo3::{Bound, BoundObject, FromPyObject, IntoPyObject, Py, PyAny, PyResult, marker::Ungil, pyclass, pymethods, types::PyList};

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
			R: IntoPyObject<'py>,
			<R as IntoPyObject<'py>>::Error: Into<pyo3::PyErr>,
	{
		let obj = match self {
			Self::Detection(det) => {
				let dr: &AprilTagDetection = &det.get().detection;
				let res = det.py().detach(move || callback(dr));
				res.into_pyobject(det.py()).map_err(Into::into)?.into_any().unbind()
			},
			Self::Detections(dets) => {
				let dets_list = &dets.get().0.detections;
				let py = dets.py();
				let res = py.detach(move || {
					dets_list
						.iter()
						.map(|it| callback(it))
						.collect::<Vec<_>>()
				});
				PyList::new(py, res)?.into_any().unbind()
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

	fn __repr__(&self) -> String {
		format!("{:?}", &self.config)
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