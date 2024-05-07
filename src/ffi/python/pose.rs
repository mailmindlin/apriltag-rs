use cpython::{exc, py_class, PyErr, PyObject, PyResult, PythonObject};

use crate::{pose, util::math::{mat::Mat33, Vec3}};

py_class!(pub class PoseEstimator |py| {
	data config: pose::PoseParams;
	def __new__(_cls, cx: f64, cy: f64, fx: f64, fy: f64, tagsize: f64) -> PyResult<Self> {
		Self::create_instance(py, pose::PoseParams {
			cx, cy, fx, fy, tagsize
		})
	}

	def estimate_pose_for_tag_homography(&self, detection: PyObject) -> PyResult<PyObject> {
		if let Ok(detection) = detection.extract::<super::PyDetection>(py) {
			// Estimate single pose
			let config = self.config(py);
			let at_detection = detection.detection_ref(py);
			let pose = py.allow_threads(|| pose::estimate_pose_for_tag_homography(at_detection, config));
			Ok(AprilTagPose::create_instance(py, pose)?.into_object())
		// } else if let Ok(detections) = detection.extract::<super::PyDetections>(py) {
			// Multiple poses

		} else {
			Err(PyErr::new::<exc::ValueError, _>(py, format!("Unknown argument type")))
		}
	}

	def estimate_tag_pose_orthogonal_iteration(&self, detection: &super::PyDetection, n_iters: usize = 50) -> PyResult<PyObject> {
		// Estimate single pose
		let config = self.config(py);
		let at_detection = detection.detection_ref(py);
		let pose = py.allow_threads(|| pose::estimate_tag_pose_orthogonal_iteration(at_detection, config, n_iters));
		Ok(OrthogonalIterationResult::create_instance(py, pose)?.into_object())
	}

	/// Estimate tag pose, returning best option
	def estimate_tag_pose(&self, detection: &super::PyDetection) -> PyResult<AprilTagPoseWithError> {
		let config = self.config(py);
		let at_detection = detection.detection_ref(py);
		let pose = py.allow_threads(|| pose::estimate_tag_pose(at_detection, config));
		AprilTagPoseWithError::create_instance(py, pose)
	}
});

py_class!(pub class OrthogonalIterationResult |py| {
	data inner: pose::OrthogonalIterationResult;

	/// Best pose solution
	@property
	def solution1(&self) -> PyResult<AprilTagPoseWithError> {
		let solution1 = self.inner(py).solution1;
		AprilTagPoseWithError::create_instance(py, solution1)
	}

	/// Second-best pose solution
	@property
    def solution2(&self) -> PyResult<Option<AprilTagPoseWithError>> {
        match self.inner(py).solution2 {
			Some(solution2) => Ok(Some(AprilTagPoseWithError::create_instance(py, solution2)?)),
			None => Ok(None),
		}
    }

	def __len__(&self) -> PyResult<usize> {
		let len = if self.inner(py).solution2.is_some() { 2 } else { 1 };
		Ok(len)
	}

	//TODO: iter

	def __repr__(&self) -> PyResult<String> {
		Ok(format!("{:?}", self.inner(py)))
	}
});

py_class!(pub class AprilTagPoseWithError |py| {
	data inner: pose::AprilTagPoseWithError;
	/// Get mean estimated pose
	@property
	def pose(&self) -> PyResult<AprilTagPose> {
		AprilTagPose::create_instance(py, self.inner(py).pose)
	}

	/// Get pose error
	@property
	def error(&self) -> PyResult<f64> {
		Ok(self.inner(py).error)
	}

	def __repr__(&self) -> PyResult<String> {
		Ok(format!("{:?}", self.inner(py)))
	}
});

py_class!(pub class AprilTagPose |py| {
	data inner: pose::AprilTagPose;
	/// 3x3 rotation matrix
	@property
	def R(&self) -> PyResult<Mat33> {
		Ok(self.inner(py).R)
	}

	/// Translation vector
	@property
	def t(&self) -> PyResult<Vec3> {
		Ok(self.inner(py).t)
	}

	def __repr__(&self) -> PyResult<String> {
		Ok(format!("{:?}", self.inner(py)))
	}
});