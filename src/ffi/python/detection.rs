use std::{sync::Arc, ops::Deref};

use pyo3::{exceptions::{PyIndexError, PyTypeError}, pyclass, pymethods, types::PyAnyMethods, Bound, PyAny, PyErr, PyResult};
use crate::{
    AprilTagDetection,
    Detections as ATDetections, util::math::mat::Mat33
};

#[pyclass(frozen, module="apriltag_rs")]
pub(super) struct Detections(pub(super) Arc<ATDetections>);

#[pymethods]
impl Detections {
	#[getter]
    fn nquads(&self) -> u32 {
        self.0.nquads
    }

	#[getter]
    fn detections(&self) -> Vec<Detection> {
        let mut res = vec![];
        for i in 0..self.0.detections.len() {
            res.push(Detection { detection: DetectionRef::Indexed(self.0.clone(), i) });
        }

        res
    }

	#[getter]
    fn time_profile(&self) -> super::debug::TimeProfile {
        self.0.tp.clone()
    }

	/// Number of detections
    fn __len__(&self) -> usize {
        self.0.detections.len()
    }

    fn __getitem__(&self, key: &Bound<PyAny>) -> PyResult<Detection> {
        let data = &self.0;
        if let Ok(idx) = key.extract::<isize>() {
            let idx_res = if idx > 0 { idx } else { data.detections.len() as isize - idx };
            if 0 <= idx_res && (idx_res as usize) < data.detections.len() {
                Ok(Detection { detection: DetectionRef::Indexed(data.clone(), idx_res as usize) })
            } else {
                Err(PyErr::new::<PyIndexError, _>(format!("index {idx} out of range")))
            }
        } else {
            Err(PyErr::new::<PyTypeError, _>("unexpected key type"))
        }
    }

    fn __iter__(&self) -> PyResult<DetectionsIter> {
		Ok(DetectionsIter {
			detections: self.0.clone(),
			index: 0,
		})
    }

    fn __repr__(&self) -> String {
		format!("{:?}", self.0.as_ref())
    }
}

#[pyclass(module="apriltag_rs")]
pub(super) struct DetectionsIter {
	detections: Arc<ATDetections>,
	index: usize,
}

#[pymethods]
impl DetectionsIter {
	fn __next__(&mut self) -> PyResult<Option<Detection>> {
        let detections = &self.detections;
        let index = &mut self.index;
        if *index >= detections.detections.len() {
            Ok(None)
        } else {
			let result = Detection {
				detection: DetectionRef::Indexed(detections.clone(), *index),
			};
            *index += 1;
            Ok(Some(result))
        }
    }
}

pub enum DetectionRef {
    // Owned(Box<ApriltagDetection>),
    Indexed(Arc<ATDetections>, usize),
}

impl Deref for DetectionRef {
    type Target = AprilTagDetection;

    fn deref(&self) -> &Self::Target {
        match self {
            // DetectionRef::Owned(v) => v,
            DetectionRef::Indexed(list, idx) => &list.detections[*idx],
        }
    }
}

impl core::fmt::Debug for DetectionRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v: &AprilTagDetection = &*self;
        f.debug_struct("Detection")
            .field("tag_family", &v.family.name)
            .field("tag_id", &v.id)
            .field("hamming", &v.hamming)
            .field("center", &v.center)
            .finish_non_exhaustive()
    }
}

// impl Detection {
//     pub(super) fn detection_ref<'a>(&'a self, py: Python<'a>) -> &'a AprilTagDetection {
//         self.detection(py).deref()
//     }
// }

#[pyclass(frozen, module="apriltag_rs")]
pub(super) struct Detection {
	pub(super) detection: DetectionRef,
}

#[pymethods]
impl Detection {
	#[getter]
    fn tag_id(&self) -> usize {
        self.detection.id
    }

    #[getter]
    fn family(&self) -> PyResult<super::PyAprilTagFamily> {
        let det = &self.detection;
        Ok(super::PyAprilTagFamily {
			family: det.family.clone()
		})
    }

    #[getter]
    fn tag_family(&self) -> String {
        self.detection.family.name.clone().into_owned()
    }

    #[getter]
    fn hamming(&self) -> u16 {
        self.detection.hamming
    }

    #[getter]
    fn decision_margin(&self) -> f32 {
        self.detection.decision_margin
    }

    #[getter]
    fn center(&self) -> PyResult<(f64, f64)> {
        let det = &self.detection;
        Ok((det.center.x(), det.center.y()))
    }

    #[getter]
    fn corners(&self) -> PyResult<Vec<(f64, f64)>> {
        let det = &self.detection;
        let res = det.corners.as_array().map(|p| (p[0], p[1]));
        Ok(res.to_vec())
    }

    #[getter]
    fn H(&self) -> Mat33 {
        self.detection.H
    }

    fn __str__(&self) -> PyResult<String> {
        let det = &self.detection;
        Ok(format!("AprilTagDetection({} #{})", &det.family.name, det.id))
    }

    fn __repr__(&self) -> PyResult<String> {
        let det = &self.detection;
        Ok(format!("{det:?}"))
    }
}