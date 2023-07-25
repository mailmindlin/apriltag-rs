use std::{sync::Arc, ops::Deref};

use cpython::{PyResult, py_class, PyString, PythonObject, PyList};
use crate::{
    AprilTagDetection,
    Detections as ATDetections
};

py_class!(pub(super) class Detections |py| {
    data dets: Arc<ATDetections>;

    @property def nquads(&self) -> PyResult<u32> {
        let detections = self.dets(py);
        Ok(detections.nquads)
    }

    @property def detections(&self) -> PyResult<PyList> {
        let detections = self.dets(py);
        let mut py_objects = vec![];
        for i in 0..detections.detections.len() {
            let obj = Detection::create_instance(py, DetectionRef::Indexed(detections.clone(), i))?;
            py_objects.push(obj.into_object());
        }

        Ok(PyList::new(py, &py_objects))
    }

    @property def time_profile(&self) -> PyResult<super::debug::TimeProfile> {
        let data = self.dets(py);
        super::debug::TimeProfile::create_instance(py, data.tp.clone())
    }
});

pub(super) enum DetectionRef {
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

py_class!(pub(super) class Detection |py| {
    data detection: DetectionRef;

    def __str__(&self) -> PyResult<String> {
        let det = self.detection(py);
        Ok(format!("AprilTagDetection({} #{})", &det.family.name, det.id))
    }

    @property def tag_id(&self) -> PyResult<usize> {
        let det = self.detection(py);
        Ok(det.id)
    }

    @property def tag_family(&self) -> PyResult<PyString> {
        let det = self.detection(py);
        let name = PyString::new(py, &det.family.name);
        Ok(name)
    }

    @property def hamming(&self) -> PyResult<usize> {
        let det = self.detection(py);
        Ok(det.hamming as usize)
    }

    @property def decision_margin(&self) -> PyResult<f32> {
        let det = self.detection(py);
        Ok(det.decision_margin)
    }

    @property def center(&self) -> PyResult<(f64, f64)> {
        let det = self.detection(py);
        Ok((det.center.x(), det.center.y()))
    }

    @property def corners(&self) -> PyResult<Vec<(f64, f64)>> {
        let det = self.detection(py);
        let res = det.corners.map(|p| (p.x(), p.y()));
        Ok(res.to_vec())
    }
});