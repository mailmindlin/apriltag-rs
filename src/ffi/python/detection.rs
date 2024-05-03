use std::{sync::Arc, ops::{Deref, DerefMut}, cell::RefCell};

use cpython::{exc, py_class, PyErr, PyList, PyObject, PyResult, PyString, PythonObject};
use crate::{
    AprilTagDetection,
    Detections as ATDetections, util::math::mat::Mat33
};

py_class!(pub class Detections |py| {
    data raw: Arc<ATDetections>;

    @property def nquads(&self) -> PyResult<u32> {
        let detections = self.raw(py);
        Ok(detections.nquads)
    }

    @property def detections(&self) -> PyResult<PyList> {
        let detections = self.raw(py);
        let mut py_objects = vec![];
        for i in 0..detections.detections.len() {
            let obj = Detection::create_instance(py, DetectionRef::Indexed(detections.clone(), i))?;
            py_objects.push(obj.into_object());
        }

        Ok(PyList::new(py, &py_objects))
    }

    @property def time_profile(&self) -> PyResult<super::debug::TimeProfile> {
        let data = self.raw(py);
        super::debug::TimeProfile::create_instance(py, data.tp.clone())
    }

    def __len__(&self) -> PyResult<usize> {
        let data = self.raw(py);
        Ok(data.detections.len())
    }

    def __getitem__(&self, key: PyObject) -> PyResult<PyObject> {
        let data = self.raw(py);
        if let Ok(idx) = key.extract::<isize>(py) {
            let idx_res = if idx > 0 { idx } else { data.detections.len() as isize - idx };
            if 0 <= idx_res && (idx_res as usize) < data.detections.len() {
                let res = Detection::create_instance(py, DetectionRef::Indexed(data.clone(), idx_res as usize))?;
                Ok(res.into_object())
            } else {
                Err(PyErr::new::<exc::IndexError, _>(py, "index out of range"))
            }
        } else {
            Err(PyErr::new::<exc::TypeError, _>(py, "unexpected key type"))
        }
    }

    def __iter__(&self) -> PyResult<DetectionsIter> {
        DetectionsIter::create_instance(py, self.raw(py).clone(), RefCell::new(0))
    }

    def __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.raw(py).as_ref()))
    }
});

py_class!(pub class DetectionsIter |py| {
    data detections: Arc<ATDetections>;
    data index: RefCell<usize>;
    
    def __next__(&self) -> PyResult<Option<Detection>> {
        let detections = self.detections(py);
        let mut index = self.index(py).borrow_mut();
        let index = index.deref_mut();
        if *index >= detections.detections.len() {
            Ok(None)
        } else {
            let result = Detection::create_instance(py, DetectionRef::Indexed(detections.clone(), *index))?;
            *index += 1;
            Ok(Some(result))
        }
    }
});

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

impl Detection {
    pub(super) fn detection_ref<'a>(&'a self, py: cpython::Python<'a>) -> &'a AprilTagDetection {
        self.detection(py).deref()
    }
}

py_class!(pub class Detection |py| {
    data detection: DetectionRef;

    @property def tag_id(&self) -> PyResult<usize> {
        let det = self.detection(py);
        Ok(det.id)
    }

    @property def family(&self) -> PyResult<super::PyAprilTagFamily> {
        let det = self.detection(py);
        super::PyAprilTagFamily::create_instance(py, RefCell::new(det.family.clone()))
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
        let res = det.corners.as_array().map(|p| (p[0], p[1]));
        Ok(res.to_vec())
    }

    @property def H(&self) -> PyResult<Mat33> {
        let det = self.detection(py);
        Ok(det.H)
    }

    def __str__(&self) -> PyResult<String> {
        let det = self.detection(py);
        Ok(format!("AprilTagDetection({} #{})", &det.family.name, det.id))
    }

    def __repr__(&self) -> PyResult<String> {
        let det = self.detection(py);
        Ok(format!("{det:?}"))
    }
});