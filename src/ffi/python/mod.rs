use std::{sync::{RwLock, Arc}, ops::Deref, cell::{RefCell}, borrow::Cow};

use cpython::{PyResult, py_class, PyString, PyBool, py_module_initializer, PySequence, exc, PyErr, PyObject, buffer::PyBuffer, PythonObject, PyList};
use crate::{AprilTagDetector, AprilTagDetection, Detections as ATDetections, AprilTagFamily as ATFamily, quickdecode::AddFamilyError, util::ImageY8, dbg::TimeProfile as ATTimeProfile};

py_class!(class TimeProfile |py| {
    data data: ATTimeProfile;

    def total_time(&self) -> PyResult<f64> {
        let data = self.data(py);
        Ok(data.total_duration().as_secs_f64())
    }

    def __str__(&self) -> PyResult<String> {
        let data = self.data(py);
        Ok(format!("{data}"))
    }
});

py_class!(class Detections |py| {
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

    @property def time_profile(&self) -> PyResult<TimeProfile> {
        let data = self.dets(py);
        TimeProfile::create_instance(py, data.tp.clone())
    }
});

enum DetectionRef {
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

py_class!(class Detection |py| {
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

py_class!(class AprilTagFamily |py| {
    data family: RefCell<Arc<ATFamily>>;

    /// Create AprilTag family for name
    @staticmethod def for_name(name: PyString) -> PyResult<PyObject> {
        let name = name.to_string(py)?;
        let family = match ATFamily::for_name(&name) {
            Some(family) => family,
            None => return Ok(py.None()),
        };
        let x = Self::create_instance(py, RefCell::new(family))?;
        Ok(x.into_object())
    }

    @property def width_at_border(&self) -> PyResult<u32> {
        let family = self.family(py).borrow();
        Ok(family.width_at_border)
    }

    @property def total_width(&self) -> PyResult<u32> {
        let family = self.family(py).borrow();
        Ok(family.total_width)
    }

    @property def reversed_border(&self) -> PyResult<bool> {
        let family = self.family(py).borrow();
        Ok(family.reversed_border)
    }

    @property def min_hamming(&self) -> PyResult<u32> {
        let family = self.family(py).borrow();
        Ok(family.min_hamming)
    }

    @property def name(&self) -> PyResult<PyString> {
        let family = self.family(py).borrow();
        Ok(PyString::new(py, &family.name))
    }

    @property def bits(&self) -> PyResult<Vec<(u32, u32)>> {
        let family = self.family(py).borrow();
        Ok(family.bits.clone())
    }

    @property def codes(&self) -> PyResult<Vec<u64>> {
        let family = self.family(py).borrow();
        Ok(family.codes.clone())
    }

    def __str__(&self) -> PyResult<String> {
        Ok(format!("AprilTagFamily {{ name = {} .. }}", self.family(py).borrow().name))
    }

    @codes.setter def set_codes(&self, value: Option<Vec<u64>>) -> PyResult<()> {
        let value = match value {
            Some(value) => value,
            None => return Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete quad_sigma")),
        };

        let mut family = self.family(py).borrow_mut();

        // Create mutated family
        let mut new_family = family.as_ref().clone();
        if !new_family.name.ends_with(" (modified)") {
            new_family.name = Cow::Owned(String::from(new_family.name) + " (modified)");
        }
        new_family.codes = value;

        *family = Arc::new(new_family);
        Ok(())
    }

    def to_image(&self, idx: usize) -> PyResult<u64> {
        let family = self.family(py).borrow();
        let img = family.to_image(idx);

        let np = py.import("numpy")?;
        np.get(py, "empty")?;
        todo!()
    }
});

py_class!(class Detector |py| {
    data detector: RwLock<AprilTagDetector>;
    def __new__(_cls, nthreads: Option<usize>, quad_decimate: Option<f32>, quad_sigma: Option<f32>, refine_edges: Option<bool>, decode_sharpening: Option<f64>, debug: Option<bool>, camera_params: Option<PySequence>) -> PyResult<Detector> {
        let mut detector = AprilTagDetector::default();
        if let Some(nthreads) = nthreads {
            detector.params.nthreads = nthreads;
        }
        if let Some(quad_decimate) = quad_decimate {
            detector.params.quad_decimate = quad_decimate;
        }
        if let Some(quad_sigma) = quad_sigma {
            detector.params.quad_sigma = quad_sigma;
        }
        if let Some(refine_edges) = refine_edges {
            detector.params.refine_edges = refine_edges;
        }
        if let Some(decode_sharpening) = decode_sharpening {
            detector.params.decode_sharpening = decode_sharpening;
        }
        if let Some(debug) = debug {
            detector.params.debug = debug;
        }
        if let Some(refine_edges) = refine_edges {
            detector.params.refine_edges = refine_edges;
        }
        Self::create_instance(py, RwLock::new(detector))
    }


    @property def nthreads(&self) -> PyResult<usize> {
        let det = self.detector(py).read().unwrap();
        Ok(det.params.nthreads)
    }

    @nthreads.setter def set_nthreads(&self, value: Option<usize>) -> PyResult<()> {
        if let Some(value) = value {
            let mut det = self.detector(py).write().unwrap();
            det.params.nthreads = value;
            Ok(())
        } else {
            Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete nthreads"))
        }
    }

    @property def quad_decimate(&self) -> PyResult<f32> {
        let det = self.detector(py).read().unwrap();
        Ok(det.params.quad_decimate)
    }

    @quad_decimate.setter def set_qd(&self, value: Option<f32>) -> PyResult<()> {
        if let Some(value) = value {
            let mut det = self.detector(py).write().unwrap();
            det.params.quad_decimate = value;
            Ok(())
        } else {
            Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete quad_decimate"))
        }
    }

    @property def quad_sigma(&self) -> PyResult<f32> {
        let det = self.detector(py).read().unwrap();
        Ok(det.params.quad_sigma)
    }

    @quad_sigma.setter def set_qs(&self, value: Option<f32>) -> PyResult<()> {
        if let Some(value) = value {
            let mut det = self.detector(py).write().unwrap();
            det.params.quad_sigma = value;
            Ok(())
        } else {
            Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete quad_sigma"))
        }
    }

    @property def refine_edges(&self) -> PyResult<PyBool> {
        let det = self.detector(py).read().unwrap();
        return if det.params.refine_edges {
            Ok(py.True())
        } else {
            Ok(py.False())
        }
    }

    @refine_edges.setter def set_rf(&self, value: Option<PyBool>) -> PyResult<()> {
        if let Some(value) = value {
            let mut det = self.detector(py).write().unwrap();
            det.params.refine_edges = value.is_true();
            Ok(())
        } else {
            Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete refine_edges"))
        }
    }

    @property def debug(&self) -> PyResult<PyBool> {
        let det = self.detector(py).read().unwrap();
        Ok(if det.params.debug {
            py.True()
        } else {
            py.False()
        })
    }

    @debug.setter def set_debug(&self, value: Option<PyBool>) -> PyResult<()> {
        if let Some(value) = value {
            let mut det = self.detector(py).write().unwrap();
            det.params.debug = value.is_true();
            Ok(())
        } else {
            Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete debug"))
        }
    }

    def add_family(&self, family: AprilTagFamily, num_bits: Option<usize>) -> PyResult<PyObject> {
        let num_bits = num_bits.unwrap_or(2);
        let mut det = self.detector(py).write().unwrap();


        // let family_name = family.to_string(py)?;
        // let family = ATFamily::for_name(&family_name)
        //     .ok_or_else(|| PyErr::new::<exc::ValueError, _>(py, format!("Unknown AprilTag family: {}", family_name)))?;

        let family = family.family(py).borrow().clone();
        match det.add_family_bits(family.clone(), num_bits) {
            Ok(_) => Ok(py.None()),
            Err(AddFamilyError::TooManyCodes(num_codes)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Too many codes ({}, max 2**16) in AprilTag family {}", num_codes, family.name))),
            Err(AddFamilyError::BigHamming(hamming)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Too many hamming bits ({}, max 3) when adding AprilTag family {}", hamming, family.name))),
            Err(AddFamilyError::QuickDecodeAllocation) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Unable to allocate memory for AprilTag family {}", family.name))),
        }
    }

    def detect(&self, image: PyObject) -> PyResult<PyObject> {
        let img_buf = PyBuffer::get(py, &image)?;
        if img_buf.dimensions() != 2 {
            return Err(PyErr::new::<exc::ValueError, _>(py, format!("Expected 2d numpy array")));
        }
        let shape = img_buf.shape();
        let img = {
            let mut img = ImageY8::zeroed_packed(shape[1], shape[0]);
            let width = img.width();
            let v = img_buf.to_vec::<u8>(py)?;
            for ((x, y), dst) in img.enumerate_pixels_mut() {
                dst.0 = [v[y * width + x]];
            }
            img
        };
        let detector = self.detector(py).read().unwrap();
        let detections = py.allow_threads(|| detector.detect(&img));
        drop(img);
        let r = Detections::create_instance(py, Arc::new(detections)).expect("foo");
        Ok(r.into_object())
    }
});


py_module_initializer!(apriltag_rs, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    // m.add(py, "sum_as_string", py_fn!(py, sum_as_string_py(a: i64, b:i64)))?;
    m.add_class::<TimeProfile>(py)?;
    m.add_class::<Detections>(py)?;
    m.add_class::<AprilTagFamily>(py)?;
    m.add_class::<Detection>(py)?;
    m.add_class::<Detector>(py)?;

    Ok(())
});