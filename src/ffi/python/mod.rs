use std::sync::RwLock;

use cpython::{PyResult, py_class, PyString, PyBool, py_module_initializer, PySequence, exc, PyErr, PyObject, buffer::PyBuffer};
use crate::{ApriltagDetector, ApriltagDetection, AprilTagFamily as ATFamily, quickdecode::AddFamilyError};



py_class!(class Detection |py| {
    data detection: ApriltagDetection;

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
});

py_class!(class Detector |py| {
    data detector: RwLock<ApriltagDetector>;
    def __new__(_cls, families: Option<PySequence>, nthreads: Option<usize>, quad_decimate: Option<f32>, quad_sigma: Option<f32>, refine_edges: Option<bool>, decode_sharpening: Option<f64>, debug: Option<bool>, camera_params: Option<PySequence>) -> PyResult<Detector> {
        let mut detector = ApriltagDetector::default();
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

        if let Some(families) = families {
            for family in families.iter(py)? {
                let family = family?;
            }
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

    def add_family(&self, family: PyString, num_bits: Option<usize>) -> PyResult<PyObject> {
        let num_bits = num_bits.unwrap_or(2);
        let mut det = self.detector(py).write().unwrap();

        let family_name = family.to_string(py)?;
        let family = ATFamily::for_name(&family_name)
            .ok_or_else(|| PyErr::new::<exc::ValueError, _>(py, format!("Unknown AprilTag family: {}", family_name)))?;
        
        match det.add_family_bits(family, num_bits) {
            Ok(_) => Ok(py.None()),
            Err(AddFamilyError::TooManyCodes(num_codes)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Too many codes ({}, max 2**16) in AprilTag family {}", num_codes, family_name))),
            Err(AddFamilyError::BigHamming(hamming)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Too many hamming bits ({}, max 3) when adding AprilTag family {}", hamming, family_name))),
            Err(AddFamilyError::QuickDecodeAllocation(e)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Unable to allocate memory for AprilTag family {}: {}", family_name, e))),
        }
    }

    def detect(&self, image: PyObject) -> PyResult<Vec<Detection>> {
        let img_buf = PyBuffer::get(py, &image)?;
        if img_buf.dimensions() != 2 {
            return Err(PyErr::new::<exc::ValueError, _>(py, format!("Expected 2d numpy array")));
        }
        println!("img_buf format: {:?}", img_buf.format());
        Ok(Vec::new())
    }
});


py_module_initializer!(apriltag_rs, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    // m.add(py, "sum_as_string", py_fn!(py, sum_as_string_py(a: i64, b:i64)))?;
    m.add_class::<Detection>(py)?;
    m.add_class::<Detector>(py)?;

    Ok(())
});