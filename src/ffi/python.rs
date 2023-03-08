use std::sync::RwLock;

use cpython::{PyResult, py_class, PyString, PyBool, py_module_initializer, PySequence, exc, PyErr};
use crate::{ApriltagDetector, ApriltagDetection};

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
    def __new__(_cls, families: PySequence, nthreads: Option<usize>, quad_decimate: Option<f32>, quad_sigma: Option<f32>, refine_edges: Option<bool>, decode_sharpening: Option<f64>, debug: Option<bool>, camera_params: Option<PySequence>) -> PyResult<Detector> {
        let detector = ApriltagDetector::default();
        Self::create_instance(py, RwLock::new(detector))
    }

    @property def nthreads(&self) -> PyResult<usize> {
        let det = self.detector(py).read().unwrap();
        Ok(det.params.nthreads)
    }

    @nthreads.setter def set_nthreads(&self, value: Option<usize>) -> PyResult<()> {
        if let Some(value) = value {
            let det = self.detector(py).write().unwrap();
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
            let det = self.detector(py).write().unwrap();
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
            let det = self.detector(py).write().unwrap();
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
            let det = self.detector(py).write().unwrap();
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
            let det = self.detector(py).write().unwrap();
            det.params.debug = value.is_true();
            Ok(())
        } else {
            Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete debug"))
        }
    }
});


py_module_initializer!(apriltag_rs, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    // m.add(py, "sum_as_string", py_fn!(py, sum_as_string_py(a: i64, b:i64)))?;
    m.add_class::<Detection>(py)?;
    m.add_class::<Detector>(py)?;

    Ok(())
});