use std::{borrow::Cow, sync::Arc};

use pyo3::{exceptions::{PyAttributeError, PyEnvironmentError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError}, pyclass, pymethods, types::{PyAnyMethods, PyString}, Bound, Py, PyAny, PyErr, PyObject, PyResult, Python, ToPyObject };
use parking_lot::RwLock;

use crate::{
    quickdecode::AddFamilyError, util::ImageY8, AccelerationRequest, AprilTagDetector, AprilTagQuadThreshParams, DetectError, DetectorBuildError, DetectorBuilder as AprilTagDetectorBuilder, DetectorConfig as AprilTagDetectorConfig
};

use super::{family::AprilTagFamily, timeout::PyTimeout};

/// Reference to some Config
pub enum ConfigRef {
    /// Owned
    Owned(RwLock<AprilTagDetectorConfig>),
    /// Borrowed from a Builder
    BuilderMut(Arc<RwLock<AprilTagDetectorBuilder>>),
    /// Borrowed from a Detector (read-only)
    DetectorRef(Py<Detector>),
}

impl ConfigRef {
	fn read<R>(&self, py: Python<'_>, callback: impl FnOnce(&AprilTagDetectorConfig) -> R) -> PyResult<R> {
		self.read_t(py, callback, PyTimeout::Blocking)
	}
    /// Run callback with read-only reference to config
    /// 
    /// May fail if [detector](variant.DetectorRef) is deallocated
    fn read_t<R>(&self, py: Python<'_>, callback: impl FnOnce(&AprilTagDetectorConfig) -> R, timeout: PyTimeout) -> PyResult<R> {
        let value = match self {
            Self::Owned(rwl) => callback(&*timeout.read_recursive(rwl)?),
            Self::BuilderMut(rwl) => callback(&timeout.read_recursive(rwl)?.config),
            Self::DetectorRef(rwl) => callback(&rwl.try_borrow(py)?.detector.params),
        };
		Ok(value)
    }

	fn write(&self, callback: impl FnOnce(&mut AprilTagDetectorConfig) -> ()) -> PyResult<()> {
		self.write_t(callback, PyTimeout::Blocking)
	}

    /// Mutates config
    /// 
    /// Fails if it's a reference to a [detector](variant.DetectorRef)
    fn write_t(&self, callback: impl FnOnce(&mut AprilTagDetectorConfig) -> (), timeout: PyTimeout) -> PyResult<()> {
        let value = match self {
            Self::Owned(rwl) => &mut timeout.write(rwl)?,
            Self::BuilderMut(rwl) => &mut timeout.write(rwl)?.config,
            Self::DetectorRef(_rwl)
                => return Err(PyErr::new::<PyAttributeError, _>("Cannot mutate AprilTagDetector"))
        };
		callback(value);
        Ok(())
    }
}

#[pyclass(module="apriltag_rs")]
pub(super) struct DetectorConfig {
	config: ConfigRef,
}

impl DetectorConfig {
	fn read_config<R>(self_: &Bound<Self>, callback: impl FnOnce(&AprilTagDetectorConfig) -> R) -> PyResult<R> {
		let py = self_.py();
		let me = self_.try_borrow()?;
		me.config.read(py, callback)
	}

	fn write_config(&self, callback: impl FnOnce(&mut AprilTagDetectorConfig) -> ()) -> PyResult<()> {
		self.config.write(callback)
	}
}

#[pymethods]
impl DetectorConfig {
	#[new]
	fn py_new() -> Self {
		Self {
			config: ConfigRef::Owned(RwLock::new(AprilTagDetectorConfig::default()))
		}
	}

    /// Number of theads to use (zero = automatic)
    #[getter]
	fn nthreads(self_:&Bound<Self>) -> PyResult<usize> {
		Self::read_config(self_, |config| config.nthreads)
    }

    #[setter]
    fn set_nthreads(&self, value: usize) -> PyResult<()> {
        self.write_config(|config| {
            config.nthreads = value;
        })
    }

    #[getter]
	fn quad_decimate(self_: &Bound<Self>) -> PyResult<f32> {
        Self::read_config(self_, |config| config.quad_decimate)
    }

    #[setter]
	fn set_quad_decimate(&self, value: f32) -> PyResult<()> {
        if value < 0. {
            Err(PyErr::new::<PyValueError, _>("quad_decimate must be non-negative"))
        } else {
            self.write_config(|config| {
                config.quad_decimate = value;
            })
        }
    }

    #[getter]
    fn quad_sigma(self_: &Bound<Self>) -> PyResult<f32> {
        Self::read_config(self_, |config| config.quad_sigma)
    }

    #[setter]
    fn set_quad_sigma(&self, value: f32) -> PyResult<()> {
        self.write_config(|config| {
            config.quad_sigma = value;
        })
    }

    #[getter]
    fn refine_edges(self_: &Bound<Self>) -> PyResult<bool> {
        Self::read_config(self_, |config| config.refine_edges)
    }

    #[setter]
    fn set_refine_edges(&self, value: bool) -> PyResult<()> {
        self.write_config(|config| {
            config.refine_edges = value;
        })
    }

    #[getter]
    fn debug(self_: &Bound<Self>) -> PyResult<bool> {
        Self::read_config(self_, |config| config.debug)
    }

    #[setter]
    fn set_debug(&self, value: bool) -> PyResult<()> {
        self.write_config(|config| {
            config.debug = value;
        })
    }

    #[getter]
    fn debug_path(self_: &Bound<Self>) -> PyResult<Option<String>> {
        Self::read_config(self_, |config| {
            match &config.debug_path {
                Some(path) => Some(path.to_string_lossy().into_owned()),
                None => None,
            }
        })
    }

    #[setter]
    fn set_debug_path(&self, value: Option<&str>) -> PyResult<()> {
        match value {
            None => {
                self.write_config(|config| {
                    config.debug_path = None;
                })
            },
            Some(value) => {
                let value = value
                    .to_string()
                    .into();
                self.write_config(|config| {
                    config.debug_path = Some(value);
                })
            }
        }
    }

    fn __str__(self_: &Bound<Self>) -> PyResult<String> {
        Self::read_config(self_, |config| format!("{config:?}"))
    }
    fn __repr__(self_: &Bound<Self>) -> PyResult<String> {
        Self::__str__(self_)
    }
}

#[pyclass(module="apriltag_rs")]
pub(super) struct QuadThresholdParams {
	config: ConfigRef,
}

#[pymethods]
impl QuadThresholdParams {
    #[getter]
    fn min_cluster_pixels(self_: &Bound<Self>) -> PyResult<u32> {
        Self::read(self_, |qtp| qtp.min_cluster_pixels)
    }

    #[setter]
    fn set_min_cluster_pixels(&self, value: u32) -> PyResult<()> {
        self.write(|qtp| {
            qtp.min_cluster_pixels = value;
        })
    }

    #[getter]
    fn max_nmaxima(self_: &Bound<Self>) -> PyResult<u8> {
        Self::read(self_, |qtp| qtp.max_nmaxima)
    }

    #[setter]
    fn set_max_nmaxima(&self, value: u8) -> PyResult<()> {
        self.write(|qtp| {
            qtp.max_nmaxima = value;
        })
    }

    #[getter]
    fn cos_critical_rad(self_: &Bound<Self>) -> PyResult<f32> {
        Self::read(self_, |qtp| qtp.cos_critical_rad)
    }

    #[setter]
    fn set_cos_critical_rad(&self, value: f32) -> PyResult<()> {
        self.write(|qtp| {
            qtp.cos_critical_rad = value;
        })
    }

    #[getter]
    fn max_line_fit_mse(self_: &Bound<Self>) -> PyResult<f32> {
        Self::read(self_, |qtp| qtp.max_line_fit_mse)
    }

    #[setter]
    fn set_max_line_fit_mse(&self, value: f32) -> PyResult<()> {
        self.write(|qtp| {
            qtp.max_line_fit_mse = value;
        })
    }

    #[getter]
    fn min_white_black_diff(self_: &Bound<Self>) -> PyResult<u8> {
        Self::read(self_, |qtp| qtp.min_white_black_diff)
    }

    #[setter]
    fn set_min_white_black_diff(&self, value: u8) -> PyResult<()> {
        self.write(|qtp| {
            qtp.min_white_black_diff = value;
        })
    }

    #[getter]
    fn deglitch(self_: &Bound<Self>) -> PyResult<bool> {
        Self::read(self_, |qtp| qtp.deglitch)
    }

    #[setter]
    fn set_deglitch(&self, value: bool) -> PyResult<()> {
        self.write(|qtp| {
            qtp.deglitch = value;
        })
    }
}

impl QuadThresholdParams {
    fn read<R>(self_: &Bound<Self>, callback: impl FnOnce(&AprilTagQuadThreshParams) -> R) -> PyResult<R> {
        self_.try_borrow()?.config.read(self_.py(), |config| callback(&config.qtp))
    }

    fn write(&self, callback: impl FnOnce(&mut AprilTagQuadThreshParams) -> ()) -> PyResult<()> {
        self.config.write(|config| callback(&mut config.qtp))
    }
}

/// Builder for AprilTag detector
#[pyclass(module="apriltag_rs")]
pub(super) struct DetectorBuilder {
	builder: Arc<RwLock<AprilTagDetectorBuilder>>
}

#[pymethods]
impl DetectorBuilder {
	#[new]
    fn __new__() -> Self {
        let builder = Arc::new(RwLock::new(AprilTagDetectorBuilder::default()));
        Self { builder }
    }

    #[getter]
    fn acceleration(self_: Bound<Self>) -> PyResult<PyObject> {
		let me = self_.try_borrow()?;
        let builder = me.builder.read();
        let text = match builder.gpu_mode() {
            AccelerationRequest::Disabled => return Ok(().to_object(self_.py())),
            AccelerationRequest::Prefer => Cow::Borrowed("prefer"),
            AccelerationRequest::Required => Cow::Borrowed("required"),
            AccelerationRequest::PreferGpu => Cow::Borrowed("prefer_gpu"),
            AccelerationRequest::RequiredGpu => Cow::Borrowed("required_gpu"),
            AccelerationRequest::PreferDeviceIdx(idx) => Cow::Owned(format!("prefer_{idx}")),
            AccelerationRequest::RequiredDeviceIdx(idx) => Cow::Owned(format!("required_{idx}")),
        };
        Ok(PyString::new_bound(self_.py(), &text).to_object(self_.py()))
    }

    #[setter]
    fn set_acceleration(self_: &Bound<Self>, value: Option<&str>) -> PyResult<()> {
        let request = match value {
            None => AccelerationRequest::Disabled,
            Some(value) => match value {
                "disable" => AccelerationRequest::Disabled,
                "prefer" => AccelerationRequest::Prefer,
                "required" => AccelerationRequest::Required,
                "prefer_gpu" => AccelerationRequest::PreferGpu,
                "required_gpu" => AccelerationRequest::RequiredGpu,
                value => {
                    let res = if let Some(idx_text) = value.strip_prefix("prefer_") {
                        if let Ok(idx) = idx_text.parse::<usize>() {
                            Some(AccelerationRequest::PreferDeviceIdx(idx))
                        } else {
                            None
                        }
                    } else if let Some(idx_text) = value.strip_prefix("required_") {
                        if let Ok(idx) = idx_text.parse::<usize>() {
                            Some(AccelerationRequest::RequiredDeviceIdx(idx))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    match res {
                        Some(res) => res,
                        None => return Err(PyErr::new::<PyValueError, _>(format!("Unsupported value {} for DetectorBuilder.acceleration", value)))
					}
                }
            }
        };
		{
			let me = self_.try_borrow()?;
			let builder = &me.builder;
			self_.py().allow_threads(|| {
				let mut builder = builder.write();
				builder.config.acceleration = request;
			});
		}

        Ok(())
    }

    #[getter]
    fn config(&self) -> PyResult<DetectorConfig> {
        Ok(DetectorConfig {
			config: ConfigRef::BuilderMut(self.builder.clone())
		})
    }

    #[setter]
    fn set_config(self_: &Bound<Self>, config: &Bound<DetectorConfig>) -> PyResult<()> {
		let config = config.try_borrow()?;
		let inner = &config.config;
		let me = self_.try_borrow()?;
		// Check this isn't *our* config to ensure that we don't deadlock
		if let ConfigRef::BuilderMut(builder) = inner {
			if Arc::ptr_eq(builder, &me.builder) {
				// No-op
				return Ok(());
			}
		}
		let builder = &me.builder;
		inner.read(self_.py(), |config| {
			let mut wg = builder.write();
			wg.config.clone_from(config);
		})
    }

    /// Add AprilTag family to detector
	#[pyo3(signature=(family, num_bits=None))]
    fn add_family(self_: &Bound<Self>, family: &Bound<PyAny>, num_bits: Option<usize>) -> PyResult<()> {
        // Parse family
        let family = if let Ok(family) = family.downcast::<super::PyAprilTagFamily>() {
            AprilTagFamily::raw_family(&family)?
        } else if let Ok(family_name) = family.downcast::<PyString>() {
            let family_name = family_name.to_string();
			//TODO: call PyAprilTagFamily::for_name directly?
			super::PyAprilTagFamily::for_name(&family_name)?.family
        } else {
            return Err(PyErr::new::<PyTypeError, _>(format!("Invalid AprilTag family type")));
        };

        let num_bits = num_bits.unwrap_or(2);
		let me = self_.try_borrow()?;
        let builder = &me.builder;

        // add_family_bits could be slow, so run it on a different thread
        let res = {
            let family = family.clone();
            self_.py().allow_threads(|| {
				let mut builder = builder.write();
                builder.add_family_bits(family, num_bits)
            })
        };
        
        match res {
            Ok(_) => Ok(()),
            Err(AddFamilyError::TooManyCodes(num_codes)) =>
                Err(PyErr::new::<PyValueError, _>(format!("Too many codes ({}, max 2**16) in AprilTag family {}", num_codes, family.name))),
            Err(AddFamilyError::BigHamming(hamming)) =>
                Err(PyErr::new::<PyValueError, _>(format!("Too many hamming bits ({}, max 3) when adding AprilTag family {}", hamming, family.name))),
            Err(AddFamilyError::QuickDecodeAllocation) =>
                Err(PyErr::new::<PyValueError, _>(format!("Unable to allocate memory for AprilTag family {}", family.name))),
        }
    }

    /// Remove all registered AprilTag families
    fn clear_families(&self) -> PyResult<()> {
        let mut builder = self.builder.write();
        builder.clear_families();
        Ok(())
    }

    fn build(self_: &Bound<Self>) -> PyResult<Detector> {
        let builder = self_.try_borrow()?
			.builder
            .read()
            .clone();// Ideally we'd be able to take the builder, but whatever

        match self_.py().allow_threads(|| builder.build()) {
            Ok(detector)
                => Ok(Detector { detector }),
            Err(DetectorBuildError::BufferAllocationFailure)
                => Err(PyErr::new::<PyMemoryError, _>("Unable to allocate buffer(s)")),
            Err(DetectorBuildError::NoTagFamilies)
                => Err(PyErr::new::<PyValueError, _>("No tag families were provided")),
            Err(DetectorBuildError::AccelerationNotAvailable)
                => Err(PyErr::new::<PyEnvironmentError, _>("Acceleration was required, but not available")),
            #[cfg(feature="opencl")]
            Err(DetectorBuildError::OpenCLError(e))
                => Err(PyErr::new::<PyEnvironmentError, _>(py, format!("OpenCL error: {e}"))),
            #[cfg(feature="wgpu")]
            Err(DetectorBuildError::WGPU(e))
                => Err(PyErr::new::<PyEnvironmentError, _>(format!("WGPU error: {e}"))),
            Err(DetectorBuildError::Threadpool(e))
                => Err(PyErr::new::<PyRuntimeError, _>(format!("Error building threadpool: {e}"))),
            Err(e)
                => Err(PyErr::new::<PyRuntimeError, _>(e.to_string())),
        }
    }
}

#[pyclass(module="apriltag_rs")]
pub(super) struct Detector {
	detector: AprilTagDetector,
}

#[pymethods]
impl Detector {
	#[staticmethod]
    fn builder() -> DetectorBuilder {
		DetectorBuilder::__new__()
    }
    
	#[new]
	#[pyo3(signature=(nthreads=None, quad_decimate=None, quad_sigma=None, refine_edges=None, decode_sharpening=None, debug=None))]
    fn __new__(nthreads: Option<usize>, quad_decimate: Option<f32>, quad_sigma: Option<f32>, refine_edges: Option<bool>, decode_sharpening: Option<f64>, debug: Option<bool>) -> PyResult<Self> {
        let mut builder = AprilTagDetector::builder();
        if let Some(nthreads) = nthreads {
            builder.config.nthreads = nthreads;
        }
        if let Some(quad_decimate) = quad_decimate {
            builder.config.quad_decimate = quad_decimate;
        }
        if let Some(quad_sigma) = quad_sigma {
            builder.config.quad_sigma = quad_sigma;
        }
        if let Some(refine_edges) = refine_edges {
            builder.config.refine_edges = refine_edges;
        }
        if let Some(decode_sharpening) = decode_sharpening {
            builder.config.decode_sharpening = decode_sharpening;
        }
        if let Some(debug) = debug {
            builder.config.debug = debug;
        }
        if let Some(refine_edges) = refine_edges {
            builder.config.refine_edges = refine_edges;
        }

        match builder.build() {
            Ok(detector) => Ok(Self {
				detector
			}),
            Err(DetectorBuildError::BufferAllocationFailure) => Err(PyErr::new::<PyMemoryError, _>("Buffer allocation failure")),
            Err(DetectorBuildError::AccelerationNotAvailable) => Err(PyErr::new::<PyRuntimeError, _>("OpenCL not available")),
            Err(DetectorBuildError::Threadpool(tpbe)) => Err(PyErr::new::<PyRuntimeError, _>(format!("Failed building threadpool: {tpbe}"))),
            Err(e)
                => Err(PyErr::new::<PyRuntimeError, _>(e.to_string())),
        }
    }

    #[getter]
    fn config(self_: &Bound<Self>) -> DetectorConfig {
		let me = self_.clone().unbind();
		DetectorConfig {
			config: ConfigRef::DetectorRef(me)
		}
    }

    fn detect(self_: &Bound<Self>, image: ImageY8) -> PyResult<super::PyDetections> {
        // let detector_ref = match self.detector.try_borrow() {
        //     Ok(det) => det,
        //     Err(e) => return Err(PyErr::new::<exc::InterruptedError, _>(py, format!("Unable to borrow detector: {e}"))),
        // };
		let me = self_.try_borrow()?;

        let res = {
            let detector: &AprilTagDetector = &me.detector;
            self_.py().allow_threads(|| {
                let res = detector.detect(&image);
                drop(image);
                res
            })
        };

        match res {
            Ok(detections) => Ok(super::PyDetections(Arc::new(detections))),
            Err(DetectError::ImageAlloc(..)) =>
                Err(PyErr::new::<PyMemoryError, _>(format!("Unable to allocate buffers"))),
            Err(DetectError::BadSourceImageDimensions(..)) =>
                Err(PyErr::new::<PyValueError, _>(format!("Source image too small"))),  
            Err(e)
                => Err(PyErr::new::<PyRuntimeError, _>(e.to_string())),
        }
    }
}