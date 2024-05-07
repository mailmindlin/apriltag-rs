use std::{borrow::{Borrow, Cow}, sync::Arc};

use cpython::{exc, py_class, PyBool, PyErr, PyObject, PyResult, PyString, Python, PythonObject, UnsafePyLeaked};
use parking_lot::RwLock;

use crate::{
    quickdecode::AddFamilyError, util::ImageY8, AccelerationRequest, AprilTagDetector, AprilTagFamily as ATFamily, AprilTagQuadThreshParams, DetectError, DetectorBuildError, DetectorBuilder as AprilTagDetectorBuilder, DetectorConfig as AprilTagDetectorConfig
};

use super::shim::readonly;

/// Reference to some Config
pub enum ConfigRef {
    /// Owned
    Owned(RwLock<AprilTagDetectorConfig>),
    /// Borrowed from a Builder
    BuilderMut(Arc<RwLock<AprilTagDetectorBuilder>>),
    /// Borrowed from a Detector (read-only)
    DetectorRef(UnsafePyLeaked<&'static AprilTagDetector>),
}

impl ConfigRef {
    /// Run callback with read-only reference to config
    /// 
    /// May fail if [detector](variant.DetectorRef) is deallocated
    fn read<R>(&self, py: Python, callback: impl FnOnce(&AprilTagDetectorConfig) -> R) -> PyResult<R> {
        match self {
            Self::Owned(rwl) => Ok(callback(&rwl.read())),
            Self::BuilderMut(rwl) => Ok(callback(&rwl.read().config)),
            Self::DetectorRef(rwl) => Ok(callback(&unsafe { rwl.try_borrow(py) }?.params))
        }
    }

    /// Mutates config
    /// 
    /// Fails if it's a reference to a [detector](variant.DetectorRef)
    fn write(&self, py: Python, callback: impl FnOnce(&mut AprilTagDetectorConfig) -> ()) -> PyResult<()> {
        match self {
            Self::Owned(rwl) => callback(&mut rwl.write()),
            Self::BuilderMut(rwl) => callback(&mut rwl.write().config),
            Self::DetectorRef(_rwl)
                => return Err(PyErr::new::<exc::AttributeError, _>(py, "Cannot mutate AprilTagDetector"))
        }
        Ok(())
    }
}

py_class!(pub class DetectorConfig |py| {
    data config: ConfigRef;

    def __new__(_cls) -> PyResult<Self> {
        Self::create_instance(py, ConfigRef::Owned(RwLock::new(AprilTagDetectorConfig::default())))
    }

    /// Number of theads to use (zero = automatic)
    @property def nthreads(&self) -> PyResult<usize> {
        self.config(py).read(py, |config| config.nthreads)
    }

    @nthreads.setter def set_nthreads(&self, value: Option<usize>) -> PyResult<()> {
        let value = readonly(py, value, "nthreads")?;
        self.config(py).write(py, |config| {
            config.nthreads = value;
        })
    }

    @property def quad_decimate(&self) -> PyResult<f32> {
        self.config(py).read(py, |config| config.quad_decimate)
    }

    @quad_decimate.setter def set_quad_decimate(&self, value: Option<f32>) -> PyResult<()> {
        let value = readonly(py, value, "quad_decimate")?;
        if value < 0. {
            Err(PyErr::new::<exc::ValueError, _>(py, "quad_decimate must be non-negative"))
        } else {
            self.config(py).write(py, |config| {
                config.quad_decimate = value;
            })
        }
    }

    @property def quad_sigma(&self) -> PyResult<f32> {
        self.config(py).read(py, |config| config.quad_sigma)
    }

    @quad_sigma.setter def set_quad_sigma(&self, value: Option<f32>) -> PyResult<()> {
        let value = readonly(py, value, "quad_sigma")?;

        self.config(py).write(py, |config| {
            config.quad_sigma = value;
        })
    }

    @property def refine_edges(&self) -> PyResult<PyBool> {
        if self.config(py).read(py, |config| config.refine_edges)? {
            Ok(py.True())
        } else {
            Ok(py.False())
        }
    }

    @refine_edges.setter def set_refine_edges(&self, value: Option<PyBool>) -> PyResult<()> {
        let value = readonly(py, value, "refine_edges")?
            .is_true();
        self.config(py).write(py, |config| {
            config.refine_edges = value;
        })
    }

    @property def debug(&self) -> PyResult<PyBool> {
        let debug = self.config(py).read(py, |config| config.debug)?;
        Ok(if debug {
            py.True()
        } else {
            py.False()
        })
    }

    @debug.setter def set_debug(&self, value: Option<PyBool>) -> PyResult<()> {
        let value = readonly(py, value, "debug")?;
        let value = value.is_true();
        self.config(py).write(py, |config| {
            config.debug = value;
        })
    }

    @property def debug_path(&self) -> PyResult<Option<PyString>> {
        self.config(py).read(py, |config| {
            match &config.debug_path {
                Some(path) => Some(PyString::new(py, &path.to_string_lossy())),
                None => None,
            }
        })
    }

    @debug_path.setter def set_debug_path(&self, value: Option<PyString>) -> PyResult<()> {
        match value {
            None => {
                self.config(py).write(py, |config| {
                    config.debug_path = None;
                })
            },
            Some(value) => {
                let value = value
                    .to_string(py)?
                    .to_string()
                    .into();
                self.config(py).write(py, |config| {
                    config.debug_path = Some(value);
                })
            }
        }
    }

    def __str__(&self) -> PyResult<String> {
        self.config(py).read(py, |config| format!("{config:?}"))
    }
    def __repr__(&self) -> PyResult<String> {
        self.__str__(py)
    }
});

py_class!(pub class QuadThresholdParams |py| {
    data config: UnsafePyLeaked<&'static ConfigRef>;

    @property def min_cluster_pixels(&self) -> PyResult<u32> {
        self.read(py, |qtp| qtp.min_cluster_pixels)
    }

    @deglitch.setter def set_min_cluster_pixels(&self, value: Option<u32>) -> PyResult<()> {
        let value = readonly(py, value, "min_cluster_pixels")?;
        self.write(py, |qtp| {
            qtp.min_cluster_pixels = value;
        })
    }

    @property def max_nmaxima(&self) -> PyResult<u8> {
        self.read(py, |qtp| qtp.max_nmaxima)
    }

    @deglitch.setter def set_max_nmaxima(&self, value: Option<u8>) -> PyResult<()> {
        let value = readonly(py, value, "max_nmaxima")?;
        self.write(py, |qtp| {
            qtp.max_nmaxima = value;
        })
    }

    @property def cos_critical_rad(&self) -> PyResult<f32> {
        self.read(py, |qtp| qtp.cos_critical_rad)
    }

    @deglitch.setter def set_cos_critical_rad(&self, value: Option<f32>) -> PyResult<()> {
        let value = readonly(py, value, "cos_critical_rad")?;
        self.write(py, |qtp| {
            qtp.cos_critical_rad = value;
        })
    }

    @property def max_line_fit_mse(&self) -> PyResult<f32> {
        self.read(py, |qtp| qtp.max_line_fit_mse)
    }

    @deglitch.setter def set_max_line_fit_mse(&self, value: Option<f32>) -> PyResult<()> {
        let value = readonly(py, value, "max_line_fit_mse")?;
        self.write(py, |qtp| {
            qtp.max_line_fit_mse = value;
        })
    }

    @property def min_white_black_diff(&self) -> PyResult<u8> {
        self.read(py, |qtp| qtp.min_white_black_diff)
    }

    @deglitch.setter def set_min_white_black_diff(&self, value: Option<u8>) -> PyResult<()> {
        let value = readonly(py, value, "min_white_black_diff")?;
        self.write(py, |qtp| {
            qtp.min_white_black_diff = value;
        })
    }

    @property def deglitch(&self) -> PyResult<bool> {
        self.read(py, |qtp| qtp.deglitch)
    }

    @deglitch.setter def set_deglitch(&self, value: Option<bool>) -> PyResult<()> {
        let value = readonly(py, value, "deglitch")?;
        self.write(py, |qtp| {
            qtp.deglitch = value;
        })
    }
});

impl QuadThresholdParams {
    fn read<R>(&self, py: Python, callback: impl FnOnce(&AprilTagQuadThreshParams) -> R) -> PyResult<R> {
        let config_py = self.config(py);
        let inner = unsafe { config_py.try_borrow(py) }?;
        inner.read(py, |config| callback(&config.qtp))
    }

    fn write(&self, py: Python, callback: impl FnOnce(&mut AprilTagQuadThreshParams) -> ()) -> PyResult<()> {
        let config_py = self.config(py);
        let inner = unsafe { config_py.try_borrow(py) }?;
        inner.write(py, |config| callback(&mut config.qtp))
    }
}

py_class!(pub class DetectorBuilder |py| {
    data builder: Arc<RwLock<AprilTagDetectorBuilder>>;

    def __new__(_cls) -> PyResult<DetectorBuilder> {
        let builder = Arc::new(RwLock::new(AprilTagDetectorBuilder::default()));
        Self::create_instance(py, builder)
    }

    @property def acceleration(&self) -> PyResult<PyObject> {
        let builder = self.builder(py).read();
        let text = match builder.gpu_mode() {
            AccelerationRequest::Disabled => return Ok(py.None()),
            AccelerationRequest::Prefer => Cow::Borrowed("prefer"),
            AccelerationRequest::Required => Cow::Borrowed("required"),
            AccelerationRequest::PreferGpu => Cow::Borrowed("prefer_gpu"),
            AccelerationRequest::RequiredGpu => Cow::Borrowed("required_gpu"),
            AccelerationRequest::PreferDeviceIdx(idx) => Cow::Owned(format!("prefer_{idx}")),
            AccelerationRequest::RequiredDeviceIdx(idx) => Cow::Owned(format!("required_{idx}")),
        };
        Ok(PyString::new(py, &text).into_object())
    }

    @acceleration.setter def set_acceleration(&self, value: Option<PyString>) -> PyResult<()> {
        let request = match value {
            None => AccelerationRequest::Disabled,
            Some(value) => match value.to_string(py)?.borrow() {
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
                        None => return Err(PyErr::new::<exc::ValueError, _>(py, format!("Unsupported value {} for DetectorBuilder.acceleration", value)))                    }
                }
            }
        };
        let builder = self.builder(py);
        py.allow_threads(|| {
            let mut builder = builder.write();
            builder.config.acceleration = request;
        });

        Ok(())
    }

    @property def config(&self) -> PyResult<DetectorConfig> {
        let builder = self.builder(py);
        DetectorConfig::create_instance(py, ConfigRef::BuilderMut(builder.clone()))
    }

    @config.setter def set_config(&self, config: Option<DetectorConfig>) -> PyResult<()> {
        match config {
            Some(config) => {
                let inner = config.config(py);
                // Check this isn't *our* config to ensure that we don't deadlock
                if let ConfigRef::BuilderMut(builder) = inner {
                    if Arc::ptr_eq(builder, self.builder(py)) {
                        // No-op
                        return Ok(());
                    }
                }
                let mut wg = self.builder(py).write();
                inner.read(py, |config| {
                    wg.config.clone_from(config);
                })
            },
            None => Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete config"))
        }
    }

    /// Add AprilTag family to detector
    def add_family(&self, family: PyObject, num_bits: Option<usize> = None) -> PyResult<PyObject> {
        // Parse family
        let family = if let Ok(family) = family.extract::<super::PyAprilTagFamily>(py) {
            family.raw_family(py)
        } else if let Ok(family_name) = family.cast_as::<PyString>(py) {
            let family_name = family_name.to_string(py)?;
            match ATFamily::for_name(&family_name) {
                Some(family) => family,
                None => return Err(PyErr::new::<exc::NameError, _>(py, format!("Unknown AprilTag family name {family_name}"))),
            }
        } else {
            return Err(PyErr::new::<exc::TypeError, _>(py, format!("Invalid AprilTag family type")));
        };

        let num_bits = num_bits.unwrap_or(2);
        let mut builder = self.builder(py).write();

        // add_family_bits could be slow, so run it on a different thread
        let res = {
            let builder_ref: &mut AprilTagDetectorBuilder = &mut builder;
            let family = family.clone();
            py.allow_threads(|| {
                builder_ref.add_family_bits(family, num_bits)
            })
        };
        
        match res {
            Ok(_) => Ok(py.None()),
            Err(AddFamilyError::TooManyCodes(num_codes)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Too many codes ({}, max 2**16) in AprilTag family {}", num_codes, family.name))),
            Err(AddFamilyError::BigHamming(hamming)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Too many hamming bits ({}, max 3) when adding AprilTag family {}", hamming, family.name))),
            Err(AddFamilyError::QuickDecodeAllocation) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Unable to allocate memory for AprilTag family {}", family.name))),
        }
    }

    /// Remove all registered AprilTag families
    def clear_families(&self) -> PyResult<PyObject> {
        let mut builder = self.builder(py).write();
        builder.clear_families();
        Ok(py.None())
    }

    def build(&self) -> PyResult<Detector> {
        let builder = self.builder(py)
            .read()
            .clone();// Ideally we'd be able to take the builder, but whatever

        match py.allow_threads(|| builder.build()) {
            Ok(detector)
                => Detector::create_instance(py, detector),
            Err(DetectorBuildError::BufferAllocationFailure)
                => Err(PyErr::new::<exc::MemoryError, _>(py, "Unable to allocate buffer(s)")),
            Err(DetectorBuildError::NoTagFamilies)
                => Err(PyErr::new::<exc::ValueError, _>(py, "No tag families were provided")),
            Err(DetectorBuildError::AccelerationNotAvailable)
                => Err(PyErr::new::<exc::EnvironmentError, _>(py, "Acceleration was required, but not available")),
            #[cfg(feature="opencl")]
            Err(DetectorBuildError::OpenCLError(e))
                => Err(PyErr::new::<exc::EnvironmentError, _>(py, format!("OpenCL error: {e}"))),
            #[cfg(feature="wgpu")]
            Err(DetectorBuildError::WGPU(e))
                => Err(PyErr::new::<exc::EnvironmentError, _>(py, format!("WGPU error: {e}"))),
            Err(DetectorBuildError::Threadpool(e))
                => Err(PyErr::new::<exc::RuntimeError, _>(py, format!("Error building threadpool: {e}"))),
            Err(e)
                => Err(PyErr::new::<exc::RuntimeError, _>(py, e.to_string())),
        }
    }
});

py_class!(pub class Detector |py| {
    @shared data detector: AprilTagDetector;

    @staticmethod def builder() -> PyResult<DetectorBuilder> {
        DetectorBuilder::create_instance(py, Arc::new(RwLock::new(AprilTagDetector::builder())))
    }
    
    def __new__(_cls, nthreads: Option<usize> = None, quad_decimate: Option<f32> = None, quad_sigma: Option<f32> = None, refine_edges: Option<bool> = None, decode_sharpening: Option<f64> = None, debug: Option<bool> = None) -> PyResult<Detector> {
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
            Ok(detector) => Self::create_instance(py, detector),
            Err(DetectorBuildError::BufferAllocationFailure) => Err(PyErr::new::<exc::MemoryError, _>(py, "Buffer allocation failure")),
            Err(DetectorBuildError::AccelerationNotAvailable) => Err(PyErr::new::<exc::RuntimeError, _>(py, "OpenCL not available")),
            Err(DetectorBuildError::Threadpool(tpbe)) => Err(PyErr::new::<exc::RuntimeError, _>(py, format!("Failed building threadpool: {tpbe}"))),
            Err(e)
                => Err(PyErr::new::<exc::RuntimeError, _>(py, e.to_string())),
        }
    }

    @property def config(&self) -> PyResult<DetectorConfig> {
        let leaked = self.detector(py).leak_immutable();
        DetectorConfig::create_instance(py, ConfigRef::DetectorRef(leaked))
    }

    def detect(&self, image: ImageY8) -> PyResult<super::PyDetections> {
        let detector_ref = match self.detector(py).try_borrow() {
            Ok(det) => det,
            Err(e) => return Err(PyErr::new::<exc::InterruptedError, _>(py, format!("Unable to borrow detector: {e}"))),
        };
        let res = {
            let detector: &AprilTagDetector = &detector_ref;
            py.allow_threads(|| {
                let res = detector.detect(&image);
                drop(image);
                res
            })
        };

        match res {
            Ok(detections) =>
                super::PyDetections::create_instance(py, Arc::new(detections)),
            Err(DetectError::ImageAlloc(..)) =>
                Err(PyErr::new::<exc::MemoryError, _>(py, format!("Unable to allocate buffers"))),
            Err(DetectError::BadSourceImageDimensions(..)) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Source image too small"))),  
            Err(e)
                => Err(PyErr::new::<exc::RuntimeError, _>(py, e.to_string())),
        }
    }
});