use std::{sync::Arc, ops::Deref, cell::RefCell, borrow::{Cow, Borrow}};

use cpython::{PyResult, py_class, PyString, PyBool, py_module_initializer, PySequence, exc, PyErr, PyObject, buffer::PyBuffer, PythonObject, PyList, Python};
use parking_lot::{RwLockReadGuard, RwLock};
use crate::{
    AprilTagDetector,
    DetectorBuilder as AprilTagDetectorBuilder,
    OpenClMode,
    DetectorConfig as AprilTagDetectorConfig,
    AprilTagDetection,
    Detections as ATDetections,
    AprilTagFamily as ATFamily,
    quickdecode::AddFamilyError, 
    util::ImageY8,
    dbg::TimeProfile as ATTimeProfile, DetectError, DetectorBuildError
};

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

struct MappedRwReadGuard<'a, U, G: 'a> {
    guard: G,
    value: &'a U
}

impl<'a, U, G: 'a> MappedRwReadGuard<'a, U, G> {
    fn from_guard<T: 'a + ?Sized>(guard: G, mapper: impl FnOnce(&'a T) -> &'a U) -> Self where G: Borrow<T> {
        let raw = guard.borrow();
        let value = mapper(raw);
        Self {
            guard,
            value,
        }
    }
}
impl<'a, T, U> MappedRwReadGuard<'a, U, RwLockReadGuard<'a, T>> where T: ?Sized {
    fn lock(lock: &'a RwLock<T>, mapper: impl FnOnce(&'a T) -> &'a U) -> Self where RwLockReadGuard<'a, T>: Borrow<T> {
        let guard = lock.read();
        Self::from_guard::<T>(guard, mapper)
    }
}


enum ConfigRef {
    Owned(RwLock<AprilTagDetectorConfig>),
    BuilderMut(Arc<RwLock<AprilTagDetectorBuilder>>),
    DetectorRef(Arc<RwLock<AprilTagDetector>>),
}
impl ConfigRef {
    fn read<R>(&self, callback: impl FnOnce(&AprilTagDetectorConfig) -> R) -> R {
        match self {
            Self::Owned(rwl) => callback(&rwl.read()),
            Self::BuilderMut(rwl) => callback(&rwl.read().config),
            Self::DetectorRef(rwl) => callback(&rwl.read().params),
        }
    }

    fn write(&self, py: Python, callback: impl FnOnce(&mut AprilTagDetectorConfig) -> ()) -> PyResult<()> {
        match self {
            Self::Owned(rwl) => callback(&mut rwl.write()),
            Self::BuilderMut(rwl) => callback(&mut rwl.write().config),
            Self::DetectorRef(rwl)
                => return Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete quad_sigma"))
        }
        Ok(())
    }
}

py_class!(class DetectorConfig |py| {
    data config: ConfigRef;

    def __new__(_cls) -> PyResult<Self> {
        Self::create_instance(py, ConfigRef::Owned(RwLock::new(AprilTagDetectorConfig::default())))
    }

    @property def nthreads(&self) -> PyResult<usize> {
        self.config(py).read(|config| Ok(config.nthreads))
    }

    @nthreads.setter def set_nthreads(&self, value: Option<usize>) -> PyResult<()> {
        match value {
            Some(value) => {
                self.config(py).write(py, |config| {
                    config.nthreads = value;
                });
                Ok(())
            },
            None => Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete nthreads"))
        }
    }

    @property def quad_decimate(&self) -> PyResult<f32> {
        self.config(py).read(|config| Ok(config.quad_decimate))
    }

    @quad_decimate.setter def set_quad_decimate(&self, value: Option<f32>) -> PyResult<()> {
        match value {
            Some(value) => {
                if value < 0. {
                    return Err(PyErr::new::<exc::ValueError, _>(py, "quad_decimate should be positive"));
                }
                
                self.config(py).write(py, |config| {
                    config.quad_decimate = value;
                });
                Ok(())
            },
            None => Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete quad_decimate"))
        }
    }

    @property def quad_sigma(&self) -> PyResult<f32> {
        self.config(py).read(|config| Ok(config.quad_sigma))
    }

    @quad_sigma.setter def set_quad_sigma(&self, value: Option<f32>) -> PyResult<()> {
        match value {
            Some(value) => {
                self.config(py).write(py, |config| {
                    config.quad_sigma = value;
                });
                Ok(())
            },
            None => Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete quad_sigma"))
        }
    }

    @property def refine_edges(&self) -> PyResult<PyBool> {
        if self.config(py).read(|config| config.refine_edges) {
            Ok(py.True())
        } else {
            Ok(py.False())
        }
    }

    @refine_edges.setter def set_refine_edges(&self, value: Option<PyBool>) -> PyResult<()> {
        match value {
            Some(value) => {
                let value = value.is_true();
                self.config(py).write(py, |config| {
                    config.refine_edges = value;
                })?;
                Ok(())
            },
            None => Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete refine_edges"))
        }
    }

    @property def debug(&self) -> PyResult<PyBool> {
        let debug = self.config(py).read(|config| config.debug);
        Ok(if debug {
            py.True()
        } else {
            py.False()
        })
    }

    @debug.setter def set_debug(&self, value: Option<PyBool>) -> PyResult<()> {
        match value {
            Some(value) => {
                let value = value.is_true();
                self.config(py).write(py, |config| {
                    config.debug = value;
                })
            },
            None => Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete debug"))
        }
    }

    @property def debug_path(&self) -> PyResult<Option<PyString>> {
        self.config(py).read(|config| {
            match config.debug_path {
                Some(path) => Ok(Some(PyString::new(py, &path))),
                None => Ok(None),
            }
        })
    }
});

py_class!(class DetectorBuilder |py| {
    data builder: Arc<RwLock<AprilTagDetectorBuilder>>;

    def __new__(_cls) -> PyResult<DetectorBuilder> {
        Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete nthreads"))
    }

    @property def opencl_mode(&self) -> PyResult<PyObject> {
        let builder = self.builder(py).read();
        let text = match builder.opencl_mode() {
            OpenClMode::Disabled => return Ok(py.None()),
            OpenClMode::Prefer => Cow::Borrowed("prefer"),
            OpenClMode::PreferGpu => Cow::Borrowed("prefer_gpu"),
            OpenClMode::PreferGpu => Cow::Borrowed("prefer_gpu"),
            OpenClMode::Required => Cow::Borrowed("required"),
            OpenClMode::RequiredGpu => Cow::Borrowed("required_gpu"),
            OpenClMode::PreferDeviceIdx(idx) => Cow::Owned(format!("prefer_{idx}")),
            OpenClMode::RequiredDeviceIdx(idx) => Cow::Owned(format!("require_{idx}")),
        };
        Ok(PyString::new(py, &text).into_object())
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
                inner.read(|config| {
                    wg.config.clone_from(config);
                });
                Ok(())
            },
            None => Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete config"))
        }
    }

    def add_family(&self, family: AprilTagFamily, num_bits: Option<usize>) -> PyResult<PyObject> {
        let num_bits = num_bits.unwrap_or(2);
        let mut builder = self.builder(py).write();

        // let family_name = family.to_string(py)?;
        // let family = ATFamily::for_name(&family_name)
        //     .ok_or_else(|| PyErr::new::<exc::ValueError, _>(py, format!("Unknown AprilTag family: {}", family_name)))?;

        let family = family.family(py)
            .borrow()
            .clone();
        // add_family_bits could be slow, so run it on a different thread
        let res = {
            let builder_ref: &mut AprilTagDetectorBuilder = &mut builder;
            let family = family.clone();
            py.allow_threads(move || {
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

    def build(&self) -> PyResult<Detector> {
        let mut builder = self.builder(py)
            .read()
            .clone();// Ideally we'd be able to take the builder, but whatever

        match py.allow_threads(|| builder.build()) {
            Ok(detector)
                => Detector::create_instance(py, RwLock::new(detector)),
            Err(DetectorBuildError::BufferAllocationFailure)
                => Err(PyErr::new::<exc::MemoryError, _>(py, "Unable to allocate buffer(s)")),
            Err(DetectorBuildError::OpenCLNotAvailable)
                => Err(PyErr::new::<exc::EnvironmentError, _>(py, "OpenCL was required, but not available")),
            Err(DetectorBuildError::Threadpool(e))
                => Err(PyErr::new::<exc::RuntimeError, _>(py, format!("Error building threadpool: {e}"))),
        }
    }
});

py_class!(class Detector |py| {
    data detector: RwLock<AprilTagDetector>;

    @staticmethod def builder() -> PyResult<DetectorBuilder> {
        DetectorBuilder::create_instance(py, Arc::new(RwLock::new(AprilTagDetector::builder())))
    }
    def __new__(_cls, nthreads: Option<usize>, quad_decimate: Option<f32>, quad_sigma: Option<f32>, refine_edges: Option<bool>, decode_sharpening: Option<f64>, debug: Option<bool>, camera_params: Option<PySequence>) -> PyResult<Detector> {
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

        let detector = match builder.build() {
            Ok(det) => det,
            Err(e) => todo!(),
        };
        Self::create_instance(py, RwLock::new(detector))
    }

    def detect(&self, image: PyObject) -> PyResult<Detections> {
        let img_buf = PyBuffer::get(py, &image)?;
        if img_buf.dimensions() != 2 {
            return Err(PyErr::new::<exc::ValueError, _>(py, format!("Expected 2d numpy array")));
        }
        let shape = img_buf.shape();
        // Copy image from PyObject
        let img = {
            let mut img = ImageY8::zeroed_packed(shape[1], shape[0]);
            let width = img.width();
            let v = img_buf.to_vec::<u8>(py)?;
            for ((x, y), dst) in img.enumerate_pixels_mut() {
                dst.0 = [v[y * width + x]];
            }
            img
        };
        let detector = self.detector(py);
        let res = py.allow_threads(|| {
            let detector = detector.read();
            let res = detector.detect(&img);
            drop(img);
            res
        });

        match res {
            Ok(detections) =>
                Detections::create_instance(py, Arc::new(detections)),
            Err(DetectError::AllocError) =>
                Err(PyErr::new::<exc::MemoryError, _>(py, format!("Unable to allocate buffers"))),
            Err(DetectError::ImageTooSmall) =>
                Err(PyErr::new::<exc::ValueError, _>(py, format!("Source image too small"))),  
        }
    }
});


py_module_initializer!(apriltag_rs, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    // m.add(py, "sum_as_string", py_fn!(py, sum_as_string_py(a: i64, b:i64)))?;
    m.add_class::<TimeProfile>(py)?;
    
    m.add_class::<AprilTagFamily>(py)?;

    m.add_class::<DetectorConfig>(py)?;
    m.add_class::<DetectorBuilder>(py)?;

    m.add_class::<Detections>(py)?;
    m.add_class::<Detection>(py)?;

    m.add_class::<Detector>(py)?;

    Ok(())
});