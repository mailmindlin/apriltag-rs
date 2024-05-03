use std::{cell::RefCell, sync::Arc, borrow::Cow};

use cpython::{py_class, PyString, PyResult, PyObject, PythonObject, exc, PyErr, Python};
use crate::{AprilTagFamily as ATFamily, util::ImageY8};

py_class!(pub class AprilTagFamily |py| {
    data family: RefCell<Arc<ATFamily>>;

    @staticmethod def names() -> PyResult<Vec<String>> {
        Ok(ATFamily::names().into_iter().map(|s| String::from(s)).collect())
    }

    /// Create AprilTag family for name
    @staticmethod def for_name(name: PyString) -> PyResult<PyObject> {
        let name = name.to_string(py)?;
        let family = match ATFamily::for_name(&name) {
            Some(family) => family,
            None => return Err(PyErr::new::<exc::ValueError, _>(py, format!("Unknown AprilTag family '{name}'"))),
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

    /// Human-readable name for family
    @property def name(&self) -> PyResult<PyString> {
        let family = self.family(py).borrow();
        Ok(PyString::new(py, &family.name))
    }

    /// The bit locations
    @property def bits(&self) -> PyResult<Vec<(u32, u32)>> {
        let family = self.family(py).borrow();
        Ok(family.bits.clone())
    }

    @property def codes(&self) -> PyResult<Vec<u64>> {
        let family = self.family(py).borrow();
        Ok(family.codes.clone())
    }

    @codes.setter def set_codes(&self, value: Option<Vec<u64>>) -> PyResult<()> {
        let value = match value {
            Some(value) => value,
            None => return Err(PyErr::new::<exc::NotImplementedError, _>(py, "Cannot delete AprilTagFamily.codes")),
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

    def to_image(&self, idx: usize) -> PyResult<ImageY8> {
        let family = self.family(py).borrow();
        let img = family.to_image(idx);

        Ok(img)
    }

    def __str__(&self) -> PyResult<String> {
        Ok(format!("AprilTagFamily {{ name = '{}' .. }}", self.family(py).borrow().name))
    }

    def __repr__(&self) -> PyResult<String> {
        self.__str__(py)
    }
});

impl AprilTagFamily {
    pub(super) fn raw_family(&self, py: Python) -> Arc<ATFamily> {
        self.family(py).borrow().clone()
    }
}