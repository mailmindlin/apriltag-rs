use std::{sync::Arc, borrow::Cow};

use pyo3::{exceptions::{PyNotImplementedError, PyValueError}, pyclass, pymethods, Bound, PyErr, PyResult};

use crate::{AprilTagFamily as ATFamily, util::ImageY8};

#[pyclass(module="apriltag_rs")]
pub(super) struct AprilTagFamily {
	pub(super) family: Arc<ATFamily>,
}

#[pymethods]
impl AprilTagFamily {
	#[staticmethod]
    fn names() -> PyResult<Vec<String>> {
        Ok(ATFamily::names().into_iter().map(|s| String::from(s)).collect())
    }

    /// Create AprilTag family for name
	#[staticmethod]
    pub(super) fn for_name(name: &str) -> PyResult<Self> {
        match ATFamily::for_name(&name) {
            Some(family) => Ok(Self { family }),
            None => Err(PyErr::new::<PyValueError, _>(format!("Unknown AprilTag family '{name}'"))),
        }
    }

    #[getter]
	fn width_at_border(&self) -> PyResult<u32> {
        Ok(self.family.width_at_border)
    }

    #[getter]
	fn total_width(&self) -> PyResult<u32> {
        Ok(self.family.total_width)
    }

    #[getter]
	fn reversed_border(&self) -> bool {
        self.family.reversed_border
    }

    #[getter]
	fn min_hamming(&self) -> u32 {
        self.family.min_hamming
    }

    /// Human-readable name for family
    #[getter]
	fn name(&self) -> String {
		self.family.name.clone().into_owned()
    }

    /// The bit locations
    #[getter]
	fn bits(&self) -> PyResult<Vec<(u32, u32)>> {
        Ok(self.family.bits.clone())
    }

    #[getter]
	fn codes(&self) -> PyResult<Vec<u64>> {
        Ok(self.family.codes.clone())
    }

	#[setter]
    fn set_codes(&mut self, value: Option<Vec<u64>>) -> PyResult<()> {
        let value = match value {
            Some(value) => value,
            None => return Err(PyErr::new::<PyNotImplementedError, _>("Cannot delete AprilTagFamily.codes")),
        };

        let family = &mut self.family;

        // Create mutated family
        let mut new_family = family.as_ref().clone();
        if !new_family.name.ends_with(" (modified)") {
            new_family.name = Cow::Owned(String::from(new_family.name) + " (modified)");
        }
        new_family.codes = value;

        *family = Arc::new(new_family);
        Ok(())
    }

    fn to_image(&self, idx: usize) -> PyResult<ImageY8> {
        let img = self.family.to_image(idx);

        Ok(img)
    }

    fn __str__(&self) -> String {
        format!("AprilTagFamily {{ name = '{}' .. }}", self.family.name)
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

impl AprilTagFamily {
    pub(super) fn raw_family(self_: &Bound<Self>) -> PyResult<Arc<ATFamily>> {
        Ok(self_.try_borrow()?.family.clone())
    }
}