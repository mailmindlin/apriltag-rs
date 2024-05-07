use std::ops::DerefMut;

use pyo3::{exceptions::PyIndexError, pyclass, pymethods,  Bound, Py, PyErr, PyResult};

pub(super) use crate::dbg::TimeProfile;

#[pymethods]
impl TimeProfile {
	#[getter]
	fn get_total_duration(&self) -> f64 {
		self.total_duration().as_secs_f64()
	}

	fn __iter__(slf: Bound<Self>) -> PyResult<TimeProfileIter> {
		Ok(TimeProfileIter {
			tp: slf.unbind(),
			index: 0,
		})
	}

	fn __len__(&self) -> usize {
		self.entries().len()
	}

	fn as_list(&self) -> Vec<TimeProfileEntry> {
		self.entries()
			.iter()
			.map(|entry| py_entry(self, entry))
			.collect()
	}

	fn __getitem__(&self, key: usize) -> PyResult<TimeProfileEntry> {
		match self.entries().get(key) {
			Some(item) => Ok(py_entry(self, item)),
			None => Err(PyErr::new::<PyIndexError, _>(format!("Index {key} out of range"))),
		}
	}

	fn __str__(&self) -> String {
		format!("{self}")
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}
}

#[pyclass(module="apriltag_rs")]
pub(super) struct TimeProfileIter {
	tp: Py<TimeProfile>,
	index: usize,
}

#[pymethods]
impl TimeProfileIter {
	fn __iter__(_self: Bound<Self>) -> Bound<Self> {
		_self
	}

	fn __len__(self_: Bound<Self>) -> PyResult<usize> {
		let slf = self_.try_borrow()?;
		let idx = slf.index;
		let tp = slf.tp.try_borrow(self_.py())?;
		Ok(tp.entries().len() - idx)
	}

	fn __next__(self_: Bound<Self>) -> PyResult<Option<TimeProfileEntry>> {
		let mut me = self_.try_borrow_mut()?;
		let TimeProfileIter { tp, index } = me.deref_mut();
		let tp = tp.try_borrow(self_.py())?;
		
		Ok(match tp.entries().get(*index) {
			Some(entry) => {
				*index += 1;
				Some(py_entry(&tp, entry))
			},
			None => None,
		})
	}
}

fn py_entry(tp: &TimeProfile, entry: &crate::dbg::TimeProfileEntry) -> TimeProfileEntry {
	TimeProfileEntry {
		name: entry.name().into(),
		duration: entry.timestamp().duration_since(tp.start()).as_secs_f64(),
	}
}

#[pyclass(frozen, get_all, module="apriltag_rs")]
pub(super) struct TimeProfileEntry {
	name: String,
	duration: f64,
}

#[pymethods]
impl TimeProfileEntry {

}