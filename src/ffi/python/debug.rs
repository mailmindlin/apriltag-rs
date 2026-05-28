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

	fn as_list(&self) -> PyResult<Vec<TimeProfileEntry>> {
		(0..self.entries().len())
			.map(|idx| py_entry(self, idx))
			.collect()
	}

	fn __getitem__(&self, key: usize) -> PyResult<TimeProfileEntry> {
		py_entry(self, key)
	}

	fn __str__(&self) -> String {
		format!("{self:#}")
	}

	fn __repr__(&self) -> String {
		format!("{self:#?}")
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
		
		match py_entry(&tp, *index) {
			Ok(entry) => {
				*index += 1;
				Ok(Some(entry))
			},
			Err(_) => Ok(None)
		}
	}
}

fn py_entry(tp: &TimeProfile, key: usize) -> PyResult<TimeProfileEntry> {
	let entry = match tp.entries().get(key) {
		Some(entry) => entry,
		None => return Err(PyErr::new::<PyIndexError, _>(format!("Index {key} out of range"))),
	};

	let ts = entry.timestamp();
	let prev_ts = if key == 0 {
		tp.start()
	} else {
		*tp.entries()[key - 1].timestamp()
	};
	Ok(TimeProfileEntry {
		id: key,
		name: entry.name().into(),
		wall_cum: ts.duration_since(tp.start()).as_secs_f64(),
		wall_part: ts.duration_since(prev_ts).as_secs_f64(),
		proc_part: None,
		proc_cum: None,
	})
}

#[pyclass(frozen, get_all, module="apriltag_rs")]
#[derive(Clone, Debug)]
pub(super) struct TimeProfileEntry {
	/// Entry index (starts at 0)
	id: usize,
	/// Entry name
	name: String,
	/// Partial wall time
	wall_part: f64,
	/// Cumulative wall time
	wall_cum: f64,
	/// Partial processtime
	proc_part: Option<f64>,
	/// Cumulative process time
	proc_cum: Option<f64>,
}

impl std::fmt::Display for TimeProfileEntry {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{:2} {} {:12.6} ms {:12.6} ms {:3.0}%",
			self.id,
			self.name,
			self.wall_part * 1000.,
			self.wall_cum * 1000.,
			0,
		)?;
		if let Some(proc_part) = self.proc_part {
			write!(f, " {:12.6} ms", proc_part * 1000.)?;
		}
		
		if let Some(proc_cum) = self.proc_cum {
			write!(f, " {:12.6} ms", proc_cum * 1000.)?;
		}
		Ok(())
	}
}

#[pymethods]
impl TimeProfileEntry {
	fn __str__(&self) -> String {
		format!("{self}")
	}
	fn __repr__(&self) -> String {
		format!("{self:?}")
	}
}