use std::{cell::RefCell, ops::{DerefMut, Deref}, time::Duration};

use cpython::{py_class, PyResult, UnsafePyLeaked, PyTuple, PyString, PyFloat, PythonObject, PyList, exc, PyErr, Python, PyObject};
use crate::dbg::TimeProfile as ATTimeProfile;

fn py_duration(py: Python, value: Duration) -> PyResult<PyObject> {
    let value_secs = value.as_secs_f64();
    Ok(PyFloat::new(py, value_secs).into_object())
}

fn py_entry(py: Python, tp: &ATTimeProfile, idx: usize) -> PyResult<PyTuple> {
    let entry = tp.entries().get(idx)
        .ok_or_else(|| PyErr::new::<exc::IndexError, _>(py, format!("Index {idx} out of range")))?;

    let start = *tp.start();
    let res = PyTuple::new(py, &[
        PyString::new(py, entry.name()).into_object(),
        py_duration(py, entry.timestamp().duration_since(start))?,
    ]);
    Ok(res)
}

py_class!(pub class TimeProfile |py| {
    @shared data data: ATTimeProfile;
    
    def total_duration(&self) -> PyResult<f64> {
        let data = self.data(py)
            .borrow();
        Ok(data.total_duration().as_secs_f64())
    }

    def __iter__(&self) -> PyResult<TimeProfileIter> {
        let data = self.data(py).leak_immutable();
        TimeProfileIter::create_instance(py, data, RefCell::new(0))
    }

    def __len__(&self) -> PyResult<usize> {
        let data = self.data(py).borrow();
        Ok(data.entries().len())
    }

    def as_list(&self) -> PyResult<PyList> {
        let data = self.data(py).borrow();
        let mut res_vec = Vec::with_capacity(data.entries().len());
        for i in 0..data.entries().len() {
            res_vec.push(py_entry(py, &data, i)?.into_object());
        }
        Ok(PyList::new(py, &res_vec))
    }

    def __getitem__(&self, key: usize) -> PyResult<PyTuple> {
        let data = self.data(py).borrow();
        py_entry(py, &data, key)
    }

    def __str__(&self) -> PyResult<String> {
        let data = self.data(py)
            .borrow();
        Ok(format!("{data}"))
    }
});

py_class!(pub class TimeProfileIter |py| {
    data tp: UnsafePyLeaked<&'static ATTimeProfile>;
    data index: RefCell<usize>;

    def __len__(&self) -> PyResult<usize> {
        let idx = *self.index(py).borrow().deref();
        let tp = self.tp(py);
        let tp = unsafe { tp.try_borrow(py) }?;
        Ok(tp.entries().len() - idx)
    }

    def __next__(&self) -> PyResult<Option<PyTuple>> {
        let (key, ts) = {
            let mut idx_ref = self.index(py).borrow_mut();
            let idx = idx_ref.deref_mut();

            let tp = self.tp(py);
            let tp = unsafe { tp.try_borrow(py) }?;
            
            let entry = match tp.entries().get(*idx) {
                Some(entry) => entry,
                None => return Ok(None),
            };
            *idx += 1;
            let key = PyString::new(py, entry.name());
            let ts = entry.timestamp().duration_since(*tp.start());
            (key, ts)
        };
        let value = PyFloat::new(py, ts.as_secs_f64());
        let res = PyTuple::new(py, &[key.into_object(), value.into_object()]);
        Ok(Some(res))
    }
});