use std::{cell::RefCell, ops::DerefMut};

use cpython::{py_class, PyResult, UnsafePyLeaked, PyTuple, PyString, PyFloat, PythonObject};
use crate::dbg::TimeProfile as ATTimeProfile;

py_class!(pub(super) class TimeProfile |py| {
    @shared data data: ATTimeProfile;

    def total_time(&self) -> PyResult<f64> {
        let data = self.data(py)
            .borrow();
        Ok(data.total_duration().as_secs_f64())
    }

    def __iter__(&self) -> PyResult<TimeProfileIter> {
        let data = self.data(py).leak_immutable();
        TimeProfileIter::create_instance(py, RefCell::new((data, 0)))
    }

    def __str__(&self) -> PyResult<String> {
        let data = self.data(py)
            .borrow();
        Ok(format!("{data}"))
    }
});

py_class!(pub(super) class TimeProfileIter |py| {
    data data: RefCell<(UnsafePyLeaked<&'static ATTimeProfile>, usize)>;

    def __next__(&self) -> PyResult<Option<PyTuple>> {
        let (key, ts) = {
            let mut data = self.data(py).borrow_mut();
            let (tp, idx) = data.deref_mut();
            let tp = unsafe { tp.try_borrow_mut(py) }?;
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