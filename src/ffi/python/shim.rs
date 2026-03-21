use pyo3::{Borrowed, Bound, FromPyObject, IntoPyObject, PyAny, PyErr, PyResult, Python, buffer::PyBuffer, exceptions::{PyNotImplementedError, PyTypeError, PyValueError}, types::{PyAnyMethods, PyFloat, PyList, PyString}};

use crate::util::{math::{mat::Mat33, Vec3}, ImageY8};


pub(super) fn readonly<T>(_py: Python, value: Option<T>, name: &'static str) -> PyResult<T> {
    match value {
        Some(value) => Ok(value),
        None => Err(PyErr::new::<PyNotImplementedError, _>(format!("Cannot delete {name}"))),
    }
}

impl<'s, 'py> FromPyObject<'s, 'py> for ImageY8 {
	type Error = PyErr;
	
    fn extract(obj: Borrowed<'s, 'py, PyAny>) -> PyResult<Self> {
        //TODO: support other parameter types
        let img_buf = PyBuffer::get(&obj)?;
        if img_buf.dimensions() != 2 {
            return Err(PyErr::new::<PyValueError, _>("Expected 2d array"));
        }
        if img_buf.item_size() != std::mem::size_of::<u8>() {
            return Err(PyErr::new::<PyTypeError, _>("Expected array of bytes"));
        }

        let shape = img_buf.shape();
        // Copy image from PyObject
        let mut img = ImageY8::zeroed_packed(shape[1], shape[0]);
        img_buf.copy_to_slice(obj.py(), &mut img.data)?;
        Ok(img)
    }

	fn type_input() -> pyo3::inspect::types::TypeInfo {
		use pyo3::inspect::types::{TypeInfo, ModuleName};
		let t_int = TypeInfo::Class {
			module: ModuleName::Builtin,
			name: "int".into(),
			type_vars: vec![],
		};

		let mod_np = ModuleName::Module("numpy".into());

		// np.ndarray[tuple[int, int], np.dtype[np.uint8]]
		TypeInfo::Class {
			module: mod_np.clone(),
			name: "ndarray".into(),
			type_vars: vec![
				TypeInfo::Tuple(Some(vec![
					t_int.clone(),
					t_int,
				])),
				TypeInfo::Class {
					module: mod_np.clone(),
					name: "dtype".into(),
					type_vars: vec![
						TypeInfo::Class {
							module: mod_np,
							name: "uint8".into(),
							type_vars: vec![],
						}
					]
				}
			]
		}
	}
}

impl<'py> IntoPyObject<'py> for ImageY8 {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fn np_image<'py>(py: Python<'py>, img: &ImageY8) -> PyResult<Bound<'py, PyAny>> {
            let np = py.import("numpy")?;
            let np_empty = np.get_item("empty")?;
            let np_uint8 = np.get_item("uint8")?;
            let result = np_empty.call((
                (img.width(), img.height()), // Dims
                np_uint8, // dtype
                PyString::new(py, "C") // Order
            ), None)?;

            {
                let res_buf = PyBuffer::get(&result)?;
                assert_eq!(res_buf.dimensions(), 2);
                assert_eq!(res_buf.shape(), &[img.width(), img.height()]);
                assert_eq!(res_buf.item_size(), std::mem::size_of::<u8>());

                if img.width() == img.stride() {
                    res_buf.copy_from_slice(py, &img.data)?;
                } else if res_buf.strides()[1] == (std::mem::size_of::<u8>() as isize) {
                    py.detach(|| {
                        for (idx, row) in img.rows() {
                            let src_row = row.as_slice();

                            let dst_row = unsafe {
                                let dr_start = res_buf.get_ptr(&[idx, 0]);
                                debug_assert_eq!(dr_start.byte_add((img.width() - 1) * std::mem::size_of::<u8>()), res_buf.get_ptr(&[idx, img.width() - 1]));
                                std::slice::from_raw_parts_mut(dr_start as *mut u8, img.width())
                            };
                            dst_row.copy_from_slice(src_row);
                        }
                    });
                } else {
                    panic!("TODO: negative strides: {:?}", res_buf.strides())
                }
            }

            Ok(result)
        }
        match np_image(py, &self) {
            Ok(obj) => Ok(obj),
            Err(_) => todo!("Python ImageY8 without numpy"),
        }
    }

    fn type_output() -> pyo3::inspect::types::TypeInfo {
        Self::type_input()
    }
}

impl<'py> IntoPyObject<'py> for Mat33 {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // List of lists
        let elem = |idx: (usize, usize)| PyFloat::new(py, self[idx]);
        let arr = PyList::new(py, &[
            PyList::new(py, &[elem((0, 0)), elem((0, 1)), elem((0, 2))])?.into_any(),
            PyList::new(py, &[elem((1, 0)), elem((1, 1)), elem((1, 2))])?.into_any(),
            PyList::new(py, &[elem((2, 0)), elem((2, 1)), elem((2, 2))])?.into_any(),
        ])?;

        // Try converting it to a numpy matrix object
        fn np_mat33<'py>(py: Python<'py>, raw: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyAny>> {
            let np = py.import("numpy")?;
            let np_asarray = np.get_item("asarray")?;
            np_asarray.call((raw, ), None)
        }

        match np_mat33(py, &arr) {
            Ok(res) => Ok(res),
            Err(_) => Ok(arr.into_any()),
        }
    }
}

impl<'py> IntoPyObject<'py> for Vec3 {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = PyList::new(py, &[
            PyFloat::new(py, self.0),
            PyFloat::new(py, self.1),
            PyFloat::new(py, self.2),
        ])?;

        // Try converting it to a numpy matrix object
        fn np_vec3<'py>(py: Python<'py>, raw: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyAny>> {
            let np = py.import("numpy")?;
            let np_asarray = np.get_item("asarray")?;
            let np_f64 = np.get_item("float64")?;
            np_asarray.call((raw, np_f64), None)
        }

        match np_vec3(py, &arr) {
            Ok(res) => Ok(res),
            Err(_) => Ok(arr.into_any()),
        }
    }
}