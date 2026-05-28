use pyo3::{buffer::PyBuffer, exceptions::{PyNotImplementedError, PyTypeError, PyValueError}, types::{PyAnyMethods, PyFloat, PyList, PyString}, Bound, FromPyObject, IntoPy, Py, PyAny, PyErr, PyResult, Python, ToPyObject};

use crate::util::{math::{mat::Mat33, Vec3}, ImageY8};


pub(super) fn readonly<T>(_py: Python, value: Option<T>, name: &'static str) -> PyResult<T> {
    match value {
        Some(value) => Ok(value),
        None => Err(PyErr::new::<PyNotImplementedError, _>(format!("Cannot delete {name}"))),
    }
}

impl<'s> FromPyObject<'s> for ImageY8 {
    fn extract_bound(obj: &Bound<'s, PyAny>) -> PyResult<Self> {
        //TODO: support other parameter types
        let img_buf = PyBuffer::get_bound(obj)?;
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
        // let v = img_buf.to_vec::<u8>(py)?;
        // for ((x, y), dst) in img.enumerate_pixels_mut() {
        //     dst.0 = [v[y * width + x]];
        // }
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

impl ToPyObject for ImageY8 {
    fn to_object(&self, py: Python<'_>) -> Py<PyAny> {
        fn np_image(py: Python, img: &ImageY8) -> PyResult<Py<PyAny>> {
            let np = py.import_bound("numpy")?;
            let np_empty = np.get_item("empty")?;
            let np_uint8 = np.get_item("uint8")?;
            let result = np_empty.call((
                (img.width(), img.height()), // Dims
                np_uint8, // dtype
                PyString::new_bound(py, "C") // Order
            ), None)?;
    
            {
                let res_buf = PyBuffer::get_bound(&result)?;
                assert_eq!(res_buf.dimensions(), 2);
                assert_eq!(res_buf.shape(), &[img.width(), img.height()]);
                assert_eq!(res_buf.item_size(), std::mem::size_of::<u8>());
                
                if img.width() == img.stride() {
                    res_buf.copy_from_slice(py, &img.data)?;
                } else if res_buf.strides()[1] == (std::mem::size_of::<u8>() as _) {
                    py.allow_threads(|| {
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
    
            Ok(result.to_object(py))
        }
        if let Ok(obj) = np_image(py, &self) {
            obj
        } else {
            todo!("Python ImageY8 without numpy");
        }
    }
}

impl IntoPy<Py<PyAny>> for ImageY8 {
	fn into_py(self, py: Python<'_>) -> Py<PyAny> {
		self.to_object(py)
	}

	fn type_output() -> pyo3::inspect::types::TypeInfo {
		Self::type_input()
	}
}

impl ToPyObject for Mat33 {
    fn to_object(&self, py: Python) -> Py<PyAny> {
        // List of lists
        let elem = |idx| PyFloat::new_bound(py, self[idx]).to_object(py);
        let arr = PyList::new_bound(py, &[
            PyList::new_bound(py, &[elem((0, 0)), elem((0, 1)), elem((0, 2))]).to_object(py),
            PyList::new_bound(py, &[elem((1, 0)), elem((1, 1)), elem((1, 2))]).to_object(py),
            PyList::new_bound(py, &[elem((2, 0)), elem((2, 1)), elem((2, 2))]).to_object(py),
        ]);

        // Try converting it to a numpy matrix object
        fn np_mat33(py: Python, raw: &Bound<PyList>) -> PyResult<Py<PyAny>> {
            let np = py.import_bound("numpy")?;
            let np_matrix = np.get_item("asarray")?;
            let res = np_matrix.call( (raw, ), None)?;
			Ok(res.to_object(py))
        }

        if let Ok(res) = np_mat33(py, &arr) {
            res
        } else {
            arr.to_object(py)
        }
    }
}

impl IntoPy<Py<PyAny>> for Mat33 {
	fn into_py(self, py: Python<'_>) -> Py<PyAny> {
		self.to_object(py)
	}
}

impl ToPyObject for Vec3 {
    fn to_object(&self, py: Python) -> Py<PyAny> {
        // List of lists
        let arr = PyList::new_bound(py, &[
            PyFloat::new_bound(py, self.0).to_object(py),
            PyFloat::new_bound(py, self.1).to_object(py),
            PyFloat::new_bound(py, self.2).to_object(py),
        ]);

        // Try converting it to a numpy matrix object
        fn np_vec3(py: Python, raw: &Bound<PyList>) -> PyResult<Py<PyAny>> {
            let np = py.import_bound("numpy")?;
            let np_matrix = np.get_item("asarray")?;
            let np_f64 = np.get_item("float64")?;
            let res = np_matrix.call((raw, np_f64), None)?;
			Ok(res.to_object(py))
        }

        if let Ok(res) = np_vec3(py, &arr) {
            res
        } else {
            arr.to_object(py)
        }
    }
}

impl IntoPy<Py<PyAny>> for Vec3 {
	fn into_py(self, py: Python<'_>) -> Py<PyAny> {
		self.to_object(py)
	}
}