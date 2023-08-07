use cpython::{Python, exc, PyErr, PyResult, PyObject, buffer::PyBuffer, FromPyObject, ToPyObject, ObjectProtocol, PyString};

use crate::util::ImageY8;


pub(super) fn readonly<T>(py: Python, value: Option<T>, name: &'static str) -> PyResult<T> {
    match value {
        Some(value) => Ok(value),
        None => Err(PyErr::new::<exc::NotImplementedError, _>(py, format!("Cannot delete {name}"))),
    }
}

impl<'s> FromPyObject<'s> for ImageY8 {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self> {
        //TODO: support other parameter types
        let img_buf = PyBuffer::get(py, &obj)?;
        if img_buf.dimensions() != 2 {
            return Err(PyErr::new::<exc::ValueError, _>(py, "Expected 2d array"));
        }
        if img_buf.item_size() != std::mem::size_of::<u8>() {
            return Err(PyErr::new::<exc::TypeError, _>(py, "Expected array of bytes"));
        }

        let shape = img_buf.shape();
        // Copy image from PyObject
        let mut img = ImageY8::zeroed_packed(shape[1], shape[0]);
        img_buf.copy_to_slice(py, &mut img.data)?;
        // let v = img_buf.to_vec::<u8>(py)?;
        // for ((x, y), dst) in img.enumerate_pixels_mut() {
        //     dst.0 = [v[y * width + x]];
        // }
        Ok(img)
    }
}

impl ToPyObject for ImageY8 {
    type ObjectType = PyObject;

    fn to_py_object(&self, py: Python) -> Self::ObjectType {
        fn np_image(py: Python, img: &ImageY8) -> PyResult<PyObject> {
            let np = py.import("numpy")?;
            let np_empty = np.get(py, "empty")?;
            let np_uint8 = np.get(py, "uint8")?;
            let result = np_empty.call(py, (
                (img.width(), img.height()), // Dims
                np_uint8, // dtype
                PyString::new(py, "C") // Order
            ), None)?;
    
            {
                let res_buf = PyBuffer::get(py, &result)?;
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
    
            Ok(result)
        }
        if let Ok(obj) = np_image(py, &self) {
            obj
        } else {
            todo!("Python ImageY8 without numpy");
        }
    }
}