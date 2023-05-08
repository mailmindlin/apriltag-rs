use std::mem;

use libc::{size_t, c_int, c_char, c_uint, c_double};

use crate::util::math::mat::Mat;

use super::FFIConvertError;

#[repr(C)]
pub struct zarray {
    /// size of each element
    el_sz: size_t,

    /// how many elements?
    size: c_int,
    /// we've allocated storage for how many elements?
    alloc: c_int,
    data: *mut c_char,
}

impl<T: Sized> From<Vec<T>> for zarray {
    fn from(value: Vec<T>) -> Self {
        let el_sz = mem::size_of::<T>();
        let (data, size, alloc) = value.into_raw_parts();
        Self {
            el_sz,
            size: size as _,
            alloc: alloc as _,
            data: data as _,
        }
    }
}

impl<T> From<zarray> for Vec<T> {
    fn from(value: zarray) -> Self {
        assert_eq!(value.el_sz as usize, std::mem::size_of::<T>());
        unsafe {
            Vec::from_raw_parts(value.data as *mut T, value.size as _, value.alloc as _)
        }
    }
}

#[repr(C)]
pub struct matd_t {
    nrows: c_uint,
    ncols: c_uint,
    data: [c_double],
}

impl matd_t {
    fn new(nrows: c_uint, ncols: c_uint, data: Box<[c_double]>) -> Box<matd_t> {
        //See: https://stackoverflow.com/questions/67171086/how-can-a-dynamically-sized-object-be-constructed-on-the-heap
        use std::alloc::Layout;
        let base_layout = {
            #[repr(C)]
            struct matd_like<T: ?Sized> {
                nrows: c_uint,
                ncols: c_uint,
                data: T
            }

            const MATD_EMPTY: matd_like<[c_double; 0]> = matd_like {
                nrows: 0,
                ncols: 0,
                data: []
            };
            Layout::for_value(&MATD_EMPTY)
        };

        let (layout, arr_offset) = base_layout.extend(Layout::array::<c_double>(data.len()).unwrap()).unwrap();
        todo!()
/*

        let data_size = mem::size_of_val(&*data);
        let (data_ptr, metadata): (*const (), <T as Pointee>::Metadata) =
            (&*data as *const T).to_raw_parts();

        // UNSOUNDLY assume that the metadata must surely be the same, and 
        // reinterpret so the types match.
        let coerced_metadata: <IdAndData<T> as Pointee>::Metadata = unsafe {
            *(&metadata as *const _ as *const <IdAndData<T> as Pointee>::Metadata)
        };

        // Figure out what we need to allocate.
        // Safety: Layout::for_value_raw doesn't say the pointer per se needs to be
        // valid, just that its metadata does.
        let result_layout = unsafe {
            Layout::for_value_raw::<IdAndData<T>>(std::ptr::from_raw_parts_mut(
                &mut () as *mut (), // Not valid, but not used.
                coerced_metadata,
            ))
        };

        // Actually allocate the memory for our future Box,
        // and attach the metadata.
        let result_mem = unsafe { alloc::alloc(result_layout) }; 
        if result_mem.is_null() {
            alloc::handle_alloc_error(result_layout);
        }
        let result_ptr: *mut IdAndData<T> = std::ptr::from_raw_parts_mut(
            result_mem as *mut (),
            coerced_metadata,
        );

        // Copy each field into the new allocation.
        unsafe {
            std::ptr::write(addr_of_mut!((*result_ptr).id), id);
            std::ptr::copy_nonoverlapping(
                data_ptr as *const u8,
                addr_of_mut!((*result_ptr).data) as *mut u8,
                data_size,
            );
        }

        // Free the old `data` box, without running drop on its contents which we
        // just copied. (Is there a better way to do this, maybe with ManuallyDrop?)
        let data_layout = Layout::for_value(&*data);
        unsafe {
            alloc::dealloc(Box::into_raw(data) as *mut u8, data_layout);
        };

        // result_ptr is now initialized so we can let Box manage it.
        unsafe { Box::from_raw(result_ptr) }*/
    }
    pub(super) fn convert(src: &Mat) -> *mut matd_t {
        todo!()
    }
}

impl TryFrom<*const matd_t> for Mat {
    type Error = FFIConvertError;

    fn try_from(value: *const matd_t) -> Result<Self, Self::Error> {
        todo!()
    }
}