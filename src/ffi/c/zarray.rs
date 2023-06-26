use std::{mem, marker::PhantomData};

use libc::{size_t, c_int, c_char};

#[repr(C)]
pub struct ZArray<T> {
    /// size of each element
    pub el_sz: size_t,

    /// how many elements?
    pub size: c_int,
    /// we've allocated storage for how many elements?
    pub alloc: c_int,
    pub data: *mut c_char,
    elem: PhantomData<T>,
}

impl<T: Sized> From<Vec<T>> for ZArray<T> {
    fn from(value: Vec<T>) -> Self {
        let el_sz = mem::size_of::<T>();
        let (data, size, alloc) = value.into_raw_parts();
        Self {
            el_sz,
            size: size as _,
            alloc: alloc as _,
            data: data as _,
            elem: PhantomData,
        }
    }
}

impl<T> From<ZArray<T>> for Vec<T> {
    fn from(value: ZArray<T>) -> Self {
        assert_eq!(value.el_sz as usize, std::mem::size_of::<T>());
        unsafe {
            Vec::from_raw_parts(value.data as *mut T, value.size as _, value.alloc as _)
        }
    }
}

impl<T> Drop for ZArray<T> {
    fn drop(&mut self) {
        assert_eq!(self.el_sz, std::mem::size_of::<T>());
        let vec = unsafe {
            Vec::from_raw_parts(self.data as *mut T, self.size as _, self.alloc as _)
        };
        drop(vec);
    }
}
