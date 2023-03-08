use std::ops::Deref;

use libc::{c_uint};

use crate::util::Image;

use super::{drop_boxed_mut, drop_array_mut};

#[repr(C)]
pub struct image_u8_t {
    width: i32,
    height: i32,
    stride: i32,
    buf: *mut u8,
}

impl Drop for image_u8_t {
    fn drop(&mut self) {
        let len = (self.height as usize) * (self.stride as usize);
        drop_array_mut(&mut self.buf, len);
    }
}

impl From<Image<u8>> for image_u8_t {
    fn from(value: Image<u8>) -> Self {
        let (buf, len, cap) = value.buf.into_vec().into_raw_parts();
        assert_eq!(len, cap);
        Self {
            width: value.width as i32,
            height: value.height as i32,
            stride: value.stride as i32,
            buf,
        }
    }
}

impl image_u8_t {
    pub(super) fn pretend_ref<'a>(&'a self) -> FakeImageGuard<'a> {
        todo!()
    }
}

pub(super) struct FakeImageGuard<'a> {
    raw: &'a image_u8_t,
    pretend: Image<u8>,
}

impl<'a> Deref for FakeImageGuard<'a> {
    type Target = Image<u8>;

    fn deref(&self) -> &Self::Target {
        &self.pretend
    }
}

#[no_mangle]
pub unsafe extern "C" fn image_u8_create(width: c_uint, height: c_uint) -> *mut image_u8_t {
    let width = width as usize;
    let height = height as usize;
    let img = Image::<u8>::create(width, height);
    
    Box::into_raw(box image_u8_t::from(img))
}

#[no_mangle]
pub unsafe extern "C" fn image_u8_destroy(mut im: *mut image_u8_t) {
    drop_boxed_mut(&mut im);
}