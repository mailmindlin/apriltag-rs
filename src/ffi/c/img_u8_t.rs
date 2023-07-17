use std::{ops::Deref, mem::ManuallyDrop, slice};

use libc::c_uint;
use raw_parts::RawParts;

use crate::util::image::{ImageBuffer, Luma, ImageY8};

use super::super::util::{drop_boxed_mut, drop_array_mut};

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

impl From<ImageBuffer<Luma<u8>>> for image_u8_t {
    fn from(value: ImageBuffer<Luma<u8>>) -> Self {
        let width = value.width() as i32;
        let height = value.height() as i32;
        let stride = value.stride() as i32;
        
        let RawParts { ptr, length, capacity } = RawParts::from_vec(value.data.into_vec());
        assert_eq!(length, capacity, "Vector not compact");

        Self {
            width,
            height,
            stride,
            buf: ptr,
        }
    }
}

impl image_u8_t {
    pub(super) fn pretend_ref<'a>(&'a self) -> FakeImageGuard<'a> {
        let mut pretend = ManuallyDrop::new(ImageY8::zeroed(self.width as usize, self.height as usize));
        let buf_ptr = unsafe { slice::from_raw_parts_mut(self.buf, pretend.len()) };
        pretend.data = unsafe { Box::from_raw(buf_ptr as *mut _) };
        FakeImageGuard {
            raw: self,
            pretend,
        }
    }
}

pub(super) struct FakeImageGuard<'a> {
    raw: &'a image_u8_t,
    pretend: ManuallyDrop<ImageBuffer<Luma<u8>>>,
}

impl<'a> Drop for FakeImageGuard<'a> {
    fn drop(&mut self) {
        let pb = std::mem::take(&mut self.pretend.data);
        let pretend_ptr = pb.as_ptr();

        assert_eq!(pretend_ptr, self.raw.buf);
        
        std::mem::forget(pb);// Don't free pb, because it's really owned by `raw`
    }
}

impl<'a> Deref for FakeImageGuard<'a> {
    type Target = ImageBuffer<Luma<u8>>;

    fn deref(&self) -> &Self::Target {
        &self.pretend
    }
}

#[no_mangle]
pub unsafe extern "C" fn image_u8_create(width: c_uint, height: c_uint) -> *mut image_u8_t {
    let width = width as usize;
    let height = height as usize;
    let img = ImageY8::zeroed(width, height);
    
    Box::into_raw(Box::new(image_u8_t::from(img)))
}

#[no_mangle]
pub unsafe extern "C" fn image_u8_destroy(mut im: *mut image_u8_t) {
    drop_boxed_mut(&mut im);
}