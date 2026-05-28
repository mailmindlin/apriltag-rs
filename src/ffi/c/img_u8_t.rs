use std::slice;

use libc::c_uint;
use raw_parts::RawParts;

use crate::util::image::{ImageBuffer, Luma, ImageY8, ImageRefY8};

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
	pub(super) fn as_ref<'a>(&'a self) -> ImageRefY8<'a> {
		let width = self.width as usize;
		let height = self.height as usize;
		let stride = self.stride as usize;

		let data = unsafe { slice::from_raw_parts_mut(self.buf, stride * height) };
		ImageRefY8::wrap(data, width, height, stride)
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