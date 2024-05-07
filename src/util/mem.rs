use std::{mem::MaybeUninit, alloc::AllocError};

/// Marker trait for types that zeroed-out memory is a valid representation
pub trait SafeZero {}

impl SafeZero for u8 {}
impl SafeZero for u16 {}
impl SafeZero for u32 {}
impl SafeZero for u64 {}
impl SafeZero for i8 {}
impl SafeZero for i16 {}
impl SafeZero for i32 {}
impl SafeZero for i64 {}
impl SafeZero for f32 {}
impl SafeZero for f64 {}
impl SafeZero for isize {}
impl SafeZero for usize {}

/// Arrays of SafeZero can be allocated
impl<T: SafeZero, const N: usize> SafeZero for [T; N] {}
//TODO: double check this is valid
impl<T: SafeZero> SafeZero for Option<T> {}
impl<T> SafeZero for MaybeUninit<T> {}

/// Safely allocate a zeroed slice
pub(crate) fn calloc<T: SafeZero>(size: usize) -> Box<[T]> {
	let res = Box::new_zeroed_slice(size);
	unsafe { res.assume_init() }
}

pub(crate) fn try_calloc<T: SafeZero>(size: usize) -> Result<Box<[T]>, AllocError> {
	let res = Box::try_new_zeroed_slice(size)?;
	Ok(unsafe { res.assume_init() })
}