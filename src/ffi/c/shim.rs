use std::{fmt::Debug, marker::PhantomData, panic::{UnwindSafe, RefUnwindSafe, Location}, alloc::AllocError};

use super::FFIConvertError;

#[repr(C)]
#[derive(Default)]
pub struct IncompleteArrayField<T>(::std::marker::PhantomData<T>, [T; 0]);
impl<T> IncompleteArrayField<T> {
    #[inline]
    pub const fn new() -> Self {
        IncompleteArrayField(::std::marker::PhantomData, [])
    }
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self as *const _ as *const T
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut _ as *mut T
    }
    #[inline]
    pub unsafe fn as_slice(&self, len: usize) -> &[T] {
        ::std::slice::from_raw_parts(self.as_ptr(), len)
    }
    #[inline]
    pub unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
        ::std::slice::from_raw_parts_mut(self.as_mut_ptr(), len)
    }
}
impl<T> ::std::fmt::Debug for IncompleteArrayField<T> {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        fmt.write_str("__IncompleteArrayField")
    }
}

/// Error generated by FFI
pub enum CFFIError {
    /// Allocation failed
    Alloc,
    /// Required argument was null
    NullArgument(&'static str),
    /// Type conversion failure
    BadConversion(FFIConvertError),
    /// Required argument conversion failure
    BadInput {
        param_name: &'static str,
        inner: FFIConvertError,
    },
    /// Message extracted from catching a `panic`
    PanicStr(String),
}

impl From<AllocError> for CFFIError {
    fn from(value: AllocError) -> Self {
        Self::Alloc
    }
}

pub(super) mod param {
    use crate::ffi::c::FFIConvertError;

    use super::CFFIError;

    pub(in super::super) fn try_read<R, T>(value: T, name: &'static str) -> Result<R, CFFIError> where R: TryFrom<T>, FFIConvertError: From<<R as TryFrom<T>>::Error> {
        match <R as TryFrom<T>>::try_from(value) {
            Ok(res) => Ok(res),
            Err(e) => {
                let inner = <FFIConvertError as From<_>>::from(e);
                Err(CFFIError::BadInput { param_name: name, inner })
            }
        }
    }
}

#[repr(transparent)]
pub struct InPtr<'a, T>{
    ptr: *const T,
    lifetime: PhantomData<&'a T>,
}

#[repr(transparent)]
pub struct OutPtr<'a, T> {
    ptr: *mut T,
    lifetime: PhantomData<&'a mut T>,
}

pub struct InOutPtr<'a, T> {
    ptr: *mut T,
    lifetime: PhantomData<&'a mut T>,
}

impl<'a, T> RefUnwindSafe for InPtr<'a, T> {}

pub(super) trait ReadPtr<'a, T> {
    fn ptr(&self) -> *const T;

    unsafe fn try_read<R>(&self, name: &'static str) -> Result<R, CFFIError> where R: TryFrom<*const T>, <R as TryFrom<*const T>>::Error: Debug {
        todo!()
    }

    unsafe fn try_array(&self, len: usize, name: &'static str) -> Result<&'a [T], CFFIError> {
        let ptr = self.ptr();
        if ptr.is_null() {
            Err(CFFIError::NullArgument(name))
        } else {
            Ok(std::slice::from_raw_parts(ptr, len))
        }
    }

    unsafe fn try_ref(&self, name: &'static str) -> Result<&'a T, CFFIError> {
        match self.ptr().as_ref() {
            Some(r) => Ok(r),
            None => Err(CFFIError::NullArgument(name)),
        }
    }
}

impl<'a, T> ReadPtr<'a, T> for InPtr<'a, T> {
    fn ptr(&self) -> *const T {
        self.ptr
    }
}

impl<'a, T> ReadPtr<'a, T> for InOutPtr<'a, T> {
    fn ptr(&self) -> *const T {
        self.ptr
    }
}

impl<'a, T> OutPtr<'a, T> {
    pub(super) unsafe fn maybe_write(&self, value: T) {
        if let Some(dst) = self.ptr.as_mut() {
            match value.try_into() {
                Ok(v) => {
                    std::ptr::write(dst, v);
                },
                Err(e) => {
                    eprintln!("{e:?}");
                }
            }
        }
    }

    pub(super) unsafe fn maybe_try_write<V: TryInto<T>>(&self, value: V) -> Result<(), CFFIError> where CFFIError: From<<V as TryInto<T>>::Error> {
        if let Some(dst) = self.ptr.as_mut() {
            let v = value.try_into()?;
            std::ptr::write(dst, v);
        }
        Ok(())
    }
}

fn cffi_catch<R>(callback: impl (FnOnce() -> Result<R, CFFIError>) + UnwindSafe) -> Result<R, CFFIError> {
    match std::panic::catch_unwind(callback) {
        Ok(result) => result,
        Err(panic_data) => {
            todo!()
        }
    }
}

fn handle_cffi_error(location: &Location, error: CFFIError) {
    
}

#[track_caller]
pub(super) fn cffi_wrapper<R>(callback: impl (FnOnce() -> Result<R, CFFIError>) + UnwindSafe) -> R where R: Default {
    match cffi_catch(callback) {
        Ok(result) => result,
        Err(error) => {
            // Constrain codegen
            handle_cffi_error(std::panic::Location::caller(), error);
            R::default()
        }
    }
}

// pub(super) fn try_read<'a, R, T>(ptr: *const T, param_name: &'static str) -> &'a T where R: TryFrom<T>, <R as TryFrom<T>>::Error: Debug {

// }


// pub(super) fn try_read_mut<'a, R, T>(ptr: *mut T, param_name: &'static str) -> &'a mut T where R: TryFrom<T>, <R as TryFrom<T>>::Error: Debug {

// }