use std::{sync::Arc, mem::transmute, ops::Deref, marker::PhantomData, ptr::NonNull, panic::{UnwindSafe, RefUnwindSafe}};

use jni::sys::jlong;

use super::JavaError;

#[derive(Debug, Clone, Copy)]
pub struct PointerNullError;

impl From<PointerNullError> for JavaError {
    fn from(_value: PointerNullError) -> Self {
        JavaError::NullPointerException("Missing data pointer".to_owned())
    }
}

pub trait RawPtr<T> {
    unsafe fn from_raw(ptr: *mut T) -> Self;
    unsafe fn to_raw(self) -> *mut T;
}

impl<T> RawPtr<T> for Arc<T> {
    unsafe fn from_raw(ptr: *mut T) -> Self {
        let p1: *const T = ptr as _;
        Arc::from_raw(p1)
    }

    unsafe fn to_raw(self) -> *mut T {
        transmute(Arc::into_raw(self))
    }
}

impl<T> RawPtr<T> for Box<T> {
    unsafe fn from_raw(ptr: *mut T) -> Self {
        Box::from_raw(ptr)
    }

    unsafe fn to_raw(self) -> *mut T {
        Box::into_raw(self)
    }
}

impl<T> RawPtr<T> for &T {
    unsafe fn from_raw(ptr: *mut T) -> Self {
        ptr.as_mut().unwrap()
    }

    unsafe fn to_raw(self) -> *mut T {
        let cp: *const T = &*self;
        transmute(cp)
    }
}

impl<T> RawPtr<T> for &mut T {
    unsafe fn from_raw(ptr: *mut T) -> Self {
        ptr.as_mut().unwrap()
    }

    unsafe fn to_raw(self) -> *mut T {
        &mut *self
    }
}

#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct JPtrManaged<'a, R: Deref + RawPtr<<R as Deref>::Target>>(JPtrMut<'a, R>) where <R as Deref>::Target: Sized;

#[allow(suspicious_auto_trait_impls)]
impl<'a, T> UnwindSafe for JPtrManaged<'a, Box<T>> {}
#[allow(suspicious_auto_trait_impls)]
impl<'a, T> RefUnwindSafe for JPtrManaged<'a, Box<T>> {}

impl<'a, R: Deref + RawPtr<<R as Deref>::Target>> From<R> for JPtrManaged<'a, R> where <R as Deref>::Target: Sized {
    fn from(value: R) -> Self {
        Self(JPtrMut::from(value))
    }
}

impl<'a, R: Deref + RawPtr<<R as Deref>::Target>> Default for JPtrManaged<'a, R> where <R as Deref>::Target: Sized {
    fn default() -> Self {
        Self(JPtrMut::default())
    }
}

impl<'a, R: Deref + RawPtr<<R as Deref>::Target>> JPtrManaged<'a, R> where <R as Deref>::Target: Sized {
    pub fn as_jni(&self) -> jlong {
        self.0.internal
    }

    pub fn take(mut self) -> Result<R, PointerNullError> {
        let ptr_jl = std::mem::replace(&mut self.0.internal, 0);
        let ptr = (ptr_jl as usize) as *mut <R as Deref>::Target;
        if ptr.is_null() {
            Err(PointerNullError)
        } else {
            Ok(unsafe { RawPtr::from_raw(ptr) })
        }
    }
}

impl<'a, R: Deref + RawPtr<<R as Deref>::Target>> Drop for JPtrManaged<'a, R> where <R as Deref>::Target: Sized {
    fn drop(&mut self) {
        if let Ok(ptr) = self.0.as_ptr() {
            let x = unsafe { R::from_raw(ptr.as_ptr()) };
            drop(x);
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct JPtr<'a, T> {
    internal: jlong,
    lifetime: PhantomData<&'a T>,
}

impl<'a, T> Default for JPtr<'a, T> {
    fn default() -> Self {
        Self { internal: 0, lifetime: PhantomData }
    }
}

impl<'a, T> JPtr<'a, *const T> {
    pub(in super::super) fn from_ptr(ptr: *const T) -> Result<Self, JavaError> {
        let ptr_i = ptr as usize;
        let ptr_jl: jlong = ptr_i.try_into()
            .map_err(|e| JavaError::IllegalStateException(format!("Unable to return pointer: {e}")))?;
        Ok(Self {
            internal: ptr_jl,
            lifetime: PhantomData,
        })
    }
}

impl<'a, R: Deref> JPtr<'a, R> where <R as Deref>::Target: Sized, R: RawPtr<<R as Deref>::Target> {
    pub(crate) fn as_ptr(&self) -> Result<NonNull<R::Target>, PointerNullError> {
        let ptr = (self.internal as usize) as *mut R::Target;
        match NonNull::new(ptr) {
            Some(ptr) => Ok(ptr),
            None => Err(PointerNullError)
        }
    }
    
    pub(crate) fn as_ref(&self) -> Result<&'a R::Target, PointerNullError> {
        let ptr = self.as_ptr()?;
        Ok(unsafe { ptr.as_ref() })
    }
}

#[allow(suspicious_auto_trait_impls)]
impl<'a, T> UnwindSafe for JPtr<'a, Arc<T>> {}
#[allow(suspicious_auto_trait_impls)]
impl<'a, T> RefUnwindSafe for JPtr<'a, Arc<T>> {}
#[allow(suspicious_auto_trait_impls)]
impl<'a, T> UnwindSafe for JPtr<'a, Box<T>> {}
#[allow(suspicious_auto_trait_impls)]
impl<'a, T> RefUnwindSafe for JPtr<'a, Box<T>> {}


#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct JPtrMut<'a, T> {
    internal: jlong,
    lifetime: PhantomData<&'a T>,
}

impl<'a, T> Default for JPtrMut<'a, T> {
    fn default() -> Self {
        Self { internal: 0, lifetime: PhantomData }
    }
}

impl<'a, R: Deref> JPtrMut<'a, R> where <R as Deref>::Target: Sized, R: RawPtr<<R as Deref>::Target> {
    pub(super) fn as_ptr(&self) -> Result<NonNull<R::Target>, PointerNullError> {
        let ptr = (self.internal as usize) as *mut R::Target;
        match NonNull::new(ptr) {
            Some(ptr) => Ok(ptr),
            None => Err(PointerNullError)
        }
    }

    pub(super) fn as_mut(&self) -> Result<&'a mut R::Target, PointerNullError> {
        let mut ptr = self.as_ptr()?;
        Ok(unsafe { ptr.as_mut() })
    }
}

impl<'a, R: Deref + RawPtr<<R as Deref>::Target>> From<R> for JPtrMut<'a, R> where <R as Deref>::Target: Sized {
    fn from(value: R) -> Self {
        let ptr = unsafe { RawPtr::to_raw(value) };
        let internal: jlong = (ptr as usize).try_into()
            .expect("Pointer conversion error");
        Self {
            internal,
            lifetime: PhantomData
        }
    }
}

// impl<T> From<Box<T>> for JPtrMut<'static, Box<T>> {
//     fn from(value: Box<T>) -> Self {
//         let ptr = Box::into_raw(value);
//         Self {
//             internal: unsafe { transmute(ptr) },
//             lifetime: PhantomData,
//         }
//     }
// }

// impl<T> From<Arc<T>> for JPtrMut<'static, Arc<T>> {
//     fn from(value: Arc<T>) -> Self {
//         let ptr = Arc::into_raw(value);
//         Self {
//             internal: unsafe { transmute(ptr) },
//             lifetime: PhantomData,
//         }
//     }
// }