mod buffer;

use std::{sync::{PoisonError, Arc}, mem::{self, transmute}, marker::PhantomData, ptr::NonNull, ops::Deref};

use jni::{JNIEnv, sys::jlong};

pub(super) use buffer::get_buffer;

pub(super) enum JavaError {
    GenericException(String),
    Exception(String, String),
    ConcurrentModificationException,
    IllegalArgumentException(String),
    NullPointerException(String),
    IllegalStateException(String),
    Internal(jni::errors::Error),
}

impl JavaError {
    fn throw(&self, env: &mut JNIEnv) {
        match self {
            JavaError::GenericException(msg) => env.throw_new("java/lang/Exception", msg),
            JavaError::Exception(class, msg) => env.throw_new(class, msg),
            Self::NullPointerException(msg) => env.throw_new("java/lang/NullPointerException", msg),
            JavaError::ConcurrentModificationException => env.throw_new("java/util/ConcurrentModificationException", ""),
            JavaError::IllegalArgumentException(msg) => env.throw_new("java/lang/IllegalArgumentException", msg),
            JavaError::IllegalStateException(msg) => env.throw_new("java/lang/IllegalStateException", msg),
            JavaError::Internal(_) => return,
        }.unwrap();
    }
}

impl<T> From<PoisonError<T>> for JavaError {
    fn from(_value: PoisonError<T>) -> Self {
        JavaError::ConcurrentModificationException
    }
}

impl From<jni::errors::Error> for JavaError {
    fn from(value: jni::errors::Error) -> Self {
        Self::Internal(value)
    }
}


pub trait RawPtr<T> {
    unsafe fn from_raw(ptr: *mut T) -> Self;
    unsafe fn to_raw(self) -> *mut T;
}

impl<T> RawPtr<T> for Arc<T> {
    unsafe fn from_raw(ptr: *mut T) -> Self {
        let p1: *const T = transmute(ptr);
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

impl<'a, R: Deref> JPtr<'a, R> where <R as Deref>::Target: Sized, R: RawPtr<<R as Deref>::Target> {
    pub(super) fn as_ptr(&self) -> Result<NonNull<R::Target>, JavaError> {
        let ptr: *mut R::Target = unsafe { mem::transmute(self.internal) };
        match NonNull::new(ptr) {
            Some(ptr) => Ok(ptr),
            None => Err(JavaError::NullPointerException("Missing data pointer".to_owned()))
        }
    }
    pub(super) fn as_ref(&self) -> Result<&'a R::Target, JavaError> {
        let ptr = self.as_ptr()?;
        Ok(unsafe { ptr.as_ref() })
    }

    pub(super) fn as_mut(&self) -> Result<&'a mut R::Target, JavaError> {
        let mut ptr = self.as_ptr()?;
        Ok(unsafe { ptr.as_mut() })
    }

    pub(super) fn take(self) -> Result<R, JavaError> {
        let ptr = self.as_ptr()?;
        Ok(unsafe { R::from_raw(ptr.as_ptr()) })
    }
}

impl<T> From<Box<T>> for JPtr<'static, Box<T>> {
    fn from(value: Box<T>) -> Self {
        let ptr = Box::into_raw(value);
        Self {
            internal: unsafe { transmute(ptr) },
            lifetime: PhantomData,
        }
    }
}

impl<T> From<Arc<T>> for JPtr<'static, Arc<T>> {
    fn from(value: Arc<T>) -> Self {
        let ptr = Arc::into_raw(value);
        Self {
            internal: unsafe { transmute(ptr) },
            lifetime: PhantomData,
        }
    }
}


type JavaResult<T> = Result<T, JavaError>;

// pub(super) fn get_ptr_mut(ptr: jlong) -> JavaResult<&'static mut JniDetectorData> {
//     let ptr: *mut JniDetectorData = unsafe { mem::transmute(ptr) };
//     match unsafe { ptr.as_mut::<'static>() } {
//         Some(data) => Ok(data),
//         None => Err(JavaError::NullPointerException("Missing data pointer".to_owned()))
//     }
// }

// pub(super) fn get_ptr<T>(ptr: jlong) -> JavaResult<&'static T> {
//     let ptr: *mut T = unsafe { mem::transmute(ptr) };
//     match unsafe { ptr.as_mut::<'static>() } {
//         Some(data) => Ok(data),
//         None => Err(JavaError::NullPointerException("Missing data pointer".to_owned()))
//     }
// }

pub(super) fn jni_wrap_simple<'a, R: Default>(env: &mut JNIEnv<'a>, inner: impl FnOnce(&mut JNIEnv<'a>) -> JavaResult<R>) -> R {
    let res = inner(env);
    match res {
        Ok(v) => v,
        Err(e) => {
            e.throw(env);
            R::default()
        },
    }
}

pub(super) fn jni_wrap<'a, T: Deref, R: Default>(env: &mut JNIEnv<'a>, ptr: JPtr<'a, T>, inner: impl FnOnce(&mut JNIEnv<'a>, &mut T::Target) -> JavaResult<R>) -> R where T::Target: Sized, T: RawPtr<T::Target> {
    jni_wrap_simple(env, |env| {
        let value = ptr.as_mut()?;
        // let mut detector = data.detector.write()?;
        inner(env, value)
    })
}