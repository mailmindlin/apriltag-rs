use std::{sync::{PoisonError, RwLock}, mem};

use jni::{JNIEnv, sys::jlong};

use crate::ApriltagDetector;

pub(super) struct JniDetectorData {
    pub(super) detector: RwLock<ApriltagDetector>,
    // field_ptr: Mutex<Option<JFieldID>>,
}

pub(super) enum JavaError {
    Exception(String),
    ConcurrentModificationException,
    IllegalArgumentException(String),
    Internal(jni::errors::Error),
}

impl JavaError {
    fn throw(&self, env: &mut JNIEnv) {
        match self {
            JavaError::Exception(msg) => env.throw_new("java/lang/Exception", msg),
            JavaError::ConcurrentModificationException => env.throw_new("java/util/ConcurrentModificationException", ""),
            JavaError::IllegalArgumentException(msg) => env.throw_new("java/lang/IllegalArgumentException", msg),
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

type JavaResult<T> = Result<T, JavaError>;

pub(super) fn get_ptr_mut(ptr: jlong) -> JavaResult<&'static mut JniDetectorData> {
    let ptr: *mut JniDetectorData = unsafe { mem::transmute(ptr) };
    match unsafe { ptr.as_mut::<'static>() } {
        Some(data) => Ok(data),
        None => Err(JavaError::IllegalArgumentException("Missing data pointer".to_owned()))
    }
}

pub(super) fn get_ptr(ptr: jlong) -> JavaResult<&'static JniDetectorData> {
    let ptr: *mut JniDetectorData = unsafe { mem::transmute(ptr) };
    match unsafe { ptr.as_mut::<'static>() } {
        Some(data) => Ok(data),
        None => Err(JavaError::IllegalArgumentException("Missing data pointer".to_owned()))
    }
}

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

pub(super) fn jni_wrap<'a, R: Default>(env: &mut JNIEnv<'a>, ptr: jlong, inner: impl FnOnce(&mut JNIEnv<'a>, &mut ApriltagDetector) -> JavaResult<R>) -> R {
    jni_wrap_simple(env, |env| {
        let data = get_ptr(ptr)?;
        let mut detector = data.detector.write()?;
        inner(env, &mut detector)
    })
}