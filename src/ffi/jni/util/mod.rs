mod buffer;
mod jptr;

use std::{sync::PoisonError, ops::Deref, panic::{UnwindSafe, AssertUnwindSafe}};
use jni::JNIEnv;

pub(super) use buffer::get_buffer;

use self::jptr::{JPtrMut, RawPtr};
pub(super) use self::jptr::{PointerNullError, JPtr, JPtrManaged};

pub(super) enum JavaError {
    GenericException(String),
    // Exception(String, String),
    ConcurrentModificationException,
    IllegalArgumentException(String),
    NullPointerException(String),
    IllegalStateException(String),
    Internal(jni::errors::Error),
    OutOfMemoryError(String),
}

impl JavaError {
    fn throw(&self, env: &mut JNIEnv) {
        match self {
            JavaError::GenericException(msg) => env.throw_new("java/lang/Exception", msg),
            // JavaError::Exception(class, msg) => env.throw_new(class, msg),
            Self::NullPointerException(msg) => env.throw_new("java/lang/NullPointerException", msg),
            JavaError::ConcurrentModificationException => env.throw_new("java/util/ConcurrentModificationException", ""),
            JavaError::IllegalArgumentException(msg) => env.throw_new("java/lang/IllegalArgumentException", msg),
            JavaError::IllegalStateException(msg) => env.throw_new("java/lang/IllegalStateException", msg),
            JavaError::OutOfMemoryError(msg) => env.throw_new("java/lang/RuntimeException", msg),
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

pub(super) fn jni_wrap_simple<'a, R: Default>(mut env: JNIEnv<'a>, inner: impl FnOnce(&mut JNIEnv<'a>) -> JavaResult<R> + UnwindSafe) -> R {
    let res = std::panic::catch_unwind(AssertUnwindSafe(|| inner(&mut env)));
    match res {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            e.throw(&mut env);
            R::default()
        },
        Err(e) => {
            let res = match e.downcast_ref::<&str>() {
                Some(msg) => env.throw_new("java/lang/RuntimeException", msg),
                None => env.throw_new("java/lang/RuntimeException", "Rust panic occurred"),
            };
            if let Err(e) = res {
                env.fatal_error(format!("Unable to throw error from panic: {e}"));
            }
            R::default()
        },
    }
}

pub(super) trait JavaBorrow<'a> {
    type Borrowed: 'a;
    fn borrow(self) -> Result<Self::Borrowed, PointerNullError>;
}

impl<'a, T: Deref + RawPtr<<T as Deref>::Target>> JavaBorrow<'a> for JPtrMut<'a, T> where <T as Deref>::Target: Sized {
    type Borrowed = &'a mut <T as Deref>::Target;

    fn borrow(self) -> Result<Self::Borrowed, PointerNullError> {
        self.as_mut()
    }
}

impl<'a, T: Deref + RawPtr<<T as Deref>::Target>> JavaBorrow<'a> for JPtr<'a, T> where <T as Deref>::Target: Sized {
    type Borrowed = &'a <T as Deref>::Target;
    fn borrow(self) -> Result<Self::Borrowed, PointerNullError> {
        self.as_ref()
    }
}

impl<'a, T: Deref + RawPtr<<T as Deref>::Target>> JavaBorrow<'a> for JPtrManaged<'a, T> where <T as Deref>::Target: Sized {
    type Borrowed = T;
    fn borrow(self) -> Result<Self::Borrowed, PointerNullError> {
        self.take()
    }
}

pub(super) fn jni_wrap<'a, B: JavaBorrow<'a>, R: Default>(env: JNIEnv<'a>, ptr: B, inner: impl FnOnce(&mut JNIEnv<'a>, B::Borrowed) -> JavaResult<R> + UnwindSafe) -> R
    where B: UnwindSafe
{
    jni_wrap_simple(env, |env| {
        let value = ptr.borrow()?;
        inner(env, value)
    })
}