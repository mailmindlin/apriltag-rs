#[cfg(feature="jni")]
pub(crate) mod jni;

#[cfg(feature="python")]
pub(crate) mod python;

#[cfg(feature="cffi")]
pub(crate) mod c;

mod util;