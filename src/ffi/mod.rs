#[cfg(feature="jni")]
pub(crate) mod jni;

#[cfg(feature="python")]
pub(crate) mod python;

#[cfg(all(feature="cffi", not(feature="cbindgen")))]
mod c;
#[cfg(all(feature="cbindgen"))]
pub mod c;

mod util;