
pub(crate) trait SafeZero {}

impl SafeZero for u8 {}
impl SafeZero for u32 {}
impl SafeZero for i32 {}
impl SafeZero for f32 {}
impl SafeZero for f64 {}
impl SafeZero for usize {}

impl<T: SafeZero, const N: usize> SafeZero for [T; N] {}

pub(crate) fn calloc<T: SafeZero>(size: usize) -> Box<[T]> {
    let res = Box::new_zeroed_slice(size);
    unsafe { res.assume_init() }
}