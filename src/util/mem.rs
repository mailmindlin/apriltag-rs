/// Marker trait for types that zeroed-out memory is a valid representation
pub(crate) trait SafeZero {}

impl SafeZero for u8 {}
impl SafeZero for u32 {}
impl SafeZero for i32 {}
impl SafeZero for f32 {}
impl SafeZero for f64 {}
impl SafeZero for usize {}

/// Arrays of SafeZero can be allocated
impl<T: SafeZero, const N: usize> SafeZero for [T; N] {}
//TODO: double check this is valid
impl<T: SafeZero> SafeZero for Option<T> {}

pub(crate) fn calloc<T: SafeZero>(size: usize) -> Box<[T]> {
    let res = Box::new_zeroed_slice(size);
    unsafe { res.assume_init() }
}