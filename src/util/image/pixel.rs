use std::mem::MaybeUninit;

use crate::util::mem::SafeZero;

use super::{Rgb, Luma};

pub trait Primitive: Copy + Clone {
    const DEFAULT_MAX_VALUE: Self;
    const DEFAULT_MIN_VALUE: Self;
}

impl Primitive for u8 {
    const DEFAULT_MAX_VALUE: Self = u8::MAX;

    const DEFAULT_MIN_VALUE: Self = u8::MIN;
}

impl Primitive for f64 {
    const DEFAULT_MAX_VALUE: Self = f64::MIN;

    const DEFAULT_MIN_VALUE: Self = f64::MAX;
}

pub trait Pixel: Copy {
    type Subpixel: Primitive;

    type Value;

    /// Number of subpixels per pixel
    const CHANNEL_COUNT: usize;

    fn channels(&self) -> &[Self::Subpixel];
    fn channels_mut(&mut self) -> &mut [Self::Subpixel];

    fn to_value(self) -> Self::Value;

    fn from_slice<'a>(slice: &'a [Self::Subpixel]) -> &'a Self;

    fn slice_to_value<'a>(slice: &'a [Self::Subpixel]) -> &'a Self::Value;

    fn from_slice_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self;

    fn slice_to_value_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self::Value;
}

pub trait PixelConvert: Pixel {
    fn to_rgb(&self) -> Rgb<Self::Subpixel>;
    fn to_luma(&self) -> Luma<Self::Subpixel>;
}

pub trait DefaultAlignment: Pixel {
    const DEFAULT_ALIGNMENT: usize;
}

impl<T: Primitive> Primitive for MaybeUninit<T> {
    const DEFAULT_MAX_VALUE: Self = MaybeUninit::new(<T as Primitive>::DEFAULT_MAX_VALUE);

    const DEFAULT_MIN_VALUE: Self = MaybeUninit::new(<T as Primitive>::DEFAULT_MAX_VALUE);
}

impl<P: Primitive, const N: usize> Pixel for [P; N] {
    type Subpixel = P;

    type Value = [P; N];

    const CHANNEL_COUNT: usize = N;

    fn channels(&self) -> &[Self::Subpixel] {
        self
    }

    fn channels_mut(&mut self) -> &mut [Self::Subpixel] {
        self
    }

    fn to_value(self) -> Self::Value {
        self
    }

    fn from_slice<'a>(slice: &'a [Self::Subpixel]) -> &'a Self {
        slice.try_into().unwrap()
    }

    fn slice_to_value<'a>(slice: &'a [Self::Subpixel]) -> &'a Self::Value {
        slice.try_into().unwrap()
    }

    fn from_slice_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self {
        slice.try_into().unwrap()
    }

    fn slice_to_value_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self::Value {
        slice.try_into().unwrap()
    }
}

impl<P: Pixel> Pixel for MaybeUninit<P> where P::Subpixel: SafeZero {
    type Subpixel = MaybeUninit<P::Subpixel>;

    type Value = MaybeUninit<P::Value>;

    const CHANNEL_COUNT: usize = P::CHANNEL_COUNT;

    fn channels(&self) -> &[Self::Subpixel] {
        unsafe {
            std::mem::transmute(P::channels(self.assume_init_ref()))
        }
    }

    fn channels_mut(&mut self) -> &mut [Self::Subpixel] {
        unsafe {
            std::mem::transmute(P::channels_mut(self.assume_init_mut()))
        }
    }

    // fn to_rgb(&self) -> Rgb<Self::Subpixel> {
    //     unsafe {
    //         let rgb = P::to_rgb(self.assume_init_ref())
    //             .channels();

    //         *Rgb::from_slice(uninit_array(rgb))
    //     }
    // }

    fn to_value(self) -> Self::Value {
        MaybeUninit::new(P::to_value(unsafe { self.assume_init() }))
    }

    // fn to_luma(&self) -> Luma<Self::Subpixel> {
    //     unsafe {
    //         let rgb = P::to_luma(self.assume_init_ref())
    //             .channels();

    //         *Luma::from_slice(uninit_array(rgb))
    //     }
    // }

    fn from_slice<'a>(slice: &'a [Self::Subpixel]) -> &'a Self {
        let slice = unsafe { MaybeUninit::slice_assume_init_ref(slice) };
        let inner = P::from_slice(slice);
        unsafe { std::mem::transmute(inner) }
    }

    fn slice_to_value<'a>(slice: &'a [Self::Subpixel]) -> &'a Self::Value {
        let slice = unsafe { MaybeUninit::slice_assume_init_ref(slice) };
        let inner = P::slice_to_value(slice);
        unsafe { std::mem::transmute(inner) }
    }

    fn from_slice_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self {
        let slice = unsafe { MaybeUninit::slice_assume_init_mut(slice) };
        let inner = P::from_slice_mut(slice);
        unsafe { std::mem::transmute(inner) }
    }

    fn slice_to_value_mut<'a>(slice: &'a mut [Self::Subpixel]) -> &'a mut Self::Value {
        let slice = unsafe { MaybeUninit::slice_assume_init_mut(slice) };
        let inner = P::slice_to_value_mut(slice);
        unsafe { std::mem::transmute(inner) }
    }
}