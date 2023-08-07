#![cfg(feature="compare_reference")]

#[cfg(feature="cffi")]
compile_error!("Feature \"cffi\" and \"compare_reference\" are mutually exclusive");

use std::{ptr::NonNull, marker::PhantomData, mem::{transmute, ManuallyDrop}, fmt::Debug};

use apriltag_sys::apriltag_family;

use crate::{AprilTagDetector, util::{ImageY8, math::mat::Mat}, AprilTagFamily};

pub(crate) trait PtrDrop {
    unsafe fn drop_ptr(ptr: *mut Self);
}

impl PtrDrop for apriltag_sys::apriltag_detector {
    unsafe fn drop_ptr(ptr: *mut Self) {
        // println!("Calling apriltag_sys::apriltag_detector_destroy({ptr:p})");
        // let families = if let Some(zfamilies) = ZArraySys::<SysPtr<apriltag_family>>::wrap(ptr.as_ref().unwrap().tag_families) {
        //     ManuallyDrop::new(zfamilies);
        //     // ManuallyDrop::new(zfamilies).unsafe_as_vec().unwrap()
        //     Vec::<SysPtr<apriltag_family>>::new()
        // } else {
        //     Vec::new()
        // };

        apriltag_sys::apriltag_detector_destroy(ptr);
    }
}

impl PtrDrop for apriltag_sys::matd_t {
    unsafe fn drop_ptr(ptr: *mut Self) {
        apriltag_sys::matd_destroy(ptr)
    }
}

impl PtrDrop for apriltag_sys::image_u8 {
    unsafe fn drop_ptr(ptr: *mut Self) {
        // println!("Calling apriltag_sys::image_u8_destroy");
        apriltag_sys::image_u8_destroy(ptr)
    }
}

impl PtrDrop for apriltag_sys::apriltag_family {
    unsafe fn drop_ptr(ptr: *mut Self) {
        println!("Calling apriltag_sys::destroy family");
        if ptr.is_null() {
            return;
        }
        let mut fam = Box::from_raw(ptr);
        let codes = Vec::from_raw_parts(std::mem::replace(&mut fam.codes, std::ptr::null_mut()), fam.ncodes as _, fam.ncodes as _);
        drop(codes);
        let bit_x = Vec::from_raw_parts(std::mem::replace(&mut fam.bit_x, std::ptr::null_mut()), fam.nbits as _, fam.nbits as _);
        drop(bit_x);
        let bit_y = Vec::from_raw_parts(std::mem::replace(&mut fam.bit_y, std::ptr::null_mut()), fam.nbits as _, fam.nbits as _);
        drop(bit_y);
        drop(fam);
    }
}

#[repr(transparent)]
pub(crate) struct SysPtr<T: PtrDrop>(NonNull<T>);

impl<T: PtrDrop> SysPtr<T> {
    pub(crate) fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
    pub(crate) fn as_ref(&self) -> &T {
        unsafe { self.0.as_ref() }
    }
    pub(crate) fn as_mut(&mut self) -> &mut T {
        unsafe { self.0.as_mut() }
    }
}

impl SysPtr<apriltag_sys::matd_t> {
    pub(crate) fn new(rows: usize, cols: usize) -> Option<Self> {
        let res = Self(NonNull::new(unsafe { apriltag_sys::matd_create(rows as _, cols as _) })?);

        Some(res)
    }

    pub(crate) fn new_like(src: &Mat) -> Option<Self> {
        let res = Self(NonNull::new(unsafe {
            apriltag_sys::matd_create_data(src.rows() as _, src.cols() as _, src.data.as_ptr() as *mut f64)
        })?);
        Some(res)
    }
}

impl SysPtr<apriltag_sys::apriltag_family> {
    pub(crate) fn new(src: &AprilTagFamily) -> Option<Self> {
        let mut codes = src.codes.clone();
        codes.shrink_to_fit();

        let (bit_x, bit_y) = src.split_bits();

        let (codes, _, ncodes) = Vec::into_raw_parts(codes);
        let (bit_x, _, _) = Vec::into_raw_parts(bit_x);
        let (bit_y, _, _) = Vec::into_raw_parts(bit_y);

        let family = Box::new(apriltag_sys::apriltag_family {
            ncodes: ncodes as _,
            codes,
            width_at_border: src.width_at_border as _,
            total_width: src.total_width as _,
            reversed_border: src.reversed_border,
            nbits: src.bits.len() as _,
            bit_x,
            bit_y,
            h: src.min_hamming as _,
            name: std::ptr::null_mut(),//TODO
            impl_: std::ptr::null_mut(),
        });
        Some(Self(NonNull::new(Box::into_raw(family))?))
    }
}

impl SysPtr<apriltag_sys::apriltag_detector> {
    pub(crate) fn new() -> Option<Self> {
        let res = Self(NonNull::new(unsafe { apriltag_sys::apriltag_detector_create() })?);
        Some(res)
    }
    pub(crate) fn new_like(src: &AprilTagDetector) -> Option<Self> {
        let mut res = Self::new()?;
        // println!("Created apriltag_detector: {:p}", res.as_ptr());
        let v = unsafe { res.0.as_mut() };
        v.nthreads = src.params.nthreads as _;
        v.quad_decimate = src.params.quad_decimate;
        v.quad_sigma = src.params.quad_sigma;
        v.refine_edges = src.params.refine_edges;
        v.decode_sharpening = src.params.decode_sharpening;
        v.debug = src.params.debug;

        v.qtp.min_cluster_pixels = src.params.qtp.min_cluster_pixels as _;
        v.qtp.max_nmaxima = src.params.qtp.max_nmaxima as _;
        v.qtp.cos_critical_rad = src.params.qtp.cos_critical_rad;
        v.qtp.max_line_fit_mse = src.params.qtp.max_line_fit_mse;
        v.qtp.min_white_black_diff = src.params.qtp.min_white_black_diff as _;
        v.qtp.deglitch = src.params.qtp.deglitch as _;

        Some(res)
    }

    pub(crate) fn new_with_families(src: &AprilTagDetector) -> Option<(Self, Vec<SysPtr<apriltag_family>>)> {
        let mut res = Self::new_like(src)?;
        let mut families = Vec::new();
        for qd in src.tag_families.iter() {
            let family_sys = SysPtr::<apriltag_family>::new(&qd.family).unwrap();
            families.push(family_sys);

            unsafe {
                apriltag_sys::apriltag_detector_add_family_bits(res.as_ptr(), families.last().unwrap().as_ptr(), qd.bits_corrected as _);
            }
        }

        Some((res, families))
    }
}

impl<T: PtrDrop> Drop for SysPtr<T> {
    fn drop(&mut self) {
        unsafe {
            T::drop_ptr(self.0.as_ptr())
        }
    }
}

pub(crate) type AprilTagDetectorSys = SysPtr<apriltag_sys::apriltag_detector>;


impl SysPtr<apriltag_sys::image_u8> {
    pub(crate) fn new(src: &ImageY8) -> Option<Self> {
        let ptr = NonNull::new(unsafe { apriltag_sys::image_u8_create_stride(src.width() as _, src.height() as _, src.stride() as _)})?;
        let buf = unsafe {
            core::slice::from_raw_parts_mut(ptr.as_ref().buf, src.width() * src.stride())
        };
        for y in 0..src.height() {
            for x in 0..src.width() {
                let px = src[(x, y)];
                buf[y * src.stride() + x] = px;
            }
        }
        Some(Self(ptr))
    }
}

pub(crate) type ImageU8Sys = SysPtr<apriltag_sys::image_u8>;

#[repr(transparent)]
pub(crate) struct ZArraySys<T> {
    ptr: NonNull<apriltag_sys::zarray>,
    t: PhantomData<T>,
}

impl<T: Debug> Debug for ZArraySys<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = unsafe { self.ptr.as_ref() };
        f.debug_struct("ZArraySys")
            .field("el_sz", &r.el_sz)
            .field("len", &r.size)
            .field("cap", &r.alloc)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T> ZArraySys<T> {
    pub(crate) fn new(src: Vec<T>) -> Option<Self> {
        let ptr = NonNull::new(unsafe {apriltag_sys::zarray_create(std::mem::size_of::<T>())})?;
        let mut res = Self {
            ptr,
            t: PhantomData
        };

        let len = src.len().try_into().unwrap();
        unsafe {
            apriltag_sys::zarray_ensure_capacity(res.as_ptr(), len);
        }
        if std::mem::align_of::<T>() == std::mem::size_of::<T>() && false {
            // Packed, we can use memmove
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr() as *mut i8, res.as_ptr() as *mut i8, src.len() * std::mem::size_of::<T>());
                let src_drop: Vec<ManuallyDrop<T>> = transmute(src);
                res.ptr.as_mut().size = len;
                drop(src_drop);
            }
        } else {
            // We need to move elements one-by-one
            for (i, el) in src.into_iter().enumerate() {
                let r = unsafe { res.ptr.as_mut() };
                unsafe {
                    *(r.data.byte_add(i * std::mem::size_of::<T>()) as *mut T) = el;
                    r.size += 1;
                }
            }
        }
        Some(res)
    }

    pub(crate) fn wrap(ptr: *mut apriltag_sys::zarray) -> Option<Self> {
        let ptr = NonNull::new(ptr)?;
        assert_eq!(unsafe { ptr.as_ref() }.el_sz, std::mem::size_of::<T>());
        Some(Self {
            ptr,
            t: PhantomData,
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut apriltag_sys::zarray {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe {
            let r = self.ptr.as_ref();
            std::slice::from_raw_parts(r.data as *const T, r.size as _)
        }
    }

    unsafe fn unsafe_as_vec(&self) -> Result<Vec<T>, ()> {
        let (len, cap, el_sz, ptr) = unsafe {
            let r = self.ptr.as_ref();
            (r.size as usize, r.alloc as usize, r.el_sz as usize, r.data)
        };
        debug_assert_eq!(el_sz, std::mem::size_of::<T>());
        if cap < len {
            return Err(());
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        
        let mut result = Box::<[T]>::new_zeroed_slice(len);
        if std::mem::align_of::<T>() == el_sz {
            // Packed, we can use memmove
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, result.as_ptr() as *mut i8, len * std::mem::size_of::<T>());
            }
        } else {
            // We need to move elements one-by-one
            for i in 0..len {
                let value = unsafe {
                    std::ptr::read(ptr.byte_add(i * el_sz) as *const T)
                };
                result[i].write(value);
            }
        }
        let res = unsafe { result.assume_init() };
        Ok(res.into_vec())
    }
}

impl<T> TryFrom<ZArraySys<T>> for Vec<T> {
    type Error = ();

    fn try_from(mut value: ZArraySys<T>) -> Result<Self, Self::Error> {
        unsafe {
            let res = value.unsafe_as_vec()?;
            value.ptr.as_mut().size = 0; // Don't free elements when dropping
            Ok(res)
        }
    }
}

impl<T> Drop for ZArraySys<T> {
    fn drop(&mut self) {
        unsafe {
            if std::mem::needs_drop::<T>() {
                let r = self.ptr.as_mut();
                for i in 0..(r.size as usize) {
                    let ptr = r.data.byte_add(i * r.el_sz) as *mut T;
                    std::ptr::drop_in_place(ptr);
                }
            }
            apriltag_sys::zarray_destroy(self.ptr.as_ptr());
        }
    }
}