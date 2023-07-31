use std::{sync::{atomic::AtomicPtr, Arc}, marker::PhantomData, mem::{self}, ptr, ops::Deref, cell::UnsafeCell};

pub trait Container: Deref {
    fn into_raw(this: Self) -> *const <Self as Deref>::Target;
    unsafe fn from_raw(ptr: *const <Self as Deref>::Target) -> Self;
}

impl<T> Container for Arc<T> {
    fn into_raw(this: Self) -> *const T {
        Arc::into_raw(this)
    }

    unsafe fn from_raw(ptr: *const T) -> Self {
        Arc::from_raw(ptr)
    }
}

impl<T> Container for Box<T> {
    fn into_raw(this: Self) -> *const T {
        Box::into_raw(this)
    }

    unsafe fn from_raw(ptr: *const T) -> Self {
        Box::from_raw(ptr as *mut T)
    }
}

#[repr(transparent)]
pub struct ManagedPtr<C: Container, F=<C as Deref>::Target> where <C as Deref>::Target: Sized {
    pub ptr: UnsafeCell<*const F>,
    container: PhantomData<C>,
}

impl<C: Container, F> ManagedPtr<C, F> where <C as Deref>::Target: Sized {
    pub(super) const unsafe fn wrap_ptr(ptr: *const F) -> Self {
        Self {
            ptr: UnsafeCell::new(ptr),
            container: PhantomData,
        }
    }
    pub(super) fn from(value: C) -> Self {
        let ptr = C::into_raw(value) as *const F;
        unsafe { Self::wrap_ptr(ptr) }
    }
    pub(super) const fn null() -> Self {
        unsafe { Self::wrap_ptr(ptr::null()) }
    }
    fn unwrap_ptr(ptr: *const F) -> Option<C> {
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { C::from_raw(ptr as *const <C as Deref>::Target) })
        }
    }
    pub(super) fn borrow(&self) -> Option<C> where C: Clone {
        let ptr = mem::replace(unsafe { self.ptr.get().as_mut().unwrap() }, ptr::null_mut());
        let value = Self::unwrap_ptr(ptr)?;
        let res = value.clone();
        unsafe { *self.ptr.get() = C::into_raw(value) as _; }
        Some(res)
    }
    pub(super) fn take(&mut self) -> Option<C> {
        let ptr = mem::replace(self.ptr.get_mut(), ptr::null_mut());
        Self::unwrap_ptr(ptr)
    }
}

impl<C: Container, F> Drop for ManagedPtr<C, F> where <C as Deref>::Target: Sized {
    fn drop(&mut self) {
        self.take();
    }
}

#[repr(transparent)]
pub(super) struct AtomicManagedPtr<C: Container, F=<C as Deref>::Target> where <C as Deref>::Target: Sized {
    pub ptr: AtomicPtr<F>,
    container: PhantomData<C>,
}

impl<C: Container, F> AtomicManagedPtr<C, F> where <C as Deref>::Target: Sized {
    pub(super) const fn wrap(ptr: *mut <C as Deref>::Target) -> Self {
        Self {
            ptr: AtomicPtr::new(ptr as *mut F),
            container: PhantomData,
        }
    }

    pub(super) fn from(value: C) -> Self {
        Self::wrap(C::into_raw(value) as *mut <C as Deref>::Target)
    }

    pub(super) const fn null() -> Self {
        Self::wrap(ptr::null_mut())
    }

    fn load(&self) -> *mut <C as Deref>::Target {
        self.ptr.load(std::sync::atomic::Ordering::SeqCst) as _
    }

    pub(super) fn is_null(&self) -> bool {
        self.load().is_null()
    }

    pub(super) fn borrow(&self) -> Option<C> where C: Clone {
        let value = self.take()?;
        self.swap(value.clone());
        Some(value)
    }
    fn unwrap_ptr(ptr: *mut F) -> Option<C> {
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { C::from_raw(ptr as *const <C as Deref>::Target) })
        }
    }
    pub(super) fn take(&self) -> Option<C> {
        let prev = self.ptr.swap(ptr::null_mut(), std::sync::atomic::Ordering::SeqCst);
        
        Self::unwrap_ptr(prev)
    }
    // pub(super) fn into_inner(self) -> Option<C> {
    //     let ptr = self.ptr.into_inner();
    //     Self::unwrap_ptr(ptr)
    // }
    pub(super) fn swap(&self, value: C) -> Option<C> {
        let next = C::into_raw(value);
        let prev = self.ptr.swap(next as _, std::sync::atomic::Ordering::SeqCst);
        Self::unwrap_ptr(prev)
    }
}

impl<C: Container, F> Drop for AtomicManagedPtr<C, F> where <C as Deref>::Target: Sized {
    fn drop(&mut self) {
        self.take();
    }
}

pub(super) fn drop_boxed_mut<T>(ptr: &mut *mut T) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null_mut()) as *mut T;
    let boxed = unsafe { Box::from_raw(ptr) };
    drop(boxed);
}

pub(super) fn drop_array<T>(ptr: &mut *const T, len: usize) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null()) as *mut T;
    unsafe { Vec::from_raw_parts(ptr, len, len) };
}

pub(super) fn drop_array_mut<T>(ptr: &mut *mut T, len: usize) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null_mut()) as *mut T;
    unsafe { Vec::from_raw_parts(ptr, len, len) };
}