//! Utilities for safely managing Rust-owned heap allocations across the FFI boundary.
//!
//! The core types — [`ManagedPtr`] and [`AtomicManagedPtr`] — store smart pointers
//! (`Arc`, `Box`) as raw pointers inside `#[repr(C)]` structs so that C callers can
//! hold opaque handles to Rust objects. Ownership is reclaimed automatically on drop
//! or explicitly via [`ManagedPtr::take`] / [`AtomicManagedPtr::take`].

use std::{cell::UnsafeCell, marker::PhantomData, mem, ops::Deref, panic::RefUnwindSafe, ptr, sync::{atomic::AtomicPtr, Arc}};

/// Abstraction over smart pointer types (`Arc`, `Box`) that can be converted
/// to and from raw pointers. Used by [`ManagedPtr`] and [`AtomicManagedPtr`]
/// to manage ownership of heap-allocated values across the FFI boundary.
pub trait Container: Deref {
    /// Consumes the container and returns a raw pointer to the managed value.
    fn into_raw(this: Self) -> *const <Self as Deref>::Target;
    /// Reconstructs the container from a raw pointer previously obtained via [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    /// The pointer must have been produced by `into_raw` on the same container type,
    /// and must not have already been reclaimed.
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

/// A smart-pointer wrapper that stores a [`Container`] (e.g. `Arc<T>` or `Box<T>`) as a raw
/// pointer, suitable for embedding in `#[repr(C)]` FFI structs.
///
/// The optional type parameter `F` allows the stored pointer to be cast to a different
/// type than the container's target (e.g. an opaque FFI handle type), while still
/// reconstructing the correct `C` on drop/take.
///
/// Ownership semantics:
/// - [`from`](Self::from) — takes ownership of a container, storing it as a raw pointer.
/// - [`take`](Self::take) — reclaims ownership, leaving the pointer null.
/// - [`borrow`](Self::borrow) — clones the container without giving up ownership.
/// - On [`Drop`], any non-null pointer is reclaimed and freed.
/// cbindgen:no-export
#[repr(transparent)]
pub struct ManagedPtr<C: Container, F=<C as Deref>::Target> where <C as Deref>::Target: Sized {
    pub ptr: UnsafeCell<*const F>,
    container: PhantomData<C>,
}

//TODO: check for correctness
impl <C: Container, F> RefUnwindSafe for ManagedPtr<C, F>
    where
        <C as Deref>::Target: Sized,
        C: RefUnwindSafe,
        F: RefUnwindSafe {}

impl <C: Container, F> Default for ManagedPtr<C, F> where <C as Deref>::Target: Sized {
    fn default() -> Self {
        Self {
            ptr: UnsafeCell::new(ptr::null()),
            container: Default::default()
        }
    }
}

impl<C: Container, F> ManagedPtr<C, F> where <C as Deref>::Target: Sized {
    /// Wraps an already-raw pointer without taking ownership of a container.
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid for the lifetime of this
    /// `ManagedPtr`, or is null.
    pub(super) const unsafe fn wrap_ptr(ptr: *const F) -> Self {
        Self {
            ptr: UnsafeCell::new(ptr),
            container: PhantomData,
        }
    }
    /// Takes ownership of `value`, converting it to a raw pointer stored internally.
    pub(super) fn from(value: C) -> Self {
        let ptr = C::into_raw(value) as *const F;
        unsafe { Self::wrap_ptr(ptr) }
    }
    /// Creates a `ManagedPtr` with a null pointer (no owned value).
    pub(super) const fn null() -> Self {
        unsafe { Self::wrap_ptr(ptr::null()) }
    }
    /// Attempts to reconstruct a `C` from the raw pointer, returning `None` if null.
    fn unwrap_ptr(ptr: *const F) -> Option<C> {
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { C::from_raw(ptr as *const <C as Deref>::Target) })
        }
    }
    /// Clones the owned value without releasing ownership.
    ///
    /// Temporarily reclaims the container to clone it, then stores the raw
    /// pointer again. Returns `None` if the pointer is null.
    pub(super) fn borrow(&self) -> Option<C> where C: Clone {
        let ptr = mem::replace(unsafe { self.ptr.get().as_mut().unwrap() }, ptr::null_mut());
        let value = Self::unwrap_ptr(ptr)?;
        let res = value.clone();
        unsafe { *self.ptr.get() = C::into_raw(value) as _; }
        Some(res)
    }
    /// Takes the owned value out of this pointer, leaving it null.
    /// Returns `None` if the pointer was already null.
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

/// Thread-safe variant of [`ManagedPtr`] backed by an [`AtomicPtr`].
///
/// All pointer swaps use `SeqCst` ordering. Like `ManagedPtr`, the container
/// is reclaimed on [`Drop`] if the pointer is non-null.
/// cbindgen:no-export
#[repr(transparent)]
pub(super) struct AtomicManagedPtr<C: Container, F=<C as Deref>::Target> where <C as Deref>::Target: Sized {
    pub ptr: AtomicPtr<F>,
    container: PhantomData<C>,
}

impl<C: Container, F> AtomicManagedPtr<C, F> where <C as Deref>::Target: Sized {
    /// Wraps a raw mutable pointer in an atomic managed pointer.
    pub(super) const fn wrap(ptr: *mut <C as Deref>::Target) -> Self {
        Self {
            ptr: AtomicPtr::new(ptr as *mut F),
            container: PhantomData,
        }
    }

    /// Takes ownership of `value`, storing it as an atomic raw pointer.
    pub(super) fn from(value: C) -> Self {
        Self::wrap(C::into_raw(value) as *mut <C as Deref>::Target)
    }

    /// Creates an `AtomicManagedPtr` with a null pointer (no owned value).
    pub(super) const fn null() -> Self {
        Self::wrap(ptr::null_mut())
    }

    /// Atomically loads the current raw pointer (`SeqCst`).
    fn load(&self) -> *mut <C as Deref>::Target {
        self.ptr.load(std::sync::atomic::Ordering::SeqCst) as _
    }

    /// Returns `true` if the stored pointer is null.
    pub(super) fn is_null(&self) -> bool {
        self.load().is_null()
    }

    /// Atomically clones the owned value without releasing ownership.
    ///
    /// Briefly takes and re-stores the value, so concurrent access may
    /// observe a null pointer during the clone.
    pub(super) fn borrow(&self) -> Option<C> where C: Clone {
        let value = self.take()?;
        self.swap(value.clone());
        Some(value)
    }
    /// Attempts to reconstruct a `C` from the raw pointer, returning `None` if null.
    fn unwrap_ptr(ptr: *mut F) -> Option<C> {
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { C::from_raw(ptr as *const <C as Deref>::Target) })
        }
    }
    /// Atomically takes the owned value, replacing it with null.
    /// Returns `None` if the pointer was already null.
    pub(super) fn take(&self) -> Option<C> {
        let prev = self.ptr.swap(ptr::null_mut(), std::sync::atomic::Ordering::SeqCst);

        Self::unwrap_ptr(prev)
    }
    /// Atomically stores `value` and returns the previously stored value (if any).
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

/// Drops a `Box`-allocated value behind a mutable raw pointer, setting the pointer to null.
///
/// No-op if the pointer is already null.
pub(super) fn drop_boxed_mut<T>(ptr: &mut *mut T) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null_mut());
    let boxed = unsafe { Box::from_raw(ptr) };
    drop(boxed);
}

/// Drops a `Vec`-allocated array behind a const raw pointer, setting it to null.
///
/// Reconstructs a `Vec` with the given `len` (used as both length and capacity)
/// so that each element is properly dropped. No-op if the pointer is already null.
pub(super) fn drop_array<T>(ptr: &mut *const T, len: usize) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null()) as *mut T;
    unsafe { Vec::from_raw_parts(ptr, len, len) };
}

/// Drops a `Vec`-allocated array behind a mutable raw pointer, setting it to null.
///
/// Same as [`drop_array`] but for `*mut T` pointers.
pub(super) fn drop_array_mut<T>(ptr: &mut *mut T, len: usize) {
    if ptr.is_null() {
        return;
    }
    let ptr = std::mem::replace(ptr, std::ptr::null_mut());
    unsafe { Vec::from_raw_parts(ptr, len, len) };
}