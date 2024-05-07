use std::time::{Duration, Instant};

use parking_lot::lock_api::{RawRwLock, RawRwLockRecursive, RawRwLockRecursiveTimed, RawRwLockTimed, RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::{exceptions::PyTimeoutError, PyErr, PyResult};


pub(super) enum PyTimeout {
	Blocking,
	NonBlocking,
	For(Duration),
	Until(Instant),
}

impl PyTimeout {
	fn nonblocking_err() -> PyErr {
		PyErr::new::<PyTimeoutError, _>(format!("Unable to get nonblocking lock"))
	}
	fn timeout_err(timeout: &Duration) -> PyErr {
		PyErr::new::<PyTimeoutError, _>(format!("Lock timed out after {timeout:?}"))
	}
	fn deadline_err(deadline: &Instant) -> PyErr {
		PyErr::new::<PyTimeoutError, _>(format!("Lock timed out after deadline {deadline:?}"))
	}

	
	pub(super) fn read<'a, T, R: RawRwLock + RawRwLockTimed<Duration = Duration, Instant = Instant>>(&self, lock: &'a RwLock<R, T>) -> PyResult<RwLockReadGuard<'a, R, T>> {
		match self {
			PyTimeout::Blocking => Ok(lock.read()),
			PyTimeout::NonBlocking => lock.try_read().ok_or_else(Self::nonblocking_err),
			PyTimeout::For(timeout) => lock.try_read_for(*timeout).ok_or_else(|| Self::timeout_err(timeout)),
			PyTimeout::Until(deadline) => lock.try_read_until(*deadline).ok_or_else(|| Self::deadline_err(deadline)),
		}
	}
	
	pub(super) fn read_recursive<'a, T, R: RawRwLockRecursive + RawRwLockRecursiveTimed<Duration = Duration, Instant = Instant>>(&self, lock: &'a RwLock<R, T>) -> PyResult<RwLockReadGuard<'a, R, T>> {
		match self {
			PyTimeout::Blocking => Ok(lock.read_recursive()),
			PyTimeout::NonBlocking => lock.try_read_recursive().ok_or_else(Self::nonblocking_err),
			PyTimeout::For(timeout) => lock.try_read_recursive_for(*timeout).ok_or_else(|| Self::timeout_err(timeout)),
			PyTimeout::Until(deadline) => lock.try_read_recursive_until(*deadline).ok_or_else(|| Self::deadline_err(deadline)),
		}
	}

	pub(super) fn write<'a, T, R: RawRwLock + RawRwLockTimed<Duration = Duration, Instant = Instant>>(&self, lock: &'a RwLock<R, T>) -> PyResult<RwLockWriteGuard<'a, R, T>> {
		match self {
			PyTimeout::Blocking => Ok(lock.write()),
			PyTimeout::NonBlocking => lock.try_write().ok_or_else(Self::nonblocking_err),
			PyTimeout::For(timeout) => lock.try_write_for(*timeout).ok_or_else(|| Self::timeout_err(timeout)),
			PyTimeout::Until(deadline) => lock.try_write_until(*deadline).ok_or_else(|| Self::deadline_err(deadline)),
		}
	}
}