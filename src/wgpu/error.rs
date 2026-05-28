use wgpu::{BufferAsyncError, RequestDeviceError};
use super::util::dev_select::InvalidAdapterError;

use crate::DetectError;

#[derive(Debug, thiserror::Error)]
pub enum WgpuBuildError {
	#[error("No WGPU adapters found")]
	NoAdapters,
	#[error("All adapters were bad: {0:?}")]
	BadAdapters(Vec<InvalidAdapterError>),
	#[error("Error requesting device")]
	CantRequestDevice(#[from] #[source] RequestDeviceError),
	#[error("Runtime error")]
	RuntimeError(#[source] wgpu::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum WgpuDetectError {
	#[error("Unable to map buffer")]
	BufferMap(#[from] GpuBufferFetchError),
}

impl From<WgpuDetectError> for DetectError {
    fn from(value: WgpuDetectError) -> Self {
        todo!()
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum GpuBufferFetchError {
    #[error("Unable to map buffer")]
    BufferMap(#[from] BufferAsyncError),
    #[error("Buffer is not readable")]
    NotReadable,
}
