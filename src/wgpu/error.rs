use wgpu::{BufferAsyncError, RequestAdapterError, RequestDeviceError};
use super::util::dev_select::InvalidAdapterError;

use crate::DetectError;

/// Error from initializing WGPU
#[derive(Debug, thiserror::Error)]
pub enum WgpuBuildError {
	#[error("No WGPU adapters found")]
	NoAdapters(Option<RequestAdapterError>),
	/// All the detected adapters wouldn't work
	#[error("All adapters were bad: {0:?}")]
	BadAdapters(Vec<InvalidAdapterError>),
	#[error("Error requesting device")]
	CantRequestDevice(#[from] #[source] RequestDeviceError),
	#[error("Runtime error")]
	RuntimeError(#[source] wgpu::Error),
	/// We need a workgroup that's bigger than what the device supports
	#[error("Workgroup size ({actual_x}×{actual_y}={total}) exceeds device limit ({limit_x}×{limit_y}, max {max_invocations} invocations)")]
	WorkgroupTooLarge {
		actual_x: u32,
		actual_y: u32,
		total: u32,
		limit_x: u32,
		limit_y: u32,
		max_invocations: u32,
	},
}

/// Error from detecting AprilTags on WGPU
#[derive(Debug, thiserror::Error)]
pub enum WgpuDetectError {
	#[expect(private_interfaces)]
	#[error("Unable to map buffer")]
	BufferMap(#[from] GpuBufferFetchError),
}

impl From<WgpuDetectError> for DetectError {
    fn from(_: WgpuDetectError) -> Self {
        Self::Acceleration
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum GpuBufferFetchError {
    #[error("Unable to map buffer")]
    BufferMap(#[from] BufferAsyncError),
    #[error("Buffer is not readable")]
    NotReadable,
}
