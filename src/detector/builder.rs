use std::{sync::Arc, ops::Deref};

use rayon::ThreadPoolBuildError;

use crate::{AprilTagFamily, quickdecode::QuickDecode, AddFamilyError};
use super::{AprilTagDetector, DetectorConfig, config::GpuAccelRequest};

#[derive(Clone)]
pub struct DetectorBuilder {
	/// Configuration parameters
	pub config: DetectorConfig,
	pub(crate) tag_families: Vec<Arc<QuickDecode>>,
	// static_buffers: Option<(usize, usize, usize)>,
	// dynamic_buffers: bool,
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        Self {
			config: Default::default(),
			tag_families: Default::default(),
			// static_buffers: None,
			// dynamic_buffers: true,
		}
    }
}

impl From<AprilTagDetector> for DetectorBuilder {
    fn from(value: AprilTagDetector) -> Self {
        Self {
            config: value.params,
            tag_families: value.tag_families,
			// static_buffers: None,
			// dynamic_buffers: true,
        }
    }
}

impl DetectorBuilder {
	pub fn new(config: DetectorConfig) -> Self {
		Self {
			config,
			tag_families: Vec::new(),
			// static_buffers: None,
			// dynamic_buffers: true,
		}
	}

	/// Add a family to the apriltag detector.
	/// 
	/// A single instance should only be provided to one apriltag detector instance.
	pub fn add_family_bits(&mut self, family: Arc<AprilTagFamily>, bits_corrected: usize) -> Result<(), AddFamilyError> {
        let qd = QuickDecode::new(family, bits_corrected)?;
		self.tag_families.push(Arc::new(qd));
		Ok(())
	}

	/// Add the family to this builder (reduce copying)
	pub fn with_family(mut self, fam: Arc<AprilTagFamily>, bits_corrected: usize) -> Result<Self, AddFamilyError> {
		self.add_family_bits(fam, bits_corrected)?;
		Ok(self)
	}

	/// Prefer OpenCL, if available
	/// 
	/// This function will work even if the feature-flag `opencl` is not provided
	pub fn prefer_gpu(#[allow(unused_mut)] mut self) -> Self {
		#[cfg(feature="opencl")]
		self.set_gpu_mode(GpuAccelRequest::Prefer);
		self
	}

	pub fn gpu_mode(&self) -> &GpuAccelRequest {
		&self.config.gpu
	}

	pub fn set_gpu_mode(&mut self, mode: GpuAccelRequest) {
		self.config.gpu = mode;
	}

	/// Clear AprilTag families to detect
	pub fn clear_families(&mut self) {
		self.tag_families.clear();
	}

	/// pre-allocate buffers for compute
	pub fn static_buffers(mut self, frame_dims: (usize, usize), concurrency: usize) -> Self {
		let (width, height) = frame_dims;
		// self.static_buffers = Some((width, height, concurrency));
		self
	}

	/// Allow (or not) allocating dynamic buffers
	pub fn dynamic_buffers(mut self, dynamic_allocation: bool) -> Self {
		// self.dynamic_buffers = dynamic_allocation;
		self
	}

	pub fn remove_family(&mut self, fam: &AprilTagFamily) {
		if let Some(idx) = self.tag_families.iter().position(|qd| qd.family.deref().eq(fam)) {
			self.tag_families.remove(idx);
		}
	}

	/// Build a detector with these options
	pub fn build(self) -> Result<AprilTagDetector, DetectorBuildError> {
        AprilTagDetector::new(self.config, self.tag_families)
	}
}

#[derive(Debug)]
#[non_exhaustive]
pub enum DetectorBuildError {
	/// There was an error creating the threadpool
	Threadpool(ThreadPoolBuildError),
	/// There was an error allocating buffers
	BufferAllocationFailure,
	DimensionTooSmall,
	DimensionTooBig,
	/// OpenCL was required, but is not available
	GpuNotAvailable,
	#[cfg(feature="opencl")]
	OpenCLError(ocl::Error),
}

impl From<ThreadPoolBuildError> for DetectorBuildError {
    fn from(value: ThreadPoolBuildError) -> Self {
        Self::Threadpool(value)
    }
}

#[cfg(feature="opencl")]
impl From<ocl::Error> for DetectorBuildError {
    fn from(value: ocl::Error) -> Self {
        Self::OpenCLError(value)
    }
}