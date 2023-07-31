use std::{sync::Arc, ops::Deref};

use rayon::ThreadPoolBuildError;

use crate::{AprilTagFamily, quickdecode::QuickDecode, AddFamilyError};
use super::{AprilTagDetector, DetectorConfig};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpenClMode {
	/// Do not use OpenCL
	Disabled,
	/// Attempt to use OpenCL if available
	Prefer,
	/// Attempt to use OpenCL if GPU available
	PreferGpu,
	/// Attempt to use OpenCL if device index is available
	PreferDeviceIdx(usize),
	/// Force using OpenCL (build error if unavailable)
	Required,
	/// Force using OpenCL on some GPU
	RequiredGpu,
	/// Force using a specific OpenCL device (build error if OpenCL or device is unavailable)
	RequiredDeviceIdx(usize),
}

impl OpenClMode {
	pub const fn is_required(&self) -> bool {
		match self {
			Self::Required | Self::RequiredGpu | Self::RequiredDeviceIdx(_) => true,
			_ => false,
		}
	}
}

impl Default for OpenClMode {
    fn default() -> Self {
        Self::Prefer
    }
}

#[derive(Clone)]
pub struct DetectorBuilder {
	/// Configuration parameters
	pub config: DetectorConfig,
	pub(crate) tag_families: Vec<Arc<QuickDecode>>,
	// static_buffers: Option<(usize, usize, usize)>,
	// dynamic_buffers: bool,
	#[cfg(feature="opencl")]
	ocl: OpenClMode,
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        Self {
			config: Default::default(),
			tag_families: Default::default(),
			// static_buffers: None,
			// dynamic_buffers: true,
			#[cfg(feature="opencl")]
			ocl: OpenClMode::default(),
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
			#[cfg(feature="opencl")]
			ocl: match value.ocl {
				Some(ocl) => ocl.mode,
				None => OpenClMode::default(),
			},
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
			#[cfg(feature="opencl")]
			ocl: OpenClMode::default(),
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
	pub fn prefer_opencl(mut self) -> Self {
		#[cfg(feature="opencl")]
		self.use_opencl(OpenClMode::Prefer);
		self
	}

	pub fn opencl_mode(&self) -> &OpenClMode {
		#[cfg(feature="opencl")]
		{
			&self.ocl
		}
		#[cfg(not(feature="opencl"))]
		&OpenClMode::Disabled
	}

	pub fn use_opencl(&mut self, mode: OpenClMode) {
		#[cfg(feature="opencl")]
		{
			self.ocl = mode;
		}
	}

	/// Clear AprilTag families to detect
	pub fn clear_families(&mut self) {
		self.tag_families.clear();
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
	/// OpenCL was required, but is not available
	OpenCLNotAvailable,
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