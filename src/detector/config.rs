use std::{num::{NonZeroU32, NonZeroUsize}, path::PathBuf, cmp::Ordering};

use crate::AprilTagQuadThreshParams;


/// When building the [AprilTagDetector], what kind of acceleration should we use?
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuAccelRequest {
	/// Do not use GPU acceleration
	Disabled,
	/// Attempt to use acceleration if any device is available
	/// 
	/// This will possibly use OpenCL on the CPU
	Prefer,
	/// Attempt to use acceleration if any GPU is available,
	/// but won't use the CPU
	PreferGpu,
	/// Attempt to use acceleration if device index is available
	PreferDeviceIdx(usize),
	/// Force using acceleration (build error if unavailable)
	Required,
	/// Force using acceleration on some GPU
	RequiredGpu,
	/// Force using a specific OpenCL device (build error if OpenCL or device is unavailable)
	RequiredDeviceIdx(usize),
}

impl GpuAccelRequest {
	pub const fn is_disabled(&self) -> bool {
		matches!(self, Self::Disabled)
	}
	
	/// Is acceleration required?
	/// 
	/// If `true`, emit a detector build error if acceleration is not available
	pub const fn is_required(&self) -> bool {
		match self {
			Self::Required | Self::RequiredGpu | Self::RequiredDeviceIdx(_) => true,
			_ => false,
		}
	}
}

impl Default for GpuAccelRequest {
    fn default() -> Self {
        Self::Prefer
    }
}

#[derive(Debug, Clone)]
pub enum SourceDimensions {
	/// Unknown dimensions
	Dynamic,
	Exactly {
		width: NonZeroUsize,
		height: NonZeroUsize,
	}
}

impl SourceDimensions {
	pub(crate) fn cmp_width(&self, width: usize) -> Option<Ordering> {
		if width == 0 {
			return Some(Ordering::Greater);
		}
		match self {
			SourceDimensions::Dynamic => None,
			SourceDimensions::Exactly { width: width1, .. } => Some(width1.get().cmp(&width))
		}
	}

	pub(crate) fn cmp_height(&self, height: usize) -> Option<Ordering> {
		if height == 0 {
			return Some(Ordering::Greater);
		}
		match self {
			SourceDimensions::Dynamic => None,
			SourceDimensions::Exactly { height: height1, .. } => Some(height1.get().cmp(&height))
		}
	}
}

#[derive(Debug, Clone)]
pub struct DetectorConfig {
	/// How many threads should be used?
	pub nthreads: usize,

	/// Detection of quads can be done on a lower-resolution image,
	/// improving speed at a cost of pose accuracy and a slight
	/// decrease in detection rate. Decoding the binary payload is
	/// still done at full resolution.
	pub quad_decimate: f32,

	/// What Gaussian blur should be applied to the segmented image
	/// (used for quad detection?)  Parameter is the standard deviation
	/// in pixels.  Very noisy images benefit from non-zero values
	/// (e.g. 0.8).
	pub quad_sigma: f32,

	/// When non-zero, the edges of the each quad are adjusted to "snap
	/// to" strong gradients nearby. This is useful when decimation is
	/// employed, as it can increase the quality of the initial quad
	/// estimate substantially. Generally recommended to be on (1).
	///
	/// Very computationally inexpensive. Option is ignored if
	/// quad_decimate = 1.
	pub refine_edges: bool,

	/// How much sharpening should be done to decoded images? This
	/// can help decode small tags but may or may not help in odd
	/// lighting conditions or low light conditions.
	///
	/// The default value is 0.25.
	pub decode_sharpening: f64,

	/// When non-zero, write a variety of debugging images to the
	/// current working directory at various stages through the
	/// detection process. (Somewhat slow).
	pub debug: bool,

	pub qtp: AprilTagQuadThreshParams,

	/// Path to write debug images to
	pub debug_path: Option<PathBuf>,

	pub gpu: GpuAccelRequest,
	pub allow_concurrency: bool,
	/// What size frames should we expect?
	pub source_dimensions: SourceDimensions,
}


impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
			nthreads: 1,
			quad_decimate: 1.0,
			quad_sigma: 0.0,
			refine_edges: true,
			decode_sharpening: 0.25,
			debug: false,
			qtp: AprilTagQuadThreshParams::default(),
			debug_path: None,
			gpu: Default::default(),
			allow_concurrency: true,
			source_dimensions: SourceDimensions::Dynamic,
		}
    }
}

/// Quad decimate algorithm to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QuadDecimateMode {
	/// Special case for 3/2 scaling
	ThreeHalves,
	/// Integer scaling
	Scaled(NonZeroU32),
}

impl DetectorConfig {
	#[cfg(feature="debug")]
	pub(crate) const fn debug(&self) -> bool {
		self.debug
	}

	/// Get algorithm to use for quad decimation
	pub(crate) fn quad_decimate_mode(&self) -> Option<QuadDecimateMode> {
		if self.quad_decimate > 1. {
			if self.quad_decimate == 1.5 {
				Some(QuadDecimateMode::ThreeHalves)
			} else {
				Some(QuadDecimateMode::Scaled(NonZeroU32::new(self.quad_decimate.round() as u32).unwrap()))
			}
		} else {
			None
		}
	}

	/// Should quad_sigma be applied
	pub(crate) fn do_quad_sigma(&self) -> bool {
		self.quad_sigma != 1.
	}

	#[cfg(not(feature="debug"))]
	pub(crate) const fn debug(&self) -> bool {
		false
	}

	/// Should debug images be generated
	#[cfg(feature="debug")]
	pub(crate) const fn generate_debug_image(&self) -> bool {
		self.debug
	}

	/// Should debug images be generated (always false if feature disabled)
	#[cfg(not(feature="debug"))]
	#[inline(always)]
	pub(crate) const fn generate_debug_image(&self) -> bool {
		false
	}

	/// Generate a debug image with the given name.
	#[cfg(feature="debug")]
	#[inline]
	pub(crate) fn debug_image(&self, name: &str, callback: impl FnOnce(std::fs::File) -> std::io::Result<()>) {
		if self.debug {
			let path = if let Some(pfx) = &self.debug_path {
				let mut path = PathBuf::from(pfx);
				path.push(name);
				path
			} else {
				PathBuf::from(name)
			};
			let f = match std::fs::File::create(&path) {
				Ok(f) => f,
				Err(e) => panic!("Unable to create debug file {} ({e:?})", path.display()),
			};
			callback(f)
				.expect(&format!("Error writing {name}"));
		}
	}

	#[cfg(not(feature="debug"))]
	#[inline(always)]
	pub(crate) fn debug_image(&self, name: &str, callback: impl FnOnce(File) -> std::io::Result<()>) {
		
	}

	/// Should the algorithm be run on a single thread?
	pub(crate) const fn single_thread(&self) -> bool {
		self.nthreads <= 1
	}
}