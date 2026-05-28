use std::{fmt::Debug, num::{NonZeroU32, NonZeroUsize}, ops::{Bound, Div, Range, RangeBounds}, path::PathBuf};

use crate::AprilTagQuadThreshParams;

use super::ImageDimensionError;

/// When building the [AprilTagDetector], what kind of acceleration should we use?
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AccelerationRequest {
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

impl AccelerationRequest {
	/// Is acceleration disallowed?
	pub const fn is_disabled(&self) -> bool {
		matches!(self, Self::Disabled)
	}

	/// Are high-powered devices (GPUs) preferred over lower-powered devices?
	pub const fn prefer_high_power(&self) -> bool {
		match self {
			Self::PreferGpu | Self::RequiredGpu => true,
			_ => false,
		}
	}

	/// If no GPU is available, should we fall back to CPU-with-acceleration?
	pub const fn allow_cpu(&self) -> bool {
		match self {
			Self::Prefer | Self::Required => true,
			_ => false,
		}
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

impl Default for AccelerationRequest {
    fn default() -> Self {
        Self::Prefer
    }
}

#[derive(Debug, Clone)]
pub enum SourceDimensions {
	/// Unknown dimensions
	Dynamic,
	/// All frames will be exactly this size
	Exactly {
		width: usize,
		height: usize,
	},
	Range {
		width: Range<Bound<usize>>,
		height: Range<Bound<usize>>,
	}
}

impl SourceDimensions {
	pub(crate) fn width_range(&self) -> Range<Bound<usize>> {
		match self {
			SourceDimensions::Dynamic => Range { start: Bound::Unbounded, end: Bound::Unbounded },
			SourceDimensions::Exactly { width, .. } => Range { start: Bound::Included(*width), end: Bound::Included(*width) },
			SourceDimensions::Range { width, .. } => width.clone(),
		}
	}
	pub(crate) fn height_range(&self) -> Range<Bound<usize>> {
		match self {
			SourceDimensions::Dynamic => Range { start: Bound::Unbounded, end: Bound::Unbounded },
			SourceDimensions::Exactly { height, .. } => Range { start: Bound::Included(*height), end: Bound::Included(*height) },
			SourceDimensions::Range { height, .. } => height.clone(),
		}
	}

	pub(crate) fn check(&self, width_bounds: impl RangeBounds<usize>, height_bounds: impl RangeBounds<usize>) -> Result<(), ImageDimensionError> {
		match self {
			Self::Dynamic => {},
			Self::Exactly { width, height } => {
				let width = *width;
				match width_bounds.start_bound() {
					Bound::Excluded(v) => if &width <= v { return Err(ImageDimensionError::WidthTooSmall { actual: width, minimum: v + 1 }); }
					Bound::Included(v) => if &width < v { return Err(ImageDimensionError::WidthTooSmall { actual: width, minimum: *v }); }
					Bound::Unbounded => {}
				}
				match width_bounds.end_bound() {
					Bound::Excluded(v) => if &width >= v { return Err(ImageDimensionError::WidthTooBig { actual: width, maximum: v - 1 }); }
					Bound::Included(v) => if &width > v { return Err(ImageDimensionError::WidthTooBig { actual: width, maximum: *v }); }
					Bound::Unbounded => {}
				}

				let height = *height;
				match height_bounds.start_bound() {
					Bound::Excluded(v) => if &height <= v { return Err(ImageDimensionError::HeightTooSmall { actual: height, minimum: v + 1 }); }
					Bound::Included(v) => if &height < v { return Err(ImageDimensionError::HeightTooSmall { actual: height, minimum: *v }); }
					Bound::Unbounded => {}
				}
				match height_bounds.end_bound() {
					Bound::Excluded(v) => if &height >= v { return Err(ImageDimensionError::HeightTooBig { actual: height, maximum: v - 1 }); }
					Bound::Included(v) => if &height > v { return Err(ImageDimensionError::HeightTooBig { actual: height, maximum: *v }); }
					Bound::Unbounded => {}
				}
			}
			Self::Range { .. } => todo!(),
		}
		Ok(())
	}
	
	pub(crate) fn div_ceil(&self, dw: usize, dh: usize) -> Self {
		match self {
			Self::Dynamic => Self::Dynamic,
			Self::Exactly { width, height } => Self::Exactly { width: width.div_ceil(dw), height: height.div(dh) },
			Self::Range { width, height } => {
				fn div_bound(bound: &Bound<usize>, divisor: usize) -> Bound<usize> {
					match bound {
						Bound::Included(v) => Bound::Included(v.div_ceil(divisor)),
						Bound::Excluded(v) => Bound::Excluded(v.div_ceil(divisor)),
						Bound::Unbounded => Bound::Unbounded,
					}
				}
				fn div_range(range: &Range<Bound<usize>>, divisor: usize) -> Range<Bound<usize>> {
					Range {
						start: div_bound(&range.start, divisor),
						end: div_bound(&range.end, divisor),
					}
				}
				Self::Range {
					width: div_range(width, dw),
					height: div_range(height, dh),
				}
			}
		}
	}
}

/// Configuration for [AprilTagDetector]
#[derive(Debug, Clone)]
pub struct DetectorConfig {
	/// How many threads should be used?
	/// - Zero results in autodetection
	/// - One will be single-threaded
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

	/// What kind of hardware acceleration should we use?
	pub acceleration: AccelerationRequest,
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
			acceleration: Default::default(),
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
	pub(crate) fn debug_image<E: Debug>(&self, name: &str, callback: impl FnOnce(std::fs::File) -> Result<(), E>) {
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
		self.nthreads == 1
	}

	pub(crate) fn nthreads(&self) -> NonZeroUsize {
		match NonZeroUsize::new(self.nthreads) {
			Some(v) => v,
			None => NonZeroUsize::new(rayon::max_num_threads()).expect("rayon::max_num_threads() is zero"),
		}
	}
}