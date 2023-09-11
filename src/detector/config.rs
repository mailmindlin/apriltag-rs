use std::{num::NonZeroU32, path::PathBuf};

use crate::AprilTagQuadThreshParams;

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

	/// Path to debug to
	pub debug_path: Option<PathBuf>,
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
	pub(crate) fn debug_image(&self, name: &str, callback: impl FnOnce(File) -> io::Result<()>) {
		
	}

	/// Should the algorithm be run on a single thread?
	pub(crate) const fn single_thread(&self) -> bool {
		self.nthreads <= 1
	}
}