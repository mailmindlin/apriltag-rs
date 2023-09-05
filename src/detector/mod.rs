mod builder;
mod config;
#[cfg(feature="debug")]
mod debug;

pub use config::DetectorConfig;
pub use builder::{DetectorBuilder, DetectorBuildError, OpenClMode};

use std::sync::{Mutex, Arc};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

use crate::{util::{math::{Vec2, Vec2Builder}, image::{ImageWritePNM, Pixel, ImageY8}}, quickdecode::QuickDecode, quad_decode::QuadDecodeInfo, quad_thresh::apriltag_quad_thresh, dbg::TimeProfile, Detections, detection::reconcile_detections, detector::config::QuadDecimateMode};

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DetectError {
	/// Input image was too small
	ImageTooSmall,
	/// Input image was too large
	ImageTooBig,
	/// Buffer allocation error
	AllocError,
	OpenCLError,
}

pub struct AprilTagDetector {
	pub params: DetectorConfig,

	///////////////////////////////////////////////////////////////
	// Internal variables below

	// Not freed on apriltag_destroy; a tag family can be shared
	// between multiple users. The user should ultimately destroy the
	// tag family passed into the constructor.
	pub(crate) tag_families: Vec<Arc<QuickDecode>>,

	// Used to manage multi-threading.
	pub(crate) wp: ThreadPool,
}

fn quad_sigma(img: &mut ImageY8, quad_sigma: f32) {
	if quad_sigma == 0. {
		return;
	}
	// compute a reasonable kernel width by figuring that the
	// kernel should go out 2 std devs.
	//
	// max sigma          ksz
	// 0.499              1  (disabled)
	// 0.999              3
	// 1.499              5
	// 1.999              7

	let sigma = f32::abs(quad_sigma);

	let ksz = (4. * sigma) as usize; // 2 std devs in each direction
	let ksz = if (ksz & 1) == 0 {
		ksz + 1
	} else {
		ksz
	};

	if ksz > 1 {
		if quad_sigma > 0. {
			// Apply a blur
			img.gaussian_blur(sigma as f64, ksz);
		} else {
			// SHARPEN the image by subtracting the low frequency components.
			let orig = img.clone();
			img.gaussian_blur(sigma as f64, ksz);

			for ((x, y), vorig) in orig.enumerate_pixels() {
				let vblur = img[(x, y)];

				// Prevent overflow
				let v = ((vorig.to_value() as i16) * 2).saturating_sub(vblur as i16);

				img[(x, y)] = (v.clamp(0, 255) as u8).into();
			}
		}
	}
}



impl AprilTagDetector {
	/// Create a new builder
	pub fn builder() -> DetectorBuilder {
		DetectorBuilder::default()
	}

	fn new(params: DetectorConfig, tag_families: Vec<Arc<QuickDecode>>) -> Result<AprilTagDetector, DetectorBuildError> {
		let tpb = ThreadPoolBuilder::new()
			.num_threads(params.nthreads);

		let wp = match tpb.build() {
			Ok(wp) => wp,
			Err(e) => return Err(DetectorBuildError::Threadpool(e)),
		};
		
		Ok(Self {
			params,
			tag_families,
			wp,
		})
	}

	pub fn detect(&self, im_orig: &ImageY8) -> Result<Detections, DetectError> {
		if self.tag_families.len() == 0 {
			println!("AprilTag: No tag families enabled.");
			return Ok(Detections::default());
		}

		// Statistics relating to last processed frame
		let mut tp = TimeProfile::default();

		tp.stamp("init");

		///////////////////////////////////////////////////////////
		// Step 1. Detect quads according to requested image decimation
		// and blurring parameters.
		let mut quad_im = match self.params.quad_decimate_mode() {
			None => {
				//TODO: can we not copy here?
				im_orig.clone()
			},
			Some(QuadDecimateMode::ThreeHalves) => {
				let quad_im = im_orig.decimate_three_halves();
				tp.stamp("decimate");
				quad_im
			},
			Some(QuadDecimateMode::Scaled(quad_decimate)) => {
				let quad_im = im_orig.decimate(quad_decimate.get() as _);
				tp.stamp("decimate");
				quad_im
			}
		};

		quad_sigma(&mut quad_im, self.params.quad_sigma);

		tp.stamp("blur/sharp");

		#[cfg(feature="debug")]
		self.params.debug_image("01_debug_preprocess.pnm", |mut f| quad_im.write_pnm(&mut f));

		let mut quads = apriltag_quad_thresh(self, &mut tp, quad_im.as_ref());
		drop(quad_im);

		#[cfg(feature="extra_debug")]
		println!("Found {} quads", quads.len());

		// adjust centers of pixels so that they correspond to the
		// original full-resolution image.
		match self.params.quad_decimate_mode() {
			None => {},
			Some(QuadDecimateMode::ThreeHalves) => {
				for q in quads.iter_mut() {
					q.corners *= 1.5;
				}
			},
			Some(QuadDecimateMode::Scaled(scale)) => {
				let half = Vec2::dup(0.5);
				for q in quads.iter_mut() {
					q.corners -= half;
					q.corners *= scale.get() as f64;
					q.corners += half;
				}
			}
		}

		let nquads: u32 = quads.len()
			.try_into()
			.unwrap();

		tp.stamp("quads");

		#[cfg(feature="debug")]
		self.params.debug_image("07_debug_quads.pnm", |f| debug::debug_quads(f, ImageY8::clone_like(im_orig), &quads));

		////////////////////////////////////////////////////////////////
		// Step 2. Decode tags from each quad.
		let detections = if true {
			#[cfg(feature="debug")]
			let im_samples = if self.params.generate_debug_image() { Some(Mutex::new(ImageY8::clone_like(im_orig))) } else { None };
			
			let info = QuadDecodeInfo {
				det_params: &self.params,
				tag_families: &self.tag_families,
				im_orig,
				#[cfg(feature="debug")]
				im_samples: im_samples.as_ref(),
			};

			let detections = if quads.len() > 999 && !self.params.single_thread() {
				self.wp.install(|| {
					#[cfg(feature="debug")]
					let quad_iter = quads.par_iter_mut();
					#[cfg(not(feature="debug"))]
					let quad_iter = quads.into_par_iter();

					quad_iter
						.flat_map(|#[allow(unused_mut)] mut quad| quad.decode_task(info))
						.collect::<Vec<_>>()
				})
			} else {
				#[cfg(feature="debug")]
				let quad_iter = quads.iter_mut();
				#[cfg(not(feature="debug"))]
				let quad_iter = quads.into_iter();
				let mut dets = Vec::new();

				for quad in quad_iter {
					dets.extend(quad.decode_task(info));
				}
				dets
			};

			#[cfg(feature="extra_debug")]
			println!("Found {} detections", detections.len());

			#[cfg(feature="debug")]
			if let Some(im_samples) = im_samples {
				let im_samples = im_samples.into_inner().unwrap();
				self.params.debug_image("08_debug_samples.pnm", |mut f| im_samples.write_pnm(&mut f));
			}
			detections
		} else {
			Vec::new()
		};

		tp.stamp("decode+refinement");

		#[cfg(feature="debug")]
		if self.params.generate_debug_image() {
			self.params.debug_image("09a_debug_quads_fixed.pnm", |f| debug::debug_quads_fixed(f, ImageY8::clone_like(im_orig), &quads));
			#[cfg(feature="debug_ps")]
			self.params.debug_image("09b_debug_quads.ps", |f| debug::debug_quads_ps(f, ImageY8::clone_like(im_orig), &quads));
			tp.stamp("decode+refinement (output)");
		}
		drop(quads);

		let mut detections = reconcile_detections(detections);

		tp.stamp("reconcile");

		////////////////////////////////////////////////////////////////
		// Produce final debug output
		#[cfg(feature="debug")]
		{
			#[cfg(feature="debug_ps")]
			self.params.debug_image("10a_debug_output.ps", |f| debug::debug_output_ps(f, ImageY8::clone_like(im_orig), &detections));
			self.params.debug_image("10b_debug_output.pnm", |f| debug::debug_output_pnm(f, ImageY8::clone_like(im_orig), &detections));
		}
		tp.stamp("debug output");

		detections.sort_unstable_by(|a, b| Ord::cmp(&a.id, &b.id));

		tp.stamp("cleanup");

		#[cfg(feature="compare_reference")]
		{
			use crate::sys::{AprilTagDetectorSys, ImageU8Sys, ZArraySys};
			let (td_sys, fams_sys) = AprilTagDetectorSys::new_with_families(self).unwrap();

			let im_sys = ImageU8Sys::new(im_orig).unwrap();
			let dets = ZArraySys::<*mut apriltag_sys::apriltag_detection>::wrap(unsafe { apriltag_sys::apriltag_detector_detect(td_sys.as_ptr(), im_sys.as_ptr()) }).unwrap();

			assert_eq!(td_sys.as_ref().nquads, nquads);

			drop(td_sys);
			println!("sys dets: {dets:?}");
			drop(fams_sys);
		}

		Ok(Detections {
			tp,
			nquads,
			detections,
		})
	}
}