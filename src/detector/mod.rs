mod builder;
mod config;
mod debug;

pub use config::DetectorConfig;
pub use builder::{DetectorBuilder, DetectorBuildError, OpenClMode};

use std::sync::{Mutex, Arc};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

use crate::{util::{geom::Point2D, math::{Vec2, Vec2Builder}, image::{ImageWritePNM, Pixel, ImageY8}}, quickdecode::QuickDecode, quad_decode::QuadDecodeInfo, quad_thresh::apriltag_quad_thresh, dbg::TimeProfile, Detections, detection::reconcile_detections};

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DetectError {
	ImageTooSmall,
	AllocError,
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
		let mut quad_im = if self.params.quad_decimate > 1. {
			let quad_im = im_orig.decimate(self.params.quad_decimate);

			tp.stamp("decimate");
			quad_im
		} else {
			//TODO: can we not copy here?
			im_orig.clone()
		};

		quad_sigma(&mut quad_im, self.params.quad_sigma);

		tp.stamp("blur/sharp");

		#[cfg(feature="debug")]
		self.params.debug_image("debug_preprocess.pnm", |mut f| quad_im.write_pnm(&mut f));

		let mut quads = apriltag_quad_thresh(self, &mut tp, &quad_im);
		std::mem::drop(quad_im);

		#[cfg(feature="extra_debug")]
		println!("Found {} quads", quads.len());

		// adjust centers of pixels so that they correspond to the
		// original full-resolution image.
		if self.params.quad_decimate > 1. {
			let quad_decimate = self.params.quad_decimate as f64;
			for q in quads.iter_mut() {
				for corner in q.corners.iter_mut() {
					if self.params.quad_decimate == 1.5 {
						*corner.vec_mut() *= quad_decimate;
					} else {
						let half = Vec2::dup(0.5);
						*corner = Point2D::from_vec((corner.vec() - &half) * quad_decimate + &half);
					}
				}
			}
		}

		let nquads: u32 = quads.len().try_into().unwrap();

		tp.stamp("quads");

		#[cfg(feature="debug")]
		self.params.debug_image("debug_quads.pnm", |f| debug::debug_quads(f, ImageY8::clone_like(im_orig), &quads));

		////////////////////////////////////////////////////////////////
		// Step 2. Decode tags from each quad.
		let detections = if true {
			#[cfg(feature="debug")]
			let im_samples = if self.params.generate_debug_image() { Some(Mutex::new(ImageY8::clone_like(im_orig))) } else { None };

			let detections = if quads.len() > 4 && !self.params.single_thread() {
				self.wp.install(|| {
					quads
						.par_iter_mut()
						.flat_map(|quad| {
							quad.decode_task(QuadDecodeInfo {
								det_params: &self.params,
								tag_families: &self.tag_families,
								im_orig,
								#[cfg(feature="debug")]
								im_samples: im_samples.as_ref(),
							})
						})
						.collect::<Vec<_>>()
				})
			} else {
				quads
					.iter_mut()
					.flat_map(|quad| {
						quad.decode_task(QuadDecodeInfo {
							det_params: &self.params,
							tag_families: &self.tag_families,
							im_orig,
							#[cfg(feature="debug")]
							im_samples: im_samples.as_ref(),
						})
					})
					.collect::<Vec<_>>()
			};

			#[cfg(feature="extra_debug")]
			println!("Found {} detections", detections.len());

			#[cfg(feature="debug")]
			if let Some(im_samples) = im_samples {
				let im_samples = im_samples.into_inner().unwrap();
				im_samples.save_to_pnm("debug_samples.pnm").unwrap();
			}
			detections
		} else {
			Vec::new()
		};

		#[cfg(feature="debug")]
        self.params.debug_image("debug_quads_fixed.pnm", |f| debug::debug_quads_fixed(f, ImageY8::clone_like(im_orig), &quads));
        #[cfg(feature="debug")]
        self.params.debug_image("debug_quads.ps", |f| debug::debug_quads_ps(f, ImageY8::clone_like(im_orig), &quads));
		std::mem::drop(quads);

		tp.stamp("decode+refinement");

		let mut detections = reconcile_detections(detections);

		tp.stamp("reconcile");

		////////////////////////////////////////////////////////////////
		// Produce final debug output
		#[cfg(feature="debug")]
        self.params.debug_image("debug_output.ps", |f| debug::debug_output_ps(f, ImageY8::clone_like(im_orig), &detections));
        #[cfg(feature="debug")]
        self.params.debug_image("debug_output.pm,", |f| debug::debug_output_pnm(f, ImageY8::clone_like(im_orig), &detections));

		tp.stamp("debug output");

		detections.sort_by(|a, b| Ord::cmp(&a.id, &b.id));

		tp.stamp("cleanup");

		Ok(Detections {
			tp,
			nquads,
			detections,
		})
	}
}