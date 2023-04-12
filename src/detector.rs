use std::{cmp::Ordering, sync::{Arc, Mutex}};

use rand::thread_rng;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::{util::{TimeProfile, Image, geom::{Point2D, Poly2D}, math::mat::Mat, color::RandomColor, image::{ImageWritePNM, ImageWritePostscript, PostScriptWriter, ImageBuffer, ImageY8, Pixel, ImageMut, HasDimensions, pixel::PixelConvert}}, families::AprilTagFamily, quickdecode::{QuickDecode, AddFamilyError}, quad_decode::{QuadDecodeInfo, Quad}, quad_thresh::{ApriltagQuadThreshParams, apriltag_quad_thresh}};

pub struct AprilTagParams {
	/// How many threads should be used?
	pub nthreads: usize,

	/// detection of quads can be done on a lower-resolution image,
	/// improving speed at a cost of pose accuracy and a slight
	/// decrease in detection rate. Decoding the binary payload is
	/// still done at full resolution. .
	pub quad_decimate: f32,

	// What Gaussian blur should be applied to the segmented image
	// (used for quad detection?)  Parameter is the standard deviation
	// in pixels.  Very noisy images benefit from non-zero values
	// (e.g. 0.8).
	pub quad_sigma: f32,

	// When non-zero, the edges of the each quad are adjusted to "snap
	// to" strong gradients nearby. This is useful when decimation is
	// employed, as it can increase the quality of the initial quad
	// estimate substantially. Generally recommended to be on (1).
	//
	// Very computationally inexpensive. Option is ignored if
	// quad_decimate = 1.
	pub refine_edges: bool,

	/// How much sharpening should be done to decoded images? This
	/// can help decode small tags but may or may not help in odd
	/// lighting conditions or low light conditions.
	///
	/// The default value is 0.25.
	pub decode_sharpening: f64,

	// When non-zero, write a variety of debugging images to the
	// current working directory at various stages through the
	// detection process. (Somewhat slow).
	pub debug: bool,
}

impl Default for AprilTagParams {
    fn default() -> Self {
        Self {
			nthreads: 1,
			quad_decimate: 1.0,
			quad_sigma: 0.0,
			refine_edges: true,
			decode_sharpening: 0.25,
			debug: false,
		}
    }
}

impl AprilTagParams {
	#[cfg(feature="debug")]
	pub(crate) const fn generate_debug_image(&self) -> bool {
		self.debug
	}

	#[cfg(not(feature="debug"))]
	pub(crate) const fn generate_debug_image(&self) -> bool {
		false
	}
}

pub struct ApriltagDetector {
	pub params: AprilTagParams,

	pub qtp: ApriltagQuadThreshParams,

	///////////////////////////////////////////////////////////////
	// Internal variables below

	// Not freed on apriltag_destroy; a tag family can be shared
	// between multiple users. The user should ultimately destroy the
	// tag family passed into the constructor.
	pub(crate) tag_families: Vec<QuickDecode>,

	// Used to manage multi-threading.
	pub(crate) wp: ThreadPool,
}

impl Default for ApriltagDetector {
    fn default() -> Self {
        Self {
			params: AprilTagParams::default(),
			qtp: ApriltagQuadThreshParams::default(),
			tag_families: Vec::new(),
			// NB: defer initialization of self.wp so that the user can
			// override self.nthreads.
			wp: ThreadPoolBuilder::new().build().unwrap(),//TODO
		}
    }
}

impl ApriltagDetector {
	pub fn remove_family(&mut self, fam: &AprilTagFamily) {
		// quick_decode_uninit(fam);
		if let Some(idx) = self.tag_families.iter().position(|qd| qd.family.as_ref() == fam) {
			self.tag_families.remove(idx);
		}
	}

	/// Add a family to the apriltag detector.
	/// 
	/// A single instance should only be provided to one apriltag detector instance.
	pub fn add_family_bits(&mut self, fam: Arc<AprilTagFamily>, bits_corrected: usize) -> Result<(), AddFamilyError> {
		let qd = QuickDecode::init(fam, bits_corrected)?;
		self.tag_families.push(qd);
		Ok(())
	}

	pub fn clear_families(&mut self) {
		self.tag_families.clear();
	}

	pub fn detect(&self, im_orig: &ImageY8) -> Detections {
		if self.tag_families.len() == 0 {
			println!("AprilTag: No tag families enabled.");
			return Detections::default();
		}

		// Statistics relating to last processed frame
		let mut tp = TimeProfile::default();

		tp.stamp("init");

		///////////////////////////////////////////////////////////
		// Step 1. Detect quads according to requested image decimation
		// and blurring parameters.
		let mut quad_im = if self.params.quad_decimate > 1. {
			println!("Decimate value {}", self.params.quad_decimate);
			let quad_im = im_orig.decimate(self.params.quad_decimate);

			tp.stamp("decimate");
			quad_im
		} else {
			//TODO: can we not copy here?
			im_orig.clone()
		};

		if self.params.quad_sigma != 0. {
			// compute a reasonable kernel width by figuring that the
			// kernel should go out 2 std devs.
			//
			// max sigma          ksz
			// 0.499              1  (disabled)
			// 0.999              3
			// 1.499              5
			// 1.999              7

			let sigma = f32::abs(self.params.quad_sigma);

			let ksz = (4. * sigma) as usize; // 2 std devs in each direction
			let ksz = if (ksz & 1) == 0 {
				ksz + 1
			} else {
				ksz
			};

			if ksz > 1 {
				if self.params.quad_sigma > 0. {
					// Apply a blur
					quad_im.gaussian_blur(sigma as f64, ksz);
				} else {
					// SHARPEN the image by subtracting the low frequency components.
					let orig = quad_im.clone();
					quad_im.gaussian_blur(sigma as f64, ksz);

					for ((x, y), vorig) in orig.enumerate_pixels() {
						let vblur = quad_im[(x, y)];

						// Prevent overflow
						let v = ((vorig.to_value() as i16) * 2).saturating_sub(vblur as i16);

						quad_im[(x, y)] = (v.clamp(0, 255) as u8).into();
					}
				}
			}
		}

		tp.stamp("blur/sharp");

		if self.params.generate_debug_image() {
			quad_im.save_to_pnm("debug_preprocess.pnm").unwrap();
		}

		let mut quads = apriltag_quad_thresh(self, &mut tp, &quad_im);
		std::mem::drop(quad_im);

		println!("Found {} quads", quads.len());

		// adjust centers of pixels so that they correspond to the
		// original full-resolution image.
		if self.params.quad_decimate > 1. {
			for q in quads.iter_mut() {
				for i in 0..4 {
					*q.corners[i].vec_mut() *= self.params.quad_decimate as f64;
				}
			}
		}

		let nquads: u32 = quads.len().try_into().unwrap();

		tp.stamp("quads");

		if self.params.debug {
			let mut im_quads = im_orig.clone();
			im_quads.darken();
			im_quads.darken();

			let mut rng = thread_rng();

			for quad in quads.iter() {
				let color = rng.gen_color_gray(100).into();

				im_quads.draw_line(quad.corners[0], quad.corners[1], &color, 1);
				im_quads.draw_line(quad.corners[1], quad.corners[2], &color, 1);
				im_quads.draw_line(quad.corners[2], quad.corners[3], &color, 1);
				im_quads.draw_line(quad.corners[3], quad.corners[0], &color, 1);
			}

			im_quads.save_to_pnm("debug_quads_raw.pnm").unwrap();
		}

		////////////////////////////////////////////////////////////////
		// Step 2. Decode tags from each quad.
		let detections = if true {
			let im_samples = if self.params.generate_debug_image() { Some(Mutex::new(im_orig.clone())) } else { None };

			println!("Has im_samples: {}", im_samples.is_some());

			let detections = self.wp.install(|| {
				quads.iter_mut()
					.flat_map(|quad| {
						quad.decode_task(QuadDecodeInfo {
							det_params: &self.params,
							tag_families: &self.tag_families,
							im: im_orig,
							im_samples: im_samples.as_ref(),
						})
					})
					.collect::<Vec<_>>()
			});

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

		if self.params.generate_debug_image() {
			let mut im_quads = im_orig.clone();
			im_quads.darken();
			im_quads.darken();

			let mut rng = thread_rng();

			for quad in quads.iter() {
				let color = rng.gen_color_gray(100).into();

				im_quads.draw_line(quad.corners[0], quad.corners[1], &color, 1);
				im_quads.draw_line(quad.corners[1], quad.corners[2], &color, 1);
				im_quads.draw_line(quad.corners[2], quad.corners[3], &color, 1);
				im_quads.draw_line(quad.corners[3], quad.corners[0], &color, 1);

			}

			im_quads.save_to_pnm("debug_quads_fixed.pnm").unwrap();
		}

		tp.stamp("decode+refinement");

		////////////////////////////////////////////////////////////////
		// Step 3. Reconcile detections--- don't report the same tag more
		// than once. (Allow non-overlapping duplicate detections.)
		let detections = if true {
			let mut drop_idxs = Vec::new();
			'outer: for (i0, det0) in detections.iter().enumerate() {
				if drop_idxs.contains(&i0) {
					continue;
				}
				let poly0 = Poly2D::of(&det0.corners);

				for i1 in (i0+1)..detections.len() {
					if drop_idxs.contains(&i1) {
						continue;
					}
					let det1 = &detections[i1];

					if det0.id != det1.id || det0.family != det1.family {
						// They can't be the same detection
						continue;
					}

					let poly1 = Poly2D::of(&det1.corners);

					if poly0.overlaps_polygon(&poly1) {
						// the tags overlap. Delete one, keep the other.

						let mut pref = Ordering::Equal; // Equal means undecided which one we'll keep.
						pref = prefer_smaller(pref, det0.hamming, det1.hamming);     // want small hamming
						pref = prefer_smaller(pref, -det0.decision_margin, -det1.decision_margin);      // want bigger margins

						// if we STILL don't prefer one detection over the other, then pick
						// any deterministic criterion.
						for i in 0..4 {
							pref = prefer_smaller(pref, det0.corners[i].x(), det1.corners[i].x());
							pref = prefer_smaller(pref, det0.corners[i].y(), det1.corners[i].y());
						}


						fn prefer_smaller<T: PartialOrd>(pref: Ordering, q0: T, q1: T) -> std::cmp::Ordering {
							if pref.is_ne() {
								// already prefer something? exit.
								return pref;
							}

							T::partial_cmp(&q0, &q1)
								.unwrap_or(Ordering::Equal)
								.reverse()
						}

						if pref.is_eq() {
							// at this point, we should only be undecided if the tag detections
							// are *exactly* the same. How would that happen?
							println!("uh oh, no preference for overlappingdetection");
						}

						if pref.is_lt() {
							// keep det0, destroy det1
							drop_idxs.push(i1);
							continue;
						} else {
							// keep det1, destroy det0
							drop_idxs.push(i0);
							continue 'outer;
						}
					}
				}
			}

			if !drop_idxs.is_empty() {
				detections
					.into_iter()
					.enumerate()
					.filter(|(idx, _)| !drop_idxs.contains(idx))
					.map(|(_idx, det)| det)
					.collect::<Vec<_>>()
			} else {
				detections
			}
		} else {
			detections
		};

		tp.stamp("reconcile");

		////////////////////////////////////////////////////////////////
		// Produce final debug output
		#[cfg(feature="debug")]
		if self.params.generate_debug_image() {
			debug_output_ps(im_orig, &detections)
				.expect("Error generating debug_output.ps");
			debug_output_pnm(im_orig, &detections)
				.expect("Error generating debug_output.pnm");
			debug_quads_ps(im_orig, &quads)
				.expect("Error generating debug_quads.ps");
		}

		tp.stamp("debug output");
		std::mem::drop(quads);

		let mut detections = detections;
		detections.sort_by(|a, b| Ord::cmp(&a.id, &b.id));

		tp.stamp("cleanup");

		Detections {
			tp,
			nquads,
			detections,
		}
	}
}

#[cfg(feature="debug")]
fn debug_output_ps(im_orig: &ImageY8, detections: &[ApriltagDetection]) -> std::io::Result<()> {
	// assume letter, which is 612x792 points.
	let mut f = std::fs::File::create("debug_output.ps")?;
	let mut f = PostScriptWriter::new(&mut f)?;
	{
		let mut darker = im_orig.clone();
		darker.darken();
		darker.darken();

		let scale = f32::min(612.0f32 / darker.width() as f32, 792.0f32 / darker.height() as f32);
		f.scale(scale, scale)?;
		f.translate(0., darker.height() as f32)?;
		f.scale(1., -1.)?;
		
		darker.write_postscript(&mut f)?;
	}

	let mut rng = thread_rng();
	for det in detections.iter() {
		let rgb = rng.gen_color_rgb(100);

		f.setrgbcolor(&rgb)?;
		f.command(|mut c| {
			c.moveto(&det.corners[0])?;
			c.lineto(&det.corners[1])?;
			c.lineto(&det.corners[2])?;
			c.lineto(&det.corners[3])?;
			c.lineto(&det.corners[0])?;
			c.stroke()
		})?;
	}

	f.showpage()
}

#[cfg(feature="debug")]
fn debug_output_pnm(im_orig: &ImageY8, detections: &[ApriltagDetection]) -> std::io::Result<()> {
	let mut darker = im_orig.clone();
	darker.darken();
	darker.darken();

	let mut out = darker.map(|p| p.to_rgb());

	let mut rng = thread_rng();
	for det in detections.iter() {
		let rgb = rng.gen_color_rgb::<u8>(100);

		for j in 0..4 {
			let k = (j + 1) & 3;
			out.draw_line(
				det.corners[j], det.corners[k],
				&rgb,
				1
			);
		}
	}

	out.save_to_pnm("debug_output.pnm")
}

#[cfg(feature="debug")]
fn debug_quads_ps(im_orig: &ImageY8, quads: &[Quad]) -> std::io::Result<()> {
	let mut f = std::fs::File::create("debug_quads.ps")?;
	let mut f = PostScriptWriter::new(&mut f)?;
	let mut rng = thread_rng();

	{
		let mut darker = im_orig.clone();
		darker.darken();
		darker.darken();

		// assume letter, which is 612x792 points.
		let scale = f32::min(612.0f32/darker.width() as f32, 792.0f32/darker.height() as f32);
		f.scale(scale, scale)?;
		f.translate(0., darker.height() as f32)?;
		f.scale(1., -1.)?;

		darker.write_postscript(&mut f)?;
	}

	for q in quads.iter() {
		let rgb = rng.gen_color_rgb(100);

		f.setrgbcolor(&rgb)?;
		f.command(|mut c| {
			c.moveto(&q.corners[0])?;
			c.lineto(&q.corners[1])?;
			c.lineto(&q.corners[2])?;
			c.lineto(&q.corners[3])?;
			c.lineto(&q.corners[0])?;
			c.stroke()
		})?;
	}

	f.showpage()
}

/// Represents the detection of a tag. These are returned to the user
/// and must be individually destroyed by the user.
pub struct ApriltagDetection {
	/// a pointer for convenience. not freed by apriltag_detection_destroy.
	pub family: Arc<AprilTagFamily>,

	/// The decoded ID of the tag
	pub id: usize,

	/// How many error bits were corrected? Note: accepting large numbers of
	/// corrected errors leads to greatly increased false positive rates.
	/// NOTE: As of this implementation, the detector cannot detect tags with
	/// a hamming distance greater than 2.
	pub hamming: i16,

	/// A measure of the quality of the binary decoding process: the
	/// average difference between the intensity of a data bit versus
	/// the decision threshold. Higher numbers roughly indicate better
	/// decodes. This is a reasonable measure of detection accuracy
	/// only for very small tags-- not effective for larger tags (where
	/// we could have sampled anywhere within a bit cell and still
	/// gotten a good detection.)
	pub decision_margin: f32,

	/// The 3x3 homography matrix describing the projection from an
	/// "ideal" tag (with corners at (-1,-1), (1,-1), (1,1), and (-1,
	/// 1)) to pixels in the image. This matrix will be freed by
	/// apriltag_detection_destroy.
	pub H: Mat,

	/// The center of the detection in image pixel coordinates.
	pub center: Point2D,

	/// The corners of the tag in image pixel coordinates. These always
	/// wrap counter-clock wise around the tag.
	pub corners: [Point2D; 4],
}

#[derive(Default)]
pub struct Detections {
	pub tp: TimeProfile,
	pub nquads: u32,
	pub detections: Vec<ApriltagDetection>,
}