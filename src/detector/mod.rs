mod builder;
pub(crate) mod config;
mod debug;

pub use config::{DetectorConfig, GpuAccelRequest};
pub use builder::{DetectorBuilder, DetectorBuildError};

use std::sync::{Mutex, Arc};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

use crate::{util::{math::{Vec2, Vec2Builder}, image::{ImageWritePNM, Pixel, ImageY8, ImageAllocError, ImageRefY8, Luma}, ImageBuffer}, quickdecode::QuickDecode, quad_decode::QuadDecodeInfo, quad_thresh::{unionfind::connected_components, Clusters, gradient_clusters, debug_unionfind, quads_from_clusters}, dbg::TimeProfile, Detections, detection::reconcile_detections, wgpu::WGPUDetector};

use self::config::QuadDecimateMode;

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DetectError {
	/// Input image was too small
	ImageTooSmall,
	/// Input image was too large
	ImageTooBig,
	/// Buffer allocation error
	BufferAlloc,
	ImageAlloc(ImageAllocError),
	OpenCLError,
}

pub(crate) trait Preprocessor {
	fn preprocess(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<(ImageY8, ImageY8), DetectError>;
	fn cluster(&self, config: &DetectorConfig, tp: &mut TimeProfile, image: ImageRefY8) -> Result<(ImageY8, Clusters), DetectError>;
}

impl From<ImageAllocError> for DetectError {
    fn from(value: ImageAllocError) -> Self {
        Self::ImageAlloc(value)
    }
}

enum GpuAccelerator {
	None,
	#[cfg(feature="opencl")]
	OpenCL(Box<crate::ocl::OpenCLDetector>),
	#[cfg(feature="wgpu")]
	WebGPU(Box<WGPUDetector>),
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
	gpu: GpuAccelerator,
}

pub(crate) fn quad_sigma_kernel(quad_sigma: f32) -> Option<Vec<u8>> {
	if quad_sigma == 0. {
		return None;
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

	let kernel_size = (4. * sigma) as usize; // 2 std devs in each direction
	let kernel_size = if (kernel_size % 2) == 0 {
		kernel_size + 1
	} else {
		kernel_size
	};

	crate::util::image::luma::quad_sigma_kernel(quad_sigma as f64, kernel_size)
}

/// Apply blur/sharp filter
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

#[cfg(test)]
pub fn quad_sigma_cpu(img: &mut ImageY8, quad_sigma_v: f32) {
	quad_sigma(img, quad_sigma_v)
}


impl AprilTagDetector {
	/// Create a new builder
	pub fn builder() -> DetectorBuilder {
		DetectorBuilder::default()
	}

	pub fn has_gpu(&self) -> bool {
		!matches!(self.gpu, GpuAccelerator::None)
	}

	fn new(params: DetectorConfig, tag_families: Vec<Arc<QuickDecode>>) -> Result<AprilTagDetector, DetectorBuildError> {
		let wp = {
			let tpb = ThreadPoolBuilder::new()
				.num_threads(params.nthreads);

			match tpb.build() {
				Ok(wp) => wp,
				Err(e) => return Err(DetectorBuildError::Threadpool(e)),
			}
		};

		let gpu = {
			let mut gpu = GpuAccelerator::None;
			let mut gpu_err = None;

			#[cfg(feature="opencl")]
			if !params.gpu.is_disabled() && matches!(gpu, GpuAccelerator::None) {
				println!("trying to build OpenCL");
				match crate::ocl::OpenCLDetector::new(&params) {
					Ok(det) => {
						gpu = GpuAccelerator::OpenCL(Box::new(det));
					},
					Err(DetectorBuildError::GpuNotAvailable) => {},
					Err(e) => {
						eprintln!("GPU init error: {e:?}");
						gpu_err = Some(e);
					}
				}
			}

			#[cfg(feature="wgpu")]
			if (!params.gpu.is_disabled()) && matches!(gpu, GpuAccelerator::None) {
				println!("trying to build WGPU: {:?}", params.gpu);
				match WGPUDetector::new(&params) {
					Ok(det) => {
						gpu = GpuAccelerator::WebGPU(Box::new(det));
					},
					Err(DetectorBuildError::GpuNotAvailable) => {},
					Err(e) => {
						eprintln!("GPU init error: {e:?}");
						gpu_err = Some(e);
					}
				}
			}

			if params.gpu.is_required() && matches!(gpu, GpuAccelerator::None) {
				match gpu_err {
					Some(err) => return Err(err),
					None => return Err(DetectorBuildError::GpuNotAvailable),
				}
			}
			gpu
		};

		Ok(Self {
			params,
			tag_families,
			wp,
			gpu,
		})
	}

	fn preprocess_image(&self, tp: &mut TimeProfile, im_orig: ImageRefY8) -> Result<(ImageY8, ImageY8), DetectError> {
		///////////////////////////////////////////////////////////
		// Step 1. Detect quads according to requested image decimation
		// and blurring parameters.
		let mut quad_im = match self.params.quad_decimate_mode() {
			None => {
				//TODO: can we not copy here?
				ImageY8::clone_packed(&im_orig)
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

		////////////////////////////////////////////////////////
		// step 1. threshold the image, creating the edge image.
		let threshim = super::quad_thresh::threshold::threshold(&self.params, tp, quad_im.as_ref())?;

		#[cfg(feature="debug")]
        self.params.debug_image("02_debug_threshold.pnm", |mut f| threshim.write_pnm(&mut f));

		Ok((quad_im, threshim))
	}

	/// Preprocess and segment image
	/// 
	/// Returns preprocessed image & ccl
	fn segment_image(&self, tp: &mut TimeProfile, im_orig: ImageRefY8) -> Result<(ImageY8, Clusters), DetectError> {
		match &self.gpu {
			#[cfg(feature="opencl")]
			GpuAccelerator::OpenCL(ocl) => {
				match ocl.cluster(&self.params, tp, im_orig) {
					Ok(res) => {
						return Ok(res);
					},
					Err(e) => {
						if self.params.gpu.is_required() {
							return Err(e);
						} else {
							// Swallow
							eprintln!("OpenCL error: {e:?}");
						}
					}
				}
			},
			#[cfg(feature="wgpu")]
			GpuAccelerator::WebGPU(wgpu) => {
				match wgpu.preprocess(&self.params, tp, im_orig) {
					Ok((quad_im, threshim)) => {
						let mut uf = connected_components(&self.params, &threshim);
						tp.stamp("unionfind");

						// make segmentation image.
						#[cfg(feature="debug")]
						debug_unionfind(&self.params, tp, threshim.dimensions(), &mut uf);

						let clusters = gradient_clusters(&self.params, &threshim.as_ref(), uf);
						return Ok((quad_im, clusters));
					},
					Err(e) => {
						if self.params.gpu.is_required() {
							return Err(e);
						} else {
							// Swallow
							eprintln!("OpenCL error: {e:?}");
						}
					}
				}
			}
			GpuAccelerator::None => {},
		}

		#[cfg(feature="debug")]
		self.params.debug_image("00_debug_src.pnm", |mut f| im_orig.write_pnm(&mut f));

		// CPU
		let (quad_im, threshim) = self.preprocess_image(tp, im_orig)?;
		let mut uf = connected_components(&self.params, &threshim);
		tp.stamp("unionfind");

		// make segmentation image.
		#[cfg(feature="debug")]
		debug_unionfind(&self.params, tp, threshim.dimensions(), &mut uf);

		let clusters = gradient_clusters(&self.params, &threshim.as_ref(), uf);

		Ok((quad_im, clusters))
	}

	pub fn detect<Container: AsRef<[u8]>>(&self, im: &ImageBuffer<Luma<u8>, Container>) -> Result<Detections, DetectError> {
		self.detect_inner(&im.as_ref())
	}

	/// Detect AprilTags
	/// 
	/// ## Steps:
	/// ### 1. Decimate
	/// Downsample image, as per `quad_decimate`
	/// 
	/// ### 2. Blur / Sharpen
	/// Blur or sharpen image, as per `quad_sigma`
	/// 
	/// ### 3. Threshold
	/// 
	fn detect_inner(&self, im_orig: &ImageRefY8) -> Result<Detections, DetectError> {
		if self.tag_families.len() == 0 {
			println!("AprilTag: No tag families enabled.");
			return Ok(Detections::default());
		}

		// Statistics relating to last processed frame
		let mut tp = TimeProfile::default();
		tp.stamp("init");

		let mut quads = {
			let (quad_im, clusters) = self.segment_image(&mut tp, im_orig.as_ref())?;

			quads_from_clusters(self, &mut tp, quad_im.as_ref(), clusters)
		};

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
		self.params.debug_image("07_debug_quads.pnm", |f| debug::debug_quads(f, ImageY8::clone_packed(im_orig), &quads));

		////////////////////////////////////////////////////////////////
		// Step 2. Decode tags from each quad.
		let detections = if true {
			#[cfg(feature="debug")]
			let im_samples = if self.params.generate_debug_image() { Some(Mutex::new(ImageY8::clone_packed(im_orig))) } else { None };
			
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
				self.params.debug_image("08_debug_samples.pnm", |mut f| {
					let im_samples = im_samples.into_inner().unwrap();
					im_samples.write_pnm(&mut f)
				});
			}
			detections
		} else {
			Vec::new()
		};

		tp.stamp("decode+refinement");

		#[cfg(feature="debug")]
		if self.params.generate_debug_image() {
			self.params.debug_image("09a_debug_quads_fixed.pnm", |f| debug::debug_quads_fixed(f, ImageY8::clone_packed(im_orig), &quads));
			#[cfg(feature="debug_ps")]
			self.params.debug_image("09b_debug_quads.ps", |f| debug::debug_quads_ps(f, ImageY8::clone_like(im_orig), &quads));
			tp.stamp("decode+refinement (output)");
		}
		drop(quads);

		let mut detections = reconcile_detections(detections);

		tp.stamp("reconcile");

		detections.sort_unstable_by(|a, b| Ord::cmp(&a.id, &b.id));

		////////////////////////////////////////////////////////////////
		// Produce final debug output
		#[cfg(feature="debug")]
		{
			#[cfg(feature="debug_ps")]
			self.params.debug_image("10a_debug_output.ps", |f| debug::debug_output_ps(f, ImageY8::clone_like(im_orig), &detections));
			self.params.debug_image("10b_debug_output.pnm", |f| debug::debug_output_pnm(f, ImageY8::clone_packed(im_orig), &detections));
		}
		tp.stamp("debug output");

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