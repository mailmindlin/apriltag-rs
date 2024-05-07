pub(crate) mod unionfind;
mod linefit;
mod quadfit;
mod grad_cluster;
pub(super) mod threshold;
pub(super) use grad_cluster::{gradient_clusters, Clusters};
use std::{fs::File, f64::consts as f64c};

use crate::{detector::AprilTagDetector, util::{mem::calloc, color::RandomColor, image::{ImageWritePNM, ImageBuffer, Rgb, ImageY8, ImageDimensions, ImageRefY8}}, quad_decode::Quad, dbg::{TimeProfile, debug_images}, DetectorConfig};

use self::{unionfind::connected_components, quadfit::fit_quads};

#[cfg(feature="debug")]
use self::unionfind::UnionFindStatic;

/// Minimum size for blobs
const MIN_CLUSTER_SIZE: usize = 24;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "cffi", repr(C))]
pub struct AprilTagQuadThreshParams {
	/// Reject quads containing too few pixels
	pub min_cluster_pixels: u32,

	/// How many corner candidates to consider when segmenting a group
	/// of pixels into a quad.
	pub max_nmaxima: u8,

	/// Reject quads where pairs of edges have angles that are close to
	/// straight or close to 180 degrees. Zero means that no quads are
	/// rejected. (In radians).
	pub cos_critical_rad: f32,

	/// When fitting lines to the contours, what is the maximum mean
	/// squared error allowed?
	/// This is useful in rejecting contours that are far from being
	/// quad shaped; rejecting these quads "early" saves expensive
	/// decoding processing.
	pub max_line_fit_mse: f32,

	/// When we build our model of black & white pixels, we add an
	/// extra check that the white model must be (overall) brighter
	/// than the black model. 
	/// How much brighter? (in pixel values, [0,255]).
	pub min_white_black_diff: u8,

	/// Should the thresholded image be deglitched?
	/// Only useful for very noisy images
	pub deglitch: bool,
}

impl Default for AprilTagQuadThreshParams {
	fn default() -> Self {
		Self {
			min_cluster_pixels: 5,
			max_nmaxima: 10,
			cos_critical_rad: (10. * f64c::PI / 180.).cos() as f32,
			max_line_fit_mse: 10.,
			min_white_black_diff: 5,
			deglitch: false,
		}
	}
}

#[cfg(feature="debug")]
fn debug_segmentation(mut f: File, w: usize, h: usize, uf: &impl UnionFindStatic<(u32, u32), Id = u32>, qtp: &AprilTagQuadThreshParams) -> std::io::Result<()> {
	use rand::{rngs::StdRng, SeedableRng};

	let mut d = ImageBuffer::<Rgb<u8>>::zeroed(w, h);

	let mut colors = calloc::<Option<Rgb<u8>>>(d.num_pixels());
	for ((x, y), dst) in d.enumerate_pixels_mut() {
		let (v, v_size) = uf.get_set_static((x as _, y as _));

		if v_size < qtp.min_cluster_pixels {
			continue;
		}

		*dst = {
			let color_ref = &mut colors[v as usize];

			if let Some(color) = color_ref {
				*color
			} else {
				// Deterministic color from position
				let color = {
					let mut rng = StdRng::seed_from_u64((y * w + x) as _);
					rng.gen_color_rgb(50u8)
				};
				*color_ref = Some(color);
				color
			}
		};
	}

	d.write_pnm(&mut f)
}

/// Generate a debug image showing how many hops we take to resolve each pixel in the UnionFind
#[cfg(feature="debug")]
fn debug_unionfind_depth(mut f: File, w: usize, h: usize, uf: &impl UnionFindStatic<(u32, u32)>) -> std::io::Result<()> {
	let mut d = ImageBuffer::<Rgb<u8>>::zeroed_packed(w, h);
	let mut max_hops = 0;
	for ((x, y), dst) in d.enumerate_pixels_mut() {
		let hops = uf.get_set_hops((x as u32, y as u32));
		if hops < 1 {
			*dst = Rgb([0;3]);
		} else if hops == usize::MAX {
			*dst = Rgb([255, 0, 0]);
		} else {
			max_hops = std::cmp::max(hops, max_hops);
			*dst = Rgb([1u8 << (hops - 1).clamp(0, 7) as u8; 3]);
		}
	}
	println!("Max hops: {max_hops}");

	d.write_pnm(&mut f)
}

#[cfg(feature="debug")]
fn debug_clusters(mut f: File, w: usize, h: usize, clusters: &grad_cluster::Clusters) -> std::io::Result<()> {
	use rand::{rngs::StdRng, SeedableRng};

	use crate::util::image::ImageRGB8;
	
	// Deterministically sort clusters
	let mut clusters1 = clusters.values()
		.collect::<Vec<_>>();
	clusters1.sort_by_cached_key(|cluster| {
		let mut top = u16::MAX;
		let mut left = u16::MAX;
		for p in *cluster {
			if p.x < left {
				left = p.x;
			}
			if p.y < top {
				top = p.y;
			}
		}
		(top as usize) * h + (left as usize)
	});

	let mut d = ImageRGB8::zeroed(w, h);
	for cluster in clusters1.into_iter() {
		// Deterministically generate color from value for better comparison
		let color = {
			let mut min_x = u16::MAX;
			let mut min_y = u16::MAX;
			for p in cluster {
				min_x = min_x.min(p.x);
				min_y = min_y.min(p.y);
			}
			// This is effectively a hash
			let mut rng = StdRng::seed_from_u64((min_y as u64) * (w as u64) + (min_x as u64));
			rng.gen_color_rgb(50u8)
		};

		for p in cluster {
			let x = (p.x / 2) as usize;
			let y = (p.y / 2) as usize;
			d.pixels_mut()[(x, y)] = color;
		}
	}

	d.write_pnm(&mut f)
}

#[cfg(feature="debug_ps")]
fn debug_lines(mut f: File, im: ImageY8, quads: &[Quad]) -> std::io::Result<()> {
	use crate::util::image::{ImageWritePostscript, VectorPathWriter, PostScriptWriter};

	let mut ps = PostScriptWriter::new(&mut f)?;

	let mut im2 = im.clone();
	im2.darken();
	im2.darken();

	// assume letter, which is 612x792 points.
	let scale = f32::min(612.0/im.width() as f32, 792.0/im2.height() as f32);
	ps.scale(scale, scale)?;
	ps.translate(0., im2.height() as f32)?;
	ps.scale(1., -1.)?;

	im.write_postscript(&mut ps)?;

	let mut rng = rand::thread_rng();

	for q in quads.iter() {
		ps.setrgbcolor(&rng.gen_color_rgb(100))?;
		ps.path(|c| {
			c.move_to(&q.corners[0])?;
			c.line_to(&q.corners[0])?;
			c.line_to(&q.corners[1])?;
			c.line_to(&q.corners[2])?;
			c.line_to(&q.corners[3])?;
			c.line_to(&q.corners[0])?;
			c.stroke()
		})?;
	}

	Ok(())
}

#[cfg(feature="debug")]
pub(crate) fn debug_unionfind(config: &DetectorConfig, tp: &mut TimeProfile, dims: &ImageDimensions, uf: &impl UnionFindStatic<(u32, u32), Id = u32>) {
	if !config.generate_debug_image() {
		return;
	}
	config.debug_image(debug_images::UNIONFIND_DEPTH, |f| debug_unionfind_depth(f, dims.width, dims.height, uf));
	config.debug_image(debug_images::SEGMENTATION,   |f| debug_segmentation(f, dims.width, dims.height, uf, &config.qtp));
	tp.stamp("unionfind (output)");
}

pub(crate) fn quads_from_clusters(td: &AprilTagDetector, tp: &mut TimeProfile, im: ImageRefY8, clusters: Clusters) -> Vec<Quad> {
	#[cfg(feature="extra_debug")]
	println!("{} gradient clusters", clusters.len());
	
	#[cfg(feature="debug")]
	td.params.debug_image(debug_images::CLUSTERS, |f| debug_clusters(f, im.width(), im.height(), &clusters));
	tp.stamp("make clusters");

	////////////////////////////////////////////////////////
	// step 3. process each connected component.
	let quads = fit_quads(td, clusters, &im);

	#[cfg(feature="extra_debug")]
	for quad in quads.iter() {
		println!("Quad corner: {:?}", quad.corners);
	}

	#[cfg(feature="debug_ps")]
	td.params.debug_image(debug_images::LINES, |f| debug_lines(f, ImageY8::clone_packed(&im), &quads));

	tp.stamp("fit quads to clusters");

	quads
}

pub(crate) fn apriltag_quad_thresh(td: &AprilTagDetector, tp: &mut TimeProfile, im: &ImageY8, threshim: ImageY8) -> Vec<Quad> {
	////////////////////////////////////////////////////////
	// step 2. find connected components.
	let mut uf = connected_components(&td.params, &threshim);

	tp.stamp("unionfind");

	// make segmentation image.
	#[cfg(feature="debug")]
	debug_unionfind(&td.params, tp, threshim.dimensions(), &mut uf);

	let clusters = gradient_clusters(&td.params, &threshim.as_ref(), uf);
	drop(threshim);

	quads_from_clusters(td, tp, im.as_ref(), clusters)
}
