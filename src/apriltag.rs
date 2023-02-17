use crate::{families::AprilTagFamily, util::{homography::homography_project, math::{mat::matd_t}, Image}, quad_decode::Quad};


struct evaluate_quad_ret {
	rcode: i64,
	score: f64,
	H: matd_t,
	Hinv: matd_t,

	decode_status: isize,
	e: QuickDecodeEntry,
}

// compute a "score" for a quad that is independent of tag family
// encoding (but dependent upon the tag geometry) by considering the
// contrast around the exterior of the tag.
fn quad_goodness(family: &AprilTagFamily, im: &Image, quad: Quad) -> f64 {
	// when sampling from the white border, how much white border do
	// we actually consider valid, measured in bit-cell units? (the
	// outside portions are often intruded upon, so it could be advantageous to use
	// less than the "nominal" 1.0. (Less than 1.0 not well tested.)

	// XXX Tunable
	let white_border = 1f32;

	// in tag coordinates, how big is each bit cell?
	let bit_size = 2.0 / ((2*family.black_border + family.dimensions) as f64);
//    double inv_bit_size = 1.0 / bit_size;

	let mut xmin = i32::MAX;
	let mut xmax = 0;
	let mut ymin = i32::MAX;
	let mut ymax = 0;
	for i in 0..4 {
		let tx = if (i == 0 || i == 3) { -1 - bit_size } else { 1 + bit_size };
		let ty = if (i == 0 || i == 1) { -1 - bit_size } else { 1 + bit_size };

		let (x, y) = homography_project(&quad.H, tx, ty);
		xmin = i32::min(xmin, x);
		xmax = i32::max(xmax, x);
		ymin = i32::min(ymin, y);
		ymax = i32::max(ymax, y);
	}

	// clamp bounding box to image dimensions
	xmin = i32::max(0, xmin);
	xmax = i32::min(im.width-1, xmax);
	ymin = i32::max(0, ymin);
	ymax = i32::min(im.height-1, ymax);

//    int nbits = family.d * family.d;
	let mut W1 = 0i64;
	let mut B1 = 0i64;
	let mut Wn = 0i64;
	let mut Bn = 0i64;

	let wsz = bit_size * white_border;
	let bsz = bit_size * family.black_border;

	let Hinv = &quad.Hinv;
//    matd_t *H = quad.H;

	// iterate over all the pixels in the tag. (Iterating in pixel space)
	for y in ymin..=ymax {
		// we'll incrementally compute the homography
		// projections. Begin by evaluating the homogeneous position
		// [(xmin - .5f), y, 1]. Then, we'll update as we stride in
		// the +x direction.
		let Hx = Hinv[(0,0)] * (0.5 + (xmin as f64)) +
			Hinv[(0, 1)] * (y as f64 + 0.5) + Hinv[(0, 2)];
		let Hy = Hinv[(1, 0)] * (0.5 + (xmin as f64)) +
			Hinv[(1, 1)] * (y as f64 + 0.5) + Hinv[(1, 2)];
		let Hh = Hinv[(2, 0)] * (0.5 + (xmin as f64)) +
			Hinv[(2, 1)] * (y as f64 + 0.5) + Hinv[(2, 2)];

		for x in xmin ..= xmax {
			// project the pixel center.

			// divide by homogeneous coordinate
			let tx = Hx / Hh;
			let ty = Hy / Hh;

			// if we move x one pixel to the right, here's what
			// happens to our three pre-normalized coordinates.
			Hx += Hinv[(0, 0)];
			Hy += Hinv[(1, 0)];
			Hh += Hinv[(2, 0)];

			let txa = f32::abs(tx);
			let tya = f32::abs(ty);
			let xymax = f32::max(txa, tya);

//            if (txa >= 1 + wsz || tya >= 1 + wsz)
			if xymax >= 1. + wsz {
				continue;
			}

			let v = im[(x as usize, y as usize)];

			// it's within the white border?
//            if (txa >= 1 || tya >= 1) {
			if xymax >= 1. {
				W1 += v;
				Wn += 1;
				continue;
			}

			// it's within the black border?
//            if (txa >= 1 - bsz || tya >= 1 - bsz) {
			if xymax >= 1. - bsz as f32 {
				B1 += v;
				Bn += 1;
				continue;
			}

			// it must be a data bit. We don't do anything with these.
			continue;
		}
	}


	// score = average margin between white and black pixels near border.
	let margin = 1.0 * W1 / Wn - 1.0 * B1 / Bn;
//    printf("margin %f: W1 %f, B1 %f\n", margin, W1, B1);

	return margin;
}