#![cfg(feature="debug")]

use rand::thread_rng;

use crate::{util::{ImageY8, image::{ImageWritePNM, PostScriptWriter, ImageWritePostscript, pixel::PixelConvert}, color::RandomColor}, quad_decode::Quad, AprilTagDetection};
use std::{fs::File, io};

pub(super) fn debug_quads(mut f: File, mut im_quads: ImageY8, quads: &[Quad]) -> io::Result<()> {
	im_quads.darken();
	im_quads.darken();

	let mut rng = thread_rng();

	for quad in quads.iter() {
		let color = rng.gen_color_gray(100).into();

		println!("Corners: {:?}", quad.corners);

		im_quads.draw_line(quad.corners[0], quad.corners[1], &color, 1);
		im_quads.draw_line(quad.corners[1], quad.corners[2], &color, 1);
		im_quads.draw_line(quad.corners[2], quad.corners[3], &color, 1);
		im_quads.draw_line(quad.corners[3], quad.corners[0], &color, 1);
	}

	im_quads.write_pnm(&mut f)
}

pub(super) fn debug_quads_fixed(mut f: File, mut im_quads: ImageY8, quads: &[Quad]) -> io::Result<()> {
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

	im_quads.write_pnm(&mut f)
}

pub(super) fn debug_output_ps(mut f: File, mut img: ImageY8, detections: &[AprilTagDetection]) -> std::io::Result<()> {
	// assume letter, which is 612x792 points.
	let mut f = PostScriptWriter::new(&mut f)?;
	{
		img.darken();
		img.darken();

		let scale = f32::min(612.0f32 / img.width() as f32, 792.0f32 / img.height() as f32);
		f.scale(scale, scale)?;
		f.translate(0., img.height() as f32)?;
		f.scale(1., -1.)?;
		
		img.write_postscript(&mut f)?;
	}

	let mut rng = thread_rng();
	for det in detections.iter() {
		let rgb = rng.gen_color_rgb(100);

		f.setrgbcolor(&rgb)?;
		f.command(|c| {
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

pub(super) fn debug_output_pnm(mut f: File, mut img: ImageY8, detections: &[AprilTagDetection]) -> std::io::Result<()> {
	img.darken();
	img.darken();

	let mut out = img.map(|p| p.to_rgb());

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

	out.write_pnm(&mut f)
}

pub(super) fn debug_quads_ps(mut f: File, mut img: ImageY8, quads: &[Quad]) -> std::io::Result<()> {
	let mut f = PostScriptWriter::new(&mut f)?;
	let mut rng = thread_rng();

	{
		img.darken();
		img.darken();

		// assume letter, which is 612x792 points.
		let scale = f32::min(612.0f32/img.width() as f32, 792.0f32/img.height() as f32);
		f.scale(scale, scale)?;
		f.translate(0., img.height() as f32)?;
		f.scale(1., -1.)?;

		img.write_postscript(&mut f)?;
	}

	for q in quads.iter() {
		let rgb = rng.gen_color_rgb(100);

		f.setrgbcolor(&rgb)?;
		f.command(|c| {
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