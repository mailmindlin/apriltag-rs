mod uf;
mod linefit;
mod quadfit;
mod grad_cluster;
mod threshold;
use std::{fs::{File}, f64::consts as f64c};

use rand::thread_rng;

use crate::{detector::ApriltagDetector, util::{mem::calloc, color::RandomColor, image::{ImageWritePNM, ImageBuffer, Rgb, ImageMut, HasDimensions, ImageRGB8, PostScriptWriter, ImageY8}, TimeProfile}, quad_decode::Quad};

use self::{uf::{connected_components, UnionFind2D}, grad_cluster::gradient_clusters, quadfit::fit_quads, linefit::Pt};

pub struct ApriltagQuadThreshParams {
    /// Reject quads containing too few pixels
    pub min_cluster_pixels: u32,

    /// How many corner candidates to consider when segmenting a group
    /// of pixels into a quad.
    pub max_nmaxima: u8,

    /// Reject quads where pairs of edges have angles that are close to
    /// straight or close to 180 degrees. Zero means that no quads are
    /// rejected. (In radians).
    #[deprecated]
    pub critical_rad: f32,
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

impl Default for ApriltagQuadThreshParams {
    fn default() -> Self {
        #[allow(deprecated)]
        Self {
            min_cluster_pixels: 5,
            max_nmaxima: 10,
            critical_rad: Default::default(),
            cos_critical_rad: (10. * f64c::PI / 180.).cos() as f32,
            max_line_fit_mse: 10.,
            min_white_black_diff: 5,
            deglitch: false,
        }
    }
}

#[cfg(feature="debug")]
fn debug_segmentation(w: usize, h: usize, uf: &UnionFind2D, qtp: &ApriltagQuadThreshParams) -> std::io::Result<()> {
    let mut d = ImageBuffer::<Rgb<u8>>::create(w, h);
    let mut rng = thread_rng();

    let mut colors = calloc::<Option<Rgb<u8>>>(d.len());
    for ((x, y), dst) in d.enumerate_pixels_mut() {
        let v = uf.get_representative(x, y);

        if uf.get_set_size(v) < qtp.min_cluster_pixels {
            continue;
        }

        *dst = {
            let color_ref = &mut colors[v as usize];

            if let Some(color) = color_ref {
                *color
            } else {
                let color = rng.gen_color_rgb(50u8);
                *color_ref = Some(color);
                color
            }
        };
    }

    d.save_to_pnm("debug_segmentation.pnm")
}

#[cfg(feature="debug")]
fn debug_clusters(w: usize, h: usize, clusters: &[Vec<Pt>]) -> std::io::Result<()> {
    let mut d = ImageRGB8::create(w, h);
    let mut rng = thread_rng();
    for cluster in clusters.iter() {
        let color = rng.gen_color_rgb(50u8);
        for p in cluster.iter() {
            let x = (p.x / 2) as usize;
            let y = (p.y / 2) as usize;
            d.pixels_mut()[(x, y)] = color;
        }
    }

    d.save_to_pnm("debug_clusters.pnm")
}

#[cfg(feature="debug")]
fn debug_lines(im: &ImageY8, quads: &[Quad]) -> std::io::Result<()> {
    let mut f = File::create("debug_lines.ps")?;
    let mut ps = PostScriptWriter::new(&mut f)?;

    let mut im2 = im.clone();
    im2.darken();
    im2.darken();

    // assume letter, which is 612x792 points.
    let scale = f32::min(612.0/im.width() as f32, 792.0/im2.height() as f32);
    ps.scale(scale, scale)?;
    ps.translate(0., im2.height() as f32)?;
    ps.scale(1., -1.)?;

    // im.write_postscript(&mut f); //FIXME

    let mut rng = thread_rng();

    for q in quads.iter() {
        ps.setrgbcolor(&rng.gen_color_rgb(100));
        ps.command(|mut c| {
            c.moveto(&q.corners[0])?;
            c.lineto(&q.corners[0])?;
            c.lineto(&q.corners[1])?;
            c.lineto(&q.corners[2])?;
            c.lineto(&q.corners[3])?;
            c.lineto(&q.corners[0])?;
            c.stroke()
        })?;
    }

    Ok(())
}

pub(crate) fn apriltag_quad_thresh(td: &ApriltagDetector, tp: &mut TimeProfile, im: &ImageY8) -> Vec<Quad> {
    ////////////////////////////////////////////////////////
    // step 1. threshold the image, creating the edge image.

    let w = im.width();
    let h = im.height();

    let threshim = threshold::threshold(&td.qtp, tp, im);

    #[cfg(feature="debug")]
    if td.params.generate_debug_image() {
        threshim.save_to_pnm("debug_threshold.pnm")
            .expect("Error writing debug_threshold.pnm");
    }

    ////////////////////////////////////////////////////////
    // step 2. find connected components.

    let mut uf = connected_components(td, &threshim);

    // make segmentation image.
    #[cfg(feature="debug")]
    if td.params.generate_debug_image() {
        debug_segmentation(w, h, &uf, &td.qtp)
            .expect("Error generating debug_segmentation.pnm");
    }

    tp.stamp("unionfind");

    let clusters = gradient_clusters(td, &threshim, &mut uf);
    println!("{} gradient clusters", clusters.len());
    
    #[cfg(feature="debug")]
    if td.params.generate_debug_image() {
        debug_clusters(w, h, &clusters)
            .expect("Error generating debug_clusters.pnm");
    }

    std::mem::drop(threshim);
    tp.stamp("make clusters");

    ////////////////////////////////////////////////////////
    // step 3. process each connected component.
    let quads = fit_quads(td, clusters, im);

    #[cfg(feature="debug")]
    if td.params.generate_debug_image() {
        debug_lines(im, &quads)
            .expect("Error generating debug_lines.ps");
    }

    tp.stamp("fit quads to clusters");

    std::mem::drop(uf);

    quads
}
