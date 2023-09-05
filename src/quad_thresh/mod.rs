mod unionfind;
mod linefit;
mod quadfit;
mod grad_cluster;
mod threshold;
use std::{fs::File, f64::consts as f64c};

use rand::thread_rng;

use crate::{detector::AprilTagDetector, util::{mem::calloc, color::RandomColor, image::{ImageWritePNM, ImageBuffer, Rgb, ImageY8, Luma}}, quad_decode::Quad, dbg::TimeProfile};

use self::{unionfind::{connected_components, UnionFind, UnionFindStatic}, grad_cluster::gradient_clusters, quadfit::fit_quads};

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
fn debug_segmentation(mut f: File, w: usize, h: usize, uf: &mut impl UnionFind<(u32, u32)>, qtp: &AprilTagQuadThreshParams) -> std::io::Result<()> {
    let mut d = ImageBuffer::<Rgb<u8>>::new(w, h);
    let mut rng = thread_rng();

    let mut colors = calloc::<Option<Rgb<u8>>>(d.pixel_count());
    for ((x, y), dst) in d.enumerate_pixels_mut() {
        let (v, v_size) = uf.get_set((x as _, y as _));

        if v_size < qtp.min_cluster_pixels {
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

    d.write_pnm(&mut f)
}

#[cfg(feature="debug")]
fn debug_unionfind_depth(mut f: File, w: usize, h: usize, uf: &mut impl UnionFindStatic<(u32, u32)>) -> std::io::Result<()> {
    let mut d = ImageBuffer::<Luma<u8>>::zeroed_packed(w, h);
    let mut max_hops = 0;
    for ((x, y), dst) in d.enumerate_pixels_mut() {
        let hops = uf.get_set_hops((x as u32, y as u32));
        max_hops = std::cmp::max(hops, max_hops);
        *dst = Luma([(1 << hops).clamp(0, 255) as u8]);
    }
    println!("Max hops: {max_hops}");

    d.write_pnm(&mut f)
}

#[cfg(feature="debug")]
fn debug_clusters(mut f: File, w: usize, h: usize, clusters: &grad_cluster::Clusters) -> std::io::Result<()> {
    use crate::util::image::ImageRGB8;

    let mut d = ImageRGB8::new(w, h);
    let mut rng = thread_rng();
    for cluster in clusters.values() {
        let color = rng.gen_color_rgb(50u8);
        for p in cluster.iter() {
            let x = (p.x / 2) as usize;
            let y = (p.y / 2) as usize;
            d.pixels_mut()[(x, y)] = color;
        }
    }

    d.write_pnm(&mut f)
}

#[cfg(feature="debug_ps")]
fn debug_lines(mut f: File, im: &ImageY8, quads: &[Quad]) -> std::io::Result<()> {
    use crate::util::image::{ImageWritePostscript, VectorPathWriter};

    let mut ps = PostScriptWriter::new(&mut f)?;

    let mut im2 = im.clone();
    im2.darken();
    im2.darken();

    // assume letter, which is 612x792 points.
    let scale = f32::min(612.0/im.width() as f32, 792.0/im2.height() as f32);
    ps.scale(scale, scale)?;
    ps.translate(0., im2.height() as f32)?;
    ps.scale(1., -1.)?;

    im.write_postscript(&mut f); //FIXME

    let mut rng = thread_rng();

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

pub(crate) fn apriltag_quad_thresh(td: &AprilTagDetector, tp: &mut TimeProfile, im: &ImageY8) -> Vec<Quad> {
    let clusters = {
        ////////////////////////////////////////////////////////
        // step 1. threshold the image, creating the edge image.

        let threshim = threshold::threshold(&td.params.qtp, tp, im);

        #[cfg(feature="debug")]
        td.params.debug_image("02_debug_threshold.pnm", |mut f| threshim.write_pnm(&mut f));

        ////////////////////////////////////////////////////////
        // step 2. find connected components.

        let mut uf = connected_components(&td.params, &threshim);

        tp.stamp("unionfind");

        // make segmentation image.
        #[cfg(feature="debug")]
        if td.params.generate_debug_image() {
            td.params.debug_image("03a_debug_segmentation.pnm", |f| debug_segmentation(f, im.width(), im.height(), &mut uf, &td.params.qtp));

            td.params.debug_image("03b_debug_uniofind_depth.pnm", |f| debug_unionfind_depth(f, im.width(), im.height(), &mut uf));
            tp.stamp("unionfind (output)");
        }

        gradient_clusters(&td.params, &threshim, uf)
    };

    #[cfg(feature="extra_debug")]
    println!("{} gradient clusters", clusters.len());
    
    #[cfg(feature="debug")]
    td.params.debug_image("04_debug_clusters.pnm", |f| debug_clusters(f, im.width(), im.height(), &clusters));
    tp.stamp("make clusters");

    ////////////////////////////////////////////////////////
    // step 3. process each connected component.
    let quads = fit_quads(td, clusters, im);

    #[cfg(feature="extra_debug")]
    for quad in quads.iter() {
        println!("Quad corner: {:?}", quad.corners);
    }

    #[cfg(feature="debug_ps")]
    td.params.debug_image("05_debug_lines.ps", |f| debug_lines(f, im, &quads));

    tp.stamp("fit quads to clusters");

    quads
}
