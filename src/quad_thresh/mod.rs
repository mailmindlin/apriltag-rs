mod uf;
mod linefit;
mod quadfit;
mod grad_cluster;
mod threshold;
use std::{fs::OpenOptions, f64::consts as f64c, io::{BufWriter, Write}, path::PathBuf};

use crate::{detector::ApriltagDetector, util::{Image, mem::calloc, color::random_color}, quad_decode::Quad};

use self::{uf::connected_components, grad_cluster::gradient_clusters, quadfit::fit_quads};

const APRILTAG_TASKS_PER_THREAD_TARGET: usize = 10;

pub(crate) struct ApriltagQuadThreshParams {
    // reject quads containing too few pixels
    min_cluster_pixels: u32,

    // how many corner candidates to consider when segmenting a group
    // of pixels into a quad.
    max_nmaxima: u8,

    // Reject quads where pairs of edges have angles that are close to
    // straight or close to 180 degrees. Zero means that no quads are
    // rejected. (In radians).
    #[deprecated]
    critical_rad: f32,
    cos_critical_rad: f32,

    // When fitting lines to the contours, what is the maximum mean
    // squared error allowed?  This is useful in rejecting contours
    // that are far from being quad shaped; rejecting these quads "early"
    // saves expensive decoding processing.
    max_line_fit_mse: f32,

    // When we build our model of black & white pixels, we add an
    // extra check that the white model must be (overall) brighter
    // than the black model.  How much brighter? (in pixel values,
    // [0,255]). .
    min_white_black_diff: u8,

    // should the thresholded image be deglitched? Only useful for
    // very noisy images
    deglitch: bool,
}

impl Default for ApriltagQuadThreshParams {
    fn default() -> Self {
        Self {
            min_cluster_pixels: 5,
            max_nmaxima: 10,
            critical_rad: Default::default(),
            cos_critical_rad: (10. * f64c::PI / 180.).cos(),
            max_line_fit_mse: 10.,
            min_white_black_diff: 5,
            deglitch: false,
        }
    }
}

// #ifndef M_PI
// # define M_PI 3.141592653589793238462643383279502884196
// #endif

// struct quad_task {
//     zarray_t *clusters;
//     int cidx0, cidx1; // [cidx0, cidx1)
//     zarray_t *quads;
//     td: &ApriltagDetector;
//     int w, h;

//     Image *im;
// };

/*
struct unionfind_task {
    y0: usize,
    y1: usize,
    w: usize,
    h: usize,
    s: usize,
    uf: &mut UnionFind,
    im: &Image,
}

impl unionfind_task {
    #[inline(always)]
    fn do_unionfind(&mut self, dx: isize, dy: isize) {
        if (self.im[(x + dx, y + dy)] == v)
            self.uf.connect(y*w + x, (y + dy)*w + x + dx);
    }
    fn do_unionfind_line(&mut self, y: usize) {
        assert!(y+1 < self.im.height);
    
        for x in 1..(self.w-1) {
            let v = self.im[(x, y)];
    
            if (v == 127) {
                continue;
            }
    
            // (dx,dy) pairs for 8 connectivity:
            //          (REFERENCE) (1, 0)
            // (-1, 1)    (0, 1)    (1, 1)
            //
            self.do_unionfind(1, 0);
            self.do_unionfind(0, 1);
            if (v == 255) {
                self.do_unionfind(-1, 1);
                self.do_unionfind(1, 1);
            }
        }
    }

    fn execute(&self) { // do_unionfind_task
        for y in self.y0..self.y1 {
            self.do_unionfind_line(y);
        }
    }
}*/

pub fn apriltag_quad_thresh(td: &ApriltagDetector, im: &Image) -> Vec<Quad> {
    ////////////////////////////////////////////////////////
    // step 1. threshold the image, creating the edge image.

    let w = im.width;
    let h = im.height;

    let threshim = threshold::threshold(&td.qtp, &mut td.tp, im);

    if td.params.debug {
        threshim.write_pnm(&PathBuf::from("debug_threshold.pnm")).unwrap();
    }

    ////////////////////////////////////////////////////////
    // step 2. find connected components.

    let mut uf = connected_components(td, &threshim);

    // make segmentation image.
    if td.params.debug {
        let mut d = Image::<[u8; 3]>::create(w, h);

        let colors = calloc::<u32>(w * h);
        for y in 0..h {
            for x in 0..w {
                let v = uf.get_representative(y*w + x);

                if uf.get_set_size(v) < td.qtp.min_cluster_pixels {
                    continue;
                }

                let color = colors[v as usize];
                if color == 0 {
                    let c = random_color(50i16);
                    colors[v] = (c[0] << 16) | (c[1] << 8) | c[2];
                }

                let r = (0xFF & (color >> 16)) as u8;
                let g = (0xFF & (color >> 8)) as u8;
                let b = (0xFF & (color)) as u8;

                d[(x, y)] = [r, g, b];
            }
        }

        d.write_pnm(&PathBuf::from("debug_segmentation.pnm")).unwrap();
    }

    td.tp.stamp("unionfind");

    let clusters = gradient_clusters(td, &threshim, &uf);
    
    if td.params.debug {
        let mut d = Image::<[u8; 3]>::create(w, h);

        for cluster in clusters {
            let color = random_color(50);
            for p in cluster.iter() {
                let x = p.x / 2;
                let y = p.y / 2;
                d[(x, y)] = color;
            }
        }

        d.write_pnm(&PathBuf::from("debug_clusters.pnm")).unwrap();
    }

    std::mem::drop(threshim);
    td.tp.stamp("make clusters");

    ////////////////////////////////////////////////////////
    // step 3. process each connected component.
    let quads = fit_quads(td, w, h, clusters, im);

    if td.params.debug {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .open("debug_lines.ps")
            .unwrap();
        write!(f, "%%!PS\n\n");

        let mut im2 = im.clone();
        im2.darken();
        im2.darken();

        // assume letter, which is 612x792 points.
        let scale = std::cmp::min(612.0/im.width as f32, 792.0/im2.height as f32);
        write!(f, "{:.15} {:.15} scale\n", scale, scale);
        write!(f, "0 {} translate\n", im2.height);
        write!(f, "1 -1 scale\n");

        im.write_postscript(&mut f);

        for q in quads.iter() {
            let rgb = random_color::<f32>(100.);

            write!(f, "{} {} {} setrgbcolor\n", rgb[0]/255.0f32, rgb[1]/255.0f32, rgb[2]/255.0f32);
            write!(f, "{:.15} {:.15} moveto {:.15} {:.15} lineto {:.15} {:.15} lineto {:.15} {:.15} lineto {:.15} {:.15} lineto stroke\n",
                    q.corners[0].x(), q.corners[0].y(),
                    q.corners[1].x(), q.corners[1].y(),
                    q.corners[2].x(), q.corners[2].y(),
                    q.corners[3].x(), q.corners[3].y(),
                    q.corners[0].x(), q.corners[0].y());
        }
    }

    td.tp.stamp("fit quads to clusters");

    std::mem::drop(uf);
    std::mem::drop(clusters);

    quads
}
