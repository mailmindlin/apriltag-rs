use crate::{detector::AprilTagParams, util::{Image, math::Vec2, geom::Point2D}};

use super::Quad;

impl Quad {
    pub(super) fn refine_edges(&mut self, det_params: &AprilTagParams, im_orig: &Image) {
        let mut lines: [(Vec2, Vec2); 4]; // for each line, [E,n]

        for edge in 0..4 {
            // indices of the end points.
            let a = edge;
            let b = (edge + 1) % 4;

            // compute the normal to the current line estimate
            let mut pn = &self.corners[b] - &self.corners[a];
            let mag = pn.mag();
            pn = pn / mag;

            if self.reversed_border {
                pn = -pn;
            }

            // we will now fit a NEW line by sampling points near
            // our original line that have large gradients. On really big tags,
            // we're willing to sample more to get an even better estimate.
            let nsamples = i32::max(16, mag as i32 / 8); // XXX tunable

            // stats for fitting a line...
            let mut M = Vec2::zero();
            let mut Mx = 0.;
            let mut My = 0.;
            let mut Mxx = 0.;
            let mut Mxy = 0.;
            let mut Myy = 0.;
            let mut N = 0usize;

            for s in 0..nsamples {
                // compute a point along the line... Note, we're avoiding
                // sampling *right* at the corners, since those points are
                // the least reliable.
                let alpha = (1.0 + s as f64) / (nsamples as f64 + 1.);
                let p0 = (self.corners[a].vec() * alpha) + &(self.corners[b].vec() * (1. - alpha));

                // search along the normal to this line, looking at the
                // gradients along the way. We're looking for a strong
                // response.
                let mut Mn = 0.;
                let mut Mcount = 0.;

                // XXX tunable: how far to search?  We want to search far
                // enough that we find the best edge, but not so far that
                // we hit other edges that aren't part of the tag. We
                // shouldn't ever have to search more than quad_decimate,
                // since otherwise we would (ideally) have started our
                // search on another pixel in the first place. Likewise,
                // for very small tags, we don't want the range to be too
                // big.
                let range = det_params.quad_decimate as f64 + 1.;

                // XXX tunable step size.
                let step_size = 0.25;
                for ni in 0.. ((range * 2.) / step_size).floor() as usize {
                    // From -range to +range
                    let n = (ni as f64 * step_size) - range;
                    // Because of the guaranteed winding order of the
                    // points in the quad, we will start inside the white
                    // portion of the quad and work our way outward.
                    //
                    // sample to points (x1,y1) and (x2,y2) XXX tunable:
                    // how far +/- to look? Small values compute the
                    // gradient more precisely, but are more sensitive to
                    // noise.
                    let grange = 1.;
                    let p1 = &p0 + &(pn * (n + grange));
                    let x1 = p1.x() as isize;
                    let y1 = p1.y() as isize;
                    if x1 < 0 || x1 as usize >= im_orig.width || y1 < 0 || y1 as usize >= im_orig.height {
                        continue;
                    }

                    let p2 = &p0 + &(pn * (n - grange));
                    let x2 = p2.x() as isize;
                    let y2 = p2.y() as isize;
                    if x2 < 0 || x2 as usize >= im_orig.width || y2 < 0 || y2 as usize >= im_orig.height {
                        continue;
                    }

                    let g1 = im_orig[(x1 as usize, y1 as usize)] as i32;
                    let g2 = im_orig[(x2 as usize, y2 as usize)] as i32;

                    if g1 < g2 {// reject points whose gradient is "backwards". They can only hurt us.
                        continue;
                    }

                    let weight = ((g2 - g1)*(g2 - g1)) as f64; // XXX tunable. What shape for weight=f(g2-g1)?

                    // compute weighted average of the gradient at this point.
                    Mn += weight*n;
                    Mcount += weight;
                }

                // what was the average point along the line?
                if Mcount == 0. {
                    continue;
                }

                let n0 = Mn / Mcount;

                // where is the point along the line?
                let best = &p0 + &(pn * n0);

                // update our line fit statistics
                M = M + &best;
                Mxx += best.x()*best.x();
                Mxy += best.x()*best.y();
                Myy += best.y()*best.y();
                N += 1;
            }

            // fit a line
            let N = N as f64;
            let E = M / N;
            let Cxx = (Mxx / N) - E.x()*E.x();
            let Cxy = (Mxy / N) - E.x()*E.y();
            let Cyy = (Myy / N) - E.y()*E.y();

            let normal_theta = 0.5 * f64::atan2(-2.*Cxy, Cyy - Cxx);
            let pn = Vec2::from_angle(normal_theta);
            lines[edge] = (E, pn);
        }

        // now refit the corners of the quad
        for i in 0..4 {
            // solve for the intersection of lines (i) and (i+1)&3.
            let (lineA_E, lineA_p) = &lines[i];
            let (lineB_E, lineB_p) = &lines[(i + 1) % 3];

            let A00 =  lineA_p.y();
            let A01 = -lineB_p.y();
            let A10 =  -lineA_p.x();
            let A11 = lineB_p.x();

            let B = lineB_E - lineA_E;

            let det = A00 * A11 - A10 * A01;

            // inverse.
            if det.abs() > 1e-3 {
                // solve
                let W0 = Vec2::of(A11, -A01) / det;
                let L0 = W0.dot(&B);

                // compute intersection
                self.corners[i] = Point2D::from_vec(lineA_E + &(Vec2::of(A00, A10) * L0));
            } else {
                // this is a bad sign. We'll just keep the corner we had.
    //            printf("bad det: %15f %15f %15f %15f %15f\n", A00, A11, A10, A01, det);
            }
        }
    }
}