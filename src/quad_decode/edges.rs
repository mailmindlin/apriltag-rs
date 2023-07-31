use arrayvec::ArrayVec;

use crate::{detector::DetectorConfig, util::{math::{Vec2, Vec2Builder, FMA}, geom::Point2D, image::ImageY8}};

use super::Quad;

struct FRange {
    current: f64,
    step: f64,
    stop: f64,
}

impl Iterator for FRange {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.current;
        if result <= self.stop {
            self.current += self.step;
            Some(result)
        } else {
            None
        }
    }
}

impl Quad {
    pub(super) fn refine_edges(&mut self, det_params: &DetectorConfig, im_orig: &ImageY8) {
        // println!("refine_edges corners: {:?}", self.corners);
        #[cfg(feature="compare_reference")]
        let mut quad_sys = apriltag_sys::quad {
            p: self.corners.as_array_f32(),
            reversed_border: self.reversed_border,
            H: std::ptr::null_mut(),
            Hinv: std::ptr::null_mut(),
        };
        #[cfg(feature="compare_reference")]
        {
            println!("\n=== Start refine_edges ===");
            // Fix divergence that arises from libapriltag storing corners as f32's
            self.corners = crate::geom::Quadrilateral::from_array(&quad_sys.p);
        }

        let lines = {
            let mut lines = ArrayVec::<(Vec2, Vec2), 4>::new(); // for each line, [E,n]

            for edge in 0..4 {
                // indices of the end points.
                let a = edge;
                let b = (edge + 1) % 4;
                // println!(" R refine_edges: Edge {edge} ({a}, {b})");

                // compute the normal to the current line estimate
                let mut pn = (&self.corners[b] - &self.corners[a]).rev_negx();
                let mag = pn.mag();
                pn = pn / mag;

                if self.reversed_border {
                    pn = -pn;
                }
                // println!(" R refine_edges: \tn={pn:.15?}");

                // we will now fit a NEW line by sampling points near
                // our original line that have large gradients. On really big tags,
                // we're willing to sample more to get an even better estimate.
                let nsamples = i32::max(16, mag as i32 / 8); // XXX tunable

                // stats for fitting a line...
                let mut M = Vec2::zero();
                let mut Mxx = 0.;
                let mut Mxy = 0.;
                let mut Myy = 0.;
                let mut N = 0.;

                for s in 0..nsamples {
                    // compute a point along the line... Note, we're avoiding
                    // sampling *right* at the corners, since those points are
                    // the least reliable.
                    let alpha = (1.0 + s as f64) / (nsamples as f64 + 1.);
                    let p0 = (self.corners[a].vec() * alpha) + &(self.corners[b].vec() * (1. - alpha));
                    // println!(" R refine_edges: \tp0={p0:.15?}");

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
                    for n in (FRange { current: -range, stop: range, step: step_size}) {
                    // for ni in 0.. ((range * 2.) / step_size).floor() as usize {
                        // From -range to +range
                        // let n = (ni as f64 * step_size) - range;

                        // Because of the guaranteed winding order of the
                        // points in the quad, we will start inside the white
                        // portion of the quad and work our way outward.
                        //
                        // sample to points (x1,y1) and (x2,y2) XXX tunable:
                        // how far +/- to look? Small values compute the
                        // gradient more precisely, but are more sensitive to
                        // noise.
                        let grange = 1.;
                        let p1 = p0.fma(pn, Vec2::dup(n + grange));
                        let x1 = p1.x() as isize;
                        let y1 = p1.y() as isize;
                        if x1 < 0 || x1 as usize >= im_orig.width() || y1 < 0 || y1 as usize >= im_orig.height() {
                            continue;
                        }

                        let p2 = p0.fma(pn, Vec2::dup(n - grange));
                        let x2 = p2.x() as isize;
                        let y2 = p2.y() as isize;
                        if x2 < 0 || x2 as usize >= im_orig.width() || y2 < 0 || y2 as usize >= im_orig.height() {
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
                    // println!(" R refine_edges: \tMn={Mn} Mcount={Mcount}");

                    // what was the average point along the line?
                    if Mcount == 0. {
                        continue;
                    }

                    let n0 = Mn / Mcount;

                    // where is the point along the line?
                    let best = p0.fma(Vec2::dup(n0), pn);

                    // update our line fit statistics
                    M += best;
                    // Mgrad += best.gradient();
                    Mxx += best.x()*best.x();
                    Mxy += best.x()*best.y();
                    Myy += best.y()*best.y();
                    N += 1.;
                }

                // fit a line
                // let N = N as f64;
                let E = M / N;
                let Cxx = (Mxx / N) - E.x()*E.x();
                let Cxy = (Mxy / N) - E.x()*E.y();
                let Cyy = (Myy / N) - E.y()*E.y();

                let normal_theta = 0.5 * f32::atan2((-2.*Cxy) as f32, (Cyy - Cxx) as f32);
                let pn = Vec2::of(f32::cos(normal_theta) as _, f32::sin(normal_theta) as _);
                // println!(" R refine_edges: \tN={N} M={M:?}  Mxx={} Mxy={} Myy={} E={E:?} Cxx={:.15} Cxy={:.15} Cyy={:.15} theta={normal_theta} pn={pn:?}", Mxx, Mxy, Myy, Cxx, Cxy, Cyy);
                // println!(" R refine_edges: \tatan2({:.15}, {:.15}) {:.15} {:.15}", -2. * Cxy, (Cyy - Cxx), Mxy/N, E.x() * E.y());
                // let pn = Vec2::from_angle(normal_theta);
                lines.push((E, pn));
            }
            lines.into_inner().unwrap()
        };
        // println!(" R refine_edges: Lines = {lines:?}\n");

        // now refit the corners of the quad
        for i in 0..4 {
            // solve for the intersection of lines (i) and (i+1)&3.
            let (lineA_E, lineA_p) = &lines[i]; // Current
            let (lineB_E, lineB_p) = &lines[(i + 1) % 4]; // Next

            // let Ax0 = lineA_p.rev_negx();
            // let Ax1 = lineB_p.rev_negy();
            let A00 =  lineA_p.y();
            let A10 =  -lineA_p.x();
            let A01 = -lineB_p.y();
            let A11 = lineB_p.x();

            let B = lineB_E - lineA_E;

            // Ay * Bx - (-Ax * -By)
            let det = A00 * A11 - A10 * A01;

            // println!(" R refine_edges[{i}]: A00={A00:.15} A01={A01:.15} A10={A10:.15} A11={A11:.15} B={B:.15?} det={det:.15}");

            // inverse.
            if det.abs() > 1e-3 {
                // solve
                let W0 = Vec2::of(A11, -A01) / det;
                let L0 = W0.dot(B);
                // println!(" R refine_edges: \tW0={W0:.15?} L0={L0:.15}");

                // compute intersection
                self.corners[(i + 1) % 4] = Point2D::from_vec(lineA_E + &(Vec2::of(A00, A10) * L0));
            } else {
                // this is a bad sign. We'll just keep the corner we had.
                #[cfg(feature="extra_debug")]
                eprintln!("bad det {i}: {A00:15} {A11:15} {A10:15} {A01:15} {det}");
            }
        }

        #[cfg(feature="compare_reference")]
        {
            use crate::sys::{AprilTagDetectorSys, ImageU8Sys};
            use float_cmp::assert_approx_eq;

            let mut td_sys = AprilTagDetectorSys::new().unwrap();
            td_sys.as_mut().quad_decimate = det_params.quad_decimate;
            let im_sys = ImageU8Sys::new(im_orig).unwrap();
            
            println!("Calling apriltag_sys::refine_edges");
            std::io::stdout().flush().unwrap();
            unsafe {
                apriltag_sys::refine_edges(td_sys.as_ptr(), im_sys.as_ptr(), &mut quad_sys);
            }
            std::io::stdout().flush().unwrap();

            let corners_sys = Quadrilateral::from_array(&quad_sys.p);
            // assert_approx_eq!(Quadrilateral, self.corners, corners_sys, epsilon = 1e-6);
            println!("=== End refine_edges ===\n");
        }
    }
}