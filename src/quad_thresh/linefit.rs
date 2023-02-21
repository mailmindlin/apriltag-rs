use std::cmp::Ordering;

use crate::util::mem::SafeZero;

#[derive(Clone, Copy)]
pub(super) struct Pt {
    // Note: these represent 2*actual value.
    pub x: u16,
    pub y: u16,
    // pub theta: f32,
    pub gx: i16,
    pub gy: i16,
    pub slope: f32,
}

impl SafeZero for Pt {}

impl Pt {
    fn compare_angle(&self, rhs: &Pt) -> Ordering {
        f32::total_cmp(&self.slope, &rhs.slope)
    }
}

#[inline]
pub(super) fn ptsort(pts: &mut [Pt]) {
    #[inline(always)]
    fn MAYBE_SWAP(arr: &mut [Pt], apos: usize, bpos: usize) {
        if Pt::compare_angle(&arr[apos], &arr[bpos]).is_gt() {
            arr.swap(apos, bpos);
        }
    }

    match pts.len() {
        0 | 1 => {
            // Already sorted
            return;
        }
        2 => {
            MAYBE_SWAP(pts, 0, 1);
            return;
        }
        // NB: Using less-branch-intensive sorting networks here on the
        // hunch that it's better for performance.
        3 => {
            // 3 element bubble sort is optimal
            MAYBE_SWAP(pts, 0, 1);
            MAYBE_SWAP(pts, 1, 2);
            MAYBE_SWAP(pts, 0, 1);
            return;
        }
        4 => {
            // 4 element optimal sorting network.
            MAYBE_SWAP(pts, 0, 1); // sort each half, like a merge sort
            MAYBE_SWAP(pts, 2, 3);
            MAYBE_SWAP(pts, 0, 2); // minimum value is now at 0.
            MAYBE_SWAP(pts, 1, 3); // maximum value is now at end.
            MAYBE_SWAP(pts, 1, 2); // that only leaves the middle two.
            return;
        }
        5 => {
            // this 9-step swap is optimal for a sorting network, but two
            // steps slower than a generic sort.
            MAYBE_SWAP(pts, 0, 1); // sort each half (3+2), like a merge sort
            MAYBE_SWAP(pts, 3, 4);
            MAYBE_SWAP(pts, 1, 2);
            MAYBE_SWAP(pts, 0, 1);
            MAYBE_SWAP(pts, 0, 3); // minimum element now at 0
            MAYBE_SWAP(pts, 2, 4); // maximum element now at end
            MAYBE_SWAP(pts, 1, 2); // now resort the three elements 1-3.
            MAYBE_SWAP(pts, 2, 3);
            MAYBE_SWAP(pts, 1, 2);
            return;
        }
        _ => {
            // Fall back to merge sort
            pts.sort_unstable_by(Pt::compare_angle);
        }
    }
}

#[derive(Default)]
pub(super) struct LineFitPoint {
    pub(super) Mx: f64,
    pub(super) My: f64,
    pub(super) Mxx: f64,
    pub(super) Myy: f64,
    pub(super) Mxy: f64,
    /// Total weight
    pub(super) W: f64,
}

pub(super) struct LineFitData {
    pub(crate) lineparm: [f64; 4],
    pub(crate) err: f64,
    /// mean squared error
    pub(crate) mse: f64,
}


/// lfps contains *cumulative* moments for N points, with
/// index j reflecting points [0,j] (inclusive).
///
/// fit a line to the points [i0, i1] (inclusive). i0, i1 are both [0,
/// sz) if i1 < i0, we treat this as a wrap around.
pub(super) fn fit_line(lfps: &[LineFitPoint], i0: usize, i1: usize) -> LineFitData {
    let sz = lfps.len();
    assert_ne!(i0, i1);
    assert!(i0 >= 0 && i1 >= 0 && i0 < sz && i1 < sz);

    let Mx;
    let My;
    let Mxx;
    let Myy;
    let Mxy;
    let W;
    let N; // how many points are included in the set?

    if i0 < i1 {
        N = i1 - i0 + 1;

        Mx  = lfps[i1].Mx;
        My  = lfps[i1].My;
        Mxx = lfps[i1].Mxx;
        Mxy = lfps[i1].Mxy;
        Myy = lfps[i1].Myy;
        W   = lfps[i1].W;

        if i0 > 0 {
            Mx  -= lfps[i0-1].Mx;
            My  -= lfps[i0-1].My;
            Mxx -= lfps[i0-1].Mxx;
            Mxy -= lfps[i0-1].Mxy;
            Myy -= lfps[i0-1].Myy;
            W   -= lfps[i0-1].W;
        }

    } else {
        // i0 > i1, e.g. [15, 2]. Wrap around.
        assert!(i0 > 0);

        Mx  = lfps[sz-1].Mx   - lfps[i0-1].Mx;
        My  = lfps[sz-1].My   - lfps[i0-1].My;
        Mxx = lfps[sz-1].Mxx  - lfps[i0-1].Mxx;
        Mxy = lfps[sz-1].Mxy  - lfps[i0-1].Mxy;
        Myy = lfps[sz-1].Myy  - lfps[i0-1].Myy;
        W   = lfps[sz-1].W    - lfps[i0-1].W;

        Mx  += lfps[i1].Mx;
        My  += lfps[i1].My;
        Mxx += lfps[i1].Mxx;
        Mxy += lfps[i1].Mxy;
        Myy += lfps[i1].Myy;
        W   += lfps[i1].W;

        N = sz - i0 + i1 + 1;
    }

    assert!(N >= 2);

    let Ex = Mx / W;
    let Ey = My / W;
    let Cxx = Mxx / W - Ex*Ex;
    let Cxy = Mxy / W - Ex*Ey;
    let Cyy = Myy / W - Ey*Ey;

    let (nx, ny) = if true {
        // on iOS about 5% of total CPU spent in these trig functions.
        // 85 ms per frame on 5S, example.pnm
        //
        // XXX this was using the double-precision atan2. Was there a case where
        // we needed that precision? Seems doubtful.
        let normal_theta = 0.5 * f32::atan2(-2.*Cxy as f32, (Cyy - Cxx) as f32);
        let (ny,nx) = normal_theta.sin_cos();
        (nx as f64, ny as f64)
    } else {
        // 73.5 ms per frame on 5S, example.pnm
        let ty = -2.*Cxy;
        let mut tx = Cyy - Cxx;
        let mag = ty*ty + tx*tx;

        if mag == 0. {
            (1., 0.)
        } else {
            let norm = f64::hypot(tx, ty);
            tx /= norm;

            // ty is now sin(2theta)
            // tx is now cos(2theta). We want sin(theta) and cos(theta)

            // due to precision err, tx could still have slightly too large magnitude.
            if tx > 1. {
                (1., 0.)
            } else if tx < -1. {
                (0., 1.)
            } else {
                // half angle formula
                let mut ny = ((1. - tx)/2.).sqrt();
                let nx = ((1. + tx)/2.).sqrt();

                // pick a consistent branch cut
                if ty < 0. {
                    ny = -ny;
                }
                (nx, ny)
            }
        }
    };

    let lineparm = [
        Ex,
        Ey,
        nx,
        ny,
    ];

    // sum of squared errors =
    //
    // SUM_i ((p_x - ux)*nx + (p_y - uy)*ny)^2
    // SUM_i  nx*nx*(p_x - ux)^2 + 2nx*ny(p_x -ux)(p_y-uy) + ny*ny*(p_y-uy)*(p_y-uy)
    //  nx*nx*SUM_i((p_x -ux)^2) + 2nx*ny*SUM_i((p_x-ux)(p_y-uy)) + ny*ny*SUM_i((p_y-uy)^2)
    //
    //  nx*nx*N*Cxx + 2nx*ny*N*Cxy + ny*ny*N*Cyy

    // sum of squared errors
    let err = {
        let N = N as f64;
        nx*nx*N*Cxx + 2.*nx*ny*N*Cxy + ny*ny*N*Cyy
    };

    let mse = nx*nx*Cxx + 2.*nx*ny*Cxy + ny*ny*Cyy;
    LineFitData {
        lineparm,
        err,
        mse,
    }
}