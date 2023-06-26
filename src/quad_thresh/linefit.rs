use std::cmp::Ordering;

use crate::util::mem::SafeZero;

#[derive(Clone, Copy, PartialEq, Debug)]
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
    pub(crate) fn compare_angle(&self, rhs: &Pt) -> Ordering {
        f32::total_cmp(&self.slope, &rhs.slope)
    }
}

#[inline]
pub(super) fn ptsort(pts: &mut [Pt]) {
    //TODO: speed test
    pts.sort_unstable_by(Pt::compare_angle);
    return;
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

#[derive(Default, Debug, Clone, Copy)]
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


#[inline(always)]
fn sqrtf(x: f64) -> f64 {
    f32::sqrt(x as f32) as f64
}

fn get_point(lfps: &[LineFitPoint], i0: usize, i1: usize) -> (LineFitPoint, usize) {
    assert_ne!(i0, i1, "i0 and i1 equal");
    assert!(i0 < lfps.len(), "i0 out of bounds");
    assert!(i1 < lfps.len(), "i0 out of bounds");

    if i0 < i1 {
        let N = i1 - i0 + 1;

        let pt = if i0 > 0 {
            let a = &lfps[i1];
            let b = &lfps[i0 - 1];
            LineFitPoint {
                Mx: a.Mx - b.Mx,
                My: a.My - b.My,
                Mxx: a.Mxx - b.Mxx,
                Myy: a.Myy - b.Myy,
                Mxy: a.Mxy - b.Mxy,
                W: a.W - b.W,
            }
        } else {
            lfps[i1].clone()
        };
        (pt, N)
    } else {
        // i0 > i1, e.g. [15, 2]. Wrap around.
        debug_assert!(i0 > 0);

        let pt_last = lfps.last().unwrap();
        let pt_i0 = &lfps[i0 - 1];
        let pt_i1 = &lfps[i1];

        let pt = LineFitPoint {
            Mx:  (pt_last.Mx   - pt_i0.Mx)   + pt_i1.Mx,
            My:  (pt_last.My   - pt_i0.My)   + pt_i1.My,
            Mxx: (pt_last.Mxx  - pt_i0.Mxx)  + pt_i1.Mxx,
            Myy: (pt_last.Mxy  - pt_i0.Mxy)  + pt_i1.Mxy,
            Mxy: (pt_last.Myy  - pt_i0.Myy)  + pt_i1.Myy,
            W:   (pt_last.W    - pt_i0.W)    + pt_i1.W,
        };

        let N = lfps.len() - i0 + i1 + 1;
        (pt, N)
    }
}

pub(crate) struct LineFitError {
    pub(crate) err: f64,
    /// mean squared error
    pub(crate) mse: f64,
}

pub(super) fn fit_line_error(lfps: &[LineFitPoint], i0: usize, i1: usize) -> LineFitError {
    let (pt, N) = get_point(lfps, i0, i1);

    assert!(N >= 2);

    let W = pt.W;
    let Ex = pt.Mx / W;
    let Ey = pt.My / W;
    let Cxx = pt.Mxx / W - Ex*Ex;
    let Cxy = pt.Mxy / W - Ex*Ey;
    let Cyy = pt.Myy / W - Ey*Ey;

    // let eig = 0.5*((pt.Mxx / W - (pt.Mx / W)*(pt.Mx / W)) + (pt.Myy / W - (pt.My / W)*(pt.My / W)) + sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy));

    // Instead of using the above cos/sin method, pose it as an eigenvalue problem.
    let eig = 0.5*(Cxx + Cyy + sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy));

    // sum of squared errors =
    //
    // SUM_i ((p_x - ux)*nx + (p_y - uy)*ny)^2
    // SUM_i  nx*nx*(p_x - ux)^2 + 2nx*ny(p_x -ux)(p_y-uy) + ny*ny*(p_y-uy)*(p_y-uy)
    //  nx*nx*SUM_i((p_x -ux)^2) + 2nx*ny*SUM_i((p_x-ux)(p_y-uy)) + ny*ny*SUM_i((p_y-uy)^2)
    //
    //  nx*nx*N*Cxx + 2nx*ny*N*Cxy + ny*ny*N*Cyy

    // sum of squared errors
    let err = N as f64 * eig;
    LineFitError { err, mse: eig }
}

/// lfps contains *cumulative* moments for N points, with
/// index j reflecting points [0,j] (inclusive).
///
/// fit a line to the points [i0, i1] (inclusive). i0, i1 are both [0,
/// sz) if i1 < i0, we treat this as a wrap around.
pub(super) fn fit_line(lfps: &[LineFitPoint], i0: usize, i1: usize) -> LineFitData {
    let (pt, N) = get_point(lfps, i0, i1);

    let W = pt.W;
    let Ex = pt.Mx / W;
    let Ey = pt.My / W;
    let Cxx = pt.Mxx / W - Ex*Ex;
    let Cxy = pt.Mxy / W - Ex*Ey;
    let Cyy = pt.Myy / W - Ey*Ey;

    // Instead of using the above cos/sin method, pose it as an eigenvalue problem.
    let eig = 0.5*(Cxx + Cyy + sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy));
    let nx1 = Cxx - eig;
    let ny1 = Cxy;
    let M1 = nx1*nx1 + ny1*ny1;
    let nx2 = Cxy;
    let ny2 = Cyy - eig;
    let M2 = nx2*nx2 + ny2*ny2;

    let (nx, ny, M) = if M1 > M2 {
        (nx1, ny1, M1)
    } else {
        (nx2, ny2, M2)
    };


    let length = sqrtf(M);

    let lineparm = [
        Ex,
        Ey,
        nx / length,
        ny / length,
    ];


    // sum of squared errors =
    //
    // SUM_i ((p_x - ux)*nx + (p_y - uy)*ny)^2
    // SUM_i  nx*nx*(p_x - ux)^2 + 2nx*ny(p_x -ux)(p_y-uy) + ny*ny*(p_y-uy)*(p_y-uy)
    //  nx*nx*SUM_i((p_x -ux)^2) + 2nx*ny*SUM_i((p_x-ux)(p_y-uy)) + ny*ny*SUM_i((p_y-uy)^2)
    //
    //  nx*nx*N*Cxx + 2nx*ny*N*Cxy + ny*ny*N*Cyy

    // sum of squared errors
    let err = N as f64 * eig;

    // mean squared error
    let mse = eig;

    LineFitData {
        lineparm,
        err,
        mse,
    }
}