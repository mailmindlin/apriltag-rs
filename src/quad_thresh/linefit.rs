use std::cmp::Ordering;

use crate::util::mem::SafeZero;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub(crate) struct Pt {
    pub slope: f32,
    // Note: these represent 2*actual value.
    pub x: u16,
    pub y: u16,
    // pub theta: f32,
    pub gx: i16,
    pub gy: i16,
}

impl SafeZero for Pt {}

impl Pt {
    pub(crate) fn compare_angle(&self, rhs: &Pt) -> Ordering {
        f32::total_cmp(&self.slope, &rhs.slope)
    }
}

#[cfg(feature="compare_reference")]
impl From<Pt> for apriltag_sys::pt {
    fn from(pt: Pt) -> Self {
        Self {
            x: pt.x,
            y: pt.y,
            gx: pt.gx,
            gy: pt.gy,
            slope: pt.slope,
        }
    }
}

#[cfg(feature="compare_reference")]
impl From<apriltag_sys::pt> for Pt {
    fn from(pt: apriltag_sys::pt) -> Self {
        Self {
            x: pt.x,
            y: pt.y,
            gx: pt.gx,
            gy: pt.gy,
            slope: pt.slope,
        }
    }
}

#[inline]
#[cfg(feature="compare_reference")]
fn ptsort_inner(pts: &mut [Pt]) {
    //TODO: speed test
    #[inline(always)]
    fn MAYBE_SWAP(arr: &mut [Pt], apos: usize, bpos: usize) {
        if Pt::compare_angle(&arr[apos], &arr[bpos]).is_le() {
            arr.swap(apos, bpos);
        }
    }

    match pts.len() {
        0 | 1 => {
            // Already sorted
            return;
        },
        2 => {
            MAYBE_SWAP(pts, 0, 1);
        },
        // NB: Using less-branch-intensive sorting networks here on the
        // hunch that it's better for performance.
        3 => {
            // 3 element bubble sort is optimal
            MAYBE_SWAP(pts, 0, 1);
            MAYBE_SWAP(pts, 1, 2);
            MAYBE_SWAP(pts, 0, 1);
            return;
        },
        4 => {
            // 4 element optimal sorting network.
            MAYBE_SWAP(pts, 0, 1); // sort each half, like a merge sort
            MAYBE_SWAP(pts, 2, 3);
            MAYBE_SWAP(pts, 0, 2); // minimum value is now at 0.
            MAYBE_SWAP(pts, 1, 3); // maximum value is now at end.
            MAYBE_SWAP(pts, 1, 2); // that only leaves the middle two.
            return;
        },
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
        },
        _ => {
            // Fall back to merge sort
            // pts.sort_unstable_by(Pt::compare_angle);
            let mut tmp = pts.to_vec();

            let (A, B) = tmp.split_at_mut(pts.len() / 2);
            ptsort_inner(A);
            ptsort_inner(B);

            let mut Apos = 0;
            let mut Bpos = 0;
            let mut Rpos = 0;

            let mut MERGE = |Apos: &mut usize, Bpos: &mut usize| {
                if Pt::compare_angle(&A[*Apos], &B[*Bpos]).is_lt() {
                    pts[Rpos] = A[*Apos];
                    *Apos += 1;
                } else {
                    pts[Rpos] = B[*Bpos];
                    *Bpos += 1;
                }
                Rpos += 1;
            };
            while Apos + 8 < A.len() && Bpos + 8 < B.len() {
                MERGE(&mut Apos, &mut Bpos); MERGE(&mut Apos, &mut Bpos);
                MERGE(&mut Apos, &mut Bpos); MERGE(&mut Apos, &mut Bpos);
                MERGE(&mut Apos, &mut Bpos); MERGE(&mut Apos, &mut Bpos);
                MERGE(&mut Apos, &mut Bpos); MERGE(&mut Apos, &mut Bpos);
            }

            while Apos < A.len() && Bpos < B.len() {
                MERGE(&mut Apos, &mut Bpos);
            }

            if Apos < A.len() {
                let count = A.len() - Apos;
                pts[Rpos..Rpos+count].copy_from_slice(&A[Apos..]);
                Rpos += count;
            }

            if Bpos < B.len() {
                let count = B.len() - Bpos;
                pts[Rpos..Rpos+count].copy_from_slice(&B[Bpos..]);
                Rpos += count;
            }
            assert_eq!(Rpos, pts.len());
        }
    }
}

#[cfg(not(feature="compare_reference"))]
#[inline]
pub(super) fn ptsort(pts: &mut [Pt]) {
    pts.sort_unstable_by(Pt::compare_angle);
}

#[cfg(feature="compare_reference")]
pub(super) fn ptsort(pts: &mut [Pt]) {
    let initial = pts.to_vec();
    let mut pts_sys = Vec::with_capacity(pts.len());
    for pt in pts.iter() {
        pts_sys.push(apriltag_sys::pt::from(*pt));
    }
    assert_eq!(pts.len(), pts_sys.len());

    ptsort_inner(pts);

    unsafe {
        apriltag_sys::ptsort(pts_sys.as_mut_ptr(), pts_sys.len() as _);
    }

    // fn compare_indices<T: Into<Pt> + Clone>(base: &[Pt], shuffled: &[T]) -> Vec<usize> {
    //     assert_eq!(base.len(), shuffled.len());
    //     let mut results = Vec::with_capacity(base.len());
    //     let mut idx_used = vec![false; base.len()];
    //     for i in 0..shuffled.len() {
    //         let shv: Pt = shuffled[i].clone().into();
    //         let idx = if base[i] == shv && idx_used[i] {
    //             i
    //         } else {
    //             base.iter()
    //                 .enumerate()
    //                 .find(|(i, bv)| !idx_used[*i] && *bv == &shv)
    //                 .map(|(i, _)| i)
    //                 .unwrap()
    //         };
    //         idx_used[idx] = true;
    //         results.push(idx);
    //     }
    //     results
    // }
    // println!(" R ptsort: {:?}", compare_indices(&initial, pts));
    // println!(" C ptsort: {:?}", compare_indices(&initial, &pts_sys));

    for (i, (rs, sys)) in pts.iter().zip(pts_sys.iter()).enumerate() {
        let sys = Pt::from(*sys);
        assert_eq!(*rs, sys, "Mismatch at index {i}");
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

#[cfg(feature="compare_reference")]
impl From<apriltag_sys::line_fit_pt> for LineFitPoint {
    fn from(value: apriltag_sys::line_fit_pt) -> Self {
        Self {
            Mx: value.Mx,
            My: value.My,
            Mxx: value.Mxx,
            Myy: value.Myy,
            Mxy: value.Mxy,
            W: value.W,
        }
    }
}

#[cfg(feature="compare_reference")]
impl From<LineFitPoint> for apriltag_sys::line_fit_pt {
    fn from(value: LineFitPoint) -> Self {
        Self {
            Mx: value.Mx,
            My: value.My,
            Mxx: value.Mxx,
            Myy: value.Myy,
            Mxy: value.Mxy,
            W: value.W,
        }
    }
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
            Myy: (pt_last.Myy  - pt_i0.Myy)  + pt_i1.Myy,
            Mxy: (pt_last.Mxy  - pt_i0.Mxy)  + pt_i1.Mxy,
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

#[cfg(feature="compare_reference")]
fn fit_line_sys(lfps: &[LineFitPoint], i0: usize, i1: usize) -> LineFitData {
    // println!("Calling apriltag_sys::fit_line(#{}, {i0}, {i1})", lfps.len());
    let mut lfps_sys = Vec::new();
    for lfp in lfps.iter() {
        lfps_sys.push(apriltag_sys::line_fit_pt {
            Mx: lfp.Mx,
            My: lfp.My,
            Mxx: lfp.Mxx,
            Myy: lfp.Myy,
            Mxy: lfp.Mxy,
            W: lfp.W,
        });
    }
    let mut lineparm = [0f64; 4];
    let mut err = f64::NAN;
    let mut mse = f64::NAN;
    unsafe {
        apriltag_sys::fit_line(
            lfps_sys.as_mut_ptr(),
            lfps_sys.len().try_into().unwrap(),
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            lineparm.as_mut_ptr(),
            &mut err,
            &mut mse
        );
    }
    LineFitData { lineparm, err, mse }
}

pub(super) fn fit_line_error(lfps: &[LineFitPoint], i0: usize, i1: usize) -> LineFitError {
    let (pt, N) = get_point(lfps, i0, i1);
    // println!("->Mx={:.15} My={:.15} Mxx={:.15} Mxy={:.15} Myy={:.15} W={:.15} N={}", pt.Mx, pt.My, pt.Mxx, pt.Mxy, pt.Myy, pt.W, N);

    assert!(N >= 2);

    let W = pt.W;
    let Ex = pt.Mx / W;
    let Ey = pt.My / W;
    let Cxx = pt.Mxx / W - Ex*Ex;
    let Cxy = pt.Mxy / W - Ex*Ey;
    let Cyy = pt.Myy / W - Ey*Ey;
    // println!("->E=({Ex:.15}, {Ey:.15}) Cxx={Cxx:.15} Cxy={Cxy:.15} Cyy={Cyy:.15}");

    // let eig = 0.5*((pt.Mxx / W - (pt.Mx / W)*(pt.Mx / W)) + (pt.Myy / W - (pt.My / W)*(pt.My / W)) + sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy));

    // Instead of using the above cos/sin method, pose it as an eigenvalue problem.
    let eig = 0.5*(Cxx + Cyy - sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy));
    // println!("->Inner {:.15} -> {:.15} -> {eig:.15}", (Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy, sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy));

    // sum of squared errors =
    //
    // SUM_i ((p_x - ux)*nx + (p_y - uy)*ny)^2
    // SUM_i  nx*nx*(p_x - ux)^2 + 2nx*ny(p_x -ux)(p_y-uy) + ny*ny*(p_y-uy)*(p_y-uy)
    //  nx*nx*SUM_i((p_x -ux)^2) + 2nx*ny*SUM_i((p_x-ux)(p_y-uy)) + ny*ny*SUM_i((p_y-uy)^2)
    //
    //  nx*nx*N*Cxx + 2nx*ny*N*Cxy + ny*ny*N*Cyy

    // sum of squared errors
    let err = N as f64 * eig;
    let mse = eig;

    #[cfg(feature="compare_reference")]
    {
        use float_cmp::assert_approx_eq;
        let res_sys = fit_line_sys(lfps, i0, i1);

        assert_approx_eq!(f64, mse, res_sys.mse, epsilon = 1e-5);
        assert_approx_eq!(f64, err, res_sys.err, epsilon = 1e-5);
    }

    LineFitError { err, mse }
}

/// lfps contains *cumulative* moments for N points, with
/// index j reflecting points [0,j] (inclusive).
///
/// fit a line to the points [i0, i1] (inclusive). i0, i1 are both [0,
/// sz) if i1 < i0, we treat this as a wrap around.
pub(super) fn fit_line(lfps: &[LineFitPoint], i0: usize, i1: usize) -> LineFitData {
    let (pt, N) = get_point(lfps, i0, i1);

    // println!("\t R fit_line: Mx={:.15} My={:.15} Mxx={:.15} Mxy={:.15} Myy={:.15} W={:.15} N={}", pt.Mx, pt.My, pt.Mxx, pt.Mxy, pt.Myy, pt.W, N);

    let W = pt.W;
    let Ex = pt.Mx / W;
    let Ey = pt.My / W;
    let Cxx = pt.Mxx / W - Ex*Ex;
    let Cxy = pt.Mxy / W - Ex*Ey;
    let Cyy = pt.Myy / W - Ey*Ey;

    let eig_small = 0.5*(Cxx + Cyy - sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4. * Cxy*Cxy));

    // println!("\t R fit_line: \tE=({:.15}, {:.15}) Cxx={:.15} Cxy={:.15} Cyy={:.15} Esm={eig_small:.15}", Ex, Ey, Cxx, Cxy, Cyy);

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
    let err = N as f64 * eig_small;

    // mean squared error
    let mse = eig_small;

    // println!("\t R fit_line: \tL={lineparm:.15?} err={err:.15} mse={mse:.15}");

    #[cfg(feature="compare_reference")]
    {
        use float_cmp::assert_approx_eq;
        let res_sys = fit_line_sys(lfps, i0, i1);

        const EPSILON: f64 = 1e-6;

        assert_approx_eq!(f64, mse, res_sys.mse, epsilon = EPSILON);
        assert_approx_eq!(f64, err, res_sys.err, epsilon = EPSILON);
        assert_approx_eq!(&[f64], &lineparm, &res_sys.lineparm, epsilon = EPSILON);
    }

    LineFitData {
        lineparm,
        err,
        mse,
    }
}