use std::ops::Mul;

use crate::util::{math::{mat::{Mat33, Mat, SvdOptions}, Vec3}, geom::Point2D};

//void homography_project(const matd_t *H, double x, double y, double *ox, double *oy);
#[inline]
pub(super) fn homography_project(H: &Mat33, x: f64, y: f64) -> Point2D {
    let v = H.mul(&Vec3::of(x, y, 1.));

    let res = Point2D::of(
        v.0 / v.2, // x
        v.1 / v.2, // y
    );

    #[cfg(feature="compare_reference")]
    {
        use float_cmp::assert_approx_eq;
        let H1: Mat = (*H).into();
        let mat_sys = crate::sys::SysPtr::<apriltag_sys::matd_t>::new_like(&H1).unwrap();

        let mut ox = f64::NAN;
        let mut oy = f64::NAN;
        unsafe {
            apriltag_sys::homography_project(mat_sys.as_ptr(), x, y, &mut ox, &mut oy);
        }
        assert_approx_eq!(f64, ox, res.x(), epsilon=1e-6);
        assert_approx_eq!(f64, oy, res.y(), epsilon=1e-6);
    }

    res
}

enum HomographyMode {
    INVERSE,
    SVD,
}

// correspondences is a list of float[4]s, consisting of the points x
// and y concatenated. We will compute a homography such that y = Hx
fn homography_compute(correspondences: &[[f64; 4]], mode: HomographyMode) -> Mat33 {
    // compute centroids of both sets of points (yields a better
    // conditioned information matrix)
    let mut x_cx = 0f64;
    let mut x_cy = 0f64;
    let mut y_cx = 0f64;
    let mut y_cy = 0f64;

    for c in correspondences {
        x_cx += c[0];
        x_cy += c[1];
        y_cx += c[2];
        y_cy += c[3];
    }

    let sz = correspondences.len() as f64;
    x_cx /= sz;
    x_cy /= sz;
    y_cx /= sz;
    y_cy /= sz;

    // NB We don't normalize scale; it seems implausible that it could
    // possibly make any difference given the dynamic range of IEEE
    // doubles.

    let mut A = Mat::zeroes(9,9);
    for c in correspondences {
        // (below world is "x", and image is "y")
        let worldx = c[0] - x_cx;
        let worldy = c[1] - x_cy;
        let imagex = c[2] - y_cx;
        let imagey = c[3] - y_cy;

        let a03 = -worldx;
        let a04 = -worldy;
        let a05 = -1.;
        let a06 = worldx*imagey;
        let a07 = worldy*imagey;
        let a08 = imagey;

        A[(3, 3)] += a03*a03;
        A[(3, 4)] += a03*a04;
        A[(3, 5)] += a03*a05;
        A[(3, 6)] += a03*a06;
        A[(3, 7)] += a03*a07;
        A[(3, 8)] += a03*a08;
        A[(4, 4)] += a04*a04;
        A[(4, 5)] += a04*a05;
        A[(4, 6)] += a04*a06;
        A[(4, 7)] += a04*a07;
        A[(4, 8)] += a04*a08;
        A[(5, 5)] += a05*a05;
        A[(5, 6)] += a05*a06;
        A[(5, 7)] += a05*a07;
        A[(5, 8)] += a05*a08;
        A[(6, 6)] += a06*a06;
        A[(6, 7)] += a06*a07;
        A[(6, 8)] += a06*a08;
        A[(7, 7)] += a07*a07;
        A[(7, 8)] += a07*a08;
        A[(8, 8)] += a08*a08;

        let a10: f64 = worldx;
        let a11: f64 = worldy;
        let a12: f64 = 1.;
        let a16: f64 = -worldx*imagex;
        let a17: f64 = -worldy*imagex;
        let a18: f64 = -imagex;

        A[(0, 0)] += a10*a10;
        A[(0, 1)] += a10*a11;
        A[(0, 2)] += a10*a12;
        A[(0, 6)] += a10*a16;
        A[(0, 7)] += a10*a17;
        A[(0, 8)] += a10*a18;
        A[(1, 1)] += a11*a11;
        A[(1, 2)] += a11*a12;
        A[(1, 6)] += a11*a16;
        A[(1, 7)] += a11*a17;
        A[(1, 8)] += a11*a18;
        A[(2, 2)] += a12*a12;
        A[(2, 6)] += a12*a16;
        A[(2, 7)] += a12*a17;
        A[(2, 8)] += a12*a18;
        A[(6, 6)] += a16*a16;
        A[(6, 7)] += a16*a17;
        A[(6, 8)] += a16*a18;
        A[(7, 7)] += a17*a17;
        A[(7, 8)] += a17*a18;
        A[(8, 8)] += a18*a18;

        let a20: f64 = -worldx*imagey;
        let a21: f64 = -worldy*imagey;
        let a22: f64 = -imagey;
        let a23: f64 = worldx*imagex;
        let a24: f64 = worldy*imagex;
        let a25: f64 = imagex;

        A[(0, 0)] += a20*a20;
        A[(0, 1)] += a20*a21;
        A[(0, 2)] += a20*a22;
        A[(0, 3)] += a20*a23;
        A[(0, 4)] += a20*a24;
        A[(0, 5)] += a20*a25;
        A[(1, 1)] += a21*a21;
        A[(1, 2)] += a21*a22;
        A[(1, 3)] += a21*a23;
        A[(1, 4)] += a21*a24;
        A[(1, 5)] += a21*a25;
        A[(2, 2)] += a22*a22;
        A[(2, 3)] += a22*a23;
        A[(2, 4)] += a22*a24;
        A[(2, 5)] += a22*a25;
        A[(3, 3)] += a23*a23;
        A[(3, 4)] += a23*a24;
        A[(3, 5)] += a23*a25;
        A[(4, 4)] += a24*a24;
        A[(4, 5)] += a24*a25;
        A[(5, 5)] += a25*a25;
    }

    // make symmetric
    for i in 0..9 {
        for j in (i+1)..9 {
            A[(j,i)] = A[(i,j)];
        }
    }

    let mut H = Mat33::zeroes();

    match mode {
        HomographyMode::INVERSE => {
            // compute singular vector by (carefully) inverting the rank-deficient matrix.

            if true {
                let Ainv = A.inv().unwrap();
                let mut scale = 0.;

                for i in 0..9 {
                    let value = Ainv[(i,0)];
                    scale += value * value;
                }
                scale = scale.sqrt();

                for i in 0..3 {
                    for j in 0..3 {
                        H[(i,j)] = Ainv[(3*i+j, 0)] / scale;
                    }
                }

                std::mem::drop(Ainv);
            } else {
                let b = Mat::create(9, 1, &[ 1., 0., 0., 0., 0., 0., 0., 0., 0. ]);

                let Ainv = if false {
                    let lu = A.plu();
                    lu.solve(&b)
                } else {
                    let chol = A.chol();
                    chol.solve(&b)
                };

                let scale = (0..9)
                    .map(|i| Ainv[(i, 0)])
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .recip();

                for i in 0..3 {
                    for j in 0..3 {
                        H[(i,j)] = Ainv[(3*i+j,0)] * scale;
                    }
                }
            }
        },
        HomographyMode::SVD => {
            // compute singular vector using SVD. A bit slower, but more accurate.
            let svd = A.svd_with_flags(SvdOptions { suppress_warnings: true });

            for i in 0..3 {
                for j in 0..3 {
                    H[(i,j)] = svd.U[(3*i+j,8)];
                }
            }
        }
    }

    let Tx = {
        let mut Tx = Mat33::identity();
        Tx[(0,2)] = -x_cx;
        Tx[(1,2)] = -x_cy;
        Tx
    };

    let Ty = {
        let mut Ty = Mat33::identity();
        Ty[(0,2)] = -y_cx;
        Ty[(1,2)] = -y_cy;
        Ty
    };

    Ty.matmul(&H).matmul(&Tx)
}