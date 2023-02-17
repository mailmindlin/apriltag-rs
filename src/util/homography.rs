use super::{math::mat::{Mat, SvdOptions}, geom::Point2D};



//void homography_project(const matd_t *H, double x, double y, double *ox, double *oy);
#[inline]
pub(crate) fn homography_project(H: &Mat, x: f64, y: f64) -> Point2D {
    let xx = H[(0, 0)]*x + H[(0, 1)]*y + H[(0, 2)];
    let yy = H[(1, 0)]*x + H[(1, 1)]*y + H[(1, 2)];
    let zz = H[(2, 0)]*x + H[(2, 1)]*y + H[(2, 2)];

    return Point2D::of(
        xx / zz, // x
        yy / zz, // y
    );
}

pub(crate) enum HomographyMode {
    INVERSE,
    SVD,
}

// correspondences is a list of float[4]s, consisting of the points x
// and y concatenated. We will compute a homography such that y = Hx
pub fn homography_compute(correspondences: &[[f64; 4]], mode: HomographyMode) -> Mat {
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

    let A = Mat::zeroes(9,9);
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

    let H = Mat::zeroes(3,3);

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

                let mut scale = 0.;

                for i in 0..9 {
                    let value = Ainv[(i,0)];
                    scale += value * value;
                }
                scale = scale.sqrt();

                for i in 0..3 {
                    for j in 0..3 {
                        H[(i,j)] = Ainv[(3*i+j,0)] / scale;
                    }
                }

                std::mem::drop(b);
                std::mem::drop(Ainv);
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

    let Tx = Mat::identity(3);
    Tx[(0,2)] = -x_cx;
    Tx[(1,2)] = -x_cy;

    let Ty = Mat::identity(3);
    Ty[(0,2)] = -y_cx;
    Ty[(1,2)] = -y_cy;

    Mat::op("M*M*M", &[&Ty, &H, &Tx]).unwrap()
}



/// assuming that the projection matrix is:
/// [ fx 0  cx 0 ]
/// [  0 fy cy 0 ]
/// [  0  0  1 0 ]
///
/// And that the homography is equal to the projection matrix times the
/// model matrix, recover the model matrix (which is returned). Note
/// that the third column of the model matrix is missing in the
/// expresison below, reflecting the fact that the homography assumes
/// all points are at z=0 (i.e., planar) and that the element of z is
/// thus omitted.  (3x1 instead of 4x1).
///
/// [ fx 0  cx 0 ] [ R00  R01  TX ]    [ H00 H01 H02 ]
/// [  0 fy cy 0 ] [ R10  R11  TY ] =  [ H10 H11 H12 ]
/// [  0  0  1 0 ] [ R20  R21  TZ ] =  [ H20 H21 H22 ]
///                [  0    0    1 ]
///
/// fx*R00 + cx*R20 = H00   (note, H only known up to scale; some additional adjustments required; see code.)
/// fx*R01 + cx*R21 = H01
/// fx*TX  + cx*TZ  = H02
/// fy*R10 + cy*R20 = H10
/// fy*R11 + cy*R21 = H11
/// fy*TY  + cy*TZ  = H12
/// R20 = H20
/// R21 = H21
/// TZ  = H22
pub fn homography_to_pose(H: &Mat, fx: f64, fy: f64, cx: f64, cy: f64) -> Mat {
    // Note that every variable that we compute is proportional to the scale factor of H.
    let mut R20 = H[(2, 0)];
    let mut R21 = H[(2, 1)];
    let mut TZ  = H[(2, 2)];
    let mut R00 = (H[(0, 0)] - cx*R20) / fx;
    let mut R01 = (H[(0, 1)] - cx*R21) / fx;
    let mut TX  = (H[(0, 2)] - cx*TZ)  / fx;
    let mut R10 = (H[(1, 0)] - cy*R20) / fy;
    let mut R11 = (H[(1, 1)] - cy*R21) / fy;
    let mut TY  = (H[(1, 2)] - cy*TZ)  / fy;

    // compute the scale by requiring that the rotation columns are unit length
    // (Use geometric average of the two length vectors we have)
    let length1 = f64::sqrt(R00*R00 + R10*R10 + R20*R20);
    let length2 = f64::sqrt(R01*R01 + R11*R11 + R21*R21);
    let mut s = f64::sqrt((length1 * length2) as f64).recip();

    // get sign of S by requiring the tag to be in front the camera;
    // we assume camera looks in the -Z direction.
    if TZ > 0. {
        s *= -1.;
    }

    R20 *= s;
    R21 *= s;
    TZ  *= s;
    R00 *= s;
    R01 *= s;
    TX  *= s;
    R10 *= s;
    R11 *= s;
    TY  *= s;

    // now recover [R02 R12 R22] by noting that it is the cross product of the other two columns.
    let mut R02 = R10*R21 - R20*R11;
    let mut R12 = R20*R01 - R00*R21;
    let mut R22 = R00*R11 - R10*R01;

    // Improve rotation matrix by applying polar decomposition.
    if true {
        // do polar decomposition. This makes the rotation matrix
        // "proper", but probably increases the reprojection error. An
        // iterative alignment step would be superior.

        let R = {
            let svd = {
                let R = Mat::create(3, 3, &[ R00, R01, R02,
                                                               R10, R11, R12,
                                                               R20, R21, R22 ]);
        
                R.svd()
            };
            Mat::op("M*M'", &[&svd.U, &svd.V]).unwrap()
        };

        R00 = R[(0, 0)];
        R01 = R[(0, 1)];
        R02 = R[(0, 2)];
        R10 = R[(1, 0)];
        R11 = R[(1, 1)];
        R12 = R[(1, 2)];
        R20 = R[(2, 0)];
        R21 = R[(2, 1)];
        R22 = R[(2, 2)];
    }

    return Mat::create(4, 4, &[
        R00, R01, R02, TX,
        R10, R11, R12, TY,
        R20, R21, R22, TZ,
         0.,  0.,  0., 1.,
    ]);
}

/// Similar to above
/// Recover the model view matrix assuming that the projection matrix is:
///
/// [ F  0  A  0 ]     (see glFrustrum)
/// [ 0  G  B  0 ]
/// [ 0  0  C  D ]
/// [ 0  0 -1  0 ]
pub fn homography_to_model_view(H: &Mat, F: f64, G: f64, A: f64, B: f64, C: f64, D: f64) -> Mat {
    // Note that every variable that we compute is proportional to the scale factor of H.
    let R20 = -H[(2, 0)];
    let R21 = -H[(2, 1)];
    let TZ  = -H[(2, 2)];
    let R00 = (H[(0, 0)] - A*R20) / F;
    let R01 = (H[(0, 1)] - A*R21) / F;
    let TX  = (H[(0, 2)] - A*TZ)  / F;
    let R10 = (H[(1, 0)] - B*R20) / G;
    let R11 = (H[(1, 1)] - B*R21) / G;
    let TY  = (H[(1, 2)] - B*TZ)  / G;

    // compute the scale by requiring that the rotation columns are unit length
    // (Use geometric average of the two length vectors we have)
    let length1 = f64::sqrt(R00*R00 + R10*R10 + R20*R20);
    let length2 = f64::sqrt(R01*R01 + R11*R11 + R21*R21);
    let s = f64::sqrt((length1 * length2) as f64).recip();

    // get sign of S by requiring the tag to be in front of the camera
    // (which is Z < 0) for our conventions.
    if TZ > 0. {
        s *= -1.;
    }

    R20 *= s;
    R21 *= s;
    TZ  *= s;
    R00 *= s;
    R01 *= s;
    TX  *= s;
    R10 *= s;
    R11 *= s;
    TY  *= s;

    // now recover [R02 R12 R22] by noting that it is the cross product of the other two columns.
    let R02 = R10*R21 - R20*R11;
    let R12 = R20*R01 - R00*R21;
    let R22 = R00*R11 - R10*R01;

    // TODO XXX: Improve rotation matrix by applying polar decomposition.

    Mat::create(4, 4, &[
        R00, R01, R02, TX,
        R10, R11, R12, TY,
        R20, R21, R22, TZ,
        0., 0., 0., 1.
    ])
}

// Only uses the upper 3x3 matrix.
/*
static void matrix_to_quat(const matd_t *R, double q[4])
{
    // see: "from quaternion to matrix and back"

    // trace: get the same result if R is 4x4 or 3x3:
    double T = MATD_EL(R, 0, 0) + MATD_EL(R, 1, 1) + MATD_EL(R, 2, 2) + 1;
    double S = 0;

    double m0  = MATD_EL(R, 0, 0);
    double m1  = MATD_EL(R, 1, 0);
    double m2  = MATD_EL(R, 2, 0);
    double m4  = MATD_EL(R, 0, 1);
    double m5  = MATD_EL(R, 1, 1);
    double m6  = MATD_EL(R, 2, 1);
    double m8  = MATD_EL(R, 0, 2);
    double m9  = MATD_EL(R, 1, 2);
    double m10 = MATD_EL(R, 2, 2);

    if (T > 0.0000001) {
        S = sqrtf(T) * 2;
        q[1] = -( m9 - m6 ) / S;
        q[2] = -( m2 - m8 ) / S;
        q[3] = -( m4 - m1 ) / S;
        q[0] = 0.25 * S;
    } else if ( m0 > m5 && m0 > m10 )  {	// Column 0:
        S  = sqrtf( 1.0 + m0 - m5 - m10 ) * 2;
        q[1] = -0.25 * S;
        q[2] = -(m4 + m1 ) / S;
        q[3] = -(m2 + m8 ) / S;
        q[0] = (m9 - m6 ) / S;
    } else if ( m5 > m10 ) {			// Column 1:
        S  = sqrtf( 1.0 + m5 - m0 - m10 ) * 2;
        q[1] = -(m4 + m1 ) / S;
        q[2] = -0.25 * S;
        q[3] = -(m9 + m6 ) / S;
        q[0] = (m2 - m8 ) / S;
    } else {
        // Column 2:
        S  = sqrtf( 1.0 + m10 - m0 - m5 ) * 2;
        q[1] = -(m2 + m8 ) / S;
        q[2] = -(m9 + m6 ) / S;
        q[3] = -0.25 * S;
        q[0] = (m4 - m1 ) / S;
    }

    double mag2 = 0;
    for (int i = 0; i < 4; i++)
        mag2 += q[i]*q[i];
    double norm = 1.0 / sqrtf(mag2);
    for (int i = 0; i < 4; i++)
        q[i] *= norm;
}
*/

/// overwrites upper 3x3 area of matrix M. Doesn't touch any other elements of M.
pub fn quat_to_matrix(q: &[f64; 4], M: &mut Mat) {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];

    M[(0, 0)] = w*w + x*x - y*y - z*z;
    M[(0, 1)] = 2.*x*y - 2.*w*z;
    M[(0, 2)] = 2.*x*z + 2.*w*y;

    M[(1, 0)] = 2.*x*y + 2.*w*z;
    M[(1, 1)] = w*w - x*x + y*y - z*z;
    M[(1, 2)] = 2.*y*z - 2.*w*x;

    M[(2, 0)] = 2.*x*z - 2.*w*y;
    M[(2, 1)] = 2.*y*z + 2.*w*x;
    M[(2, 2)] = w*w - x*x - y*y + z*z;
}
