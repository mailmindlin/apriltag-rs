use std::ops::Mul;

use super::{math::{mat::{Mat, Mat33, SvdOptions}, Vec3}, geom::Point2D};

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
pub(crate) fn homography_to_pose1(H: &Mat33, fx: f64, fy: f64, cx: f64, cy: f64) -> Mat {
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
            let svd = Mat33::of([
                R00, R01, R02,
                R10, R11, R12,
                R20, R21, R22,
            ]).svd();
            svd.U.matmul_transpose(&svd.V)
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
pub(crate) fn homography_to_model_view(H: &Mat, F: f64, G: f64, A: f64, B: f64, _C: f64, _D: f64) -> Mat {
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
    let s = if TZ > 0. {
        -s
    } else { s };

    let R20 = s * R20;
    let R21 = s * R21;
    let TZ  = s * TZ;
    let R00 = s * R00;
    let R01 = s * R01;
    let TX  = s * TX;
    let R10 = s * R10;
    let R11 = s * R11;
    let TY  = s * TY;

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
fn quat_to_matrix(q: &[f64; 4], M: &mut Mat) {
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
