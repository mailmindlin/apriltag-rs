use super::math::mat::Mat;

/// Similar to above
/// Recover the model view matrix assuming that the projection matrix is:
///
/// [ F  0  A  0 ]     (see glFrustrum)
/// [ 0  G  B  0 ]
/// [ 0  0  C  D ]
/// [ 0  0 -1  0 ]
pub fn homography_to_model_view(H: &Mat, F: f64, G: f64, A: f64, B: f64, _C: f64, _D: f64) -> Mat {
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
