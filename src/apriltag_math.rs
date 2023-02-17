
/// Computes the cholesky factorization of A, putting the lower
/// triangular matrix into R.
#[inline]
fn mat33_chol(A: &[[f64; 3]; 3]) -> [f64; 9] {
    // A[0] = R[0]*R[0]
    let R0 = A[0][0].sqrt();

    // A[1] = R[0]*R[3];
    let R3 = A[0][1] / R0;

    // A[2] = R[0]*R[6];
    let R6 = A[0][2] / R0;

    // A[4] = R[3]*R[3] + R[4]*R[4]
    let R4 = f64::sqrt(A[1][1] - R3*R3);

    // A[5] = R[3]*R[6] + R[4]*R[7]
    let R7 = (A[1][2] - R3*R6) / R4;

    // A[8] = R[6]*R[6] + R[7]*R[7] + R[8]*R[8]
    let R8 = f64::sqrt(A[2][2] - R6*R6 - R7*R7);

    [
        R0,
        0.0,
        0.0,
        R3,
        R4,
        0.0,
        R6,
        R7,
        R8,
    ]
}

#[inline]
fn mat33_lower_tri_inv(A: &[f64]) -> [f64; 9] {
    // A[0]*R[0] = 1
    let R0 = A[0].recip();

    // A[3]*R0 + A[4]*R3 = 0
    let R3 = -A[3]*R0 / A[4];

    // A[4]*R4 = 1
    let R4 = A[4].recip();

    // A[6]*R0 + A[7]*R3 + A[8]*R6 = 0
    let R6 = (-A[6]*R0 - A[7]*R3) / A[8];

    // A[7]*R4 + A[8]*R7 = 0
    let R7 = -A[7]*R4 / A[8];

    // A[8]*R8 = 1
    let R8 = A[8].recip();

    [
        R0,
        0.0,
        0.0,
        R3,
        R4,
        0.0,
        R6,
        R7,
        R8,
    ]
}


pub(crate) fn mat33_sym_solve(A: &[[f64; 3]; 3], B: &[f64; 3]) -> [f64; 3] {
    let L = mat33_chol(A);

    let M = mat33_lower_tri_inv(&L);

    let t0 = M[0]*B[0];
    let t1 = M[3]*B[0] + M[4]*B[1];
    let t2 = M[6]*B[0] + M[7]*B[1] + M[8]*B[2];

    [
        M[0]*t0 + M[3]*t1 + M[6]*t2,
                  M[4]*t1 + M[7]*t2,
                            M[8]*t2,
    ]
}
