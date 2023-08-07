use std::{ops::{Index, IndexMut, MulAssign, AddAssign, Sub, Add, Mul}, mem::swap};

use crate::util::math::Vec3;

use super::{OutOfBoundsError, MatDims, Mat};


/// 3x3 matrix
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Mat33(pub [f64; 9]);


/// Try to convert to fixed-size matrix
impl TryFrom<super::Mat> for Mat33 {
    type Error = OutOfBoundsError;

    fn try_from(value: super::Mat) -> Result<Self, Self::Error> {
        if value.dims != (MatDims { rows: 3, cols: 3 }) {
            return Err(OutOfBoundsError { dims: value.dims, index: super::MatIndex { row: 2, col: 2 } });
        }
        let a: &[f64] = &value.data;
        let b: Result<[f64; 9], _> = a.try_into();
        match b {
            Ok(elems) => Ok(Self(elems)),
            Err(_) => Err(OutOfBoundsError { dims: value.dims, index: super::MatIndex { row: 2, col: 2 } })
        }
    }
}

/// Dynamic-sized matrix with same data
impl From<Mat33> for Mat {
    fn from(value: Mat33) -> Self {
        Self::create(3, 3, &value.0)
    }
}

#[cfg(feature="compare_reference")]
impl float_cmp::ApproxEq for Mat33 {
    type Margin = float_cmp::F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        <&[f64] as float_cmp::ApproxEq>::approx_eq(&self.0, &other.0, margin)
    }
}

pub(crate) struct Mat33SVD {
    pub U: Mat33,
    #[allow(unused)]
    pub S: Mat33,
    pub V: Mat33
}

impl Mat33 {
    /// Create matrix with all zeroes
    pub(crate) const fn zeroes() -> Self {
        Self([0.; 9])
    }
    
    /// Create from array
    pub(crate) const fn of(v: [f64; 9]) -> Self {
        Self(v)
    }

    pub(crate) const fn data(&self) -> &[f64] {
        &self.0
    }

    /// Create an identity matrix
    pub(crate) const fn identity() -> Self {
        Self([
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.,
        ])
    }

    pub(crate) fn from_quaternion(q: [f64; 4]) -> Self {
        let w = q[3];
        let x = q[0];
        let y = q[1];
        let z = q[2];
    
        let qxx = x*x;
        let qyy = y*y;
        let qzz = z*z;
        let qxz = x*z;
        let qxy = x*y;
        let qyz = y*z;
        let qwx = w*x;
        let qwy = w*y;
        let qwz = w*z;
    
        Self::of([
            1. - 2.*(qyy + qzz), 2.*(qxy - qwz),     2.*(qxz + qwy),
            2.*(qxy + qwz),     1. - 2.*(qxx + qzz), 2.*(qyz - qwx),
            2.*(qxz - qwy),     2.*(qyz + qwx),     1. - 2.*(qxx + qyy),
        ])
    }

    /// Determinant
    pub(crate) fn det(&self) -> f64 {
		0.
			+ self.0[0] * self.0[4] * self.0[8]
			- self.0[0] * self.0[5] * self.0[7]
			+ self.0[1] * self.0[5] * self.0[6]
			- self.0[1] * self.0[3] * self.0[8]
			+ self.0[2] * self.0[3] * self.0[7]
			- self.0[2] * self.0[4] * self.0[6]
    }

    /// Transpose
    pub(crate) const fn transposed(&self) -> Self {
        Self([
            self.0[0], self.0[3], self.0[6],
            self.0[1], self.0[4], self.0[7],
            self.0[2], self.0[5], self.0[8],
        ])
    }

    /// In-place transpose
    pub fn transpose_mut(&mut self) {
        self.0.swap(1, 3); // (2, 1) <-> (1, 2)
        self.0.swap(2, 6); // (3, 1) <-> (1, 3)
        self.0.swap(5, 7); // (3, 2) <-> (2, 3)
    }

    /// Matrix multipliation
    pub(crate) fn matmul(&self, rhs: &Mat33) -> Self {
        Self([
            self.0[0]*rhs.0[0] + self.0[1]*rhs.0[3] + self.0[2]*rhs.0[6], self.0[0]*rhs.0[1] + self.0[1]*rhs.0[4] + self.0[2]*rhs.0[7], self.0[0]*rhs.0[2] + self.0[1]*rhs.0[5] + self.0[2]*rhs.0[8],
            self.0[3]*rhs.0[0] + self.0[4]*rhs.0[3] + self.0[5]*rhs.0[6], self.0[3]*rhs.0[1] + self.0[4]*rhs.0[4] + self.0[5]*rhs.0[7], self.0[3]*rhs.0[2] + self.0[4]*rhs.0[5] + self.0[5]*rhs.0[8],
            self.0[6]*rhs.0[0] + self.0[7]*rhs.0[3] + self.0[8]*rhs.0[6], self.0[6]*rhs.0[1] + self.0[7]*rhs.0[4] + self.0[8]*rhs.0[7], self.0[6]*rhs.0[2] + self.0[7]*rhs.0[5] + self.0[8]*rhs.0[8],
        ])
    }

    /// Matrix multiplication (equivalent to `self.transposed().matmul(rhs)`)
    pub(crate) fn transpose_matmul(&self, rhs: &Mat33) -> Self {
        Self([
            self.0[0]*rhs.0[0] + self.0[3]*rhs.0[3] + self.0[6]*rhs.0[6], self.0[0]*rhs.0[1] + self.0[3]*rhs.0[4] + self.0[6]*rhs.0[7], self.0[0]*rhs.0[2] + self.0[3]*rhs.0[5] + self.0[6]*rhs.0[8],
            self.0[1]*rhs.0[0] + self.0[4]*rhs.0[3] + self.0[7]*rhs.0[6], self.0[1]*rhs.0[1] + self.0[4]*rhs.0[4] + self.0[7]*rhs.0[7], self.0[1]*rhs.0[2] + self.0[4]*rhs.0[5] + self.0[7]*rhs.0[8],
            self.0[2]*rhs.0[0] + self.0[5]*rhs.0[3] + self.0[8]*rhs.0[6], self.0[2]*rhs.0[1] + self.0[5]*rhs.0[4] + self.0[8]*rhs.0[7], self.0[2]*rhs.0[2] + self.0[5]*rhs.0[5] + self.0[8]*rhs.0[8],
        ])
    }

    /// Matrix multiplication (equivalent to `self.matmul(rhs.transposed())`)
    pub(crate) fn matmul_transpose(&self, rhs: &Mat33) -> Self {
        //TODO
        self.matmul(&rhs.transposed())
    }

    /// QR decomposition
    fn decompose_qr(&self) -> (Mat33, Mat33) {
        fn QRGivensQuaternion(a1: f64, a2: f64) -> (f64, f64) {
            // a1 = pivot point on diagonal
            // a2 = lower triangular entry we want to annihilate
            const EPSILON: f64 = 1e-6;
            let rho = f64::hypot(a1, a2);

            let sh = if rho > EPSILON { a2 } else { 0. };
            let ch = a1.abs() + f64::max(rho, EPSILON);
            let (sh, ch) = if a1 < 0. {
                (ch, sh)
            } else {
                (sh, ch)
            };
            let w = f64::hypot(ch, sh);
            (ch * w, sh * w)
        }

        let [b11, b12, b13, b21, b22, b23, b31, b32, b33] = self.0;

        // first givens rotation (ch,0,0,sh)
        let (ch1,sh1) = QRGivensQuaternion(b11,b21);
        let a=1.-2.*sh1*sh1;
        let b=2.*ch1*sh1;
        // apply B = Q' * B
        let r11=a*b11+b*b21;  let r12=a*b12+b*b22;    let r13=a*b13+b*b23;
        let r21=-b*b11+a*b21;      let r22=-b*b12+a*b22; let r23=-b*b13+a*b23;
        let r31=b31;          let r32=b32;            let r33=b33;

        // second givens rotation (ch,0,-sh,0)
        let (ch2,sh2) = QRGivensQuaternion(r11,r31);
        let a=1.-2.*sh2*sh2;
        let b=2.*ch2*sh2;
        // apply B = Q' * B;
        let b11=a*r11+b*r31; let b12=a*r12+b*r32;  let b13=a*r13+b*r33;
        let b21=r21;           let b22=r22;           let b23=r23;
        let b31=-b*r11+a*r31; let b32=-b*r12+a*r32; let b33=-b*r13+a*r33;

        // third givens rotation (ch,sh,0,0)
        let (ch3,sh3) = QRGivensQuaternion(b22,b32);
        let a=1.-2.*sh3*sh3;
        let b=2.*ch3*sh3;
        // R is now set to desired value
        let R = Self::of([
            b11,          b12,          b13,
            a*b21+b*b31,  a*b22+b*b32,  a*b23+b*b33,
            -b*b21+a*b31, -b*b22+a*b32, -b*b23+a*b33,
        ]);

        // construct the cumulative rotation Q=Q1 * Q2 * Q3
        // the number of floating point operations for three quaternion multiplications
        // is more or less comparable to the explicit form of the joined matrix.
        // certainly more memory-efficient!
        let sh12=sh1*sh1;
        let sh22=sh2*sh2;
        let sh32=sh3*sh3;

        let Q = Self::of([
            (-1.+2.*sh12)*(-1.+2.*sh22),
            4.*ch2*ch3*(-1.+2.*sh12)*sh2*sh3+2.*ch1*sh1*(-1.+2.*sh32),
            4.*ch1*ch3*sh1*sh3-2.*ch2*(-1.+2.*sh12)*sh2*(-1.+2.*sh32),

            2.*ch1*sh1*(1.-2.*sh22),
            -8.*ch1*ch2*ch3*sh1*sh2*sh3+(-1.+2.*sh12)*(-1.+2.*sh32),
            -2.*ch3*sh3+4.*sh1*(ch3*sh1*sh3+ch1*ch2*sh2*(-1.+2.*sh32)),

            2.*ch2*sh2,
            2.*ch3*(1.-2.*sh22)*sh3,
            (-1.+2.*sh22)*(-1.+2.*sh32),
        ]);

        (Q, R)
    }

    pub(crate) fn svd(&self) -> Mat33SVD {
        const GAMMA: f64 = 5.828427124; // FOUR_GAMMA_SQUARED = sqrt(8)+3;
        const CS_STAR: f64 = 0.923879532; // cos(pi/8)
        const SS_STAR: f64 = 0.3826834323; // sin(p/8)

        /// This is a novel and fast routine for the reciprocal square root of an
        /// IEEE float (single precision).
        /// http://www.lomont.org/Math/Papers/2003/InvSqrt.pdf
        /// http://playstation2-linux.com/download/p2lsd/fastrsqrt.pdf
        /// http://www.beyond3d.com/content/articles/8/
        #[inline]
        fn rsqrt(x: f32) -> f32 {
            // int ihalf = *(int *)&x - 0x00800000; // Alternative to next line,
            // float xhalf = *(float *)&ihalf;      // for sufficiently large nos.
            let xhalf = 0.5 * x;
            let i = f32::to_bits(x);     // View x as an int.
            // i = 0x5f3759df - (i >> 1);          // Initial guess (traditional).
            let i = 0x5f375a82 - (i >> 1);    // Initial guess (slightly better).
            let x = f32::from_bits(i);        // View i as float.
            x*(1.5 - xhalf*x*x)    // Newton step.
            // x = x*(1.5008908 - xhalf*x*x);  // Newton step for a balanced error.
        }

        #[inline]
        fn jacobiConjugation(x: usize, y: usize, z: usize, s: &mut Mat33, qV: &mut [f64; 4]) {
            let (ch, sh) = {
                // Given givens angle computed by approximateGivensAngles,
                // compute the corresponding rotation quaternion.
                let ch = 2. * (s.0[0] - s.0[4]);
                let sh = s.0[5];
                let b = GAMMA * sh * sh < ch * ch;
                // fast rsqrt function suffices
                // rsqrt2 (https://code.google.com/p/lppython/source/browse/algorithm/HDcode/newCode/rsqrt.c?r=26)
                // is even faster but results in too much error
                let w = rsqrt((ch*ch+sh*sh) as _) as f64;
                if b {
                    (
                        (w * ch),
                        (w * sh)
                    )
                } else {
                    (CS_STAR, SS_STAR)
                }
            };

            let scale = ch*ch+sh*sh;
            let a = (ch*ch-sh*sh)/scale;
            let b = (2.*sh*ch)/scale;

            // make temp copy of S
            let _s11 = s.0[0];
            let _s21 = s.0[3]; let _s22 = s.0[4];
            let _s31 = s.0[6]; let _s32 = s.0[7]; let _s33 = s.0[8];

            // perform conjugation S = Q'*S*Q
            // Q already implicitly solved from a, b
            s.0[0] =a*(a*_s11 + b*_s21) + b*(a*_s21 + b*_s22);
            s.0[3] =a*(-b*_s11 + a*_s21) + b*(-b*_s21 + a*_s22);	s.0[4]=-b*(-b*_s11 + a*_s21) + a*(-b*_s21 + a*_s22);
            s.0[6] =a*_s31 + b*_s32;								s.0[7]=-b*_s31 + a*_s32; s.0[8]=_s33;

            // update cumulative rotation qV
            let tmp = [
                qV[0]*sh,
                qV[1]*sh,
                qV[2]*sh,
            ];
            let sh = sh * qV[3];

            qV[0] *= ch;
            qV[1] *= ch;
            qV[2] *= ch;
            qV[3] *= ch;

            // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
            // for (p,q) = ((0,1),(1,2),(0,2))
            qV[z] += sh;
            qV[3] -= tmp[z]; // w
            qV[x] += tmp[y];
            qV[y] -= tmp[x];

            // re-arrange matrix for next iteration
            let _s11 = s.0[4];
            let _s21 = s.0[7]; let _s22 = s.0[8];
            let _s31 = s.0[3]; let _s32 = s.0[6]; let _s33 = s.0[0];
            s.0[0] = _s11;
            s.0[3] = _s21; s.0[4] = _s22;
            s.0[6] = _s31; s.0[7] = _s32; s.0[8] = _s33;
        }

        /// finds transformation that diagonalizes a symmetric matrix
        #[inline]
        fn jacobiEigenanlysis(s: &mut Mat33) -> [f64; 4] {
            // follow same indexing convention as GLM
            let mut qV = [0., 0., 0., 1.,];
            for _ in 0..4 {
                // we wish to eliminate the maximum off-diagonal element
                // on every iteration, but cycling over all 3 possible rotations
                // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
                //  asymptotic convergence
                jacobiConjugation(0,1,2, s, &mut qV); // p,q = 0,1
                jacobiConjugation(1,2,0, s, &mut qV); // p,q = 1,2
                jacobiConjugation(2,0,1, s, &mut qV); // p,q = 0,2
            }
            qV
        }

        // normal equations matrix
        let mut ata = self.transpose_matmul(self);

        // symmetric eigenalysis
        let mut V = {
            // qV is the quaternion representation of V
            let qV = jacobiEigenanlysis(&mut ata);
            Self::from_quaternion(qV)
        };

        let mut b = self.matmul(&V);

        // sort singular values and find V
        {
            let mut rho1 = Vec3(b.0[0], b.0[3], b.0[6]).mag_sq();
            let mut rho2 = Vec3(b.0[1], b.0[4], b.0[7]).mag_sq();
            let mut rho3 = Vec3(b.0[2], b.0[5], b.0[8]).mag_sq();

            #[inline(always)]
            fn negSwap<const X: usize, const Y: usize>(b: &mut Mat33) {
                let z = -b.0[X];
                b.0[X] = b.0[Y];
                b.0[Y] = z;
            }
            
            if rho1 < rho2 {
                negSwap::<0, 1>(&mut b);
                negSwap::<0, 1>(&mut V);
                negSwap::<3, 4>(&mut b);
                negSwap::<3, 4>(&mut V);
                negSwap::<6, 7>(&mut b);
                negSwap::<6, 7>(&mut V);
                swap(&mut rho1, &mut rho2);
            }
            
            if rho1 < rho3 {
                negSwap::<0, 2>(&mut b);
                negSwap::<0, 2>(&mut V);
                negSwap::<3, 5>(&mut b);
                negSwap::<3, 5>(&mut V);
                negSwap::<6, 8>(&mut b);
                negSwap::<6, 8>(&mut V);
                swap(&mut rho1, &mut rho3);
            }
            if rho2 < rho3 {
                negSwap::<1, 2>(&mut b);
                negSwap::<1, 2>(&mut V);
                negSwap::<4, 5>(&mut b);
                negSwap::<4, 5>(&mut V);
                negSwap::<7, 8>(&mut b);
                negSwap::<7, 8>(&mut V);
            }
        }

        // QR decomposition
        let (U, S) = b.decompose_qr();

        Mat33SVD { U, S, V }
    }

    /// Matrix inverse
    /// 
    /// Returns None if this matrix is not invertible
    pub(crate) fn inv(&self) -> Option<Mat33> {
        let det = self.det();
        if det == 0. {
            return None;
        }

        let mut res = Self::of([
             ((self.0[4]*self.0[8]) - (self.0[5]*self.0[7])), -((self.0[1]*self.0[8]) - (self.0[2]*self.0[7])),  ((self.0[1]*self.0[5]) - (self.0[2]*self.0[4])),
            -((self.0[3]*self.0[8]) - (self.0[5]*self.0[6])),  ((self.0[0]*self.0[8]) - (self.0[2]*self.0[6])), -((self.0[0]*self.0[5]) - (self.0[2]*self.0[3])),
             ((self.0[3]*self.0[7]) - (self.0[4]*self.0[6])), -((self.0[0]*self.0[7]) - (self.0[1]*self.0[6])),  ((self.0[0]*self.0[4]) - (self.0[1]*self.0[3])),
        ]);

        res.scale_inplace(1. / det);
        Some(res)
    }
    
    /// Computes the cholesky factorization of A, putting the lower
    /// triangular matrix into R.
    #[inline]
    fn chol(&self) -> Self {
        // A[0] = R[0]*R[0]
        let R0 = self.0[0].sqrt();

        // A[1] = R[0]*R[3];
        let R3 = self.0[1] / R0;

        // A[2] = R[0]*R[6];
        let R6 = self.0[2] / R0;

        // A[4] = R[3]*R[3] + R[4]*R[4]
        let R4 = (self.0[4] - R3*R3).sqrt();

        // A[5] = R[3]*R[6] + R[4]*R[7]
        let R7 = (self.0[5] - R3*R6) / R4;

        // A[8] = R[6]*R[6] + R[7]*R[7] + R[8]*R[8]
        let R8 = f64::sqrt(self.0[8] - R6*R6 - R7*R7);

        Self([
            R0,
            0.0,
            0.0,
            R3,
            R4,
            0.0,
            R6,
            R7,
            R8,
        ])
    }

    #[inline]
    fn lower_tri_inv(&self) -> Self {
        // A[0]*R[0] = 1
        let R0 = self.0[0].recip();

        // A[3]*R0 + A[4]*R3 = 0
        let R3 = -self.0[3]*R0 / self.0[4];

        // A[4]*R4 = 1
        let R4 = self.0[4].recip();

        // A[6]*R0 + A[7]*R3 + A[8]*R6 = 0
        let R6 = (-self.0[6]*R0 - self.0[7]*R3) / self.0[8];

        // A[7]*R4 + A[8]*R7 = 0
        let R7 = -self.0[7]*R4 / self.0[8];

        // A[8]*R8 = 1
        let R8 = self.0[8].recip();

        Self([
            R0,
            0.0,
            0.0,
            R3,
            R4,
            0.0,
            R6,
            R7,
            R8,
        ])
    }

    #[inline]
    fn map_elements(&self, map_fn: impl Fn(f64) -> f64) -> Self {
        Self::of([
            map_fn(self.0[0]),
            map_fn(self.0[1]),
            map_fn(self.0[2]),
            map_fn(self.0[3]),
            map_fn(self.0[4]),
            map_fn(self.0[5]),
            map_fn(self.0[6]),
            map_fn(self.0[7]),
            map_fn(self.0[8]),
        ])
    }

    pub(crate) fn scale(&self, scalar: f64) -> Self {
        self.map_elements(|e| e * scalar)
    }

    pub(crate) fn scale_inplace(&mut self, scalar: f64) {
        for e in self.0.iter_mut() {
            *e = *e * scalar;
        }
    }

    pub(crate) fn sym_solve(&self, B: &[f64; 3]) -> [f64; 3] {
        let L = self.chol();
    
        let M = L.lower_tri_inv();
    
        let t0 = M.0[0]*B[0];
        let t1 = M.0[3]*B[0] + M.0[4]*B[1];
        let t2 = M.0[6]*B[0] + M.0[7]*B[1] + M.0[8]*B[2];
    
        [
            M.0[0]*t0 + M.0[3]*t1 + M.0[6]*t2,
                        M.0[4]*t1 + M.0[7]*t2,
                                    M.0[8]*t2,
        ]
    }
}

impl Index<(usize, usize)> for Mat33 {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(row < 3);
        debug_assert!(col < 3);
        let idx = row * 3 + col;
        &self.0[idx]
    }
}

impl IndexMut<(usize, usize)> for Mat33 {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(row < 3);
        debug_assert!(col < 3);
        let idx = row * 3 + col;
        &mut self.0[idx]
    }
}

impl Add<&Mat33> for Mat33 {
    type Output = Mat33;

    fn add(self, rhs: &Mat33) -> Self::Output {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
            self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6],
            self.0[7] + rhs.0[7],
            self.0[8] + rhs.0[8],
        ])
    }
}

impl Sub<&Mat33> for &Mat33 {
    type Output = Mat33;

    fn sub(self, rhs: &Mat33) -> Self::Output {
        Mat33([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
            self.0[5] - rhs.0[5],
            self.0[6] - rhs.0[6],
            self.0[7] - rhs.0[7],
            self.0[8] - rhs.0[8],
        ])
    }
}

impl Sub<&Mat33> for Mat33 {
    type Output = Mat33;

    fn sub(self, rhs: &Mat33) -> Self::Output {
        <&Mat33 as Sub<&Mat33>>::sub(&self, rhs)
    }
}

impl Mul<&Vec3> for &Mat33 {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        Vec3::of(
            self.0[0] * rhs.0 + self.0[1] * rhs.1 + self.0[2] * rhs.2,
            self.0[3] * rhs.0 + self.0[4] * rhs.1 + self.0[5] * rhs.2,
            self.0[6] * rhs.0 + self.0[7] * rhs.1 + self.0[8] * rhs.2,
        )
    }
}

impl AddAssign<&Mat33> for Mat33 {
    fn add_assign(&mut self, rhs: &Mat33) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
        self.0[4] += rhs.0[4];
        self.0[5] += rhs.0[5];
        self.0[6] += rhs.0[6];
        self.0[7] += rhs.0[7];
        self.0[8] += rhs.0[8];
    }
}

impl MulAssign<f64> for Mat33 {
    fn mul_assign(&mut self, rhs: f64) {
        self.scale_inplace(rhs);
    }
}