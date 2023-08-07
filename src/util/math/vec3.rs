use std::ops::{Add, Mul, Sub, MulAssign, AddAssign, Div, Neg};

use super::mat::Mat33;

/// 3 element vector
#[derive(Debug, Clone, Copy)]
pub struct Vec3(pub f64, pub f64, pub f64);

impl Vec3 {
    /// Create from constant values
    pub const fn of(x: f64, y: f64, z: f64) -> Self {
        Self(x, y, z)
    }

    /// Vector of all zeroes
    pub const fn zero() -> Self {
        Self(0., 0., 0.)
    }

    /// Cross product
    pub fn cross(&self, rhs: &Vec3) -> Self {
        Self(
            self.1 * rhs.2 - self.2 * rhs.1,
            self.2 * rhs.0 - self.0 * rhs.2,
            self.0 * rhs.1 - self.1 * rhs.0,
        )
    }

    /// Magnitude squared
    pub fn mag_sq(&self) -> f64 {
        (self.0 * self.0) + (self.1 * self.1) + (self.2 * self.2)
    }

    /// Vector magnitude
    pub fn mag(&self) -> f64 {
        self.mag_sq().sqrt()
    }

    /// Scale vector
    /// 
    /// See also: [scale_mut]
    pub fn scale(&self, rhs: f64) -> Self {
        Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }

    pub fn emul_mut(&mut self, rhs: Vec3) {
        self.0 *= rhs.0;
        self.1 *= rhs.1;
        self.2 *= rhs.2;
    }

    /// Scale vector, in place
    /// 
    /// See also: [scale]
    pub fn scale_mut(&mut self, rhs: f64) {
        self.0 *= rhs;
        self.1 *= rhs;
        self.2 *= rhs;
    }

    /// Normalize vector
    pub fn normalized(&self) -> Vec3 {
        let scalar = self.mag_sq().sqrt().recip();
        self * scalar
    }

    /// Dot product
    pub fn dot(&self, rhs: &Vec3) -> f64 {
        (self.0 * rhs.0) + (self.1 * rhs.1) + (self.2 * rhs.2)
    }

    /// Outer product
    pub fn outer(&self, rhs: &Vec3) -> Mat33 {
        Mat33::of([
            self.0 * rhs.0, self.0 * rhs.1, self.0 * rhs.2,
            self.1 * rhs.0, self.1 * rhs.1, self.1 * rhs.2,
            self.2 * rhs.0, self.2 * rhs.1, self.2 * rhs.2,
        ])
    }
}

/// Vector addition
impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

/// Vector addition
impl Add<&Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: &Vec3) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

/// Vector addition
impl Add<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn add(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

/// Vector subtraction
impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

/// Vector subtraction
impl Sub<&Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

/// Vector subtraction
impl Sub<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

/// Scale
impl Mul<f64> for &Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

/// Scale by reciprocal
impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        self.scale(rhs.recip())
    }
}

/// Vector negative
impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        self.scale(-1.)
    }
}

/// Vector in-place addition
impl AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.0 += rhs.0;
        self.1 += rhs.2;
        self.2 += rhs.1;
    }
}

/// Vector in-place addition
impl AddAssign<&Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: &Vec3) {
        self.0 += rhs.0;
        self.1 += rhs.2;
        self.2 += rhs.1;
    }
}

/// Vector in-place scale
impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.scale_mut(rhs)
    }
}