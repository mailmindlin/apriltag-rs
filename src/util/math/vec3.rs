use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

use super::mat::Mat33;

/// 3 element vector
#[derive(Debug, Clone, Copy)]
pub(crate) struct Vec3(pub f64, pub f64, pub f64);

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
    pub fn cross(self, rhs: Vec3) -> Self {
        Self(
            self.1 * rhs.2 - self.2 * rhs.1,
            self.2 * rhs.0 - self.0 * rhs.2,
            self.0 * rhs.1 - self.1 * rhs.0,
        )
    }

    /// Magnitude squared
    pub fn mag_sq(self) -> f64 {
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
    pub fn scale_inplace(&mut self, rhs: f64) {
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
    pub fn dot(self, rhs: Vec3) -> f64 {
        (self.0 * rhs.0) + (self.1 * rhs.1) + (self.2 * rhs.2)
    }

    /// Outer product
    pub fn outer(self, rhs: Vec3) -> Mat33 {
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
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

/// Vector in-place addition
impl AddAssign<&Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: &Vec3) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl std::iter::Sum for Vec3 {
	fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
		// This lets us optimize away the first 
		let Some(mut result) = iter.next() else { return Self::zero() };
		for item in iter {
			result += item;
		}
		result
	}
}

impl<'a> std::iter::Sum<&'a Self> for Vec3 {
	fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
		// This lets us optimize away the first 
		let Some(mut result) = iter.next().copied() else { return Self::zero() };
		for item in iter {
			result += item;
		}
		result
	}
}

/// Vector in-place scale
impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.scale_inplace(rhs)
    }
}

#[cfg(test)]
mod test {
    use super::Vec3;

    const EPS: f64 = 1e-10;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < EPS, "{a} != {b}");
    }

    fn assert_vec_close(a: Vec3, b: Vec3) {
        assert_close(a.0, b.0);
        assert_close(a.1, b.1);
        assert_close(a.2, b.2);
    }

    #[test]
    fn test_add() {
        let a = Vec3::of(1., 2., 3.);
        let b = Vec3::of(4., 5., 6.);
        assert_vec_close(a + b, Vec3::of(5., 7., 9.));
    }

    #[test]
    fn test_sub() {
        let a = Vec3::of(4., 5., 6.);
        let b = Vec3::of(1., 2., 3.);
        assert_vec_close(a - b, Vec3::of(3., 3., 3.));
    }

    #[test]
    fn test_add_assign() {
        let mut a = Vec3::of(1., 2., 3.);
        a += Vec3::of(10., 20., 30.);
        assert_vec_close(a, Vec3::of(11., 22., 33.));
    }

    #[test]
    fn test_add_assign_ref() {
        let mut a = Vec3::of(1., 2., 3.);
        let b = Vec3::of(10., 20., 30.);
        a += &b;
        assert_vec_close(a, Vec3::of(11., 22., 33.));
    }

    #[test]
    fn test_neg() {
        let a = Vec3::of(1., -2., 3.);
        assert_vec_close(-a, Vec3::of(-1., 2., -3.));
    }

    #[test]
    fn test_scale() {
        let a = Vec3::of(1., 2., 3.);
        assert_vec_close(a.scale(2.), Vec3::of(2., 4., 6.));
    }

    #[test]
    fn test_div() {
        let a = Vec3::of(2., 4., 6.);
        assert_vec_close(a / 2., Vec3::of(1., 2., 3.));
    }

    #[test]
    fn test_dot() {
        let a = Vec3::of(1., 0., 0.);
        let b = Vec3::of(0., 1., 0.);
        assert_close(a.dot(&b), 0.);

        let c = Vec3::of(1., 2., 3.);
        let d = Vec3::of(4., 5., 6.);
        assert_close(c.dot(&d), 32.);
    }

    #[test]
    fn test_cross() {
        let i = Vec3::of(1., 0., 0.);
        let j = Vec3::of(0., 1., 0.);
        let k = Vec3::of(0., 0., 1.);
        assert_vec_close(i.cross(&j), k);
        assert_vec_close(j.cross(&k), i);
        assert_vec_close(k.cross(&i), j);
        // Anti-commutativity
        assert_vec_close(j.cross(&i), -k);
    }

    #[test]
    fn test_mag() {
        assert_close(Vec3::of(3., 4., 0.).mag(), 5.);
        assert_close(Vec3::zero().mag(), 0.);
        assert_close(Vec3::of(1., 1., 1.).mag(), 3f64.sqrt());
    }

    #[test]
    fn test_normalized() {
        let v = Vec3::of(3., 4., 0.);
        let n = v.normalized();
        assert_close(n.mag(), 1.);
        assert_close(n.0, 0.6);
        assert_close(n.1, 0.8);
    }

    #[test]
    fn test_outer() {
        let a = Vec3::of(1., 2., 3.);
        let b = Vec3::of(4., 5., 6.);
        let m = a.outer(&b);
        assert_close(m[(0, 0)], 4.);
        assert_close(m[(0, 1)], 5.);
        assert_close(m[(0, 2)], 6.);
        assert_close(m[(1, 0)], 8.);
        assert_close(m[(1, 1)], 10.);
        assert_close(m[(1, 2)], 12.);
        assert_close(m[(2, 0)], 12.);
        assert_close(m[(2, 1)], 15.);
        assert_close(m[(2, 2)], 18.);
    }
}