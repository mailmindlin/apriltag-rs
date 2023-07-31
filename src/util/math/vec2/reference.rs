use std::ops::{Add, Sub, Neg, Div, Mul, MulAssign, AddAssign, DivAssign, SubAssign};

use crate::util::math::Vec3;

use super::{Vec2Builder, FMA};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec2(f64, f64);

impl const Vec2Builder for Vec2 {
    #[inline]
    fn zero() -> Self {
        Self(0.,0.)
    }

    #[inline(always)]
    fn dup(v: f64) -> Self {
        Self(v, v)
    }

    #[inline]
    fn of(x: f64, y: f64) -> Self {
        Self(x, y)
    }
}

#[cfg(feature="compare_reference")]
impl float_cmp::ApproxEq for Vec2 {
    type Margin = float_cmp::F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin: Self::Margin = margin.into();
        self.0.approx_eq(other.0, margin) && self.1.approx_eq(other.1, margin)
    }
}

impl FMA for Vec2 {
    fn fma(self, u: Self, v: Self) -> Self {
        Self::of(
            self.0 + u.0 * v.0,
            self.1 + u.1 * v.1
        )
    }
}

impl Vec2 {
    pub fn from_angle(theta: f64) -> Self {
        let (x, y) = theta.sin_cos();
        Self::of(x, y)
    }

    pub const fn flip(self) -> Self {
        Self(self.1, self.0)
    }

    /// X component
    #[inline(always)]
    pub const fn x(&self) -> f64 {
        self.0
    }

    /// Y component
    #[inline(always)]
    pub const fn y(&self) -> f64 {
        self.1
    }

    #[inline]
    pub fn gradient(&self) -> Vec3 {
        let xx = self.x() * self.x();
        let xy = self.x() * self.y();
        let yy = self.y() * self.y();
        Vec3(xx, xy, yy)
    }

    pub const fn squish32(&self) -> Self {
        Self::of(self.x() as f32 as _, self.y() as f32 as _)
    }

    /// Vector magnitude
    #[inline]
    pub fn mag(&self) -> f64 {
        f64::hypot(self.x(), self.y())
    }

    /// `(y, -x)`
    pub fn rev_negx(self) -> Self {
        Self(
            self.1,
            -self.0
        )
    }

    /// `(-y, x)`
    pub fn rev_negy(self) -> Self {
        Self(
            -self.1,
            self.0
        )
    }

    pub const fn rev(self) -> Self {
        Self(
            self.1,
            self.0
        )
    }

    #[inline]
    pub fn angle(&self) -> f64 {
        f64::atan2(self.y(), self.x())
    }

    /// Vector dot product
    #[inline]
    pub fn dot(&self, other: Vec2) -> f64 {
        self.0 * other.0 + self.1 * other.1
    }

    /// This vector, normalized
    pub fn norm(&self) -> Vec2 {
        let mag = self.mag();
        self / mag
    }
}

impl Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl AddAssign<Vec2> for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vec2) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl Add<&Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: &Vec2) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Add<&Vec2> for &Vec2 {
    type Output = Vec2;

    fn add(self, rhs: &Vec2) -> Self::Output {
        Vec2(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub<&Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: &Vec2) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl Sub<f64> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: f64) -> Self::Output {
        Self(self.0 - rhs, self.1 - rhs)
    }
}

impl Sub<&Vec2> for &Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: &Vec2) -> Self::Output {
        Vec2(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl SubAssign<Vec2> for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vec2) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(mut self, rhs: f64) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<f64> for &Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f64) -> Self::Output {
        Vec2(self.0 * rhs, self.1 * rhs)
    }
}

impl MulAssign<f64> for Vec2 {
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl Div<f64> for &Vec2 {
    type Output = Vec2;
    fn div(self, rhs: f64) -> Self::Output {
        Vec2(self.0 / rhs, self.1 / rhs)
    }
}

impl Div<f64> for Vec2 {
    type Output = Vec2;
    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs, self.1 / rhs)
    }
}

impl DivAssign<f64> for Vec2 {
    fn div_assign(&mut self, rhs: f64) {
        self.0 /= rhs;
        self.1 /= rhs;
    }
}

impl Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}