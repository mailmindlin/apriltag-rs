use std::{ops::{Add, Sub, Neg, Div, Mul, MulAssign, AddAssign, DivAssign}};

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

    pub const fn squish32(&self) -> Self {
        Self::of(self.x() as f32 as _, self.y() as f32 as _)
    }

    /// Vector magnitude
    #[inline]
    pub fn mag(&self) -> f64 {
        f64::hypot(self.x(), self.y())
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
    fn add_assign(&mut self, rhs: Vec2) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl Add<&Vec2> for Vec2 {
    type Output = Vec2;

    fn add(mut self, rhs: &Vec2) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
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