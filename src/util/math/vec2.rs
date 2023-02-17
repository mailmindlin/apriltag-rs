use std::ops::{Add, Sub, Neg, Div, Mul, MulAssign};

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Vec2(f64, f64);

impl Vec2 {
    /// Zero vector
    #[inline]
    pub const fn zero() -> Self {
        Self(0.,0.)
    }
    
    /// Create vector from values
    #[inline]
    pub const fn of(x: f64, y: f64) -> Self {
        Self(x, y)
    }

    pub fn from_angle(theta: f64) -> Self {
        let (x, y) = theta.sin_cos();
        Self::of(x, y)
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
    pub const fn dot(&self, other: &Vec2) -> f64 {
        self.0 * other.0 + self.1 * other.1
    }

    /// This vector, normalized
    pub fn norm(&self) -> Vec2 {
        let mag = self.mag();
        self / mag
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

impl Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}