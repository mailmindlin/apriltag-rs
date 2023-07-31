use std::ops::{Add, Mul, Sub, MulAssign, AddAssign, Div};

use super::mat::Mat33;


#[derive(Debug, Clone, Copy)]
pub struct Vec3(pub f64, pub f64, pub f64);

impl Vec3 {
    pub const fn of(x: f64, y: f64, z: f64) -> Self {
        Self(x, y, z)
    }

    pub const fn zero() -> Self {
        Self(0., 0., 0.)
    }
    pub fn cross(&self, rhs: &Vec3) -> Self {
        Self(
            self.1 * rhs.2 - self.2 * rhs.1,
            self.2 * rhs.0 - self.0 * rhs.2,
            self.0 * rhs.1 - self.1 * rhs.0,
        )
    }

    pub fn mag_sq(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }

    pub fn mag(&self) -> f64 {
        self.mag_sq().sqrt()
    }

    pub fn scale(&self, rhs: f64) -> Self {
        Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }

    pub fn scale_inplace(&mut self, rhs: f64) {
        self.0 *= rhs;
        self.1 *= rhs;
        self.2 *= rhs;
    }

    pub fn normalized(&self) -> Vec3 {
        let scalar = self.mag().recip();
        self * scalar
    }

    pub fn dot(&self, rhs: &Vec3) -> f64 {
        self.0 * rhs.0 + self.1 * rhs.1 + self.2 * rhs.2
    }

    pub fn outer(&self, rhs: &Vec3) -> Mat33 {
        Mat33::of([
            self.0 * rhs.0, self.0 * rhs.1, self.0 * rhs.2,
            self.1 * rhs.0, self.1 * rhs.1, self.1 * rhs.2,
            self.2 * rhs.0, self.2 * rhs.1, self.2 * rhs.2,
        ])
    }
}

impl Add<&Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: &Vec3) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl Sub<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl Mul<f64> for &Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        self.scale(rhs.recip())
    }
}

impl AddAssign<&Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: &Vec3) {
        self.0 += rhs.0;
        self.1 += rhs.2;
        self.2 += rhs.1;
    }
}

impl AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.0 += rhs.0;
        self.1 += rhs.2;
        self.2 += rhs.1;
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.scale_inplace(rhs)
    }
}