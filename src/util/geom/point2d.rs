use std::ops::{Add, Sub, Neg};

use crate::util::math::Vec2;

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Point2D(Vec2);

impl Add<&Vec2> for &Point2D {
    type Output = Point2D;

    fn add(self, rhs: &Vec2) -> Self::Output {
        Point2D(self.0 + rhs)
    }
}

impl Sub<&Point2D> for Point2D {
    type Output = Vec2;

    fn sub(self, rhs: &Point2D) -> Self::Output {
        self.0 - &rhs.0
    }
}

impl Sub<&Point2D> for &Point2D {
    type Output = Vec2;

    fn sub(self, rhs: &Point2D) -> Self::Output {
        &self.0 - &rhs.0
    }
}

impl Sub<&Vec2> for Point2D {
    type Output = Point2D;

    fn sub(self, rhs: &Vec2) -> Self::Output {
        Point2D(self.0 - rhs)
    }
}

impl Neg for Point2D {
    type Output = Point2D;

    fn neg(self) -> Self::Output {
        Point2D(-self.0)
    }
}

impl Point2D {
    pub const fn zero() -> Self {
        Self(Vec2::zero())
    }

    #[inline]
    pub const fn of(x: f64, y: f64) -> Self {
        Self(Vec2::of(x, y))
    }
    
    #[inline]
    pub const fn from_vec(v: Vec2) -> Self {
        Self(v)
    }
    
    #[inline(always)]
    pub fn x(&self) -> f64 {
        self.0.x()
    }

    #[inline(always)]
    pub fn y(&self) -> f64 {
        self.0.y()
    }

    #[inline(always)]
    pub fn vec(&self) -> &Vec2 {
        &self.0
    }

    #[inline(always)]
    pub fn vec_mut(&mut self) -> &mut Vec2 {
        &mut self.0
    }

    pub fn distance_to(&self, other: &Point2D) -> f64 {
        let delta = self - other;
        delta.mag()
    }

    #[inline]
    pub fn angle_to(&self, other: &Point2D) -> f64 {
        let delta = self - other;
        delta.angle()
    }
}

impl From<Point2D> for [f64; 2] {
    fn from(value: Point2D) -> Self {
        [value.x(), value.y()]
    }
}