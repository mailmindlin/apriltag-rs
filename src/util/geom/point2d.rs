use std::{ops::{Add, Sub, Neg}, fmt::Debug};

use crate::util::math::{Vec2, Vec2Builder};

#[derive(Copy, Clone, PartialEq)]
pub struct Point2D(Vec2);

impl Debug for Point2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Point2D")
            .field(&self.0.x())
            .field(&self.0.y())
            .finish()
    }
}

#[cfg(feature="compare_reference")]
impl float_cmp::ApproxEq for Point2D {
    type Margin = float_cmp::F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        self.0.approx_eq(other.0, margin)
    }
}

impl Add<Vec2> for &Point2D {
    type Output = Point2D;

    fn add(self, rhs: Vec2) -> Self::Output {
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
    #[inline(always)]
    pub fn zero() -> Self {
        Self(Vec2::zero())
    }

    #[inline]
    pub fn of(x: f64, y: f64) -> Self {
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
    pub const fn vec(&self) -> &Vec2 {
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

    pub fn as_array(&self) -> [f64; 2] {
        [self.x(), self.y()]
    }
}

impl From<Point2D> for [f64; 2] {
    fn from(value: Point2D) -> Self {
        [value.x(), value.y()]
    }
}

impl From<[f64; 2]> for Point2D {
    fn from(value: [f64; 2]) -> Self {
        Self::of(value[0], value[1])
    }
}