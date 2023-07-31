use std::ops::{Index, MulAssign, AddAssign, SubAssign, IndexMut};

use crate::util::math::Vec2;

use super::Point2D;


#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Quadrilateral([Point2D; 4]);

#[cfg(feature="compare_reference")]
impl float_cmp::ApproxEq for Quadrilateral {
    type Margin = float_cmp::F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin: Self::Margin = margin.into();
        for (u, v) in self.0.iter().zip(other.0.iter()) {
            if !u.approx_eq(*v, margin) {
                return false;
            }
        }
        true
    }
}

impl Quadrilateral {
    pub const fn from_points(corners: [Point2D; 4]) -> Self {
        Self(corners)
    }
    pub fn from_array<T: Into<f64> + Copy>(arr: &[[T; 2]; 4]) -> Self {
        Self([
            Point2D::of(arr[0][0].into(), arr[0][1].into()),
            Point2D::of(arr[1][0].into(), arr[1][1].into()),
            Point2D::of(arr[2][0].into(), arr[2][1].into()),
            Point2D::of(arr[3][0].into(), arr[3][1].into()),
        ])
    }
    pub fn as_array(&self) -> [[f64; 2]; 4] {
        [
            [self.0[0].x(), self.0[0].y()],
            [self.0[1].x(), self.0[1].y()],
            [self.0[2].x(), self.0[2].y()],
            [self.0[3].x(), self.0[3].y()],
        ]
    }
    pub fn as_array_f32(&self) -> [[f32; 2]; 4] {
        [
            [self.0[0].x() as _, self.0[0].y() as _],
            [self.0[1].x() as _, self.0[1].y() as _],
            [self.0[2].x() as _, self.0[2].y() as _],
            [self.0[3].x() as _, self.0[3].y() as _],
        ]
    }
}

impl Index<usize> for Quadrilateral {
    type Output = Point2D;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Quadrilateral {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl AddAssign<Vec2> for Quadrilateral {
    #[inline]
    fn add_assign(&mut self, rhs: Vec2) {
        for corner in self.0.iter_mut() {
            *corner.vec_mut() += rhs;
        }
    }
}

impl SubAssign<Vec2> for Quadrilateral {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec2) {
        for corner in self.0.iter_mut() {
            *corner.vec_mut() -= rhs;
        }
    }
}

impl MulAssign<f64> for Quadrilateral {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for corner in self.0.iter_mut() {
            *corner.vec_mut() *= rhs;
        }
    }
}