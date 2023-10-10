use std::{fmt::Debug, ops::Mul};

pub trait Dimensions2D<T: Debug + Mul<T, Output = T> + Copy = usize>: Debug {
    fn width(&self) -> T;
    fn height(&self) -> T;

    fn area(&self) -> T {
        self.width() * self.height()
    }

    fn contains(&self, index: &Index2D<T>) -> bool;
}

/// Index into 2d
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct Index2D<T = usize> {
    pub x: T,
    pub y: T,
}

impl<T> From<(T, T)> for Index2D<T> {
    fn from(value: (T, T)) -> Self {
        let (x, y) = value;
        Self {
            x,
            y,
        }
    }
}