use std::{fmt::Debug, ops::Mul};

pub trait Dimensions2D<T: Debug + Mul<T, Output = T> + Copy = usize>: Debug {
    fn width(&self) -> T;
    fn height(&self) -> T;

    fn area(&self) -> T {
        self.width() * self.height()
    }

    fn contains(&self, index: &Index2D<T>) -> bool;

    fn offset(&self, index: &Index2D<T>) -> T {
        match self.offset_checked(index) {
            Some(value) => value,
            None => panic!("Index {:?} out of bounds for dimensions {:?}", index, self),
        }
    }
    fn offset_checked(&self, index: &Index2D<T>) -> Option<T> {
        if !self.contains(&index) {
            None
        } else {
            Some(self.offset_unchecked(index))
        }
    }
    fn offset_unchecked(&self, index: &Index2D<T>) -> T;

    fn index_for_offset(&self, offset: T) -> Index2D<T> {
        match self.index_for_offset_checked(offset) {
            Some(value) => value,
            None => panic!("Offset {:?} out of bounds for dimensions {:?}", offset, self),
        }
    }
    fn index_for_offset_checked(&self, offset: T) -> Option<Index2D<T>> {
        let index = self.index_for_offset_unchecked(offset);
        if !self.contains(&index) {
            None
        } else {
            Some(index)
        }
    }
    fn index_for_offset_unchecked(&self, offset: T) -> Index2D<T>;
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

impl<T: Copy> Index2D<T> {
    pub const fn transposed(&self) -> Self {
		Self {
            x: self.y,
            y: self.x,
        }
	}
}