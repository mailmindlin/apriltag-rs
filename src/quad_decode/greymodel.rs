use std::ops::Add;

use crate::{apriltag_math::mat33_sym_solve, util::math::Vec2};

/// Regresses a model of the form:
/// intensity(x,y) = C0*x + C1*y + CC2
/// The J matrix is the:
///    J = [ x1 y1 1 ]
///        [ x2 y2 1 ]
///        [ ...     ]
///  The A matrix is J'J
pub(super) struct Graymodel {
	pub(super) A: [[f64; 3]; 3],
	pub(super) B: [f64; 3],
}

impl Graymodel {
	pub const fn new() -> Self {
		Self {
			A: [[0f64; 3]; 3],
			B: [0f64; 3],
		}
	}

	pub fn add(&mut self, x: f64, y: f64, gray: f64) {
		// update upper right entries of A = J'J
		self.A[0][0] += x*x;
		self.A[0][1] += x*y;
		self.A[0][2] += x;
		self.A[1][1] += y*y;
		self.A[1][2] += y;
		self.A[2][2] += 1.;

		// update B = J'gray
		self.B[0] += x * gray;
		self.B[1] += y * gray;
		self.B[2] += gray;
	}

	pub fn solve(self) -> SolvedGraymodel {
		let C = mat33_sym_solve(&self.A, &self.B);
		SolvedGraymodel::new(C)
	}
}

pub(super) struct SolvedGraymodel {
	C: [f64; 3]
}

impl SolvedGraymodel {
	pub fn new(C: [f64; 3]) -> Self {
		Self { C }
	}
	pub fn interpolate(&self, xy: Vec2) -> f64 {
		return self.C[0]*xy.x() + self.C[1]*xy.y() + self.C[2];
	}
}

impl Add<SolvedGraymodel> for SolvedGraymodel {
    type Output = SolvedGraymodel;
    fn add(self, rhs: SolvedGraymodel) -> Self::Output {
        Self::new([
			self.C[0] + rhs.C[0],
			self.C[1] + rhs.C[1],
			self.C[2] + rhs.C[2],
		])
    }
}