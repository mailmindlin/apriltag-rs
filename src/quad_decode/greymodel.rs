use crate::apriltag_math::mat33_sym_solve;

/// Regresses a model of the form:
/// intensity(x,y) = C0*x + C1*y + CC2
/// The J matrix is the:
///    J = [ x1 y1 1 ]
///        [ x2 y2 1 ]
///        [ ...     ]
///  The A matrix is J'J
pub(super) struct Graymodel {
	A: [[f64; 3]; 3],
	B: [f64; 3],
	C: [f64; 3],
}

impl Graymodel {
	pub fn init() -> Self {
		Self {
			A: [[0f64; 3]; 3],
			B: [0f64; 3],
			C: [0f64; 3],
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

	pub fn solve(&mut self) {
		self.C = mat33_sym_solve(&self.A, &self.B);
	}
	
	pub fn interpolate(&self, x: f64, y: f64) -> f64 {
		return self.C[0]*x + self.C[1]*y + self.C[2];
	}
}