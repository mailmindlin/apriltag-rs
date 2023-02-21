use super::Mat;

pub (crate) struct MatChol {
	is_spd: bool,
	u: Mat,
}

impl MatChol {
	/// NOTE: The below implementation of Cholesky is different from the one
	/// used in NGV.
	pub fn new(mat: &Mat) -> Self {
		assert!(mat.dims.is_square());
		let N = mat.rows();

		// make upper right
		let mut U = mat.clone();

		// don't actually need to clear lower-left... we won't touch it.
	/*    for (int i = 0; i < U->nrows; i++) {
		for (int j = 0; j < i; j++) {
	//            assert(U[(i, j)] == U[(j, i)]);
	U[(i, j)] = 0;
	}
	}
	*/
		let mut is_spd = true; // (mat.nrows == mat.ncols);

		for i in 0..N {
			let d = U[(i,i)];
			is_spd &= d > 0.;

			let d = f64::max(d, Mat::EPS)
				.sqrt()
				.recip();

			for j in i..N {
				U[(i, j)] *= d;
			}

			for j in (i+1)..N {
				let s = U[(i, j)];

				if s == 0. {
					continue;
				}

				for k in j..N {
					U[(j, k)] -= U[(i, k)]*s;
				}
			}
		}

		Self {
			is_spd,
			u: U,
		}
	}

	pub fn solve(&self, b: &Mat) -> Mat {
		let ref u = self.u;
		let mut x = b.clone();

		// LUx = b

		// solve Ly = b ==> (U')y = b

		for i in 0..u.rows() {
			for j in 0..i {
				// b[i] -= L[i,j]*x[j]... replicated across columns of b
				//   ==> i.e., ==>
				// b[i,k] -= L[i,j]*x[j,k]
				for k in 0..b.cols() {
					x[(i,k)] -= u[(j,i)] * x[(j,k)];
				}
			}
			// x[i] = b[i] / L[i,i]
			for k in 0..b.cols() {
				x[(i,k)] /= u[(i,i)];
			}
		}

		// solve Ux = y
		for k in (0..u.cols()).rev() {
			let LUkk = 1.0 / u[(k,k)];
			for t in 0..b.cols() {
				x[(k,t)] *= LUkk;
			}

			for i in 0..k {
				let LUik = -u[(i,k)];
				for t in 0..b.cols() {
					x[(i,t)] += x[(k,t)] * LUik;
				}
			}
		}

		return x;
	}
}