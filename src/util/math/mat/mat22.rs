
pub(crate) struct Mat22([f64; 4]);

impl Mat22 {
	/// Compute matrix determinant
	pub(crate) fn det(&self) -> f64 {
		self.0[0] * self.0[3] - self.0[1] * self.0[2]
	}
}