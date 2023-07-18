use std::ops::{Add, Sub, Neg, Div, Mul, MulAssign};
#[cfg(target_arch="aarch64")]
use core::arch::aarch64::{float64x2_t, vdupq_n_f64, vcombine_f64, vdup_n_f64, vfmaq_f64, vgetq_lane_f64, vpaddd_f64, vmulq_f64, vget_lane_u64, vreinterpret_u64_u32, vqmovn_u64, vceqq_f64, vaddq_f64, vsubq_f64, vdivq_f64, vnegq_f64};
#[cfg(not(target_arch="aarch64"))]
use core::arch::aarch64::{float64x2_t, vdupq_n_f64, vcombine_f64, vdup_n_f64, vfmaq_f64, vgetq_lane_f64, vpaddd_f64, vmulq_f64, vget_lane_u64, vreinterpret_u64_u32, vqmovn_u64, vceqq_f64, vaddq_f64, vsubq_f64, vdivq_f64, vnegq_f64};

use super::{Vec2Builder, FMA};

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct Vec2(float64x2_t);

impl Vec2Builder for Vec2 {
	#[inline(always)]
	fn zero() -> Self {
		Self::dup(0.)
	}

	#[inline(always)]
	fn dup(value: f64) -> Self {
		Self(unsafe { vdupq_n_f64(value) })
	}

	#[inline(always)]
    fn of(x: f64, y: f64) -> Self {
        Self(unsafe {
			vcombine_f64(
				vdup_n_f64(x),
				vdup_n_f64(y)
			)
		})
    }
}

impl FMA for Vec2 {
	#[inline(always)]
	fn fma(self, u: Vec2, v: Vec2) -> Vec2 {
		Self(unsafe {
			vfmaq_f64(self.0, u.0, v.0)
		})
	}
}

impl Vec2 {
	pub fn from_angle(theta: f64) -> Self {
		let (x, y) = theta.sin_cos();
		Self::of(x, y)
	}

	/// X component
	#[inline(always)]
	pub fn x(&self) -> f64 {
		unsafe { vgetq_lane_f64::<0>(self.0) }
	}

	/// Y component
	#[inline(always)]
	pub fn y(&self) -> f64 {
		unsafe { vgetq_lane_f64::<1>(self.0) }
	}

    #[inline(always)]
    pub fn neg_y(self) -> Self {
        self * Self::of(1., -1.)
    }

	/// Vector magnitude
	#[inline]
	pub fn mag(&self) -> f64 {
		unsafe {
			vpaddd_f64(
				vmulq_f64(self.0, self.0)
			)
		}.sqrt()
	}

	/// Vector dot product
	#[inline]
	pub fn dot(&self, other: Vec2) -> f64 {
		unsafe {
			vpaddd_f64(
				vmulq_f64(self.0, other.0)
			)
		}
	}

    #[inline]
    pub fn angle(&self) -> f64 {
        f64::atan2(self.y(), self.x())
    }

	/// This vector, normalized
	pub fn norm(&self) -> Vec2 {
		let mag = self.mag();
		self / mag
	}
}

impl PartialEq for Vec2 {
    fn eq(&self, other: &Self) -> bool {
		let val = unsafe {
			vget_lane_u64::<0>(
				vreinterpret_u64_u32(
					vqmovn_u64(
						vceqq_f64(self.0, other.0)
					)
				)
			)
		};
		val == u64::MAX
    }
}
impl Add<&Vec2> for Vec2 {
	type Output = Vec2;

	fn add(self, rhs: &Vec2) -> Self::Output {
		Self(unsafe {
			vaddq_f64(self.0, rhs.0)
		})
	}
}

impl Add<&Vec2> for &Vec2 {
	type Output = Vec2;

	#[inline(always)]
	fn add(self, rhs: &Vec2) -> Self::Output {
		Vec2(unsafe {
			vaddq_f64(self.0, rhs.0)
		})
	}
}

impl Sub<&Vec2> for Vec2 {
	type Output = Vec2;
	
	#[inline(always)]
	fn sub(self, rhs: &Vec2) -> Self::Output {
		Self(unsafe {
			vsubq_f64(self.0, rhs.0)
		})
	}
}

impl Sub<&Vec2> for &Vec2 {
	type Output = Vec2;

	fn sub(self, rhs: &Vec2) -> Self::Output {
		Vec2(unsafe {
			vsubq_f64(self.0, rhs.0)
		})
	}
}

impl Sub<f64> for Vec2 {
	type Output = Vec2;
	
	#[inline(always)]
	fn sub(self, rhs: f64) -> Self::Output {
		Self(unsafe {
			vsubq_f64(self.0, vdupq_n_f64(rhs))
		})
	}
}

impl Mul<Vec2> for Vec2 {
	type Output = Vec2;
	#[inline(always)]
	fn mul(self, rhs: Vec2) -> Self::Output {
		Self(unsafe {
			vmulq_f64(self.0, rhs.0)
		})
	}
}

impl Mul<f64> for Vec2 {
	type Output = Vec2;
	#[inline(always)]
	fn mul(self, rhs: f64) -> Self::Output {
		Self(unsafe {
			vmulq_f64(self.0, vdupq_n_f64(rhs))
		})
	}
}

impl Mul<f64> for &Vec2 {
	type Output = Vec2;
	#[inline(always)]
	fn mul(self, rhs: f64) -> Self::Output {
		Vec2(unsafe {
			vsubq_f64(self.0, vdupq_n_f64(rhs))
		})
	}
}

impl MulAssign<f64> for Vec2 {
	#[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
		self.0 = unsafe {
			vmulq_f64(self.0, vdupq_n_f64(rhs))
		};
    }
}

impl Div<f64> for &Vec2 {
	type Output = Vec2;
	#[inline(always)]
	fn div(self, rhs: f64) -> Self::Output {
		Vec2(unsafe {
			vdivq_f64(self.0, vdupq_n_f64(rhs))
		})
	}
}

impl Div<f64> for Vec2 {
	type Output = Vec2;
	#[inline(always)]
	fn div(self, rhs: f64) -> Self::Output {
		Self(unsafe {
			vdivq_f64(self.0, vdupq_n_f64(rhs))
		})
	}
}

impl Neg for Vec2 {
	type Output = Vec2;
	#[inline(always)]
	fn neg(self) -> Self::Output {
		Self(unsafe {
			vnegq_f64(self.0)
		})
	}
}