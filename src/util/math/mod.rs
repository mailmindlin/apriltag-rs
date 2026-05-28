mod vec2;
mod vec3;
pub(crate) use vec2::Vec2;
pub(crate) use vec2::{Vec2Builder, FMA};
pub(crate) use vec3::Vec3;

pub(crate) mod mat;
pub(crate) use mat::{matrix_op, Matmul2, MatmulTranspose, TransposedMatmul};
pub use mat::Mat33;
pub(crate) mod poly;

use std::f64::consts::{TAU, PI};


/// Map `v` to `[0, τ]`
#[inline]
pub(crate) fn mod2pi_pos(v: f64) -> f64 {
    v - TAU * (v / TAU).floor()
}

/// Map `v` to `[-π, π]`
#[inline]
pub(crate) fn mod2pi(v: f64) -> f64 {
    mod2pi_pos(v + PI) - PI
}

#[cfg(test)]
mod test {
    use std::f64::consts::{PI, TAU};
    use super::{mod2pi_pos, mod2pi};

    const EPS: f64 = 1e-10;

    #[test]
    fn mod2pi_pos_zero() {
        assert!((mod2pi_pos(0.) - 0.).abs() < EPS);
    }

    #[test]
    fn mod2pi_pos_in_range() {
        assert!((mod2pi_pos(1.0) - 1.0).abs() < EPS);
        assert!((mod2pi_pos(PI) - PI).abs() < EPS);
    }

    #[test]
    fn mod2pi_pos_wraps_negative() {
        assert!((mod2pi_pos(-PI) - PI).abs() < EPS);
        assert!((mod2pi_pos(-TAU) - 0.).abs() < EPS);
    }

    #[test]
    fn mod2pi_pos_wraps_large() {
        let v = mod2pi_pos(TAU + 1.0);
        assert!((v - 1.0).abs() < EPS);
        let v = mod2pi_pos(100. * TAU + 0.5);
        assert!((v - 0.5).abs() < EPS);
    }

    #[test]
    fn mod2pi_zero() {
        assert!((mod2pi(0.) - 0.).abs() < EPS);
    }

    #[test]
    fn mod2pi_positive() {
        assert!((mod2pi(0.5) - 0.5).abs() < EPS);
        assert!((mod2pi(PI - 0.01) - (PI - 0.01)).abs() < EPS);
    }

    #[test]
    fn mod2pi_negative() {
        assert!((mod2pi(-0.5) - (-0.5)).abs() < EPS);
        assert!((mod2pi(-PI + 0.01) - (-PI + 0.01)).abs() < EPS);
    }

    #[test]
    fn mod2pi_wraps() {
        assert!((mod2pi(TAU + 0.5) - 0.5).abs() < EPS);
        assert!((mod2pi(-TAU - 0.5) - (-0.5)).abs() < EPS);
    }
}