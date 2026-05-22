// #[cfg(not(target_arch="aarch64"))]
mod reference;

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
mod arm;

pub(crate) trait Vec2Builder: Sized {
    /// Create new zero vector
    fn zero() -> Self;
    //     Self::dup(0.)
    // }

    /// Create vector with both x and y having the same value
    fn dup(value: f64) -> Self;
    //     Self::of(value, value)
    // }

    /// Create vector with values
    fn of(x: f64, y: f64) -> Self;
}

pub(crate) trait FMA: Sized {
    /// Compute self + (u * v)
    fn fma(self, u: Self, v: Self) -> Self;
}

// #[cfg(not(target_arch="aarch64"))]
pub(crate) use reference::Vec2;
// #[cfg(target_arch="aarch64")]
// pub(crate) use vec2_a64::Vec2;

#[cfg(test)]
mod test {
    use super::Vec2Builder;
    use super::reference::Vec2;
    #[cfg(target_arch="aarch64")]
    use super::arm::Vec2 as Vec2A64;

    const EPS: f64 = 1e-10;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < EPS, "{a} != {b}");
    }

    #[test]
    fn test_add() {
        let u = Vec2::of(1., 2.);
        let v = Vec2::of(-1., 5.);
        let w = u + v;
        assert_eq!(w.x(), 0.);
        assert_eq!(w.y(), 7.);
    }

    #[test]
    #[cfg(target_arch="aarch64")]
    fn test_add_a64() {
        let u = Vec2A64::of(1., 2.);
        let v = Vec2A64::of(-1., 5.);
        let w = u + &v;
        assert_eq!(w.x(), 0.);
        assert_eq!(w.y(), 7.);
    }

    #[test]
    fn test_mag() {
        let v = Vec2::of(3., 4.);
        assert_close(v.mag(), 5.);
        let z = Vec2::zero();
        assert_close(z.mag(), 0.);
    }

    #[test]
    fn test_dot() {
        let a = Vec2::of(1., 0.);
        let b = Vec2::of(0., 1.);
        assert_close(a.dot(b), 0.);

        let c = Vec2::of(2., 3.);
        let d = Vec2::of(4., 5.);
        assert_close(c.dot(d), 23.);
    }

    #[test]
    fn test_norm() {
        let v = Vec2::of(3., 4.);
        let n = v.norm();
        assert_close(n.mag(), 1.);
        assert_close(n.x(), 0.6);
        assert_close(n.y(), 0.8);
    }

    #[test]
    fn test_angle() {
        use std::f64::consts::{FRAC_PI_2, PI};
        assert_close(Vec2::of(1., 0.).angle(), 0.);
        assert_close(Vec2::of(0., 1.).angle(), FRAC_PI_2);
        assert_close(Vec2::of(-1., 0.).angle(), PI);
        assert_close(Vec2::of(0., -1.).angle(), -FRAC_PI_2);
    }

    #[test]
    fn test_from_angle_roundtrip() {
        use std::f64::consts::FRAC_PI_4;
        let v = Vec2::from_angle(FRAC_PI_4);
        assert_close(v.mag(), 1.);
        assert_close(v.angle(), FRAC_PI_4);
    }

    #[test]
    fn test_flip() {
        let v = Vec2::of(1., 2.);
        let f = v.flip();
        assert_eq!(f.x(), 2.);
        assert_eq!(f.y(), 1.);
    }

    #[test]
    fn test_rev_negx() {
        let v = Vec2::of(3., 5.);
        let r = v.rev_negx();
        assert_eq!(r.x(), 5.);
        assert_eq!(r.y(), -3.);
    }

    #[test]
    fn test_rev_negy() {
        let v = Vec2::of(3., 5.);
        let r = v.rev_negy();
        assert_eq!(r.x(), -5.);
        assert_eq!(r.y(), 3.);
    }
}