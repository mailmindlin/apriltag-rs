// #[cfg(not(target_arch="aarch64"))]
mod reference;

#[cfg(any(target_arch="aarch64", target_arch="armv7", target_arch="armv7"))]
mod arm;

#[cfg(target_arch="x86_64")]
mod x86;

#[const_trait]
pub(crate) trait Vec2Builder: Sized {
    /// Create new zero vector
    fn zero() -> Self {
        Self::dup(0.)
    }

    /// Create vector with both x and y having the same value
    fn dup(value: f64) -> Self {
        Self::of(value, value)
    }

    /// Create vector with values
    fn of(x: f64, y: f64) -> Self;
}

#[const_trait]
pub(crate) trait FMA: Sized {
    fn fma(self, u: Self, v: Self) -> Self;
}

// #[cfg(not(target_arch="aarch64"))]
pub(crate) use reference::Vec2;
// #[cfg(target_arch="aarch64")]
// pub(crate) use vec2_a64::Vec2;

#[cfg(all(test, target_arch="aarch64"))]
mod test {
    use super::Vec2Builder;
    use super::reference::Vec2 as Vec2Ref;
    #[cfg(target_arch="aarch64")]
    use super::arm::Vec2 as Vec2A64;

    #[test]
    fn test_add() {
        let u = Vec2Ref::of(1., 2.);
        let v = Vec2Ref::of(-1., 5.);
        let w = u + &v;
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
}