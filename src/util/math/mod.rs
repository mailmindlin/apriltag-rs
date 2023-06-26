mod vec2;
mod vec3;
pub(crate) use vec2::{Vec2, Vec2Builder, FMA};
pub(crate) use vec3::Vec3;

pub mod mat;
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