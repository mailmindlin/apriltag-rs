/// Matrix/vector expression macro.
///
/// All types are Copy — all operands are passed by value.
/// Parentheses are required for grouping (no precedence parsing).
///
/// OPERATOR SUMMARY
/// ────────────────────────────────────────────────────────────────
///  -v               v.neg()                        → Vec3
///  v1 + v2          v1.add(v2)                     → Vec3 / Mat33
///  v1 - v2          v1.sub(v2)                     → Vec3 / Mat33
///  v1 . v2          v1.dot(v2)                     → Scalar
///  v1 % v2          v1.cross(v2)                   → Vec3
///  v * s            v.scale(s)   (also M * s)      → Vec3 / Mat33
///  v / s            v.div(s)                       → Vec3
///  v1 @ v2          v1.outer(v2)                   → Mat33
///  M @ v            M.mul_vec(v)                   → Vec3
///  M1 @ M2          M1.matmul(M2)                  → Mat33
///  ^M               M.transposed()                 → Mat33
///  ^M1 @ M2         M1.transpose_matmul(M2)        → Mat33
///  M1 @ ^M2         M1.matmul_transpose(M2)        → Mat33
/// ────────────────────────────────────────────────────────────────

pub(crate) trait Matmul<T> {
	type Output;
	fn matmul(self, rhs: T) -> Self::Output;
}

pub(crate) trait MatmulTranspose<T> {
	type Output;
	fn matmul_transpose(self, rhs: T) -> Self::Output;
}
pub(crate) trait TransposedMatmul<T> {
	type Output;
	fn transposed_matmul(self, rhs: T) -> Self::Output;
}

impl<'a> Matmul<&'a Mat33> for &'a Mat33 {
	type Output = Mat33;
	fn matmul(self, rhs: &'a Mat33) -> Self::Output {
		self.matmul(rhs)
	}
}
impl<'a> MatmulTranspose<&'a Mat33> for &'a Mat33 {
	type Output = Mat33;
	fn matmul_transpose(self, rhs: &'a Mat33) -> Self::Output {
		self.matmul_transpose(rhs)
	}
}
impl<'a> TransposedMatmul<&'a Mat33> for &'a Mat33 {
	type Output = Mat33;
	fn transposed_matmul(self, rhs: &'a Mat33) -> Self::Output {
		self.transpose_matmul(rhs)
	}
}
impl TransposedMatmul<Self> for Vec3 {
	type Output = Mat33;
	fn transposed_matmul(self, rhs: Self) -> Self::Output {
		self.outer(rhs)
	}
}
/*impl<'a> TransposedMatmul<&'a Vec3> for &'a Vec3 {
	type Output = Mat33;
	fn transposed_matmul(self, rhs: &'a Vec3) -> Self::Output {
		self.outer(rhs)
	}
}
impl<'a> TransposedMatmul<Vec3> for &'a Vec3 {
	type Output = Mat33;
	fn transposed_matmul(self, rhs: Vec3) -> Self::Output {
		self.outer(&rhs)
	}
}*/

macro_rules! matrix_op {

    //
    // ── NEGATION  -a ─────────────────────────────────────────────────
    //

    ( - $a:ident )           => { $a.neg() };
    ( - ( $($e:tt)* ) )      => { matrix_op!($($e)*).neg() };

    //
    // ── TRANSPOSE  ~a  (standalone) ─────────────────────────────────
    //

    ( ~ $a:ident )           => { $a.transposed() };
    ( ~ ( $($e:tt)* ) )      => { matrix_op!($($e)*).transposed() };

    //
    // ── DOT PRODUCT  a . b  (→ Scalar) ──────────────────────────────
    //

    ( $a:ident . $b:ident )                      => { $a.dot($b) };
    ( $a:ident . ( $($r:tt)* ) )                 => { $a.dot(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) . $b:ident )                 => { matrix_op!($($l)*).dot($b) };
    ( ( $($l:tt)* ) . ( $($r:tt)* ) )            => { matrix_op!($($l)*).dot(matrix_op!($($r)*)) };

    //
    // ── CROSS PRODUCT  a % b  (→ Vec3) ──────────────────────────────
    //

    ( $a:ident % $b:ident )                      => { $a.cross($b) };
    ( $a:ident % ( $($r:tt)* ) )                 => { $a.cross(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) % $b:ident )                 => { matrix_op!($($l)*).cross($b) };
    ( ( $($l:tt)* ) % ( $($r:tt)* ) )            => { matrix_op!($($l)*).cross(matrix_op!($($r)*)) };

    //
    // ── TRANSPOSE-MATMUL  ~a @ b  (→ Mat33) ─────────────────────────
    // Must come before the general @ rules.
    //

    ( ~ $a:ident @ $b:ident )                    => { $a.transposed_matmul($b) };
    ( ~ $a:ident @ ( $($r:tt)* ) )               => { $a.transposed_matmul(matrix_op!($($r)*)) };
    ( ~ ( $($l:tt)* ) @ $b:ident )               => { matrix_op!($($l)*).transposed_matmul($b) };
    ( ~ ( $($l:tt)* ) @ ( $($r:tt)* ) )          => { matrix_op!($($l)*).transposed_matmul(matrix_op!($($r)*)) };

    //
    // ── MATMUL-TRANSPOSE  a @ ~b  (→ Mat33) ─────────────────────────
    // Must come before the general @ rules.
    //

    ( $a:ident @ ~ $b:ident )                    => { $a.matmul_transpose($b) };
    ( $a:ident @ ~ ( $($r:tt)* ) )               => { $a.matmul_transpose(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) @ ~ $b:ident )               => { matrix_op!($($l)*).matmul_transpose($b) };
    ( ( $($l:tt)* ) @ ~ ( $($r:tt)* ) )          => { matrix_op!($($l)*).matmul_transpose(matrix_op!($($r)*)) };

    //
    // ── MATMUL / OUTER / MUL-VEC  a @ b ─────────────────────────────
    // Covers: v1 @ v2 (outer), M @ v (mul_vec), M1 @ M2 (matmul).
    // Dispatched to .matmul() — resolved by type/impl at compile time.
    //

    ( $a:ident @ $b:ident )                      => { $a.matmul($b) };
    ( $a:ident @ ( $($r:tt)* ) )                 => { $a.matmul(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) @ $b:ident )                 => { matrix_op!($($l)*).matmul($b) };
    ( ( $($l:tt)* ) @ ( $($r:tt)* ) )            => { matrix_op!($($l)*).matmul(matrix_op!($($r)*)) };

    //
    // ── SCALE / MUL  a * s ───────────────────────────────────────────
    // Covers: v * s, M * s.
    //

    ( $a:ident * $b:ident )                      => { $a.mul($b) };
    ( $a:ident * ( $($r:tt)* ) )                 => { $a.mul(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) * $b:ident )                 => { matrix_op!($($l)*).mul($b) };
    ( ( $($l:tt)* ) * ( $($r:tt)* ) )            => { matrix_op!($($l)*).mul(matrix_op!($($r)*)) };

    //
    // ── DIVISION  a / s ──────────────────────────────────────────────
    //

    ( $a:ident / $b:ident )                      => { $a.div($b) };
    ( $a:ident / ( $($r:tt)* ) )                 => { $a.div(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) / $b:ident )                 => { matrix_op!($($l)*).div($b) };
    ( ( $($l:tt)* ) / ( $($r:tt)* ) )            => { matrix_op!($($l)*).div(matrix_op!($($r)*)) };

    //
    // ── ADDITION  a + b ──────────────────────────────────────────────
    //

    ( $a:ident + $b:ident )                      => { $a.add($b) };
    ( $a:ident + ( $($r:tt)* ) )                 => { $a.add(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) + $b:ident )                 => { matrix_op!($($l)*).add($b) };
    ( ( $($l:tt)* ) + ( $($r:tt)* ) )            => { matrix_op!($($l)*).add(matrix_op!($($r)*)) };

    //
    // ── SUBTRACTION  a - b ───────────────────────────────────────────
    //

    ( $a:ident - $b:ident )                      => { $a.sub($b) };
    ( $a:ident - ( $($r:tt)* ) )                 => { $a.sub(matrix_op!($($r)*)) };
    ( ( $($l:tt)* ) - $b:ident )                 => { matrix_op!($($l)*).sub($b) };
    ( ( $($l:tt)* ) - ( $($r:tt)* ) )            => { matrix_op!($($l)*).sub(matrix_op!($($r)*)) };

    //
    // ── BASE CASES ───────────────────────────────────────────────────
    //

    ( $a:ident )             => { $a };
    ( ( $($inner:tt)* ) )    => { matrix_op!($($inner)*) };
}
pub(crate) use matrix_op;

use crate::Vec3;

use super::Mat33;