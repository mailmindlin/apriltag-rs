use super::Code;


/// Assuming we are drawing the image one quadrant at a time, what would the rotated image look like?
/// Special care is taken to handle the case where there is a middle pixel of the image.
/// 
/// if the bits in w were arranged in a d*d grid and that grid was
/// rotated, what would the new bits in w be?
/// The bits are organized like this (for d = 3):
///
/// ```text
///  8 7 6       2 5 8      0 1 2
///  5 4 3  ==>  1 4 7 ==>  3 4 5    (rotate90 applied twice)
///  2 1 0       0 3 6      6 7 8
/// ```
pub const fn rotate90(w: Code, num_bits: u64) -> Code {
	/*let mut wr = 0;

	for r in (0..d).rev() {
		for c in 0..d {
			let b = r + d*c;

			wr <<= 1;

			if (w & (1u64 << b)) != 0 {
				wr |= 1;
			}
		}
	}

	wr*/

	// Odd/even
	let (p, l) = if num_bits % 4 == 1 {
		(num_bits - 1, 1)
	} else {
		(num_bits, 0)
	};

	let w = ((w >> l) << (p/4 + l)) | (w >> (3 * p/4 + l) << l) | (w & l);
	w & (1u64 << num_bits) - 1
}

pub const fn rotations(code: u64, num_bits: u64) -> [u64; 4] {
	let r0 = code;
	let r1 = rotate90(code, num_bits);
	let r2 = rotate90(r1, num_bits);
	let r3 = rotate90(r2, num_bits);
	[ r0, r1, r2, r3 ]
}


#[cfg(test)]
#[test]
fn test_rotate90b() {
	let w_orig = 51559569327u64;
	let w_r1 = rotate90(w_orig, 36);
	assert_eq!(w_r1, 10220429184);
	let w_r2 = rotate90(w_r1, 36);
	assert_eq!(w_r2, 10179510348);
	let w_r3 = rotate90(w_r2, 36);
	assert_eq!(w_r3, 57948543051);
	let w_r4 = rotate90(w_r3, 36);
	assert_eq!(w_r4, w_orig);
}