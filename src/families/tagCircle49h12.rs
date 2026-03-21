use std::borrow::Cow;

use super::AprilTagFamily;

super::impl_tag!("tagCircle49h12");
pub fn tagCircle49h12_create() -> AprilTagFamily {
	AprilTagFamily {
		bits: vec![
			(1, -3),
			(2, -3),
			(3, -3),
			(-1, -2),
			(0, -2),
			(1, -2),
			(2, -2),
			(3, -2),
			(4, -2),
			(5, -2),
			(1, 1),
			(2, 1),
			(7, 1),
			(7, 2),
			(7, 3),
			(6, -1),
			(6, 0),
			(6, 1),
			(6, 2),
			(6, 3),
			(6, 4),
			(6, 5),
			(3, 1),
			(3, 2),
			(3, 7),
			(2, 7),
			(1, 7),
			(5, 6),
			(4, 6),
			(3, 6),
			(2, 6),
			(1, 6),
			(0, 6),
			(-1, 6),
			(3, 3),
			(2, 3),
			(-3, 3),
			(-3, 2),
			(-3, 1),
			(-2, 5),
			(-2, 4),
			(-2, 3),
			(-2, 2),
			(-2, 1),
			(-2, 0),
			(-2, -1),
			(1, 3),
			(1, 2),
			(2, 2),
		],
		#[cfg(rust_analyzer)]
		codes: vec![],
		#[cfg(not(rust_analyzer))]
		codes: include!("tagCircle49h12_codes.rs").to_vec(),
		width_at_border: 5,
		total_width: 11,
		reversed_border: true,
		min_hamming: 12,
		name: Cow::Borrowed(NAME_S),
	}
}

#[cfg(test)]
mod test {
	#[cfg(feature="compare_reference")]
	#[test]
	fn compare_circle_49h12() {
		let tag = super::tagCircle49h12_create();

		unsafe {
			let tag_sys = apriltag_sys::tagCircle49h12_create();
			let ts = tag_sys.as_ref().unwrap();
			tag.assert_similar(ts);
			apriltag_sys::tagCircle49h12_destroy(tag_sys);
		};
	}
}