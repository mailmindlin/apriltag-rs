use std::borrow::Cow;

use super::AprilTagFamily;

super::impl_tag!("tagStandard41h12");
pub fn tagStandard41h12_create() -> AprilTagFamily {
	AprilTagFamily {
		bits: vec![
			(-2, -2),
			(-1, -2),
			(0, -2),
			(1, -2),
			(2, -2),
			(3, -2),
			(4, -2),
			(5, -2),
			(1, 1),
			(2, 1),
			(6, -2),
			(6, -1),
			(6, 0),
			(6, 1),
			(6, 2),
			(6, 3),
			(6, 4),
			(6, 5),
			(3, 1),
			(3, 2),
			(6, 6),
			(5, 6),
			(4, 6),
			(3, 6),
			(2, 6),
			(1, 6),
			(0, 6),
			(-1, 6),
			(3, 3),
			(2, 3),
			(-2, 6),
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
		codes: include!("tagStandard41h12_codes.rs").to_vec(),
		width_at_border: 8,
		total_width: 10,
		reversed_border: false,
		min_hamming: 12,
		name: Cow::Borrowed(NAME_S),
	}
}

#[cfg(test)]
mod test {
	#[cfg(feature="compare_reference")]
	#[test]
	fn compare_tagStandard41h12() {
		let tag = super::tagStandard41h12_create();

		unsafe {
			let tag_sys = apriltag_sys::tagStandard41h12_create();
			let ts = tag_sys.as_ref().unwrap();
			tag.assert_similar(ts);
			apriltag_sys::tagStandard41h12_destroy(tag_sys);
		};
	}
}