use std::borrow::Cow;

use super::AprilTagFamily;

super::impl_tag!("tagCustom48h12");
pub fn tagCustom48h12_create() -> AprilTagFamily {
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
			(6, -2),
			(1, 1),
			(2, 1),
			(3, 1),
			(7, -2),
			(7, -1),
			(7, 0),
			(7, 1),
			(7, 2),
			(7, 3),
			(7, 4),
			(7, 5),
			(7, 6),
			(4, 1),
			(4, 2),
			(4, 3),
			(7, 7),
			(6, 7),
			(5, 7),
			(4, 7),
			(3, 7),
			(2, 7),
			(1, 7),
			(0, 7),
			(-1, 7),
			(4, 4),
			(3, 4),
			(2, 4),
			(-2, 7),
			(-2, 6),
			(-2, 5),
			(-2, 4),
			(-2, 3),
			(-2, 2),
			(-2, 1),
			(-2, 0),
			(-2, -1),
			(1, 4),
			(1, 3),
			(1, 2),
		],
		#[cfg(rust_analyzer)]
		codes: vec![],
		#[cfg(not(rust_analyzer))]
		codes: include!("tagCustom48h12_codes.rs").to_vec(),
		width_at_border: 6,
		total_width: 10,
		reversed_border: true,
		min_hamming: 12,
		name: Cow::Borrowed(NAME_S),
	}
}

#[cfg(test)]
mod test {
	#[cfg(feature="compare_reference")]
	#[test]
	fn compare_tagCustom48h12() {
		let tag = super::tagCustom48h12_create();

		unsafe {
			let tag_sys = apriltag_sys::tagCustom48h12_create();
			let ts = tag_sys.as_ref().unwrap();
			tag.assert_similar(ts);
			apriltag_sys::tagCustom48h12_destroy(tag_sys);
		};
	}
}