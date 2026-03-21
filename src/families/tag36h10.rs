use std::borrow::Cow;

use super::AprilTagFamily;

super::impl_tag!("tag36h10");
pub fn tag36h10_create() -> AprilTagFamily {
	AprilTagFamily {
		bits: vec![
			(1, 1),
			(2, 1),
			(3, 1),
			(4, 1),
			(5, 1),
			(2, 2),
			(3, 2),
			(4, 2),
			(3, 3),
			(6, 1),
			(6, 2),
			(6, 3),
			(6, 4),
			(6, 5),
			(5, 2),
			(5, 3),
			(5, 4),
			(4, 3),
			(6, 6),
			(5, 6),
			(4, 6),
			(3, 6),
			(2, 6),
			(5, 5),
			(4, 5),
			(3, 5),
			(4, 4),
			(1, 6),
			(1, 5),
			(1, 4),
			(1, 3),
			(1, 2),
			(2, 5),
			(2, 4),
			(2, 3),
			(3, 4),
		],
		#[cfg(rust_analyzer)]
		codes: vec![],
		#[cfg(not(rust_analyzer))]
		codes: include!("tag36h10_codes.rs").to_vec(),
		width_at_border: 8,
		total_width: 10,
		reversed_border: false,
		min_hamming: 10,
		name: Cow::Borrowed(NAME_S),
  }
}

#[cfg(test)]
mod test {
	#[cfg(feature="compare_reference")]
	#[test]
	fn compare_36h10() {
		let tag = super::tag36h10_create();

		unsafe {
			let tag_sys = apriltag_sys::tag36h10_create();
			let ts = tag_sys.as_ref().unwrap();
			tag.assert_similar(ts);
			apriltag_sys::tag36h10_destroy(tag_sys);
		};
	}
}