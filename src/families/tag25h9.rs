use std::borrow::Cow;

use super::AprilTagFamily;

pub fn tag25h9_create() -> AprilTagFamily {
	AprilTagFamily {
		bits: vec![
			(1, 1),
			(2, 1),
			(3, 1),
			(4, 1),
			(2, 2),
			(3, 2),
			(5, 1),
			(5, 2),
			(5, 3),
			(5, 4),
			(4, 2),
			(4, 3),
			(5, 5),
			(4, 5),
			(3, 5),
			(2, 5),
			(4, 4),
			(3, 4),
			(1, 5),
			(1, 4),
			(1, 3),
			(1, 2),
			(2, 4),
			(2, 3),
			(3, 3),
		],
		codes: vec![
			0x000000000156f1f4_u64,
			0x0000000001f28cd5_u64,
			0x00000000016ce32c_u64,
			0x0000000001ea379c_u64,
			0x0000000001390f89_u64,
			0x000000000034fad0_u64,
			0x00000000007dcdb5_u64,
			0x000000000119ba95_u64,
			0x0000000001ae9daa_u64,
			0x0000000000df02aa_u64,
			0x000000000082fc15_u64,
			0x0000000000465123_u64,
			0x0000000000ceee98_u64,
			0x0000000001f17260_u64,
			0x00000000014429cd_u64,
			0x00000000017248a8_u64,
			0x00000000016ad452_u64,
			0x00000000009670ad_u64,
			0x00000000016f65b2_u64,
			0x0000000000b8322b_u64,
			0x00000000005d715b_u64,
			0x0000000001a1c7e7_u64,
			0x0000000000d7890d_u64,
			0x0000000001813522_u64,
			0x0000000001c9c611_u64,
			0x000000000099e4a4_u64,
			0x0000000000855234_u64,
			0x00000000017b81c0_u64,
			0x0000000000c294bb_u64,
			0x000000000089fae3_u64,
			0x000000000044df5f_u64,
			0x0000000001360159_u64,
			0x0000000000ec31e8_u64,
			0x0000000001bcc0f6_u64,
			0x0000000000a64f8d_u64,
		],
		width_at_border: 7,
		total_width: 9,
		reversed_border: false,
		min_hamming: 9,
		name: Cow::Borrowed("tag25h9"),
	}
}

#[cfg(test)]
mod test {
	#[cfg(feature="compare_reference")]
	#[test]
	fn compare_25h9() {
		let tag = super::tag25h9_create();

		unsafe {
			let tag_sys = apriltag_sys::tag25h9_create();
			let ts = tag_sys.as_ref().unwrap();
			tag.assert_similar(ts);
			apriltag_sys::tag25h9_destroy(tag_sys);
		};
	}
}