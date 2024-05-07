use std::borrow::Cow;

use super::AprilTagFamily;

pub fn tag16h5_create() -> AprilTagFamily {
	AprilTagFamily {
		bits: vec![
			(1, 1),
			(2, 1),
			(3, 1),
			(2, 2),
			(4, 1),
			(4, 2),
			(4, 3),
			(3, 2),
			(4, 4),
			(3, 4),
			(2, 4),
			(3, 3),
			(1, 4),
			(1, 3),
			(1, 2),
			(2, 3),
		],
		codes: vec![
			0x00000000000027c8_u64,
			0x00000000000031b6_u64,
			0x0000000000003859_u64,
			0x000000000000569c_u64,
			0x0000000000006c76_u64,
			0x0000000000007ddb_u64,
			0x000000000000af09_u64,
			0x000000000000f5a1_u64,
			0x000000000000fb8b_u64,
			0x0000000000001cb9_u64,
			0x00000000000028ca_u64,
			0x000000000000e8dc_u64,
			0x0000000000001426_u64,
			0x0000000000005770_u64,
			0x0000000000009253_u64,
			0x000000000000b702_u64,
			0x000000000000063a_u64,
			0x0000000000008f34_u64,
			0x000000000000b4c0_u64,
			0x00000000000051ec_u64,
			0x000000000000e6f0_u64,
			0x0000000000005fa4_u64,
			0x000000000000dd43_u64,
			0x0000000000001aaa_u64,
			0x000000000000e62f_u64,
			0x0000000000006dbc_u64,
			0x000000000000b6eb_u64,
			0x000000000000de10_u64,
			0x000000000000154d_u64,
			0x000000000000b57a_u64,
		],
		width_at_border: 6,
		total_width: 8,
		reversed_border: false,
		min_hamming: 5,
		name: Cow::Borrowed("tag16h5"),
	}
}

#[cfg(test)]
mod test {
	#[cfg(feature="compare_reference")]
	#[test]
	fn compare_16h5() {
		let tag = super::tag16h5_create();

		unsafe {
			let tag_sys = apriltag_sys::tag16h5_create();
			let ts = tag_sys.as_ref().unwrap();
			tag.assert_similar(ts);
			apriltag_sys::tag16h5_destroy(tag_sys);
		};
	}
}