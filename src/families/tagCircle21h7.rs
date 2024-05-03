use std::borrow::Cow;

use super::AprilTagFamily;

pub fn tagCircle21h7_create() -> AprilTagFamily {
    let m2 = -2i32 as u32;
	AprilTagFamily {
		bits: vec![
            (1, m2),
            (2, m2),
            (3, m2),
            (1, 1),
            (2, 1),
            (6, 1),
            (6, 2),
            (6, 3),
            (3, 1),
            (3, 2),
            (3, 6),
            (2, 6),
            (1, 6),
            (3, 3),
            (2, 3),
            (m2, 3),
            (m2, 2),
            (m2, 1),
            (1, 3),
            (1, 2),
            (2, 2),
		],
		codes: vec![
			0x0000000000157863_u64,
            0x0000000000047e28_u64,
            0x00000000001383ed_u64,
            0x000000000000953c_u64,
            0x00000000000da68b_u64,
            0x00000000001cac50_u64,
            0x00000000000bb215_u64,
            0x000000000016ceee_u64,
            0x000000000005d4b3_u64,
            0x00000000001ff751_u64,
            0x00000000000efd16_u64,
            0x0000000000072b3e_u64,
            0x0000000000163103_u64,
            0x0000000000106e56_u64,
            0x00000000001996b9_u64,
            0x00000000000c0234_u64,
            0x00000000000624d2_u64,
            0x00000000001fa985_u64,
            0x00000000000344a5_u64,
            0x00000000000762fb_u64,
            0x000000000019e92b_u64,
            0x0000000000043755_u64,
            0x000000000001a4f4_u64,
            0x000000000010fad8_u64,
            0x0000000000001b52_u64,
            0x000000000017e59f_u64,
            0x00000000000e6f70_u64,
            0x00000000000ed47a_u64,
            0x00000000000c9931_u64,
            0x0000000000014df2_u64,
            0x00000000000a06f1_u64,
            0x00000000000e5041_u64,
            0x000000000012ec03_u64,
            0x000000000016724e_u64,
            0x00000000000af1a5_u64,
            0x000000000008a8ac_u64,
            0x0000000000015b39_u64,
            0x00000000001ec1e3_u64,
		],
		width_at_border: 5,
		total_width: 9,
		reversed_border: true,
		min_hamming: 7,
		name: Cow::Borrowed("tagCircle21h7"),
	}
}

#[cfg(test)]
mod test {
    #[cfg(feature="compare_reference")]
    #[test]
    fn compare_circle_21h7() {
        let tag = super::tagCircle21h7_create();

        unsafe {
            let tag_sys = apriltag_sys::tagCircle21h7_create();
            let ts = tag_sys.as_ref().unwrap();
            tag.assert_similar(ts);
            apriltag_sys::tagCircle21h7_destroy(tag_sys);
        };
    }
}