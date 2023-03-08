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
			0x000000000155cbf1u64,
			0x0000000001e4d1b6u64,
			0x00000000017b0b68u64,
			0x0000000001eac9cdu64,
			0x00000000012e14ceu64,
			0x00000000003548bbu64,
			0x00000000007757e6u64,
			0x0000000001065dabu64,
			0x0000000001baa2e7u64,
			0x0000000000dea688u64,
			0x000000000081d927u64,
			0x000000000051b241u64,
			0x0000000000dbc8aeu64,
			0x0000000001e50e19u64,
			0x00000000015819d2u64,
			0x00000000016d8282u64,
			0x000000000163e035u64,
			0x00000000009d9b81u64,
			0x000000000173eec4u64,
			0x0000000000ae3a09u64,
			0x00000000005f7c51u64,
			0x0000000001a137fcu64,
			0x0000000000dc9562u64,
			0x0000000001802e45u64,
			0x0000000001c3542cu64,
			0x0000000000870fa4u64,
			0x0000000000914709u64,
			0x00000000016684f0u64,
			0x0000000000c8f2a5u64,
			0x0000000000833ebbu64,
			0x000000000059717fu64,
			0x00000000013cd050u64,
			0x0000000000fa0ad1u64,
			0x0000000001b763b0u64,
			0x0000000000b991ceu64,
		],
		width_at_border: 7,
		total_width: 9,
		reversed_border: false,
		min_hamming: 9,
		name: Cow::Borrowed("tag25h9"),
	}
}
