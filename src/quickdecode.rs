use crate::families::AprilTagFamily;

#[derive(Clone, Copy)]
pub(crate) struct QuickDecodeEntry {
	/// the queried code
	pub rcode: u64,
	/// the tag ID (a small integer)
	pub id: u16,
	/// how many errors corrected?
	pub hamming: u8,
	/// number of rotations [0, 3]
	pub rotation: u8,
}

pub(crate) struct QuickDecode {
	entries: Vec<QuickDecodeEntry>,
}

/// if the bits in w were arranged in a d*d grid and that grid was
/// rotated, what would the new bits in w be?
/// The bits are organized like this (for d = 3):
///
/// ```
///  8 7 6       2 5 8      0 1 2
///  5 4 3  ==>  1 4 7 ==>  3 4 5    (rotate90 applied twice)
///  2 1 0       0 3 6      6 7 8
/// ```
fn rotate90(w: u64, d: u32) -> u64 {
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

	let mut p = d as u64;
    let mut l = 0u64;
    if d % 4 == 1 {
		p = d as u64 - 1;
		l = 1;
    }
    let mut w = ((w >> l) << (p/4 + l)) | (w >> (3 * p/ 4 + l) << l) | (w & l);
    w & (1u64 << d) - 1
}

impl QuickDecode {
	pub fn init(family: &AprilTagFamily, maxhamming: usize) -> Option<QuickDecode> {
		assert!(family.codes.len() < 65535);
	
		let nbits = family.bits.len();

		let capacity = {
			let ncodes = family.codes.len();
			let mut capacity = ncodes;
			if maxhamming >= 1 {
				capacity += ncodes * nbits;
			}
			if maxhamming >= 2 {
				capacity += ncodes * nbits * (nbits - 1);
			}
			if maxhamming >= 3 {
				capacity += ncodes * nbits * (nbits - 1) * (nbits - 2);
			}
			capacity
		};

		let mut qd = QuickDecode {
			entries: Vec::new(),
		};

		// Handle out-of-memory here
		match qd.entries.try_reserve_exact(capacity) {
			Ok(_) => {},
			Err(err) => {
				println!("Failed to allocate hamming decode table for family {}: {:?}", family.name, err);
				return None;
			}
		}

		for (i, code) in family.codes.iter().enumerate() {
			// add exact code (hamming = 0)
			qd.add(*code, i, 0);
			
			match maxhamming {
				0 => {},
				1 => {
					// add hamming 1
					for j in 0..nbits {
						let code_dist1 = code ^ (1u64 << j);
						qd.add(code_dist1, i, 1);
					}
				},
				2 => {
					// add hamming 2
					for j in 0..nbits {
						let code_dist1 = code ^ (1u64 << j);
						qd.add(code_dist1, i, 1);
						for k in 0..j {
							let code_dist2 = code_dist1 ^ (1u64 << k);
							qd.add(code_dist2, i, 2);
						}
					}
				},
				3 => {
					// add hamming 3
					for j in 0..nbits {
						let code_dist1 = code ^ (1u64 << j);
						qd.add(code_dist1, i, 1);
						for k in 0..j {
							let code_dist2 = code_dist1 ^ (1u64 << k);
							qd.add(code_dist2, i, 2);
							for m in 0..k {
								let code_dist3 = code_dist2 ^ (1u64 << m);
								qd.add(code_dist3, i, 3);
							}
						}
					}
				},
				_ => {
					println!("Error: maxhamming beyond 3 not supported");
					return None;
				},
			}
		}

		qd.entries.shrink_to_fit();
	
	//    printf("capacity %d, size: %.0f kB\n",
	//           capacity, qd->nentries * sizeof(struct quick_decode_entry) / 1024.0);
	
		if false {
			let mut longest_run = 0;
			let mut run = 0;
			let mut run_sum = 0;
			let mut run_count = 0;
	
			// This accounting code doesn't check the last possible run that
			// occurs at the wrap-around. That's pretty insignificant.
			for ref entry in qd.entries {
				if entry.rcode == u64::MAX {
					if run > 0 {
						run_sum += run;
						run_count += 1;
					}
					run = 0;
				} else {
					run += 1;
					longest_run = std::cmp::max(longest_run, run);
				}
			}
	
			println!("quick decode: longest run: {}, average run {:.3}\n", longest_run, (run_sum as f64) / (run_count as f64));
		}

		Some(qd)
	}

	pub fn add(&mut self, code: u64, id: usize, hamming: u8) {
		let bucket = (code as usize) % self.entries.len();

		while self.entries[bucket].rcode != u64::MAX {
			bucket = (bucket + 1) % self.entries.len();
		}
		let entry = &mut self.entries[bucket];
	
		entry.rcode = code;
		entry.id = id.try_into().unwrap();
		entry.hamming = hamming;
	}

	pub fn decode_codeword(&self, tf: &AprilTagFamily, rcode: u64) -> Option<QuickDecodeEntry> {
		for ridx in 0..4 {
			let mut bucket = (rcode as usize) % self.entries.len();
			loop {
				// I don't get this check
				if self.entries[bucket].rcode == u64::MAX {
					break;
				}

				if self.entries[bucket].rcode == rcode {
					let result = self.entries[bucket].clone();
					result.rotation = ridx;
					return Some(result);
				}

				bucket = (bucket + 1) % self.entries.len();
			}
	
			rcode = rotate90(rcode, tf.bits.len().try_into().unwrap());
		}

		None
	}
}