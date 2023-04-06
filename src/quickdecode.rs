use std::{alloc::AllocError, mem::MaybeUninit};

use crate::families::AprilTagFamily;


#[derive(Copy, Clone)]
struct QuickDecodeEntry {
	/// Queried code
	rcode: u64,
	/// Tag ID
	id: u16,
	/// How many errors were corrected?
	hamming: u8,
}

impl QuickDecodeEntry {
	#[inline]
	const fn empty() -> Self {
		Self {
			rcode: u64::MAX,
			id: 0,
			hamming: 0,
		}
	}

	#[inline]
	fn is_empty(&self) -> bool {
		self.rcode == u64::MAX
	}
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QuickDecodeResult {
	/// Queried code
	pub rcode: u64,
	/// Tag ID
	pub id: u16,
	/// How many errors were corrected?
	pub hamming: u8,
	/// Number of rotations [0, 3]
	pub rotation: u8,
}

pub(crate) struct QuickDecode {
	entries: Box<[QuickDecodeEntry]>,
}

#[derive(Debug)]
pub enum AddFamilyError {
	TooManyCodes(usize),
	QuickDecodeAllocation(AllocError),
	BigHamming(usize),
}

/// if the bits in w were arranged in a d*d grid and that grid was
/// rotated, what would the new bits in w be?
/// The bits are organized like this (for d = 3):
///
/// ```text
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
    let w = ((w >> l) << (p/4 + l)) | (w >> (3 * p/ 4 + l) << l) | (w & l);
    w & (1u64 << d) - 1
}

struct QDBucketIter {
	current: usize,
	capacity: usize,
}

impl Iterator for QDBucketIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
		let res = self.current;
        self.current = (self.current + 1) % self.capacity;
		//TODO: detect complete cycles (currently they will just spin forever)
		Some(res)
    }
}

impl QuickDecode {
	fn with_capacity(capacity: usize) -> Result<Self, AllocError> {
		let entries = {
			let mut entries = Box::try_new_zeroed_slice(capacity)?;
					// println!("Failed to allocate hamming decode table for family {}: {:?}", family.name, e);
			
			entries.fill(MaybeUninit::new(QuickDecodeEntry::empty()));

			unsafe { entries.assume_init() }
		};

		Ok(Self {
			entries,
		})
	}

	pub fn init(family: &AprilTagFamily, maxhamming: usize) -> Result<Self, AddFamilyError> {
		if family.codes.len() >= u16::MAX as usize {
			return Err(AddFamilyError::TooManyCodes(family.codes.len()));
		}
	
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

		let mut qd = Self::with_capacity(capacity)
			.map_err(|e| AddFamilyError::QuickDecodeAllocation(e))?;

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
					// println!("Error: maxhamming beyond 3 not supported");
					return Err(AddFamilyError::BigHamming(maxhamming));
				},
			}
		}
	
	//    printf("capacity %d, size: %.0f kB\n",
	//           capacity, qd->nentries * sizeof(struct quick_decode_entry) / 1024.0);
	
		if false {
			let mut longest_run = 0;
			let mut run = 0;
			let mut run_sum = 0;
			let mut run_count = 0;
	
			// This accounting code doesn't check the last possible run that
			// occurs at the wrap-around. That's pretty insignificant.
			for entry in qd.entries.iter() {
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
	
			println!("quick decode: longest run: {}, average run {:.3}", longest_run, (run_sum as f64) / (run_count as f64));
		}

		Ok(qd)
	}

	fn bucket_iter(&self, initial_value: u64) -> QDBucketIter {
		let capacity = self.entries.len();
		let initial = initial_value as usize % capacity;
		QDBucketIter {
			current: initial,
			capacity
		}
	}

	fn add(&mut self, code: u64, id: usize, hamming: u8) {
		for bucket in self.bucket_iter(code) {
			let entry = &mut self.entries[bucket];
			if entry.is_empty() {
				let id: u16 = id.try_into().unwrap(); // We already checked for this overflow
				*entry = QuickDecodeEntry {
					rcode: code,
					id,
					hamming,
				};
				return;
			}
		}
		panic!("No bucket for code");
	}

	pub fn decode_codeword(&self, tf: &AprilTagFamily, mut rcode: u64) -> Option<QuickDecodeResult> {
		let dim = tf.bits.len().try_into().expect("AprilTag family has too many bits");
		
		for ridx in 0..4 {
			let mut bucket = (rcode as usize) % self.entries.len();
			for bucket in self.bucket_iter(rcode) {
				let entry = &self.entries[bucket];
				if entry.is_empty() {
					break;
				}

				if entry.rcode == rcode {
					return Some(QuickDecodeResult {
						rcode: entry.rcode,
						id: entry.id,
						hamming: entry.hamming,
						rotation: ridx
					});
				}
			}
	
			rcode = rotate90(rcode, dim);
		}

		None
	}
}