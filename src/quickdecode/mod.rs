mod table;
use std::{alloc::AllocError, sync::Arc};

use datasize::DataSize;

use crate::families::{AprilTagFamily, Rotation, rotate90};

use self::table::LookupTable;

#[derive(Default, Copy, Clone, DataSize, Debug)]
struct QuickDecodeValue {
    /// Tag ID
	id: u16,
	/// How many errors were corrected?
	hamming: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuickDecodeResult {
	/// Tag ID
	pub id: u16,
	/// How many errors were corrected?
	pub hamming: u8,
	/// Number of rotations [0, 3]
	pub rotation: Rotation,
}

/// Helper for quickly decoding an AprilTag code
#[derive(Clone)]
pub(crate) struct QuickDecode {
	pub family: Arc<AprilTagFamily>,
	table: LookupTable<QuickDecodeValue>,
	pub bits_corrected: usize,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum AddFamilyError {
	/// Too many codes in an AprilTag family
	#[error("Too many codes in AprilTag family to create QuickDecode (actual: {0}, max: {})", QuickDecode::NUM_CODES_MAX)]
	TooManyCodes(usize),
	/// Error allocating QD table
	#[error("Error allocating QuickDecode table")]
	QuickDecodeAllocation,
	/// Hamming value was too big
	#[error("Hamming too big for QuickDecode: (actual: {0}, max: {})", QuickDecode::HAMMING_MAX)]
	BigHamming(usize),
}

impl From<AllocError> for AddFamilyError {
    fn from(_: AllocError) -> Self {
        Self::QuickDecodeAllocation
    }
}

impl QuickDecode {
    /// Maximum number of codes allowed in a family
    pub const NUM_CODES_MAX: usize = u16::MAX as usize;
    /// Maximum hamming
    pub const HAMMING_MAX: usize = 3;

	/// Create new QuickDecode for some AprilTag family
	pub(crate) fn new(family: Arc<AprilTagFamily>, bits_corrected: usize) -> Result<Self, AddFamilyError> {
        // Parameter validation
		if family.codes.len() >= Self::NUM_CODES_MAX {
			return Err(AddFamilyError::TooManyCodes(family.codes.len()));
		}
        if bits_corrected > Self::HAMMING_MAX {
            return Err(AddFamilyError::BigHamming(bits_corrected));
        }
	
		let nbits = family.bits.len();

		let capacity = {
			let ncodes = family.codes.len();
			let mut capacity = ncodes;
			if bits_corrected >= 1 {
				capacity += ncodes * nbits;
			}
			if bits_corrected >= 2 {
				capacity += ncodes * nbits * (nbits - 1);
			}
			if bits_corrected >= 3 {
				capacity += ncodes * nbits * (nbits - 1) * (nbits - 2);
			}
			capacity * 3
		};

		// Create QuickDecode with capacity
		let mut qd = {
			let table = LookupTable::with_capacity(capacity)?;
	
			Self {
				family: family.clone(),
				table,
				bits_corrected,
			}
		};


		for (i, code) in family.codes.iter().enumerate() {
			// add exact code (hamming = 0)
			qd.add(*code, i, 0);
			
			match bits_corrected {
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
					return Err(AddFamilyError::BigHamming(bits_corrected));
				},
			}
		}
	
        
        #[cfg(feature="extra_debug")]
		{
			use datasize::data_size;
            println!("quick decode: capacity {}, size {:.0} kB", capacity, data_size(&qd.table) as f64 / 1024.0);

            let (avg_run, longest_run) = qd.table.stats();
            println!("quick decode: longest run: {}, average run {:.3}", longest_run, avg_run);
		}

		Ok(qd)
	}

	fn add(&mut self, code: u64, id: usize, hamming: u8) {
        let id = id.try_into().unwrap(); // We already checked for this overflow
        self.table.add(code, QuickDecodeValue { id, hamming })
            .ok() // Drop value on error
            .expect("No bucket for code"); // Shouldn't happen
	}

	pub fn decode_codeword(&self, mut rcode: u64) -> Option<QuickDecodeResult> {
		let dim = self.family.bits.len().try_into()
            .expect("AprilTag family has too many bits");
		
		for rotation in Rotation::values() {
            if let Some(entry) = self.table.get(rcode) {
                return Some(QuickDecodeResult {
                    id: entry.id,
                    hamming: entry.hamming,
                    rotation,
                });
            }
            rcode = rotate90(rcode, dim);
		}

		None
	}
}

#[cfg(test)]
mod test {
    use crate::{AprilTagFamily, families::{Rotation, rotations}};

    use super::{QuickDecode, QuickDecodeResult};

	#[test]
	fn build_qd() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let _ = QuickDecode::new(family, 1).unwrap();
	}

	#[test]
	fn lookup_qd() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family, 0).unwrap();
		qd.decode_codeword(57948543051).unwrap();
	}

	#[test]
	fn lookup_qd_rotated() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family, 0).unwrap();
		qd.decode_codeword(51559569327).unwrap();
	}

	/// Decoding the canonical codeword for each tag should return the correct id,
	/// hamming=0, and rotation=Identity.
	#[test]
	fn decode_returns_correct_id_and_fields() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family.clone(), 0).unwrap();
		for (id, &code) in family.codes.iter().enumerate().take(10) {
			let result = qd.decode_codeword(code).unwrap_or_else(|| panic!("code[{id}] not found"));
			assert_eq!(result, QuickDecodeResult { id: id as u16, hamming: 0, rotation: Rotation::Identity },
				"wrong result for code[{id}]");
		}
	}

	/// All four rotations of a codeword must decode to the same tag id, with the
	/// rotation field indicating how many CCW turns bring the observed code back
	/// to the canonical one stored in the table.
	///
	/// The decode loop tries rotations in order [Identity, Deg90, Deg180, Deg270],
	/// rotating the input each iteration, so:
	///   rots[0] (canonical)   → found on iter 0 → Identity
	///   rots[1] (rotate once) → found on iter 3 → Deg270
	///   rots[2] (rotate twice)→ found on iter 2 → Deg180
	///   rots[3] (rotate 3x)   → found on iter 1 → Deg90
	#[test]
	fn decode_all_rotations() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family.clone(), 0).unwrap();
		let nbits = family.bits.len() as u64;

		for (id, &code) in family.codes.iter().enumerate().take(5) {
			let [r0, r1, r2, r3] = rotations(code, nbits);
			let cases = [
				(r0, Rotation::Identity),
				(r1, Rotation::Deg270),
				(r2, Rotation::Deg180),
				(r3, Rotation::Deg90),
			];
			for (rotated, expected_rotation) in cases {
				let result = qd.decode_codeword(rotated)
					.unwrap_or_else(|| panic!("code[{id}] rotation {expected_rotation:?} not found"));
				assert_eq!(result.id, id as u16);
				assert_eq!(result.hamming, 0);
				assert_eq!(result.rotation, expected_rotation,
					"wrong rotation for code[{id}] rotated to {rotated}");
			}
		}
	}

	#[test]
	fn decode_hamming1() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family.clone(), 1).unwrap();
		// min_hamming=11, so a 1-bit flip unambiguously maps back to codes[5]
		let code = family.codes[5];
		let result = qd.decode_codeword(code ^ 1).unwrap();
		assert_eq!(result, QuickDecodeResult { id: 5, hamming: 1, rotation: Rotation::Identity });
	}

	#[test]
	fn decode_hamming2() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family.clone(), 2).unwrap();
		let code = family.codes[5];
		let result = qd.decode_codeword(code ^ 0b11).unwrap();
		assert_eq!(result, QuickDecodeResult { id: 5, hamming: 2, rotation: Rotation::Identity });
	}

	#[test]
	fn decode_hamming3() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family.clone(), 3).unwrap();
		let code = family.codes[5];
		let result = qd.decode_codeword(code ^ 0b111).unwrap();
		assert_eq!(result, QuickDecodeResult { id: 5, hamming: 3, rotation: Rotation::Identity });
	}

	/// A 1-bit-corrupted code must return None when bits_corrected=0.
	#[test]
	fn decode_hamming1_rejected_when_bits_corrected_zero() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family.clone(), 0).unwrap();
		let code = family.codes[0];
		assert!(qd.decode_codeword(code ^ 1).is_none());
	}

	/// A code that differs by many bits from every valid codeword must return None.
	#[test]
	fn decode_miss_returns_none() {
		let family = AprilTagFamily::for_name("tag36h11").unwrap();
		let qd = QuickDecode::new(family.clone(), 3).unwrap();
		// Flipping 5 bits exceeds bits_corrected=3, and min_hamming=11 guarantees
		// the result is also at least 11-5=6 bits away from every other code.
		let corrupted = family.codes[0] ^ 0b1_1111;
		assert!(qd.decode_codeword(corrupted).is_none());
	}

	/// Verify correct behaviour on a family with fewer bits (tag16h5: 16 bits, min_hamming=5).
	#[test]
	fn decode_tag16h5() {
		let family = AprilTagFamily::for_name("tag16h5").unwrap();
		let nbits = family.bits.len() as u64;
		let qd = QuickDecode::new(family.clone(), 1).unwrap();
		let code = family.codes[0];

		// Exact match
		let result = qd.decode_codeword(code).unwrap();
		assert_eq!(result, QuickDecodeResult { id: 0, hamming: 0, rotation: Rotation::Identity });

		// Rotated once: decode loop finds it on iter 3 → Deg270
		let [_, r1, _, _] = rotations(code, nbits);
		let result = qd.decode_codeword(r1).unwrap();
		assert_eq!(result.id, 0);
		assert_eq!(result.hamming, 0);
		assert_eq!(result.rotation, Rotation::Deg270);

		// Hamming 1 (min_hamming=5, so unambiguous)
		let result = qd.decode_codeword(code ^ 1).unwrap();
		assert_eq!(result, QuickDecodeResult { id: 0, hamming: 1, rotation: Rotation::Identity });
	}
}