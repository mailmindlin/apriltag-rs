mod table;
use std::{alloc::AllocError, mem::MaybeUninit, sync::Arc};

use datasize::{data_size, DataSize};

use crate::families::AprilTagFamily;

use self::table::LookupTable;

#[derive(Default, Copy, Clone, DataSize)]
struct QuickDecodeValue {
    /// Tag ID
	id: u16,
	/// How many errors were corrected?
	hamming: u8,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QuickDecodeResult {
	/// Tag ID
	pub id: u16,
	/// How many errors were corrected?
	pub hamming: u8,
	/// Number of rotations [0, 3]
	pub rotation: u8,
}

/// Helper for quickly decoding an AprilTag code
pub(crate) struct QuickDecode {
	pub family: Arc<AprilTagFamily>,
	table: LookupTable<QuickDecodeValue>
}

#[derive(Debug)]
pub enum AddFamilyError {
	/// Too many codes in an AprilTag family
	TooManyCodes(usize),
	/// Error allocating QD table
	QuickDecodeAllocation(AllocError),
	/// Hamming value was too big
	BigHamming(usize),
}

impl std::fmt::Display for AddFamilyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddFamilyError::TooManyCodes(num_codes) =>
				write!(f, "Too many codes in AprilTag family to create QuickDecode (actual: {}, max: {})", num_codes, QuickDecode::NUM_CODES_MAX),
            AddFamilyError::QuickDecodeAllocation(e) =>
				write!(f, "Error allocating QuickDecode table: {}", e),
            AddFamilyError::BigHamming(hamming) =>
				write!(f, "Hamming too big for QuickDecode: (actual: {}, max: {})", hamming, QuickDecode::HAMMING_MAX),
        }
    }
}

/// Assuming we are drawing the image one quadrant at a time, what would the rotated image look like?
/// Special care is taken to handle the case where there is a middle pixel of the image.
/// 
/// if the bits in w were arranged in a d*d grid and that grid was
/// rotated, what would the new bits in w be?
/// The bits are organized like this (for d = 3):
///
/// ```text
///  8 7 6       2 5 8      0 1 2
///  5 4 3  ==>  1 4 7 ==>  3 4 5    (rotate90 applied twice)
///  2 1 0       0 3 6      6 7 8
/// ```
fn rotate90(w: u64, numBits: u64) -> u64 {
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

	/// Odd/even
    let (p, l) = if numBits % 4 == 1 {
		(numBits - 1, 1)
	} else {
		(numBits, 0)
	};

    let w = ((w >> l) << (p/4 + l)) | (w >> (3 * p/4 + l) << l) | (w & l);
    w & (1u64 << numBits) - 1
}

#[cfg(test)]
#[test]
fn test_rotate90() {
	let w_orig = 0b1010_1010_1010_1010;
	let w_rot  = 0b0000_1111_0000_1111;
	assert_eq!(rotate90(w_orig, 16), w_rot);
}

impl QuickDecode {
    /// Maximum number of codes allowed in a family
    pub const NUM_CODES_MAX: usize = u16::MAX as usize;
    /// Maximum hamming
    pub const HAMMING_MAX: usize = 3;

	/// Create new QuickDecode for some AprilTag family
	pub fn init(family: Arc<AprilTagFamily>, max_hamming: usize) -> Result<Self, AddFamilyError> {
        // Parameter validation
		if family.codes.len() >= Self::NUM_CODES_MAX {
			return Err(AddFamilyError::TooManyCodes(family.codes.len()));
		}
        if max_hamming >= Self::HAMMING_MAX {
            return Err(AddFamilyError::BigHamming(max_hamming));
        }
	
		let nbits = family.bits.len();

		let capacity = {
			let ncodes = family.codes.len();
			let mut capacity = ncodes;
			if max_hamming >= 1 {
				capacity += ncodes * nbits;
			}
			if max_hamming >= 2 {
				capacity += ncodes * nbits * (nbits - 1);
			}
			if max_hamming >= 3 {
				capacity += ncodes * nbits * (nbits - 1) * (nbits - 2);
			}
			capacity
		};

		// Create QuickDecode with capacity
		let mut qd = {
			let table = LookupTable::with_capacity(capacity)
				.map_err(|e| AddFamilyError::QuickDecodeAllocation(e))?;
	
			Self {
				family: family.clone(),
				table,
			}
		};


		for (i, code) in family.codes.iter().enumerate() {
			// add exact code (hamming = 0)
			qd.add(*code, i, 0);
			
			match max_hamming {
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
					return Err(AddFamilyError::BigHamming(max_hamming));
				},
			}
		}
	
        
        #[cfg(feature="extra_debug")]
		{
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
		
		for ridx in 0..4 {
            if let Some(entry) = self.table.get(rcode) {
                return Some(QuickDecodeResult {
                    id: entry.id,
                    hamming: entry.hamming,
                    rotation: ridx,
                });
            }
            rcode = rotate90(rcode, dim);
		}

		None
	}
}