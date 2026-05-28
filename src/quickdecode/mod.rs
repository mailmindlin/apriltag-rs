use std::{alloc::AllocError, array, iter, sync::Arc};

use datasize::DataSize;

use crate::{families::{AprilTagFamily, Rotation, rotate90}, util::mem::try_calloc};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuickDecodeResult {
	/// Tag ID
	pub id: u16,
	/// How many errors were corrected?
	pub hamming: u8,
	/// Number of rotations [0, 3]
	pub rotation: Rotation,
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

// NUM_CHUNKS must be strictly greater than HAMMING_MAX. With 4 chunks and at
// most 3 correctable bit errors, the pigeonhole principle guarantees at least
// one chunk is error-free — so the correct canonical code always appears in
// that chunk's candidate list during lookup.
const NUM_CHUNKS: usize = 4;

#[derive(Clone, Copy, DataSize)]
enum MaxHamming {
	_0 = 0,
	_1,
	_2,
	_3,
}

impl TryFrom<usize> for MaxHamming {
	type Error = AddFamilyError;

	fn try_from(value: usize) -> Result<Self, Self::Error> {
		Ok(match value {
			0 => Self::_0,
			1 => Self::_1,
			2 => Self::_2,
			3 => Self::_3,
			_ => return Err(AddFamilyError::BigHamming(value))
		})
	}
}

/// Decodes an observed AprilTag codeword to a tag ID, correcting up to
/// `maxhamming` bit errors.
///
/// # Algorithm
///
/// A naïve search would compare the observed code against every canonical code
/// in the family — O(N) full Hamming distance computations. QuickDecode reduces
/// that to O(1) average case using a chunk-indexed lookup table.
///
/// The codeword is split into `NUM_CHUNKS` equal bit-slices ("chunks"). For
/// each chunk we maintain a bucket index that maps a chunk value to the list of
/// canonical codes that have that exact value in that chunk position.
///
/// **Why this finds the answer:** with at most `maxhamming` bit errors spread
/// across `NUM_CHUNKS > maxhamming` chunks, the pigeonhole principle guarantees
/// at least one chunk is error-free. The correct canonical code will therefore
/// appear in that chunk's bucket, so iterating all `NUM_CHUNKS` buckets and
/// computing the full Hamming distance for each candidate is both necessary and
/// sufficient.
///
/// The full Hamming check is required because a chunk match is only a
/// necessary condition — a candidate whose chunk matches but whose other bits
/// differ too much is rejected here.
///
/// **Rotation:** AprilTag codes are orientation-independent. `decode_codeword`
/// tries all four 90° rotations of the input; the `rotation` field of the
/// result records how many times the input was rotated before a match was found,
/// which tells the caller how to orient the detected tag.
#[derive(Clone, DataSize)]
pub(crate) struct QuickDecode {
	pub family: Arc<AprilTagFamily>,
    /// Bitmask applied after shifting to extract one chunk: `(1 << chunk_size) - 1`.
    chunk_mask: u32,
    /// Bit shift for each chunk: `shifts[i] = i * chunk_size`.
    shifts: [usize; NUM_CHUNKS],

    /// Bucket boundaries, one table per chunk.
    ///
    /// For chunk `i` and chunk value `v`, the canonical codes whose chunk `i`
    /// equals `v` have their indices stored at
    /// `chunk_ids[i][chunk_offsets[i][v]..chunk_offsets[i][v+1]]`.
    ///
    /// Length is `(1 << chunk_size) + 1` (one extra entry for the sentinel).
    chunk_offsets: [Box<[u16]>; NUM_CHUNKS],

    /// Packed code-index arrays, one per chunk (indices into `family.codes`).
    ///
    /// Codes are grouped by their chunk value; `chunk_offsets` gives the range
    /// for each group. See [`chunk_offsets`] for the access pattern.
    chunk_ids: [Box<[u16]>; NUM_CHUNKS],

    maxhamming: MaxHamming,
}

impl QuickDecode {
    /// Maximum number of codes allowed in a family
    pub const NUM_CODES_MAX: usize = u16::MAX as usize;
    /// Maximum hamming
    pub const HAMMING_MAX: u8 = 3;

	/// Build a `QuickDecode` table for `family`, correcting up to `bits_corrected` errors.
	///
	/// The table is built with a two-pass counting sort (radix-sort style) for
	/// each chunk:
	///
	/// 1. **Frequency count** — tally how many codes fall into each chunk bucket.
	///    Counts are stored at `chunk_offsets[v+1]` so that the prefix-sum pass
	///    can write final offsets in-place without a separate shift.
	///
	/// 2. **Prefix sum** — convert raw counts into start positions. After this
	///    pass `chunk_offsets[v]` is the index in `chunk_ids` where bucket `v`
	///    begins and `chunk_offsets[v+1]` is where it ends (exclusive).
	///
	/// 3. **Populate** — iterate codes again, writing each code's index into
	///    `chunk_ids` at the position given by a per-bucket write cursor. The
	///    cursor starts at `chunk_offsets[v]` and advances with each write,
	///    filling the bucket contiguously.
	pub(crate) fn new(family: Arc<AprilTagFamily>, bits_corrected: usize) -> Result<Self, AddFamilyError> {
		if family.codes.len() >= Self::NUM_CODES_MAX {
			return Err(AddFamilyError::TooManyCodes(family.codes.len()));
		}
		let bits_corrected: MaxHamming = bits_corrected.try_into()?;

		let nbits = family.bits.len();

		let chunk_size = nbits.div_ceil(NUM_CHUNKS);
		let capacity = 1u32 << chunk_size;
		let chunk_mask = capacity - 1;

		let shifts: [_; NUM_CHUNKS] = array::from_fn(|i| i.wrapping_mul(chunk_size));

		// +1 for the sentinel entry used during lookup: offsets[v+1] - offsets[v] = bucket size.
		let mut chunk_offsets: [Box<[u16]>; NUM_CHUNKS] = array::try_from_fn(|_| try_calloc(capacity as usize + 1))?;
		let mut chunk_ids: [Box<[u16]>; NUM_CHUNKS] = array::try_from_fn(|_| try_calloc(family.codes.len()))?;

		// Pass 1: count how many codes land in each bucket.
		// Write to [v+1] so the prefix sum can run without a separate shift step.
		for &code in &family.codes {
			for (&shift, chunk_offset) in iter::zip(&shifts, &mut chunk_offsets) {
				let val = (code >> shift) as u32 & chunk_mask;
				chunk_offset[val as usize + 1] += 1;
			}
		}

		// Pass 2: prefix sum — turn counts into start offsets.
		// After this: chunk_offsets[v] = start of bucket v, chunk_offsets[v+1] = exclusive end.
		for chunk_offset in &mut chunk_offsets {
			for j in 0..capacity {
				chunk_offset[j as usize + 1] += chunk_offset[j as usize];
			}
		}

		// Pass 3: populate chunk_ids.
		// `cursors` is a mutable copy of chunk_offsets used as per-bucket write heads.
		let mut cursors = chunk_offsets.each_ref().try_map(|chunk_offset| -> Result<_, AllocError> {
			let mut cursor = try_calloc(capacity as usize + 1)?;
			cursor.copy_from_slice(&chunk_offset);
			Ok(cursor)
		})?;

		for (i, &code) in family.codes.iter().enumerate() {
			for j in 0..NUM_CHUNKS {
				let val = (code >> shifts[j]) as u32 & chunk_mask;
				let write_pos = cursors[j][val as usize];
				chunk_ids[j][write_pos as usize] = i as u16;
				cursors[j][val as usize] += 1;
			}
		}

		drop(cursors);

		let qd = Self { family, chunk_mask, shifts, chunk_offsets, chunk_ids, maxhamming: bits_corrected };

		#[cfg(feature = "extra_debug")]
		{
			use datasize::data_size;
			eprintln!("quick decode: capacity {}, size {:.1} kB", capacity, data_size(&qd) as f64 / 1024.0);

			let (run_min, run_mean, run_max) = qd.stats();
			println!("quick decode: shortest run: {}, average run {:.3}, longest run: {}", run_min, run_mean, run_max);
		}

		Ok(qd)
	}

	/// Decode an observed codeword, correcting up to `maxhamming` bit errors.
	///
	/// Returns `None` if no canonical code is within the Hamming distance threshold.
	///
	/// # Rotation handling
	///
	/// The outer loop iterates `[Identity, Deg90, Deg180, Deg270]`. On each
	/// iteration `rcode` is the observed code rotated by the current amount.
	/// When a match is found, the `rotation` field records the current
	/// `Rotation` value — i.e. how many 90° CCW turns were applied to the
	/// observed code before it matched the canonical (unrotated) form.
	///
	/// Consequently, a physically-rotated tag produces a result whose `rotation`
	/// is the *inverse* rotation: apply that many CCW turns to undo the camera
	/// orientation and recover the upright view.
	///
	/// # Uniqueness
	///
	/// The tag family's minimum inter-code Hamming distance (e.g. 11 for
	/// tag36h11) ensures that within `maxhamming ≤ 3` errors there is at most
	/// one matching code, so returning the first match found is correct.
	pub fn decode_codeword(&self, mut rcode: u64) -> Option<QuickDecodeResult> {
		let dim = self.family.bits.len().try_into()
            .expect("AprilTag family has too many bits");

		for rotation in Rotation::values() {
			// Check all NUM_CHUNKS buckets. We don't know which chunk is
			// error-free, so we must check all of them. The correct code is
			// guaranteed to appear in at least one bucket (pigeonhole), and the
			// full Hamming check below rejects false positives.
			for i in 0..NUM_CHUNKS {
				let val = (rcode >> self.shifts[i]) as usize & self.chunk_mask as usize;
				let start = self.chunk_offsets[i][val];
				let end = self.chunk_offsets[i][val + 1];

				for &id in &self.chunk_ids[i][start as usize..end as usize] {
					let correct_code = self.family.codes[id as usize];
					let hamming = (correct_code ^ rcode).count_ones() as u8;
					if hamming <= self.maxhamming as _ {
						return Some(QuickDecodeResult {
							id,
							hamming,
							rotation,
						});
					}
				}
			}
			rcode = rotate90(rcode, dim);
		}

		None
	}

	#[cfg(feature = "extra_debug")]
	fn stats(&self) -> (u16, f64, u16) {
		let mut run_min = u16::MAX;
		let mut run_max = 0;
		let mut run_sum = 0usize;
		let mut run_count = 0;
		for chunk_offset in &self.chunk_offsets {
			for &[start, end] in chunk_offset.array_windows() {
				let dist = end - start;
				if dist < run_min {
					run_min = dist;
				}
				if dist > run_max {
					run_max = dist;
				}
				run_sum += dist as usize;
				run_count += 1;
			}
		}
		(run_min, run_sum as f64 / run_count as f64, run_max)
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
		let nbits = family.bits.len() as u8;

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
		let nbits = family.bits.len() as u8;
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

	#[test]
	fn decode_all_tag25h9() {
		let family = AprilTagFamily::for_name("tag25h9").unwrap();
		let nbits = family.bits.len() as u64;
		let qd = QuickDecode::new(family.clone(), 3).unwrap();
		for (id, code) in family.codes.iter().copied().enumerate() {
			let id = id as u16;
			// Exact match
			assert_eq!(qd.decode_codeword(code), Some(QuickDecodeResult { id, hamming: 0, rotation: Rotation::Identity }));

			for i in 0..nbits {
				let code_1 = code ^ (1 << i);
				assert_eq!(qd.decode_codeword(code_1), Some(QuickDecodeResult { id, hamming: 1, rotation: Rotation::Identity }));
				for j in (i+1)..nbits {
					let code_2 = code_1 ^ (1 << j);
					assert_eq!(qd.decode_codeword(code_2), Some(QuickDecodeResult { id, hamming: 2, rotation: Rotation::Identity }));
					for k in (j+1)..nbits {
						let code_3 = code_2 ^ (1 << k);
						assert_eq!(qd.decode_codeword(code_3), Some(QuickDecodeResult { id, hamming: 3, rotation: Rotation::Identity }));
					}
				}
			}
		}
	}
}