use std::num::NonZeroU32;

pub use self::{layout::{ImageLayout, ImageLayoutError}, hamming::hamming_distance};

use super::util::rotations;

mod layout;
mod hamming;

/// Generic class for all tag encoding families
pub struct TagFamily {
    ///
    ///  What is the minimum hamming distance between any two codes
    /// (accounting for rotational ambiguity)? The code can recover
    /// (minHammingDistance-1)/2 bit errors.
    pub min_hamming_distance: NonZeroU32,
    /// The error recovery value determines our position on the ROC
    /// curve. We will report codes that are within errorRecoveryBits
    /// of a valid code. Small values mean greater rejection of bogus
    /// tags (but false negatives). Large values mean aggressive
    /// reporting of bad tags (but with a corresponding increase in
    /// false positives).
    //public int errorRecoveryBits = 1;
    
    /// The array of the codes. The id for a code is its index.
    pub codes: Vec<u64>,
    pub layout: ImageLayout,
}

fn upgrade_code(old_code: u64, bits: &[(u32, u32)], size: u32) -> u64 {
    let mut code = 0;

    for (x, y) in bits.iter() {
        code <<= 1;
        let mask = 1u64 << (size - x + (size - y) * size);
        if old_code & mask != 0 {
            code |= 1;
        }
    }
    code
}
fn upgrade_codes(old_codes: &[u64], size: u32) -> Vec<u64> {
    let classic = ImageLayout::new_classic(size + 4).unwrap();
    let bit_locations = classic.bit_locations();
    old_codes.iter()
        .copied()
        .map(|old_code| upgrade_code(old_code, &bit_locations, size))
        .collect()
}


impl TagFamily {
    /// Constructor for tags generated with previous AprilTag versions.
    pub fn new(area: NonZeroU32, min_hamming_distance: NonZeroU32, codes: Vec<u64>) -> Result<Self, ImageLayoutError> {
        let length = NonZeroU32::new(f64::sqrt(area.get() as _) as _)
            .ok_or(ImageLayoutError::NoBorder)?;
        let layout = ImageLayout::new_classic(length.get() + 4)?;
        let codes = upgrade_codes(&codes, length.get());
        Ok(Self {
            min_hamming_distance,
            codes,
            layout,
        })
    }

    pub fn get_codes(&self) -> &[u64] {
        &self.codes
    }

    /// Given an observed tag with code 'rcode', try to recover the
    /// id. The corresponding fields of TagDetection will be filled
    /// in.
    pub fn decode(&self, rcode: u64) -> Option<(u32, usize, usize)>{
        let rcodes = rotations(rcode, self.layout.num_bits() as _);
        
        let mut best: Option<(u32, usize, usize)> = None;
        for (id, code) in self.codes.iter().enumerate() {
            for (rotation, rcode) in rcodes.iter().enumerate() {
                let hamming = hamming_distance(*rcode, *code);

                let update = match best {
                    None => true,
                    Some((best_hamming, ..)) => hamming < best_hamming,
                };
                if update {
                    best = Some((hamming, id, rotation));
                }
            }
        }
        best
        //     for (int rot = 0; rot < rcodes.length; rot++) {
        //         int thishamming = hammingDistance(rcodes[rot], codes[id]);
        //         if (thishamming < besthamming) {
        //             besthamming = thishamming;
        //             bestrotation = rot;
        //             bestid = id;
        //             bestcode = codes[id];
        //         }
        //     }
        // }

        // det.id = bestid;
        // det.hammingDistance = besthamming;
        // det.rotation = bestrotation;
        // det.good = (det.hammingDistance <= errorRecoveryBits);
        // det.obsCode = rcode;
        // det.code = bestcode;
    }

    
    fn print_hamming_distances(&self) {
        let num_bits = self.layout.num_bits();

        let mut hammings = vec![0; num_bits * num_bits + 1];
        for (idx, code1) in self.codes[..self.codes.len() - 1].iter().enumerate() {
            let rcodes = rotations(*code1, num_bits as _);

            for code2 in self.codes.iter().skip(idx + 1) {
                let d = rcodes.iter()
                    .map(|rcode1| hamming_distance(*rcode1, *code2))
                    .min()
                    .unwrap();
                hammings[d as usize] += 1;
            }
        }

        for (idx, hamming) in hammings.iter().enumerate() {
            println!("{idx:10} {hamming:10}");
        }
    }
}