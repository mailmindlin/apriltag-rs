/// Compute the hamming distance between two longs.
pub const fn hamming_distance(a: u64, b: u64) -> u32 {
    let dif = a ^ b;
    dif.count_ones()
}