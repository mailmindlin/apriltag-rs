
pub(crate) fn gcd(mut a: usize, mut b: usize) -> usize {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    // find common factors of 2
    let shift = (a | b).trailing_zeros();

    // The algorithm needs positive numbers, but the minimum value
    // can't be represented as a positive one.
    // It's also a power of two, so the gcd can be
    // calculated by bitshifting in that case

    // divide n and m by 2 until odd
    a >>= a.trailing_zeros();
    b >>= b.trailing_zeros();

    while a != b {
        if a > b {
            a -= b;
            a >>= a.trailing_zeros();
        } else {
            b -= a;
            b >>= b.trailing_zeros();
        }
    }
    a << shift
}

pub(crate) fn lcm(a: usize, b: usize) -> usize {
    if a == 0 && b == 0 {
        return 0;
    }
    if a == b || a % b == 0 {
        return a;
    }
    if b % a == 0 {
        return b;
    }
    let gcd = gcd(a, b);
    a * (b / gcd)
}