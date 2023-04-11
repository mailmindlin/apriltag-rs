use crate::util::mem::calloc;


pub(super) fn sharpen(values: &mut [f64], sharpening: f64, size: usize) {
    let mut sharpened = calloc::<f64>(size * size);
    let kernel = [
        0., -1., 0.,
        -1., 4., -1.,
        0., -1., 0.
    ];

    for y in 0..size {
        for x in 0..size {
            let mut acc = 0.;

            for i in 0..3 {
                if y + i < 1 {
                    continue;
                }
                let cy = y + i - 1;
                if cy > size - 1 {
                    continue;
                }

                for j in 0..3 {
                    if x + j < 1 {
                        continue;
                    }
                    let cx = x + j - 1;
                    if cx > size - 1 {
                        continue;
                    }
                    acc += values[cy*size + cx]*kernel[i*3 + j];
                }
            }
            sharpened[y * size + x] = acc;
        }
    }


    for y in 0..size {
        for x in 0..size {
            values[y*size + x] = values[y*size + x] + sharpening * sharpened[y*size + x];
        }
    }
}