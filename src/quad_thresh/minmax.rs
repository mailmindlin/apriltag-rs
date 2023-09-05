use crate::{util::{image::Pixel, mem::{SafeZero, calloc}, ImageBuffer, ImageY8}, Image};


trait MinMax {
    fn new_accumulator() -> Self;
    fn min(&self) -> u8;
    fn max(&self) -> u8;

    fn update_one(&mut self, value: u8);
    fn update(&mut self, value: &Self);
}

impl MinMax for [u8; 2] {
    fn new_accumulator() -> Self {
        [u8::MAX, u8::MIN]
    }
    fn min(&self) -> u8 {
        self[0]
    }

    fn max(&self) -> u8 {
        self[1]
    }

    #[inline(always)]
    fn update_one(&mut self, value: u8) {
        if value < self[0] {
            self[0] = value;
        }
        if value > self[1] {
            self[1] = value;
        }
    }

    #[inline(always)]
    fn update(&mut self, value: &Self) {
        if value[0] < self[0] {
            self[0] = value[0];
        }
        if value[1] > self[1] {
            self[1] = value[1];
        }
    }
}

type MinMaxPixel = [u8; 2];

fn tile_minmax<const KW: usize>(im: &ImageY8) -> ImageBuffer<MinMaxPixel> {
    // the last (possibly partial) tiles along each row and column will
    // just use the min/max value from the last full tile.
    let tw = im.width().div_floor(KW);
    let th = im.height().div_floor(KW);

    let mut result = ImageBuffer::<MinMaxPixel>::zeroed_packed(tw, th);

    for ((tx, ty), dst) in result.enumerate_pixels_mut() {
        let base_y = ty * KW;
        let base_x = tx * KW;

        let mut acc = MinMaxPixel::new_accumulator();

        for dy in 0..KW {
            for dx in 0..KW {
                let v = im[(base_x + dx, base_y + dy)];
                acc.update_one(v);
            }
        }
        *dst = acc;
    }
    result
}

/// blur min/max values in a 3x3 grid
fn blur<M: MinMax + Pixel<Value = M> + SafeZero>(mut im: Image<M>) -> ImageBuffer<M> {
    // Check that we need to blur (in practice, this branch is kinda useless)
    if im.width() < 3 || im.height() < 3 {
        return im;
    }

    let mut buffer_U = calloc::<M>(im.width()); // Buffer of min/max values for y-1 over (x-1..x+1)
    let mut buffer_C = calloc::<M>(im.width()); // Buffer of min/max values for y   over (x-1..x+1)
    let mut buffer_D = calloc::<M>(im.width()); // Buffer of min/max values for y+1 over (x-1..x+1)
    
    // Compute top row (and populate buffer_C/buffer_D)
    {
        let mut v_CC = M::new_accumulator(); // Tracks value (x + 0, y + 0) Center Center
        let mut v_CD = M::new_accumulator(); // Tracks value (x + 0, y + 1) Center Down

        let mut v_RC = im[(0, 0)]; // Tracks value (x + 1, y + 0) Right Cener
        let mut v_RD = im[(0, 1)]; // Tracks value (x + 1, y + 1) Right Down
        
        let row0 = im.row(0);
        let row1 = im.row(1);
        for x in 0..(im.width() - 1) {
            // Shift Left <- Center
            let mut v_LC = v_CC; // Value (x - 1, y + 0) Left Center
            let mut v_LD = v_CD; // Value (x - 1, y + 1) Left Down
            // Shift Center <- Right
            v_CC = v_RC;
            v_CD = v_RD;

            // Load right
            v_RC = row0[x+1];
            v_RD = row1[x+1];

            // Compute value for buffer_C
            v_LC.update(&v_CC);
            v_LC.update(&v_RC);
            buffer_C[x] = v_LC;

            // Compute value for buffer_D
            v_LD.update(&v_CD);
            v_LD.update(&v_RD);
            buffer_D[x] = v_LD;

            // Compute value for output
            v_LC.update(&v_LD);
            im[(x, 0)] = v_LC;
        }
        // Compute top-right
        let last_x = im.width() - 1;
        let mut acc_C = v_CC;
        let mut acc_D = v_CD;

        acc_C.update(&row0[last_x]);
        buffer_C[last_x] = acc_C;

        acc_D.update(&row1[last_x]);
        buffer_D[last_x] = acc_D;

        acc_C.update(&acc_D);
        im[(last_x, 0)] = acc_C;
    }

    // Middle (we have up and down)
    for y in 1..(im.height() - 1) {
        // Rotate buffers up one
        (buffer_D, buffer_C, buffer_U) = (buffer_C, buffer_U, buffer_D);
        
        let mut v_CD = im[(y, 0)]; // Tracks value (x + 0, y + 1) Center Down
        let mut v_RD = im[(y, 1)]; // Tracks value (x + 1, y + 1) Right Down

        for x in 0..im.width() {
            let v_LD = v_CD; // Shift Left <- Center
            v_CD = v_RD;        // Shift Center <- Right
            // Load Center Right (or default if right edge)
            v_RD = if x + 1 < im.width() { im[(x + 1, y + 1)] } else { M::new_accumulator() };

            // Load up/left values
            let v_U = buffer_U[x];
            let v_C = buffer_C[x];

            // Compute down
            let v_D = {
                let mut v_D = v_LD;
                v_D.update(&v_CD);
                v_D.update(&v_RD);
                v_D
            };
            buffer_D[x] = v_D;

            let mut acc = v_U;
            acc.update(&v_C);
            acc.update(&v_D);

            im[(x, y)] = acc;
        }
    }

    let buffer_U = buffer_C;
    let buffer_C = buffer_D;

    // Bottom row
    im


    // im.map_indexed(|im: &ImageBuffer<M, Box<[u8]>>, x, y| {
    //     let mut acc = M::new_accumulator();
    //     for v in im.window(x, y, 1, 1).pixels() {
    //         acc.update(v);
    //     }
    //     acc
    // })
}