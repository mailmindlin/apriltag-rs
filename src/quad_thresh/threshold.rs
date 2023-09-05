use arrayvec::ArrayVec;
use std::fmt::Debug;
use crate::{util::{Image, mem::{calloc, SafeZero}, image::{ImageBuffer, ImageY8, Pixel}}, dbg::TimeProfile};

use super::AprilTagQuadThreshParams;

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
    #[inline(always)]
    fn min(&self) -> u8 {
        self[0]
    }
    #[inline(always)]
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
fn blur<M: MinMax + Pixel<Value = M> + SafeZero>(mut im: Image<M>) -> ImageBuffer<M> where [<M as Pixel>::Subpixel]: Debug, M: Debug {
    // Check that we need to blur (in practice, this branch is kinda useless)
    if im.width() < 3 || im.height() < 3 {
        return im;
    }

    let width = im.width();

    let mut buffer_U = calloc::<M>(im.width()); // Buffer of min/max values for y-1 over (x-1..x+1)
    let mut buffer_C = calloc::<M>(im.width()); // Buffer of min/max values for y   over (x-1..x+1)
    let mut buffer_D = calloc::<M>(im.width()); // Buffer of min/max values for y+1 over (x-1..x+1)
    
    // Compute top row (and populate buffer_C/buffer_D)
    {
        let mut v_CC = M::new_accumulator(); // Tracks value (x + 0, y + 0) Center Center
        let mut v_CD = M::new_accumulator(); // Tracks value (x + 0, y + 1) Center Down

        let mut v_RC = im[(0, 0)]; // Tracks value (x + 1, y + 0) Right Cener
        let mut v_RD = im[(0, 1)]; // Tracks value (x + 1, y + 1) Right Down
        
        let (mut row0, row1) = im.rows2_mut(0, 1);
        for x in 0..(width - 1) {
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
            row0[x] = v_LC;
        }
        // Compute top-right
        let last_x = im.width() - 1;
        let mut acc_C = v_CC;
        let mut acc_D = v_CD;

        acc_C.update(&im[(last_x, 0)]);
        buffer_C[last_x] = acc_C;

        acc_D.update(&im[(last_x, 1)]);
        buffer_D[last_x] = acc_D;

        acc_C.update(&acc_D);
        im[(last_x, 0)] = acc_C;
    }

    // Middle (we have up and down)
    for y in 1..(im.height() - 1) {
        // Rotate buffers up one
        (buffer_U, buffer_C, buffer_D) = (buffer_C, buffer_D, buffer_U);

        let mut v_CD = im[(y, 0)]; // Tracks value (x + 0, y + 1) Center Down
        let mut v_RD = im[(y, 1)]; // Tracks value (x + 1, y + 1) Right Down

        for x in 0..width {
            let v_LD = v_CD; // Shift Left <- Center
            v_CD = v_RD;        // Shift Center <- Right
            // Load Center Right (or default if right edge)
            v_RD = if x + 1 < width { im[(x + 1, y + 1)] } else { M::new_accumulator() };

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

    // Bottom row (we have up)
    {
        drop(buffer_U);
        let buffer_U = buffer_C;
        let buffer_C = buffer_D;
        let mut row = im.row_mut(im.height() - 1);
        for x in 0..width {
            let mut v_C = buffer_C[x];
            let v_U = buffer_U[x];
            v_C.update(&v_U);
            row[x] = v_C;
        }
    }
    im


    // im.map_indexed(|im: &ImageBuffer<M, Box<[u8]>>, x, y| {
    //     let mut acc = M::new_accumulator();
    //     for v in im.window(x, y, 1, 1).pixels() {
    //         acc.update(v);
    //     }
    //     acc
    // })
}



#[cfg(test)]
mod test {
    use rand::{thread_rng, Rng};

    use crate::{util::{ImageY8, ImageBuffer}, Image};

    use super::{tile_minmax, MinMax, blur};

    fn random_image(width: usize, height: usize) -> ImageY8 {
        let mut rng = thread_rng();
        let mut result = ImageY8::zeroed_packed(width, height);
        for y in 0..height {
            for x in 0..width {
                result[(x, y)] = rng.gen();
            }
        }
        result
    }

    fn check_minmax<const KSZ: usize>(width: usize, height: usize) {
        let mut im = ImageY8::zeroed(width, height);
        for i in 0..im.width() {
            for j in 0..im.height() {
                im[(i, j)] = (i * 4 + j) as u8;
            }
        }
        let res = tile_minmax::<KSZ>(&im);
        assert_eq!(res.width(), im.width().div_floor(KSZ));
        assert_eq!(res.height(), im.height().div_floor(KSZ));
        for i in 0..res.width() {
            for j in 0..res.height() {
                let mut v_min = 255;
                let mut v_max = 0;
                for i1 in (i*KSZ)..((i+1)*KSZ) {
                    for j1 in (j*KSZ)..((j+1)*KSZ) {
                        let v = (i1 * 4 + j1) as u8;
                        if v < v_min {
                            v_min = v;
                        }
                        if v > v_max {
                            v_max = v;
                        }
                    }
                }
                assert_eq!(res[(i, j)], [v_min, v_max]);
            }
        }
    }

    #[test]
    fn test_tile_minmax_aligned4() {
        check_minmax::<4>(16, 16);
    }

    #[test]
    fn test_tile_minmax_unaligned4() {
        check_minmax::<4>(15, 13);
    }

    fn check_blur(img: Image<[u8; 2]>) {
        let a = blur(img.clone());
        let b = img.map_indexed(|im: &ImageBuffer<[u8; 2], Box<[u8]>>, x, y| {
            let mut acc = <[u8; 2]>::new_accumulator();
            
            for v in im.window(x, y, 1, 1).pixels() {
                acc.update(v);
            }
            acc
        });
        assert_eq!(a, b);
    }

    #[test]
    fn test_blur() {
        let img0 = random_image(3, 3);
        let img_minmax = tile_minmax::<1>(&img0);
        drop(img0);
        check_blur(img_minmax);
    }
}

fn build_threshim<const TILESZ: usize>(im: &ImageY8, im_minmax: &Image<[u8; 2]>, qtp: &AprilTagQuadThreshParams) -> ImageY8 {
    let mut threshim = ImageY8::zeroed(im.width(), im.height());
    for ((tx, ty), [min, max]) in im_minmax.enumerate_pixels() {
        // low contrast region? (no edges)
        if max - min < qtp.min_white_black_diff {
            for dy in 0..TILESZ {
                let y = ty*TILESZ + dy;

                for dx in 0..TILESZ {
                    let x = tx*TILESZ + dx;

                    threshim[(x, y)] = 127;
                }
            }
            continue;
        }

        // otherwise, actually threshold this tile.

        // argument for biasing towards dark; specular highlights
        // can be substantially brighter than white tag parts
        let thresh = min + (max - min) / 2;

        for dy in 0..TILESZ {
            let y = ty*TILESZ + dy;

            for dx in 0..TILESZ {
                let x = tx*TILESZ + dx;

                let v = im[(x,y)];
                threshim[(x, y)] = if v > thresh { 255 } else { 0 };
            }
        }
    }

    // we skipped over the non-full-sized tiles above. Fix those now.
    if true {
        let th = im_minmax.height();
        let tw = im_minmax.width();
        for y in 0..im.height() {

            // what is the first x coordinate we need to process in this row?

            let x0 = if y >= th*TILESZ {
                0 // we're at the bottom; do the whole row.
            } else {
                tw*TILESZ // we only need to do the right most part.
            };

            // compute tile coordinates and clamp.
            let ty = std::cmp::min(y / TILESZ, th - 1);
            
            for x in x0..im.width() {
                let tx = std::cmp::min(x / TILESZ, tw - 1);

                let [min, max] = im_minmax[(tx, ty)];
                let thresh = min + (max - min) / 2;

                let v = im[(x, y)];
                threshim[(x, y)] = if v > thresh { 255 } else { 0 };
            }
        }
    }

    threshim
}
/// XXX Tunable. Generally, small tile sizee -- so long as they're
/// large enough to span a single tag edge -- seem to be a winner.
pub(crate) const TILESZ: usize = 4;


#[repr(u8)]
enum Threshold {
    Low,
    Invalid,
    High,
}

type ThresholdPacked = ();

type ThresholdImage = ImageBuffer<Threshold, >;

/// The idea is to find the maximum and minimum values in a
/// window around each pixel. If it's a contrast-free region
/// (max-min is small), don't try to binarize. Otherwise,
/// threshold according to (max+min)/2.
///
/// Mark low-contrast regions with value 127 so that we can skip
/// future work on these areas too.
/// 
/// however, computing max/min around every pixel is needlessly
/// expensive. We compute max/min for tiles. To avoid artifacts
/// that arise when high-contrast features appear near a tile
/// edge (and thus moving from one tile to another results in a
/// large change in max/min value), the max/min values used for
/// any pixel are computed from all 3x3 surrounding tiles. Thus,
/// the max/min sampling area for nearby pixels overlap by at least
/// one tile.
///
/// The important thing is that the windows be large enough to
/// capture edge transitions; the tag does not need to fit into
/// a tile.
pub(crate) fn threshold(qtp: &AprilTagQuadThreshParams, tp: &mut TimeProfile, im: &ImageY8) -> ImageY8 {
    let w = im.width();
    let h = im.height();
    assert!(w < u16::MAX as _, "Width overflow");
    assert!(h < u16::MAX as _, "Height overflow");

    // first, collect min/max statistics for each tile

    let im_minmax = tile_minmax::<TILESZ>(im);
    tp.stamp("tile_minmax");

    // second, apply 3x3 max/min convolution to "blur" these values
    // over larger areas. This reduces artifacts due to abrupt changes
    // in the threshold value.
    let im_minmax = if true {
        blur(im_minmax)
    } else {
        im_minmax
    };
    tp.stamp("blur");

    #[cfg(feature="extra_debug")]
    {
        use crate::util::image::ImageWritePNM;
        im_minmax
            .map(|v| Luma([v[0]]))
            .save_to_pnm("debug_threshim_min.pnm")
            .unwrap();
        im_minmax
            .map(|v| Luma([v[1]]))
            .save_to_pnm("debug_threshim_max.pnm")
            .unwrap();
    }

    let mut threshim = build_threshim::<TILESZ>(im, &im_minmax, qtp);
    drop(im_minmax);
    tp.stamp("build_threshim");

    // this is a dilate/erode deglitching scheme that does not improve
    // anything as far as I can tell.
    if qtp.deglitch {
        let mut tmp = ImageY8::zeroed_packed(w, h);

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let mut max = 0;
                let slice = threshim.window(x, y, 1, 1);
                for v in slice.pixels() {
                    let v = v.to_value();
                    if v > max {
                        max = v;
                    }
                }
                tmp[(x, y)] = max;
            }
        }

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let mut min = 255;
                for dy in (-1isize)..=(1isize) {
                    for dx in (-1isize)..=(1isize) {
                        let v = tmp[((x as isize + dx) as usize, (y as isize + dy) as usize)];
                        if v < min {
                            min = v;
                        }
                    }
                }
                threshim[(x, y)] = min;
            }
        }
    }

    tp.stamp("threshold");

    threshim
}

// basically the same as threshold(), but assumes the input image is a
// bayer image. It collects statistics separately for each 2x2 block
// of pixels. NOT WELL TESTED.
pub(super) fn threshold_bayer(tp: &mut TimeProfile, im: &ImageY8) -> ImageY8 {
    let w = im.width();
    let h = im.height();
    let s = im.stride();

    let mut threshim = ImageY8::zeroed_with_alignment(w, h, s);
    assert_eq!(threshim.stride(), s);

    let tilesz = 32;
    assert_eq!(tilesz & 1, 0); // must be multiple of 2

    let tw = w/tilesz + 1;
    let th = h/tilesz + 1;

    let (mut im_min, mut im_max) = {
        let mut im_min = ArrayVec::<Box<[u8]>, 4>::new();
        let mut im_max = ArrayVec::<Box<[u8]>, 4>::new();
        for _ in 0..4 {
            im_min.push(calloc(tw * th));
            im_max.push(calloc(tw * th));
        }
        (im_min.into_inner().unwrap(), im_max.into_inner().unwrap())
    };

    for ty in 0..th {
        for tx in 0..tw {
            let mut max = [u8::MIN; 4];
            let mut min = [u8::MAX; 4];
            for dy in 0..tilesz {
                if ty*tilesz+dy >= h {
                    continue;
                }

                for dx in 0..tilesz {
                    if tx*tilesz+dx >= w {
                        continue;
                    }

                    // which bayer element is this pixel?
                    let idx = 2*(dy&1) + (dx&1);

                    let v = im[(tx*tilesz + dx, ty * tilesz + dy)];
                    if v < min[idx] {
                        min[idx] = v;
                    }
                    if v > max[idx] {
                        max[idx] = v;
                    }
                }
            }

            for i in 0..4 {
                im_max[i][ty*tw+tx] = max[i];
                im_min[i][ty*tw+tx] = min[i];
            }
        }
    }

    for ty in 0..th {
        for tx in 0..tw {
            let mut max = [u8::MIN; 4];
            let mut min = [u8::MAX; 4];

            for dy in -1isize..=1isize {
                let y = ty as isize + dy;
                if y < 0 {
                    continue;
                }
                let y = y as usize;
                if y >= th {
                    continue;
                }

                for dx in -1isize..=1isize {
                    let x = tx as isize + dx;
                    if x < 0 {
                        continue;
                    }
                    let x = x as usize;
                    if x >= tw {
                        continue;
                    }

                    for i in 0..4 {
                        let m = im_max[i][y*tw+x];
                        if m > max[i] {
                            max[i] = m;
                        }
                        let m = im_min[i][y*tw+x];
                        if m < min[i] {
                            min[i] = m;
                        }
                    }
                }
            }

            // XXX CONSTANT
//            if (max - min < 30)
//                continue;

            // argument for biasing towards dark: specular highlights
            // can be substantially brighter than white tag parts
            let mut thresh = [0u8; 4];
            for i in 0..4 {
                thresh[i] = min[i] + (max[i] - min[i]) / 2;
            }

            for dy in 0..tilesz {
                let y = ty*tilesz + dy;
                if y >= h {
                    continue;
                }

                for dx in 0..tilesz {
                    let x = tx*tilesz + dx;
                    if x >= w {
                        continue;
                    }

                    // which bayer element is this pixel?
                    let idx = 2*(y&1) + (x&1);

                    let v = im[(x, y)];
                    threshim[(x, y)] = if v > thresh[idx] { 1 } else { 0 };
                }
            }
        }
    }

    tp.stamp("threshold");

    threshim
}
