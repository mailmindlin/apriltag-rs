use arrayvec::ArrayVec;

use crate::util::{TimeProfile, Image, mem::calloc, image::{Luma, ImageBuffer, ImageY8, HasDimensions, Pixel, ImageMut}};

use super::ApriltagQuadThreshParams;

fn tile_minmax<const KW: usize>(im: &impl Image<Luma<u8>>) -> ImageBuffer<[u8; 2]> {
    // the last (possibly partial) tiles along each row and column will
    // just use the min/max value from the last full tile.
    let tw = im.width().div_floor(KW);
    let th = im.height().div_floor(KW);

    let mut result = ImageBuffer::<[u8; 2]>::new_packed(tw, th);

    for ((tx, ty), dst) in result.enumerate_pixels_mut() {
        let base_y = ty * KW;
        let base_x = tx * KW;
        let mut max = u8::MIN;
        let mut min = u8::MAX;

        for dy in 0..KW {
            for dx in 0..KW {
                let v = im[(base_x + dx, base_y + dy)];
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        *dst = [min, max];
    }
    result
}

fn blur(im_minmax: impl Image<[u8; 2]>) -> ImageBuffer<[u8; 2]> {
    let mut im_dst = ImageBuffer::<[u8; 2]>::new_packed(im_minmax.width(), im_minmax.height());
    for ((x, y), dst) in im_dst.enumerate_pixels_mut() {
        let mut min = u8::MIN;
        let mut max = u8::MAX;
        for &[v_min, v_max] in im_minmax.window1(x, y, 1, 1).pixels() {
            if v_min < min {
                min = v_min;
            }
            if v_max > max {
                max = v_max;
            }
        }
        *dst = [min, max]
    }

    im_dst
}

fn build_threshim<'x, const TILESZ: usize>(im: &impl Image<Luma<u8>>, im_minmax: &'x impl Image<'x, [u8; 2]>, qtp: &ApriltagQuadThreshParams) -> ImageY8 {
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
pub(super) fn threshold(qtp: &ApriltagQuadThreshParams, tp: &mut TimeProfile, im: &impl Image<Luma<u8>>) -> ImageBuffer<Luma<u8>> {
    let w = im.width();
    let h = im.height();
    assert!(w < 32768);
    assert!(h < 32768);

    // first, collect min/max statistics for each tile

    /// XXX Tunable. Generally, small tile sizee -- so long as they're
    /// large enough to span a single tag edge -- seem to be a winner.
    const TILESZ: usize = 4;

    let im_minmax = tile_minmax::<TILESZ>(im);

    // second, apply 3x3 max/min convolution to "blur" these values
    // over larger areas. This reduces artifacts due to abrupt changes
    // in the threshold value.
    let im_minmax = if true {
        blur(im_minmax)
    } else {
        im_minmax
    };

    let mut threshim = build_threshim::<TILESZ>(im, &im_minmax, qtp);
    std::mem::drop(im_minmax);

    // this is a dilate/erode deglitching scheme that does not improve
    // anything as far as I can tell.
    if qtp.deglitch {
        let mut tmp = ImageY8::new_packed(w, h);

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let mut max = 0;
                let slice = threshim.window1(x, y, 1, 1);
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
pub(super) fn threshold_bayer(tp: &mut TimeProfile, im: &impl Image<Luma<u8>>) -> ImageY8 {
    let w = im.width();
    let h = im.height();
    let s = im.stride();

    let mut threshim = ImageY8::with_alignment(w, h, s);
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
