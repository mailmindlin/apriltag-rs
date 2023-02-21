use crate::{util::{TimeProfile, Image, mem::calloc}, ApriltagDetector};

use super::ApriltagQuadThreshParams;

pub(super) fn threshold(qtp: &ApriltagQuadThreshParams, tp: &mut TimeProfile, im: &Image) -> Image {
    let w = im.width;
    let h = im.height;
    let s = im.stride;
    assert!(w < 32768);
    assert!(h < 32768);

    let mut threshim = Image::<u8>::create_alignment(im.width, im.height, im.stride);
    assert_eq!(threshim.stride, im.stride);

    // The idea is to find the maximum and minimum values in a
    // window around each pixel. If it's a contrast-free region
    // (max-min is small), don't try to binarize. Otherwise,
    // threshold according to (max+min)/2.
    //
    // Mark low-contrast regions with value 127 so that we can skip
    // future work on these areas too.

    // however, computing max/min around every pixel is needlessly
    // expensive. We compute max/min for tiles. To avoid artifacts
    // that arise when high-contrast features appear near a tile
    // edge (and thus moving from one tile to another results in a
    // large change in max/min value), the max/min values used for
    // any pixel are computed from all 3x3 surrounding tiles. Thus,
    // the max/min sampling area for nearby pixels overlap by at least
    // one tile.
    //
    // The important thing is that the windows be large enough to
    // capture edge transitions; the tag does not need to fit into
    // a tile.

    // XXX Tunable. Generally, small tile sizes--- so long as they're
    // large enough to span a single tag edge--- seem to be a winner.
    let tilesz = 4;

    // the last (possibly partial) tiles along each row and column will
    // just use the min/max value from the last full tile.
    let tw = w / tilesz;
    let th = h / tilesz;

    let mut im_max = calloc::<u8>(tw * th);
    let mut im_min = calloc::<u8>(tw * th);

    // first, collect min/max statistics for each tile
    for ty in 0..th {
        for tx in 0..tw {
            let mut max = u8::MIN;
            let mut min = u8::MAX;

            for dy in 0..tilesz {
                for dx in 0..tilesz {
                    let v = im[(tx * tilesz + dx, ty * tilesz + dy)];
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
            }

            im_max[ty*tw+tx] = max;
            im_min[ty*tw+tx] = min;
        }
    }

    // second, apply 3x3 max/min convolution to "blur" these values
    // over larger areas. This reduces artifacts due to abrupt changes
    // in the threshold value.
    if true {
        let mut im_max_tmp = Image::<u8>::create_stride(tw, th, tw);
        let mut im_min_tmp = Image::<u8>::create_stride(tw, th, tw);

        for ty in 0..th {
            for tx in 0..tw {
                let mut max = u8::MIN;
                let mut min = u8::MAX;

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

                        max = std::cmp::max(max, im_max[y*tw+x]);
                        min = std::cmp::min(min, im_min[y*tw+x]);
                    }
                }
                im_max_tmp[(tx, ty)] = max;
                im_min_tmp[(tx, ty)] = min;
            }
        }
        im_max = im_max_tmp.buf;
        im_min = im_min_tmp.buf;
    }

    for ty in 0..th {
        for tx in 0..tw {
            let min = im_min[ty*tw + tx];
            let max = im_max[ty*tw + tx];

            // low contrast region? (no edges)
            if max - min < qtp.min_white_black_diff {
                for dy in 0..tilesz {
                    let y = ty*tilesz + dy;

                    for dx in 0..tilesz {
                        let x = tx*tilesz + dx;

                        threshim[(x, y)] = 127;
                    }
                }
                continue;
            }

            // otherwise, actually threshold this tile.

            // argument for biasing towards dark; specular highlights
            // can be substantially brighter than white tag parts
            let thresh = min + (max - min) / 2;

            for dy in 0..tilesz {
                let y = ty*tilesz + dy;

                for dx in 0..tilesz {
                    let x = tx*tilesz + dx;

                    let v = im[(x,y)];
                    threshim[(x, y)] = if v > thresh { 255 } else { 0 };
                }
            }
        }
    }

    // we skipped over the non-full-sized tiles above. Fix those now.
    if true {
        for y in 0..h {

            // what is the first x coordinate we need to process in this row?

            let x0 = if y >= th*tilesz {
                0 // we're at the bottom; do the whole row.
            } else {
                tw*tilesz // we only need to do the right most part.
            };

            // compute tile coordinates and clamp.
            let ty = std::cmp::min(y / tilesz, th - 1);
            
            for x in x0..w {
                let mut tx = x / tilesz;
                if tx >= tw {
                    tx = tw - 1;
                }

                let max = im_max[ty*tw + tx];
                let min = im_min[ty*tw + tx];
                let thresh = min + (max - min) / 2;

                let v = im[(x, y)];
                threshim[(x, y)] = if v > thresh { 255 } else { 0 };
            }
        }
    }

    std::mem::drop(im_min);
    std::mem::drop(im_max);

    // this is a dilate/erode deglitching scheme that does not improve
    // anything as far as I can tell.
    if false || qtp.deglitch {
        let mut tmp = Image::<u8>::create(w, h);

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let mut max = 0;
                for dy in (-1isize)..=(1isize) {
                    for dx in (-1isize)..=(1isize) {
                        let v = tmp[((x as isize + dx) as usize, (y as isize + dy) as usize)];
                        if v > max {
                            max = v;
                        }
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
pub(super) fn threshold_bayer(tp: &mut TimeProfile, im: &Image) -> Image {
    let w = im.width;
    let h = im.height;
    let s = im.stride;

    let mut threshim = Image::<u8>::create_alignment(w, h, s);
    assert_eq!(threshim.stride, s);

    let tilesz = 32;
    assert_eq!(tilesz & 1, 0); // must be multiple of 2

    let tw = w/tilesz + 1;
    let th = h/tilesz + 1;

    let mut im_min: [Box<[u8]>; 4];
    let mut im_max: [Box<[u8]>; 4];
    for i in 0..4 {
        im_min[i] = calloc(tw * th);
        im_max[i] = calloc(tw * th);
    }

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
            let mut thresh: [u8; 4];
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
