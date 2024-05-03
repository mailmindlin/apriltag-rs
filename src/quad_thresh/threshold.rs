use arrayvec::ArrayVec;

use crate::{util::{Image, mem::calloc, image::{Luma, ImageBuffer, ImageY8, Pixel, ImageRefY8, ImageRef, ImageWritePNM}}, dbg::{TimeProfile, debug_images}, DetectorConfig, DetectError};

use super::AprilTagQuadThreshParams;

fn tile_minmax<const KW: usize>(im: ImageRefY8) -> ImageBuffer<[u8; 2]> {
    // the last (possibly partial) tiles along each row and column will
    // just use the min/max value from the last full tile.
    let tw = im.width().div_floor(KW);
    let th = im.height().div_floor(KW);

    let mut result = ImageBuffer::<[u8; 2]>::zeroed_packed(tw, th);

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

#[cfg(test)]
pub(crate) fn tile_minmax_cpu<const KW: usize>(im: ImageRefY8) -> ImageBuffer<[u8; 2]> {
    tile_minmax::<KW>(im)
}

fn blur(im_minmax: Image<[u8; 2]>) -> ImageBuffer<[u8; 2]> {
    let mut result = ImageBuffer::clone_packed(&im_minmax);
    for y in 0..im_minmax.height() {
        for x in 0..im_minmax.width() {
            let mut min = u8::MAX;
            let mut max = u8::MIN;
            for dy in y.saturating_sub(1)..std::cmp::min(y+1, im_minmax.height()) {
                for dx in x.saturating_sub(1)..std::cmp::min(x+1, im_minmax.width()) {
                    let [v_min, v_max] = im_minmax[(dx, dy)];
                    if v_min < min {
                        min = v_min;
                    }
                    if v_max > max {
                        max = v_max;
                    }
                }
            }
            result[(x, y)] = [min, max];
        }
    }
    result
    // im_minmax.map_indexed(|im: &ImageBuffer<[u8; 2], Box<[u8]>>, x, y| {
    //     let mut min = u8::MAX;
    //     let mut max = u8::MIN;
    //     for &[v_min, v_max] in im.window(x, y, 1, 1).pixels() {
    //         if v_min < min {
    //             min = v_min;
    //         }
    //         if v_max > max {
    //             max = v_max;
    //         }
    //     }
    //     [min, max]
    // })
}

#[cfg(test)]
pub(crate) fn tile_blur_cpu(im: Image<[u8; 2]>) -> ImageBuffer<[u8; 2]> {
    blur(im)
}

fn build_threshim<const TILESZ: usize>(im: ImageRefY8, im_minmax: &Image<[u8; 2]>, qtp: &AprilTagQuadThreshParams) -> Result<ImageY8, DetectError> {
    let mut threshim = ImageY8::try_zeroed_dims(*im.dimensions())?;
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

    Ok(threshim)
}

fn deglitch(threshim: &mut ImageY8) {
    let mut tmp = ImageY8::zeroed(threshim.width(), threshim.height());

    for y in 1..(threshim.height() - 1) {
        for x in 1..(threshim.width() - 1) {
            let mut max = u8::MIN;
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

    for y in 0..(threshim.height() - 2) {
        for x in 0..(threshim.width() - 2) {
            let mut min = u8::MAX;
            for dy in 0..2 {
                for dx in 0..2 {
                    let v = tmp[(x + dx - 1, y + dy - 1)];
                    if v < min {
                        min = v;
                    }
                }
            }
            threshim[(x + 1, y + 1)] = min;
        }
    }
}
/// XXX Tunable. Generally, small tile sizee -- so long as they're
/// large enough to span a single tag edge -- seem to be a winner.
pub(crate) const TILESZ: usize = 4;

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
pub(crate) fn threshold(config: &DetectorConfig, tp: &mut TimeProfile, im: ImageRefY8) -> Result<ImageBuffer<Luma<u8>>, DetectError> {
    let w = im.width();
    let h = im.height();
    assert!(w < 32768);
    assert!(h < 32768);

    fn split_image(src: ImageRef<[u8; 2]>) -> (ImageY8, ImageY8) {
        let mut img_min = ImageY8::zeroed_packed(src.width(), src.height());
        let mut img_max = ImageY8::zeroed_packed(src.width(), src.height());
        for y in 0..src.height() {
            for x in 0..src.width() {
                let elem = src[(x, y)];
                img_min[(x, y)] = elem[0];
                img_max[(x, y)] = elem[1];
            }
        }
        (img_min, img_max)
    }

    // first, collect min/max statistics for each tile
    let im_minmax = tile_minmax::<TILESZ>(im);
    #[cfg(feature="debug")]
    if config.generate_debug_image() {
        let (img_min, img_max) = split_image(im_minmax.as_ref());
        config.debug_image(debug_images::TILE_MIN, |mut f| img_min.write_pnm(&mut f));
        config.debug_image(debug_images::TILE_MAX, |mut f| img_max.write_pnm(&mut f));
    }
    tp.stamp("tile_minmax");

    // second, apply 3x3 max/min convolution to "blur" these values
    // over larger areas. This reduces artifacts due to abrupt changes
    // in the threshold value.
    let im_minmax = if true {
        blur(im_minmax)
    } else {
        im_minmax
    };
    #[cfg(feature="debug")]
    if config.generate_debug_image() {
        let (img_min, img_max) = split_image(im_minmax.as_ref());
        config.debug_image(debug_images::BLUR_MIN, |mut f| img_min.write_pnm(&mut f));
        config.debug_image(debug_images::BLUR_MAX, |mut f| img_max.write_pnm(&mut f));
    }
    tp.stamp("blur");

    let mut threshim = build_threshim::<TILESZ>(im, &im_minmax, &config.qtp)?;
    drop(im_minmax);

    // this is a dilate/erode deglitching scheme that does not improve
    // anything as far as I can tell.
    if config.qtp.deglitch {
        tp.stamp("build_threshim");
        deglitch(&mut threshim);
    }

    tp.stamp("threshold");

    Ok(threshim)
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
