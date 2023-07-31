mod edges;
mod greymodel;
mod sharpening;

use std::sync::{Mutex, Arc};

use crate::{detector::DetectorConfig, util::{geom::Point2D, math::{mat::Mat33, Vec2, Vec2Builder}, homography::homography_project, image::{ImageBuffer, ImageY8}}, families::AprilTagFamily, quickdecode::{QuickDecode, QuickDecodeResult}, AprilTagDetection};

use greymodel::Graymodel;

use self::greymodel::SolvedGraymodel;

#[derive(Clone)]
pub(crate) struct Quad {
    pub(crate) corners: Quadrilateral,
    pub(crate) reversed_border: bool,
}
#[cfg(feature="compare_reference")]
impl float_cmp::ApproxEq for Quad {
    type Margin = float_cmp::F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin: Self::Margin = margin.into();
        self.corners.approx_eq(other.corners, margin) && self.reversed_border == other.reversed_border
    }
}

fn value_for_pixel(im: &ImageY8, p: Point2D) -> Option<f64> {
    let x1 = f64::floor(p.x() - 0.5) as isize;
    if x1 < 0 {
        return None;
    }
    let x2 = f64::ceil(p.x() - 0.5) as usize;
    if x2 >= im.width() {
        return None;
    }
    let x = p.x() - 0.5 - (x1 as f64);

    let y1 = f64::floor(p.y() - 0.5) as isize;
    if y1 < 0 {
        return None;
    }
    let y2 = f64::ceil(p.y() - 0.5) as usize;
    if y2 >= im.height() {
        return None;
    }
    let y = p.y() - 0.5 - (y1 as f64);

    let v = 0.
     + im[(x1 as usize, y1 as usize)] as f64 * (1.-x) * (1.-y)
     + im[(x2 as usize, y1 as usize)] as f64 *      x * (1.-y)
     + im[(x1 as usize, y2 as usize)] as f64 * (1.-x) * y
     + im[(x2 as usize, y2 as usize)] as f64 *      x * y;

    Some(v)
}

pub(crate) struct QuadDecodeInfo<'a> {
    /// Reference to detector parameters
    pub(crate) det_params: &'a DetectorConfig,
    /// Tag families to look for
    pub(crate) tag_families: &'a [Arc<QuickDecode>],
    pub(crate) im_orig: &'a ImageY8,
    /// Samples image (debug only)
    #[cfg(feature="debug")]
    pub(crate) im_samples: Option<&'a Mutex<ImageY8>>,
}

enum HomographySolveError {
    SingularMatrix,
    InverseH,
}

fn homography_compute2(c: [[f64; 4]; 4]) -> Result<Mat33, HomographySolveError> {
    let mut A = [
            [c[0][0], c[0][1], 1.,      0.,      0., 0., -c[0][0]*c[0][2], -c[0][1]*c[0][2], c[0][2]],
            [     0.,      0., 0., c[0][0], c[0][1], 1., -c[0][0]*c[0][3], -c[0][1]*c[0][3], c[0][3]],
            [c[1][0], c[1][1], 1.,      0.,      0., 0., -c[1][0]*c[1][2], -c[1][1]*c[1][2], c[1][2]],
            [     0.,      0., 0., c[1][0], c[1][1], 1., -c[1][0]*c[1][3], -c[1][1]*c[1][3], c[1][3]],
            [c[2][0], c[2][1], 1.,      0.,      0., 0., -c[2][0]*c[2][2], -c[2][1]*c[2][2], c[2][2]],
            [     0.,      0., 0., c[2][0], c[2][1], 1., -c[2][0]*c[2][3], -c[2][1]*c[2][3], c[2][3]],
            [c[3][0], c[3][1], 1.,      0.,      0., 0., -c[3][0]*c[3][2], -c[3][1]*c[3][2], c[3][2]],
            [     0.,      0., 0., c[3][0], c[3][1], 1., -c[3][0]*c[3][3], -c[3][1]*c[3][3], c[3][3]],
    ];

    const EPSILON: f64 = 1e-10;

    // Eliminate.
    for col in 0..8 {
        // Find best row to swap with.
        let max_val_idx = {
            let mut max_val = 0.;
            let mut max_val_idx = None;
            for row in col..8 {
                let val = A[row][col].abs();
                if val > max_val {
                    max_val = val;
                    max_val_idx = Some(row);
                }
            }

            if max_val < EPSILON {
                //TODO
                println!("Warning: Matrix is singular");
                return Err(HomographySolveError::SingularMatrix);
            }
            max_val_idx.unwrap()
        };

        // Swap to get best row.
        if max_val_idx != col {
            for i in col..9 {
                let tmp = A[col][i];
                A[col][i] = A[max_val_idx][i];
                A[max_val_idx][i] = tmp;
            }
        }

        // Do eliminate.
        for i in (col+1)..8 {
            let f = A[i][col]/A[col][col];
            A[i][col] = 0.;
            for j in (col+1)..9 {
                A[i][j] -= f * A[col][j];
            }
        }
    }

    // Back solve.
    for col in (0..=7).rev() {
        let mut sum = 0.;
        for i in (col+1)..8 {
            sum += A[col][i]*A[i][8];
        }
        A[col][8] = (A[col][8] - sum)/A[col][col];
    }

    Ok(Mat33::of([
        A[0][8], A[1][8], A[2][8],
        A[3][8], A[4][8], A[5][8],
        A[6][8], A[7][8], 1.
    ]))
}

impl Quad {
    /// We will compute a threshold by sampling known white/black cells around this tag.
    /// This sampling is achieved by considering a set of samples along lines.
    ///
    /// coordinates are given in bit coordinates. ([0, fam.d]).
    ///
    /// { initial x, initial y, delta x, delta y, WHITE=1 }
    fn sample_threshold(H: &Mat33, family: &AprilTagFamily, im: &ImageY8, im_samples: Option<&Mutex<ImageY8>>) -> (SolvedGraymodel, SolvedGraymodel) {
        struct Pattern {
            initial: Vec2,
            delta: Vec2,
            is_white: bool,
        }

        let patterns = [
            // left white column
            Pattern {
                initial: Vec2::of(-0.5, 0.5),
                delta: Vec2::of(0., 1.),
                is_white: true,
            },
            // left black column
            Pattern {
                initial: Vec2::dup(0.5),
                delta: Vec2::of(0., 1.),
                is_white: false,
            },
            // right white column
            Pattern {
                initial: Vec2::of(family.width_at_border as f64 + 0.5, 0.5),
                delta: Vec2::of(0., 1.),
                is_white: true,
            },
            // right black column
            Pattern {
                initial: Vec2::of(family.width_at_border as f64 - 0.5, 0.5),
                delta: Vec2::of(0., 1.),
                is_white: false,
            },
            // top white row
            Pattern {
                initial: Vec2::of(0.5, -0.5),
                delta: Vec2::of(1., 0.),
                is_white: true,
            },
            // top black row
            Pattern {
                initial: Vec2::dup(0.5),
                delta: Vec2::of(1., 0.),
                is_white: false,
            },
            // bottom white row
            Pattern {
                initial: Vec2::of(0.5, family.width_at_border as f64 + 0.5),
                delta: Vec2::of(1., 0.),
                is_white: true,
            },
            // bottom black row
            Pattern {
                initial: Vec2::of(0.5, family.width_at_border as f64 - 0.5),
                delta: Vec2::of(1., 0.),
                is_white: false,
            },
            // XXX double-counts the corners.
        ];

        let mut whitemodel = Graymodel::new();
        let mut blackmodel = Graymodel::new();

        for pattern in patterns {
            for i in 0..family.width_at_border {
                let tag01 = (pattern.initial + (pattern.delta * (i as f64))) / (family.width_at_border as f64);

                let tag = (tag01 - 0.5) * 2.;

                let p = homography_project(H, tag.x(), tag.y());

                // don't round
                let ix = p.x() as isize;
                let iy = p.y() as isize;
                if ix < 0 || iy < 0 {
                    continue;
                }

                let ix = ix as usize;
                let iy = iy as usize;
                if ix >= im.width() || iy >= im.height() {
                    continue;
                }

                let v = im[(ix, iy)];

                #[cfg(feature="debug")]
                if let Some(mut im_samples) = im_samples.and_then(|im_samples| im_samples.lock().ok()) {
                    im_samples[(ix, iy)] = if pattern.is_white { 0 } else { 255 };
                }

                let model = if pattern.is_white {
                    &mut whitemodel
                } else {
                    &mut blackmodel
                };
                model.add(tag.x(), tag.y(), v as f64);
            }
        }

        let whitemodel = whitemodel.solve();
        let blackmodel = if family.width_at_border > 1 {
            blackmodel.solve()
        } else {
            SolvedGraymodel::new([0., 0., blackmodel.B[2] / 4.])
        };
        (whitemodel, blackmodel)
    }

    /// compute the average decision margin (how far was each bit from
    /// the decision boundary?
    ///
    /// we score this separately for white and black pixels and return
    /// the minimum average threshold for black/white pixels. This is
    /// to penalize thresholds that are too close to an extreme.
    fn decision_margin(H: &Mat33, family: &AprilTagFamily, decode_sharpening: f64, im: &ImageY8, whitemodel: SolvedGraymodel, blackmodel: SolvedGraymodel, im_samples: Option<&Mutex<ImageY8>>) -> (u64, f32) {
        let mut black_score = 0f32;
        let mut white_score = 0f32;
        let mut black_score_count = 1usize;
        let mut white_score_count = 1usize;

        let mut values = ImageBuffer::<[f64; 1]>::zeroed_packed(family.total_width as usize, family.total_width as usize);

        let min_coord = (family.width_at_border as i32 - family.total_width as i32)/2;
        let half = Vec2::dup(0.5);

        let combined_model = whitemodel + blackmodel;

        for &(bitx, bity) in family.bits.iter() {
            let tag = (Vec2::of(bitx as _, bity as _) + half) / (family.width_at_border as f64);
            // let tag01x = (bitx as f64 + 0.5) / (family.width_at_border as f64);
            // let tag01y = (bity as f64 + 0.5) / (family.width_at_border as f64);

            // scale to [-1, 1]
            let tag = (tag - &half) * 2.;
            // let tagx = (tag01x - 0.5) * 2.;
            // let tagy = (tag01y - 0.5) * 2.;
            let p = homography_project(H, tag.x(), tag.y());
            let v = match value_for_pixel(im, p) {
                Some(v) => v,
                None => continue,
            };

            let thresh = combined_model.interpolate(tag) / 2.0;
            values[((bitx as i32 - min_coord) as usize, (bity as i32 - min_coord) as usize)] = [v - thresh];

            if let Some(im_samples_lock) = im_samples {
                let ix = p.x() as usize;
                let iy = p.y() as usize;
                let mut im_samples = im_samples_lock.lock().unwrap();
                im_samples[(ix, iy)] = if v < thresh { 255 } else { 0 };
            }
        }

        sharpening::sharpen(&mut values, decode_sharpening);

        let mut rcode = 0u64;
        for (bitx, bity) in family.bits.iter() {
            rcode <<= 1;
            let v = values[((*bitx as i32 - min_coord) as usize, (*bity as i32 - min_coord) as usize)][0];

            if v > 0. {
                white_score += v as f32;
                white_score_count += 1;
                rcode |= 1;
            } else {
                black_score -= v as f32;
                black_score_count += 1;
            }
        }

        let score = f32::min(white_score / (white_score_count as f32), black_score / (black_score_count as f32));

        (rcode, score)
    }

    /// returns the decision margin. Return `None` if the detection should be rejected.
    fn decode(&self, det_params: &DetectorConfig, qd: &QuickDecode, im: &ImageY8, im_samples: Option<&Mutex<ImageY8>>) -> Option<(f32, QuickDecodeResult)> {
        // decode the tag binary contents by sampling the pixel
        // closest to the center of each bit cell.
        let (whitemodel, blackmodel) = self.sample_threshold(&qd.family, im, im_samples);

        // XXX Tunable
        if (whitemodel.interpolate(Vec2::zero()) - blackmodel.interpolate(Vec2::zero()) < 0.) != qd.family.reversed_border {
            return None;
        }

        let (rcode, score) = self.decision_margin(&qd.family, det_params.decode_sharpening, im, whitemodel, blackmodel, im_samples);

        let entry = qd.decode_codeword(rcode)?;

        Some((score, entry))
    }

    fn decode_family(&self, info: &QuadDecodeInfo, qd: &QuickDecode, H: &Mat33, #[cfg(feature="compare_reference")] quad_sys: &mut apriltag_sys::quad) -> Option<AprilTagDetection> {
        let decode_res = self.decode(info, qd, H);

        #[cfg(feature="compare_reference")]
        {
            use crate::sys::{AprilTagDetectorSys, SysPtr, ImageU8Sys};
            let mut td_sys = AprilTagDetectorSys::new().unwrap();
            td_sys.as_mut().decode_sharpening = info.det_params.decode_sharpening;
            let family_sys = SysPtr::<apriltag_sys::apriltag_family>::new(&qd.family).unwrap();
            let im_sys = ImageU8Sys::new(info.im_orig).unwrap();
            let decode_res_sys = unsafe {
                apriltag_sys::quick_decode_init(family_sys.as_ptr(), qd.bits_corrected as _);
                let mut entry_sys: apriltag_sys::quick_decode_entry = std::mem::zeroed();
                let decode_margin_sys: f32 = apriltag_sys::quad_decode(
                    td_sys.as_ptr(),
                    family_sys.as_ptr(),
                    im_sys.as_ptr(),
                    quad_sys,
                    &mut entry_sys,
                    std::ptr::null_mut(),
                );
                apriltag_sys::quick_decode_uninit(family_sys.as_ptr());
                dbg!(decode_margin_sys);
                if decode_margin_sys < 0. || entry_sys.hamming >= 255 {
                    None
                } else {
                    Some((decode_margin_sys, QuickDecodeResult {
                        id: entry_sys.id,
                        hamming: entry_sys.hamming,
                        rotation: match entry_sys.rotation {
                            0 => crate::families::Rotation::Identity,
                            1 => crate::families::Rotation::Deg90,
                            2 => crate::families::Rotation::Deg180,
                            3 => crate::families::Rotation::Deg270,
                            _ => panic!("Invalid rotation"),
                        }
                    }))
                }
            };
            println!("Check: {decode_res:?} vs {decode_res_sys:?}");
            assert_eq!(decode_res, decode_res_sys);
        }
        let (decision_margin, entry) = decode_res?;

        if decision_margin < 0. || entry.hamming >= 255 {
            return None;
        }

        let theta = -entry.rotation.theta();
        let (s, c) = theta.sin_cos();

        // Fix the rotation of our homography to properly orient the tag
        let H = {
            let R = Mat33::of([
                c, -s,  0.,
                s,  c,  0.,
                0., 0., 1.,
            ]);

            let H = self.H.as_ref().unwrap();
            H.matmul(&R)
        };

        let center = homography_project(&H, 0., 0.);

        // [-1, -1], [1, -1], [1, 1], [-1, 1], Desired points
        // [-1, 1], [1, 1], [1, -1], [-1, -1], FLIP Y
        // adjust the points in det->p so that they correspond to
        // counter-clockwise around the quad, starting at -1,-1.
        let mut corners = [Point2D::zero(); 4];
        for i in 0..4 {
            let tcx = if i == 1 || i == 2 { 1. } else { -1. };
            let tcy = if i < 2 { 1. } else { -1. };

            corners[i] = homography_project(&H, tcx, tcy);
        }

        Some(AprilTagDetection {
            family: qd.family.clone(),
            id: entry.id.into(),
            hamming: entry.hamming.into(),
            decision_margin,
            H,
            center,
            corners,
        })
    }

    pub fn decode_task(&mut self, info: QuadDecodeInfo) -> Vec<AprilTagDetection> {
        // refine edges is not dependent upon the tag family, thus
        // apply this optimization BEFORE the other work.
        //if (td->quad_decimate > 1 && td->refine_edges) {
        if info.det_params.refine_edges {
            self.refine_edges(&info.det_params, info.im_orig);
        }

        // make sure the homographies are computed...
        if self.update_homographies().is_err() {
            println!("update_homographies error");
            return vec![];
        }

        info.tag_families
            .iter()
            .filter(|qd| qd.family.reversed_border == self.reversed_border)
            .filter_map(|qd| {
                // since the geometry of tag families can vary, start any
                // optimization process over with the original quad.
                self.decode_family(&info, qd)
            })
            .collect()
    }

    fn update_homographies(&mut self) -> Result<(), HomographySolveError> {
        let mut corr_arr = [[0f64; 4]; 4];
        for i in 0..4 {
            corr_arr[i] = [
                if i == 0 || i == 3 { -1. } else { 1. },
                if i == 0 || i == 1 { -1. } else { 1. },
                self.corners[i].x(),
                self.corners[i].y(),
            ];
        }

        // XXX Tunable
        let H = homography_compute2(corr_arr)?;
        let _Hinv = H.inv()
            .ok_or(HomographySolveError::InverseH)?;
        self.H = Some(H);
        // self.Hinv = Some(Hinv);

        Ok(())
    }
}

/*// returns score of best quad
fn optimize_quad_generic(family: &AprilTagFamily, im: &Image, quad0: Quad, stepsizes: &[f32], score: impl Fn(&AprilTagFamily, &Image, &Quad) -> f64) -> f64 {
    let best_quad = quad0.clone();
    let best_score = score(family, im, best_quad);

    for stepsize_idx in 0..stepsizes.len() {
        // when we make progress with a particular step size, how many
        // times will we try to perform that same step size again?
        // (max_repeat = 0 means ("don't repeat--- just move to the
        // next step size").
        // XXX Tunable
        let max_repeat = 1;

        for repeat in 0..=max_repeat {
            let mut improved = false;

            // wiggle point i
            for i in 0..4 {
                let stepsize = stepsizes[stepsize_idx];

                // XXX Tunable (really 1 makes the best sense since)
                let nsteps = 1;

                let mut this_best_quad = None;
                let mut this_best_score = best_score;

                for sx in (-nsteps)..=nsteps {
                    for sy in (-nsteps)..=nsteps {
                        if sx==0 && sy==0 {
                            continue;
                        }

                        let mut this_quad = best_quad.clone();
                        this_quad.corners[i] = best_quad.corners[i] + (Vec2::of(sx as f64, sy as f64) * (stepsize as f64));
                        if this_quad.update_homographies().is_err() {
                            continue;
                        }

                        let this_score = score(family, im, this_quad);

                        if this_score > this_best_score {
                            this_best_quad = this_quad;
                            this_best_score = this_score;
                        }
                    }
                }

                if this_best_score > best_score {
                    best_quad = this_best_quad;
                    best_score = this_best_score;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }

    matd_destroy(quad0.H);
    matd_destroy(quad0.Hinv);
    memcpy(quad0, best_quad, sizeof(struct quad)); // copy pointers
    return best_score;
}
*/