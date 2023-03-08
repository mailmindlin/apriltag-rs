mod edges;
mod greymodel;
mod sharpening;

use std::{f64::consts::FRAC_PI_2, sync::{Arc, Mutex}};

use crate::{detector::AprilTagParams, util::{geom::Point2D, Image, math::{mat::Mat, Vec2}, homography::homography_project}, families::AprilTagFamily, quickdecode::{QuickDecode, QuickDecodeEntry}, ApriltagDetection};

use greymodel::Graymodel;

#[derive(Clone)]
pub(crate) struct Quad {
    pub(crate) corners: [Point2D; 4],
    /// Tag coordinates ([-1,1] at the black corners) to pixels
    pub(crate) H: Option<Mat>,
    /// Pixels to tag
    pub(crate) Hinv: Option<Mat>,
    pub(crate) reversed_border: bool,
}

fn value_for_pixel(im: &Image, p: Point2D) -> Option<f64> {
    let x1 = f64::floor(p.x() - 0.5) as isize;
    let x2 = f64::ceil(p.x() - 0.5) as isize;
    let x = p.x() - 0.5 - (x1 as f64);

    let y1 = f64::floor(p.y() - 0.5) as isize;
    let y2 = f64::ceil(p.y() - 0.5) as isize;
    let y = p.y() - 0.5 - (y1 as f64);
    if x1 < 0 || x2 as usize >= im.width || y1 < 0 || y2 as usize >= im.height {
        return None;//TODO
    }

    let v = 0.
    + im[(x1 as usize, y1 as usize)] as f64 * (1.-x)*(1.-y)
    + im[(x2 as usize, y1 as usize)] as f64 * x*(1.-y)
    + im[(x1 as usize, y2 as usize)] as f64 * (1.-x)*y
    + im[(x2 as usize, y2 as usize)] as f64 * x*y;

    Some(v)
}

pub(crate) struct QuadDecodeInfo<'a> {
    pub(crate) det_params: &'a AprilTagParams,
    pub(crate) tag_families: &'a Vec<(Arc<AprilTagFamily>, QuickDecode)>,
    pub(crate) im: &'a Image,
    pub(crate) im_samples: Option<&'a Mutex<Image>>,
}

fn homography_compute2(c: [[f64; 4]; 4]) -> Option<Mat> {
    let mut A = [
            c[0][0], c[0][1], 1.,      0.,      0., 0., -c[0][0]*c[0][2], -c[0][1]*c[0][2], c[0][2],
                 0.,      0., 0., c[0][0], c[0][1], 1., -c[0][0]*c[0][3], -c[0][1]*c[0][3], c[0][3],
            c[1][0], c[1][1], 1.,      0.,      0., 0., -c[1][0]*c[1][2], -c[1][1]*c[1][2], c[1][2],
                 0.,      0., 0., c[1][0], c[1][1], 1., -c[1][0]*c[1][3], -c[1][1]*c[1][3], c[1][3],
            c[2][0], c[2][1], 1.,      0.,      0., 0., -c[2][0]*c[2][2], -c[2][1]*c[2][2], c[2][2],
                 0.,      0., 0., c[2][0], c[2][1], 1., -c[2][0]*c[2][3], -c[2][1]*c[2][3], c[2][3],
            c[3][0], c[3][1], 1.,      0.,      0., 0., -c[3][0]*c[3][2], -c[3][1]*c[3][2], c[3][2],
                 0.,      0., 0., c[3][0], c[3][1], 1., -c[3][0]*c[3][3], -c[3][1]*c[3][3], c[3][3],
    ];

    const EPSILON: f64 = 1e-10;

    // Eliminate.
    for col in 0..8 {
        // Find best row to swap with.
        let mut max_val = 0.;
        let mut max_val_idx = None;
        for row in col..8 {
            let val = A[row*9 + col].abs();
            if val > max_val {
                max_val = val;
                max_val_idx = Some(row);
            }
        }

        if max_val < EPSILON {
            //TODO
            // debug_print("WRN: Matrix is singular.\n");
            return None;
        }
        let max_val_idx = max_val_idx.unwrap();

        // Swap to get best row.
        if max_val_idx != col {
            for i in col..9 {
                let tmp = A[col*9 + i];
                A[col*9 + i] = A[max_val_idx*9 + i];
                A[max_val_idx*9 + i] = tmp;
            }
        }

        // Do eliminate.
        for i in (col+1)..8 {
            let f = A[i*9 + col]/A[col*9 + col];
            A[i*9 + col] = 0.;
            for j in (col+1)..9 {
                A[i*9 + j] -= f*A[col*9 + j];
            }
        }
    }

    // Back solve.
    for col in (0..=7).rev() {
        let mut sum = 0.;
        for i in (col+1)..8 {
            sum += A[col*9 + i]*A[i*9 + 8];
        }
        A[col*9 + 8] = (A[col*9 + 8] - sum)/A[col*9 + col];
    }
    Some(Mat::create(3, 3, &[
        A[8], A[17], A[26],
        A[35], A[44], A[53],
        A[62], A[71], 1.
    ]))
}

impl Quad {
    /// returns the decision margin. Return `None` if the detection should be rejected.
    pub fn decode(&self, det_params: &AprilTagParams, family: &AprilTagFamily, qd: &QuickDecode, im: &Image, im_samples: Option<&Mutex<Image>>) -> Option<(f32, QuickDecodeEntry)> {
        // decode the tag binary contents by sampling the pixel
        // closest to the center of each bit cell.

        // We will compute a threshold by sampling known white/black cells around this tag.
        // This sampling is achieved by considering a set of samples along lines.
        //
        // coordinates are given in bit coordinates. ([0, fam.d]).
        //
        // { initial x, initial y, delta x, delta y, WHITE=1 }

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
                initial: Vec2::of(0.5, 0.5),
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
                initial: Vec2::of(0.5, 0.5),
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

        let mut whitemodel = Graymodel::init();
        let mut blackmodel = Graymodel::init();

        for pattern in patterns {
            for i in 0..family.width_at_border {
                let tag01 = (&pattern.initial + &(pattern.delta * (i as f64))) / (family.width_at_border as f64);

                let tag = (tag01 - 0.5) * 2.;

                let p = homography_project(self.H.as_ref().unwrap(), tag.x(), tag.y());

                // don't round
                let ix = p.x() as isize;
                let iy = p.y() as isize;
                if ix < 0 || iy < 0 {
                    continue;
                }

                let ix = ix as usize;
                let iy = iy as usize;
                if ix >= im.width || iy >= im.height {
                    continue;
                }

                let v = im[(ix, iy)];

                if let Some(im_samples_lock) = im_samples {
                    let mut im_samples = im_samples_lock.lock().unwrap();
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

        whitemodel.solve();
        blackmodel.solve();

        // XXX Tunable
        if whitemodel.interpolate(0., 0.) - blackmodel.interpolate(0., 0.) < 0. {
            return None;
        }

        // compute the average decision margin (how far was each bit from
        // the decision boundary?
        //
        // we score this separately for white and black pixels and return
        // the minimum average threshold for black/white pixels. This is
        // to penalize thresholds that are too close to an extreme.
        let mut black_score = 0f32;
        let mut white_score = 0f32;
        let mut black_score_count = 1usize;
        let mut white_score_count = 1usize;

        let mut values = vec![0f64; (family.total_width*family.total_width) as usize];

        let min_coord = (family.width_at_border - family.total_width)/2;
        let half = Vec2::of(0.5, 0.5);
        for &(bitx, bity) in family.bits.iter() {
            let tag01 = (Vec2::of(bitx as f64, bity as f64) + &half) / (family.width_at_border as f64);

            // scale to [-1, 1]
            let tag = (tag01 - &half) * 2.;
            let p = homography_project(self.H.as_ref().unwrap(), tag.x(), tag.y());
            let v = match value_for_pixel(im, p) {
                Some(v) => v,
                None => continue,
            };

            let thresh = (blackmodel.interpolate(tag.x(), tag.y()) + whitemodel.interpolate(tag.x(), tag.y())) / 2.0;
            values[(family.total_width*(bity - min_coord) + bitx - min_coord) as usize] = v - thresh;

            if let Some(im_samples_lock) = im_samples {
                let ix = p.x() as usize;
                let iy = p.y() as usize;
                let mut im_samples = im_samples_lock.lock().unwrap();
                im_samples[(ix, iy)] = if v < thresh { 255 } else { 0 };
            }
        }

        sharpening::sharpen(&mut values, det_params.decode_sharpening, family.total_width as usize);

        let mut rcode = 0u64;
        for (bitx, bity) in family.bits.iter() {
            rcode <<= 1;
            let v = values[((bity - min_coord)*family.total_width + bitx - min_coord) as usize];

            if v > 0. {
                white_score += v as f32;
                white_score_count += 1;
                rcode |= 1;
            } else {
                black_score -= v as f32;
                black_score_count += 1;
            }
        }

        let entry = qd.decode_codeword(family, rcode)?;
        let score = f32::min(white_score / (white_score_count as f32), black_score / (black_score_count as f32));

        Some((score, entry))
    }

    fn decode_family(&mut self, info: &QuadDecodeInfo, family: Arc<AprilTagFamily>, qd: &QuickDecode) -> Option<ApriltagDetection> {
        let (decision_margin, entry) = self.decode(&info.det_params, &family, qd, info.im, info.im_samples)?;

        if decision_margin < 0. || entry.hamming >= 255 {
            return None;
        }

        let theta = -(entry.rotation as f64) * FRAC_PI_2;
        let (s, c) = theta.sin_cos();

        // Fix the rotation of our homography to properly orient the tag
        let H = {
            let mut R = Mat::zeroes(3, 3);
            R[(0, 0)] = c;
            R[(0, 1)] = -s;
            R[(1, 0)] = s;
            R[(1, 1)] = c;
            R[(2, 2)] = 1.;

            Mat::op("M*M", &[self.H.as_ref().unwrap(), &R]).unwrap()
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

        let det = ApriltagDetection {
            family,
            id: entry.id.into(),
            hamming: entry.hamming.into(),
            decision_margin,
            H,
            center,
            corners,
        };
        Some(det)
    }

    pub fn decode_task(&mut self, info: QuadDecodeInfo) -> Vec<ApriltagDetection> {
        // refine edges is not dependent upon the tag family, thus
        // apply this optimization BEFORE the other work.
        //if (td->quad_decimate > 1 && td->refine_edges) {
        if info.det_params.refine_edges {
            self.refine_edges(&info.det_params, info.im);
        }

        // make sure the homographies are computed...
        if self.update_homographies().is_err() {
            return vec![];
        }

        info.tag_families
            .iter()
            .filter(|(family, _qd)| family.reversed_border == self.reversed_border)
            .filter_map(|(family, qd)| {
                // since the geometry of tag families can vary, start any
                // optimization process over with the original quad.
                let mut quad = self.clone();
                quad.decode_family(&info, family.to_owned(), qd)
            })
            .collect()
    }

    /// returns non-zero if an error occurs (i.e., H has no inverse)
    fn update_homographies(&mut self) -> Result<(), ()> {
        let mut corr_arr = [[0f64; 4]; 4];
        for i in 0..4 {
            let c01 = if i == 0 || i == 3 { -1. } else { 1. };
            corr_arr[i] = [
                c01,
                c01,
                self.corners[i].x(),
                self.corners[i].y(),
            ];
        }

        // XXX Tunable
        self.H = homography_compute2(corr_arr);

        self.Hinv = match self.H.as_ref().and_then(|H| H.inv()) {
            Some(Hinv) => Some(Hinv),
            None => {
                self.H = None;
                return Err(());
            }
        };

        Ok(())
    }
}

/*// returns score of best quad
fn optimize_quad_generic(family: &AprilTagFamily, im: &Image, quad0: Quad, stepsizes: &[f32], score: impl Fn(&AprilTagFamily, &Image, &Quad) -> f64) -> f64 {
    let best_quad = quad0.clone();
    let best_score = score(family, im, best_quad, user);

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