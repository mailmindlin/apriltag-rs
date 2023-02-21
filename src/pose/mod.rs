use crate::{util::{math::{mat::Mat, poly::Poly}, homography::homography_to_pose}, ApriltagDetection};


/// Calculate projection operator from image points.
fn calculate_F(v: &Mat) -> Mat {
    let mut outer_product = Mat::op("MM", &[v,v,v,v]).unwrap();
    let inner_product = Mat::op("M'M", &[v,v]).unwrap();
    outer_product.scale_inplace(1.0 / inner_product.as_scalar().unwrap());
    outer_product
}

/// @param v Image points on the image plane.
/// @param p Object points in object space.
/// @outparam t Optimal translation.
/// @param R In/Outparam. Should be set to initial guess at R. Will be modified to be the optimal translation.
/// @param n_points Number of points.
/// @param n_steps Number of iterations.
///
/// @return Object-space error after iteration.
///
/// Implementation of Orthogonal Iteration from Lu, 2000.
fn orthogonal_iteration(v: &[Mat], p: &[Mat], t: &mut Mat, R: &mut Mat, n_points: usize, n_steps: usize) -> f64 {
    assert_eq!(v.len(), n_points);
    assert_eq!(p.len(), n_points);

    let p_mean = p
        .iter()
        .fold(Mat::zeroes(3,1), |acc, e| acc + e)
        * (1./(n_points as f64));

    let mut p_res = Vec::with_capacity(n_points);
    for i in 0..n_points {
        p_res[i] = Mat::op("M-M", &[&p[i], &p_mean]).unwrap();
    }

    // Compute M1_inv.
    let I3 = Mat::identity(3);
    
    let (F, M1_inv) = {
        let mut F = Vec::with_capacity(n_points);
        let mut avg_F = Mat::zeroes(3, 3);
        for i in 0..n_points {
            F[i] = calculate_F(&v[i]);
            avg_F += &F[i];
        }
        avg_F *= 1. / (n_points as f64);
        let M1 = I3 - &avg_F;
        let M1_inv = M1.inv().unwrap();
        (F, M1_inv)
    };

    let mut prev_error = f64::INFINITY;
    // Iterate.
    for i in 0..n_steps {
        // Calculate translation.
        *t = {
            let mut M2 = Mat::zeroes(3, 1);
            for j in 0..n_points {
                let M2_update = Mat::op("(M - M)*M*M", &[&F[j], &I3, R, &p[j]]).unwrap();
                M2 += &M2_update;
            }
            M2 *= 1./(n_points as f64);
            M1_inv.matmul(&M2)
        };

        // Calculate rotation.
        *R = {
            let mut q = Vec::with_capacity(n_points);
            let mut q_mean = Mat::zeroes(3, 1);
            for j in 0..n_points {
                q[j] = Mat::op("M*(M*M+M)", &[&F[j], R, &p[j], &t]).unwrap();
                q_mean += &q[j];
            }
            q_mean *= 1./(n_points as f64);
            
            let mut M3 = Mat::zeroes(3, 3);
            for j in 0..n_points {
                let M3_update = Mat::op("(M-M)*M'", &[&q[j], &q_mean, &p_res[j]]).unwrap();
                M3 += &M3_update;
            }
            
            let M3_svd = M3.svd();
            Mat::op("M*M'", &[&M3_svd.U, &M3_svd.V]).unwrap()
        };

        let mut error = 0.;
        for i in 0..4 {
            let err_vec = Mat::op("(M-M)(MM+M)", &[&I3, &F[i], R, &p[i], t]).unwrap();
            error += Mat::op("M'M", &[&err_vec, &err_vec])
                .unwrap()
                .as_scalar()
                .unwrap();
        }
        prev_error = error;
    }
    
    prev_error
}

/// Given a local minima of the pose error tries to find the other minima.
fn fix_pose_ambiguities(v: &[Mat], p: &[Mat], t: &mut Mat, R: &Mat, n_points: usize) -> Option<Mat> {
    let I3 = Mat::identity(3);

    // 1. Find R_t
    let R_t = {
        let R_t_3 = t.vec_normalize();

        let R_t_1 = {
            let mut e_x = Mat::zeroes(3, 1);
            e_x[(0,0)] = 1.;

            let R_t_1_tmp = Mat::op("M-(M'*M)*M", &[&e_x, &e_x, &R_t_3, &R_t_3]).unwrap();
            R_t_1_tmp.vec_normalize()
        };

        let R_t_2 = R_t_3.cross(&R_t_1);

        Mat::create(3, 3, &[
                R_t_1[(0, 0)], R_t_1[(0, 1)], R_t_1[(0, 2)],
                R_t_2[(0, 0)], R_t_2[(0, 1)], R_t_2[(0, 2)],
                R_t_3[(0, 0)], R_t_3[(0, 1)], R_t_3[(0, 2)],
        ])
    };

    // 2. Find R_z
    let R_1_prime = R_t.matmul(R);
    let mut r31 = R_1_prime[(2, 0)];
    let mut r32 = R_1_prime[(2, 1)];
    let mut hypotenuse = (r31*r31 + r32*r32).sqrt();
    if hypotenuse < 1e-100 {
        r31 = 1.;
        r32 = 0.;
        hypotenuse = 1.;
    }
    let R_z = Mat::create(3, 3, &[
        r31/hypotenuse, -r32/hypotenuse, 0.,
        r32/hypotenuse, r31/hypotenuse, 0.,
        0., 0., 1.
    ]);

    // 3. Calculate parameters of Eos
    let (R_gamma, t_initial) = {
        let R_trans = R_1_prime.matmul(&R_z);
        let sin_gamma = -R_trans[(0, 1)];
        let cos_gamma = R_trans[(1, 1)];
        let R_gamma = Mat::create(3, 3, &[
                cos_gamma, -sin_gamma, 0.,
                sin_gamma, cos_gamma, 0.,
                0., 0., 1.]);
    
        let sin_beta = -R_trans[(2, 0)];
        let cos_beta = R_trans[(2, 2)];
        let t_initial = f64::atan2(sin_beta, cos_beta);
        (R_gamma, t_initial)
    };

    let mut v_trans = Vec::with_capacity(n_points);
    let mut p_trans = Vec::with_capacity(n_points);
    let mut F_trans = Vec::with_capacity(n_points);
    let mut avg_F_trans = Mat::zeroes(3, 3);
    for i in 0..n_points {
        p_trans.push(Mat::op("M'*M", &[&R_z, &p[i]]).unwrap());
        v_trans.push(Mat::op("M*M", &[&R_t, &v[i]]).unwrap());
        let ft = calculate_F(&v_trans[i]);
        avg_F_trans += &ft;
        F_trans.push(ft);
    }
    avg_F_trans *= 1./(n_points as f64);

    let mut G = Mat::op("(M-M)^-1", &[&I3, &avg_F_trans]).unwrap();
    G *= 1./(n_points as f64);

    let M1 = Mat::create(3, 3, &[
             0., 0., 2.,
             0., 0., 0.,
            -2., 0., 0.]);
    let M2 = Mat::create(3, 3, &[
            -1., 0.,  0.,
             0., 1.,  0.,
             0., 0., -1.]);

    let (b0_, b1_, b2_) = {
        let mut b0 = Mat::zeroes(3, 1);
        let mut b1 = Mat::zeroes(3, 1);
        let mut b2 = Mat::zeroes(3, 1);
        for i in 0..n_points {
            let op_tmp1 = Mat::op("(M-M)MM", &[&F_trans[i], &I3, &R_gamma, &p_trans[i]]).unwrap();
            let op_tmp2 = Mat::op("(M-M)MMM", &[&F_trans[i], &I3, &R_gamma, &M1, &p_trans[i]]).unwrap();
            let op_tmp3 = Mat::op("(M-M)MMM", &[&F_trans[i], &I3, &R_gamma, &M2, &p_trans[i]]).unwrap();
    
            b0 += &op_tmp1;
            b1 += &op_tmp2;
            b2 += &op_tmp3;
        }
        let b0_ = G.matmul(&b0);
        let b1_ = G.matmul(&b1);
        let b2_ = G.matmul(&b2);
        (b0_, b1_, b2_)
    };

    let mut a0 = 0.;
    let mut a1 = 0.;
    let mut a2 = 0.;
    let mut a3 = 0.;
    let mut a4 = 0.;
    for i in 0..n_points {
        let c0 = Mat::op("(M-M)(MM+M)", &[&I3, &F_trans[i], &R_gamma, &p_trans[i], &b0_]).unwrap();
        let c1 = Mat::op("(M-M)(MMM+M)", &[&I3, &F_trans[i], &R_gamma, &M1, &p_trans[i], &b1_]).unwrap();
        let c2 = Mat::op("(M-M)(MMM+M)", &[&I3, &F_trans[i], &R_gamma, &M2, &p_trans[i], &b2_]).unwrap();

        a0 += Mat::op("M'M", &[&c0, &c0]).unwrap().as_scalar().unwrap();
        a1 += Mat::op("2M'M", &[&c0, &c1]).unwrap().as_scalar().unwrap();
        a2 += Mat::op("M'M+2M'M", &[&c1, &c1, &c0, &c2]).unwrap().as_scalar().unwrap();
        a3 += Mat::op("2M'M", &[&c1, &c2]).unwrap().as_scalar().unwrap();
        a4 += Mat::op("M'M", &[&c2, &c2]).unwrap().as_scalar().unwrap();
    }

    std::mem::drop(b0_);
    std::mem::drop(b1_);
    std::mem::drop(b2_);

    std::mem::drop(p_trans);
    std::mem::drop(v_trans);
    std::mem::drop(F_trans);
    std::mem::drop(avg_F_trans);
    std::mem::drop(G);


    // 4. Solve for minima of Eos.
    let minima = {
        let p0 = a1;
        let p1 = 2.*a2 - 4.*a0;
        let p2 = 3.*a3 - 3.*a1;
        let p3 = 4.*a4 - 2.*a2;
        let p4 = -a3;


        let poly = Poly::new(&[p0, p1, p2, p3, p4]);
        let roots = poly.solve_approx();

        let mut minima = Vec::with_capacity(4);
        for i in 0..roots.len() {
            let t1 = roots[i];
            let t2 = t1*t1;
            let t3 = t1*t2;
            let t4 = t1*t3;
            let t5 = t1*t4;
            // Check extrema is a minima.
            if a2 - 2.*a0 + (3.*a3 - 6.*a1)*t1 + (6.*a4 - 8.*a2 + 10.*a0)*t2 + (-8.*a3 + 6.*a1)*t3 + (-6.*a4 + 3.*a2)*t4 + a3*t5 >= 0. {
                // And that it corresponds to an angle different than the known minimum.
                let t = 2.*f64::atan(roots[i]);
                // We only care about finding a second local minima which is qualitatively
                // different than the first.
                if (t - t_initial).abs() > 0.1 {
                    minima.push(roots[i]);
                }
            }
        }
        minima
    };

    // 5. Get poses for minima.
    if minima.len() == 1 {
        let t = minima[0];
        let R_beta = M2.clone();
        R_beta *= t;
        R_beta += &M1;
        R_beta *= t;
        R_beta += &I3;
        R_beta *= 1./(1. + t*t);
        let res = Mat::op("M'MMM'", &[&R_t, &R_gamma, &R_beta, &R_z]).unwrap();
        Some(res)
    } else if minima.len() > 1  {
        // This can happen if our prior pose estimate was not very good.
        // panic!("Error, more than one new minima found.");
        None
    } else {
        //TODO: double check this is correct
        unreachable!()
    }
}
pub struct ApriltagDetectionInfo {
    detection: ApriltagDetection,
    tagsize: f64, // In meters.
    fx: f64, // In pixels.
    fy: f64, // In pixels.
    cx: f64, // In pixels.
    cy: f64, // In pixels.
}

pub struct ApriltagPose {
    pub R: Mat,
    pub t: Mat,
}

/// Estimate pose of the tag using the homography method.
fn estimate_pose_for_tag_homography(info: &ApriltagDetectionInfo) -> ApriltagPose {
    let scale = info.tagsize/2.0;

    let initial_pose = {
        let mut M_H = homography_to_pose(&info.detection.H, -info.fx, info.fy, info.cx, info.cy);
        M_H[(0, 3)] *= scale;
        M_H[(1, 3)] *= scale;
        M_H[(2, 3)] *= scale;

        let mut fix = Mat::zeroes(4, 4);
        fix[(0, 0)] =  1.;
        fix[(1, 1)] = -1.;
        fix[(2, 2)] = -1.;
        fix[(3, 3)] =  1.;

        fix.matmul(&M_H)
    };

    let R = Mat::zeroes(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            R[(i, j)] = initial_pose[(i, j)];
        }
    }

    let t = Mat::zeroes(3, 1);
    for i in 0..3 {
        t[(i, 0)] = initial_pose[(i, 3)];
    }
    
    ApriltagPose { R, t }
}

pub struct OrthogonalIteraionResult {
    solution1: (ApriltagPose, f64),
    solution2: Option<(ApriltagPose, f64)>,
}

/// Estimate tag pose using orthogonal iteration.
fn estimate_tag_pose_orthogonal_iteration(info: &ApriltagDetectionInfo, n_iters: usize) -> OrthogonalIteraionResult {
    let scale = info.tagsize/2.0;
    let p = [
        Mat::create(3, 1, &[-scale, scale, 0.]),
        Mat::create(3, 1, &[scale, scale, 0.]),
        Mat::create(3, 1, &[scale, -scale, 0.]),
        Mat::create(3, 1, &[-scale, -scale, 0.]),
    ];
    let v: [Mat; 4];
    for i in 0..4 {
        v[i] = Mat::create(3, 1, &[(info.detection.corners[i].x() - info.cx)/info.fx, (info.detection.corners[i].y() - info.cy)/info.fy, 1.]);
    }

    let mut pose1 = estimate_pose_for_tag_homography(info);
    let err1 = orthogonal_iteration(&v, &p, &mut pose1.t, &mut pose1.R, 4, n_iters);
    
    let solution2 = if let Some(R) = fix_pose_ambiguities(&v, &p, &mut pose1.t, &pose1.R, 4) {
        let t = Mat::zeroes(3, 1);
        let err2 = orthogonal_iteration(&v, &p, &mut t, &mut R, 4, n_iters);
        let solution2 = ApriltagPose {
            R,
            t,
        };
        Some((solution2, err2))
    } else {
        None
    };

    OrthogonalIteraionResult {
        solution1: (pose1, err1),
        solution2,
    }
}

/// Estimate tag pose.
pub fn estimate_tag_pose(info: &ApriltagDetectionInfo) -> (ApriltagPose, f64) {
    let ortho_res = estimate_tag_pose_orthogonal_iteration(info,50);
    let (pose1, err1) = ortho_res.solution1;

    if let Some((ref pose2, err2)) = ortho_res.solution2 {
        if err2 < err1 {
            return (*pose2, err2)
        }
    }
    return ortho_res.solution1;
}
