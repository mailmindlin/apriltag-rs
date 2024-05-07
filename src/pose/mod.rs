mod homography;

use std::ops::{Mul, Add, Sub};

use crate::{util::math::{mat::Mat33, poly::Poly, Vec3}, AprilTagDetection};

use self::homography::homography_to_pose;


/// Calculate projection operator from image points.
fn calculate_F(v: &Vec3) -> Mat33 {
	let mut outer_product = v.outer(v);
	let inner_product = v.dot(v);
	outer_product.scale_inplace(inner_product.recip());
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
fn orthogonal_iteration<const N: usize>(v: &[Vec3; N], p: &[Vec3; N], t: &mut Vec3, R: &mut Mat33, n_steps: usize) -> f64 {
	let p_res = {
		let p_mean = {
			let mut p_mean = Vec3::zero();
			for p_i in p.iter() {
				p_mean += p_i;
			}
			p_mean / (N as f64)
		};

		p.map(|p_i| p_i - &p_mean)
	};

	// Compute M1_inv.
	let (F, M1_inv) = {
		let mut F = Vec::with_capacity(N);
		let mut avg_F = Mat33::zeroes();
		// let mut avg_F = Mat::zeroes(3, 3);
		for v_i in v.iter() {
			let F_i = calculate_F(v_i);
			avg_F += &F_i;
			F.push(F_i);
		}
		avg_F *= (N as f64).recip();
		let M1 = Mat33::identity() - &avg_F;
		let M1_inv = M1.inv()
			.expect("M1 inverse");
		(F, M1_inv)
	};

	let mut prev_error = f64::INFINITY;
	// Iterate.
	for _ in 0..n_steps {
		// Calculate translation.
		*t = {
			const I3: Mat33 = Mat33::identity();
			let mut M2 = Vec3::zero();
			for j in 0..N {
				let M2_update = F[j].sub(&I3).matmul(R).mul(&p[j]);
				// let M2_update = Mat::op("(M - M)*M*M", &[&F[j], &I3, R, &p[j]]).unwrap();
				M2 += &M2_update;
			}
			M2 *= (N as f64).recip();
			M1_inv.mul(&M2)
		};

		// Calculate rotation.
		*R = {
			let M3 = {
				let mut q = Vec::with_capacity(N);
				let mut q_mean = Vec3::zero();
				for j in 0..N {
					let q_j = F[j].mul(&R.mul(&p[j]).add(&*t));
					// q[j] = Mat::op("M*(M*M+M)", &[&F[j], R, &p[j], &t]).unwrap();
					q_mean += q_j;
					q.push(q_j);
				}
				q_mean *= (N as f64).recip();
				
				let mut M3 = Mat33::zeroes();
				for j in 0..N {
					// let M3_update = Mat::op("(M-M)*M'", &[&q[j], &q_mean, &p_res[j]]).unwrap();
					let M3_update = (&q[j] - &q_mean).outer(&p_res[j]);
					M3 += &M3_update;
				}
				M3
			};
			
			let M3_svd = M3.svd();
			// Mat::op("M*M'", &[&M3_svd.U, &M3_svd.V]).unwrap()
			M3_svd.U.matmul_transpose(&M3_svd.V)
		};

		let mut error = 0.;
		for i in 0..4 {
			let err_vec = {
				let a = Mat33::identity() - &F[i];
				let b = R.mul(&p[i]).add(*t);
				&a * &b
			};
			// let err_vec = Mat::op("(M-M)(MM+M)", &[&I3, &F[i], R, &p[i], t]).unwrap();
			error += err_vec.dot(&err_vec);
			// error += Mat::op("M'M", &[&err_vec, &err_vec])
			//     .unwrap()
			//     .as_scalar()
			//     .unwrap();
		}
		prev_error = error;
	}
	
	prev_error
}

/// Given a local minima of the pose error tries to find the other minima.
fn fix_pose_ambiguities<const N: usize>(v: &[Vec3; N], p: &[Vec3; N], t: &Vec3, R: &Mat33) -> Option<Mat33> {
	// 1. Find R_t
	let R_t = {
		let R_t_3 = t.normalized();

		let R_t_1 = {
			let e_x = Vec3::of(1., 0., 0.);

			// let R_t_1_tmp = Mat::op("M-(M'*M)*M", &[&e_x, &e_x, &R_t_3, &R_t_3]).unwrap();
			let R_t_1_tmp = e_x.sub(e_x.outer(&R_t_3).mul(&R_t_3));

			R_t_1_tmp.normalized()
		};

		let R_t_2 = R_t_3.cross(&R_t_1);

		Mat33::of([
			R_t_1.0, R_t_1.1, R_t_1.2,
			R_t_2.0, R_t_2.1, R_t_2.2,
			R_t_3.0, R_t_3.1, R_t_3.2,
		])
	};

	// 2. Find R_z
	let R_1_prime = R_t.matmul(R);
	let R_z = {
		let r31 = R_1_prime[(2, 0)];
		let r32 = R_1_prime[(2, 1)];
		let hypotenuse = f64::hypot(r31, r32);
		let (u, v) = if hypotenuse < 1e-100 {
			(1., 0.)
			// r31 = 1.;
			// r32 = 0.;
			// hypotenuse = 1.;
		} else {
			(r31 / hypotenuse, r32 / hypotenuse)
		};
		Mat33::of([
			u, -v, 0.,
			v,  u, 0.,
			0., 0., 1.
		])
	};

	// 3. Calculate parameters of Eos
	let (R_gamma, t_initial) = {
		let R_trans = R_1_prime.matmul(&R_z);
		let sin_gamma = -R_trans[(0, 1)];
		let cos_gamma = R_trans[(1, 1)];
		let R_gamma = Mat33::of([
			cos_gamma, -sin_gamma, 0.,
			sin_gamma, cos_gamma, 0.,
			0., 0., 1.
		]);
	
		let sin_beta = -R_trans[(2, 0)];
		let cos_beta = R_trans[(2, 2)];
		let t_initial = f64::atan2(sin_beta, cos_beta);
		(R_gamma, t_initial)
	};

	const M1: Mat33 = Mat33::of([
		0., 0., 2.,
		0., 0., 0.,
		-2., 0., 0.]);
	const M2: Mat33 = Mat33::of([
		-1., 0.,  0.,
		0., 1.,  0.,
		0., 0., -1.]);

	let (a0, a1, a2, a3, a4) = {
		let mut Fp_trans = Vec::with_capacity(N);
		let mut avg_F_trans = Mat33::zeroes();
		for i in 0..N {
			// let pt_i = Mat::op("M'*M", &[&R_z, &p[i]]).unwrap();
			// let vt_i = Mat::op("M*M", &[&R_t, &v[i]]).unwrap();
			let pt_i = R_z.transposed().mul(&p[i]);
			let vt_i = R_t.mul(&v[i]);
			let ft = calculate_F(&vt_i);
			avg_F_trans += &ft;
			Fp_trans.push((ft, pt_i));
		}
		avg_F_trans *= (N as f64).recip();

		// let mut G = Mat::op("(M-M)^-1", &[&I3, &avg_F_trans]).unwrap();
		// G *= 1./(n_points as f64);
		let G = (Mat33::identity() - &avg_F_trans)
			.inv().expect("G inverse")
			.scale((N as f64).recip());

		let (b0_, b1_, b2_) = {
			let mut b0 = Vec3::zero();
			let mut b1 = Vec3::zero();
			let mut b2 = Vec3::zero();
			for (ft_i, pt_i) in Fp_trans.iter() {
				let op_tmp0 = (ft_i - &Mat33::identity()).matmul(&R_gamma);

				let op_tmp1 = op_tmp0.mul(pt_i);
				let op_tmp2 = op_tmp0.matmul(&M1).mul(pt_i);
				let op_tmp3 = op_tmp0.matmul(&M2).mul(pt_i);

				// let op_tmp1 = Mat::op("(M-M)MM", &[&F_trans[i], &I3, &R_gamma, &p_trans[i]]).unwrap();
				// let op_tmp2 = Mat::op("(M-M)MMM", &[&F_trans[i], &I3, &R_gamma, &M1, &p_trans[i]]).unwrap();
				// let op_tmp3 = Mat::op("(M-M)MMM", &[&F_trans[i], &I3, &R_gamma, &M2, &p_trans[i]]).unwrap();
		
				b0 += &op_tmp1;
				b1 += &op_tmp2;
				b2 += &op_tmp3;
			}
			let b0_ = G.mul(&b0);
			let b1_ = G.mul(&b1);
			let b2_ = G.mul(&b2);
			(b0_, b1_, b2_)
		};

		let mut a0 = 0.;
		let mut a1 = 0.;
		let mut a2 = 0.;
		let mut a3 = 0.;
		let mut a4 = 0.;
		for (Ft_i, Pt_i) in Fp_trans.into_iter() {
			let tmp0 = Mat33::identity() - &Ft_i;
			let c0 = &tmp0 * &(&R_gamma * &Pt_i + &b0_);
			let c1 = &tmp0 * &(&(&R_gamma.matmul(&M1) * &Pt_i + &b1_));
			let c2 = &tmp0 * &(&(&R_gamma.matmul(&M2) * &Pt_i + &b2_));
			// let c0 = Mat::op("(M-M)(MM+M)", &[&I3, &F_trans[i], &R_gamma, &p_trans[i], &b0_]).unwrap();
			// let c1 = Mat::op("(M-M)(MMM+M)", &[&I3, &F_trans[i], &R_gamma, &M1, &p_trans[i], &b1_]).unwrap();
			// let c2 = Mat::op("(M-M)(MMM+M)", &[&I3, &F_trans[i], &R_gamma, &M2, &p_trans[i], &b2_]).unwrap();

			a0 += c0.mag_sq();
			a1 += 2. * c0.dot(&c1);
			a2 += c1.mag_sq() + 2. * c0.dot(&c2);
			a3 += 2.* c1.dot(&c2);
			a4 += c2.mag_sq();
			// a0 += Mat::op("M'M", &[&c0, &c0]).unwrap().as_scalar().unwrap();
			// a1 += Mat::op("2M'M", &[&c0, &c1]).unwrap().as_scalar().unwrap();
			// a2 += Mat::op("M'M+2M'M", &[&c1, &c1, &c0, &c2]).unwrap().as_scalar().unwrap();
			// a3 += Mat::op("2M'M", &[&c1, &c2]).unwrap().as_scalar().unwrap();
			// a4 += Mat::op("M'M", &[&c2, &c2]).unwrap().as_scalar().unwrap();
		}
		(a0, a1, a2, a3, a4)
	};


	// 4. Solve for minima of Eos.
	let minima = {
		let roots = {
			let p0 = a1;
			let p1 = 2.*a2 - 4.*a0;
			let p2 = 3.*a3 - 3.*a1;
			let p3 = 4.*a4 - 2.*a2;
			let p4 = -a3;
	
	
			let poly = Poly::new(&[p0, p1, p2, p3, p4]);
			poly.solve_approx()
		};

		let mut minima = Vec::with_capacity(4);
		for root in roots.into_iter() {
			let t1 = root;
			let t2 = t1*t1;
			let t3 = t1*t2;
			let t4 = t1*t3;
			let t5 = t1*t4;
			// Check extrema is a minima.
			if a2 - 2.*a0 + (3.*a3 - 6.*a1)*t1 + (6.*a4 - 8.*a2 + 10.*a0)*t2 + (-8.*a3 + 6.*a1)*t3 + (-6.*a4 + 3.*a2)*t4 + a3*t5 >= 0. {
				// And that it corresponds to an angle different than the known minimum.
				let t = 2.*f64::atan(root);
				// We only care about finding a second local minima which is qualitatively
				// different than the first.
				if (t - t_initial).abs() > 0.1 {
					minima.push(root);
				}
			}
		}
		minima
	};

	// 5. Get poses for minima.
	if minima.len() == 1 {
		let R_beta = {
			let t_cur = minima[0];
			let mut R_beta = M2.clone();
			R_beta.scale_inplace(t_cur);
			R_beta += &M1;
			R_beta.scale_inplace(t_cur);
			R_beta += &Mat33::identity();
			R_beta *= (1. + t_cur*t_cur).recip();
			R_beta
		};

		let res = R_t
			.transpose_matmul(&R_gamma)
			.matmul(&R_beta)
			.matmul_transpose(&R_z);
		// let res = Mat::op("M'MMM'", &[&R_t, &R_gamma, &R_beta, &R_z]).unwrap();
		Some(res)
	} else if minima.len() > 1  {
		// This can happen if our prior pose estimate was not very good.
		#[cfg(debug_assertions)]
		eprintln!("Error, more than one new minima found.");
		None
	} else {
		//TODO: double check this is correct
		// unreachable!("No minima")
		None
	}
}

pub struct PoseParams {
	/// Tag size (meters?)
	pub tagsize: f64,
	pub fx: f64, // In pixels.
	pub fy: f64, // In pixels.
	pub cx: f64, // In pixels.
	pub cy: f64, // In pixels.
}

pub struct AprilTagDetectionInfo {
	/// Previous detection
	pub detection: AprilTagDetection,
	pub extrinsics: PoseParams,
}

#[derive(Copy, Clone, Debug)]
pub struct AprilTagPose {
	/// Rotation matrix
	pub R: Mat33,
	/// Translation vector
	pub t: Vec3,
}

#[derive(Copy, Clone, Debug)]
pub struct AprilTagPoseWithError {
	pub pose: AprilTagPose,
	pub error: f64,
}

/// Estimate pose of the tag using the homography method.
pub fn estimate_pose_for_tag_homography(detection: &AprilTagDetection, params: &PoseParams) -> AprilTagPose {
	let scale = params.tagsize/2.0;

	let (R, t) = {
		let (R, mut t) = homography_to_pose(&detection.H, -params.fx, params.fy, params.cx, params.cy);
		t.0 *= scale;
		t.1 *= -scale;
		t.2 *= -scale;
		// M_H[(0, 3)] *= scale;
		// M_H[(1, 3)] *= scale;
		// M_H[(2, 3)] *= scale;

		const FIX: Mat33 = Mat33::of([
			1.,  0.,  0.,
			0., -1.,  0.,
			0.,  0., -1.,
		]);
		// let fix = Mat::create(4, 4, &[
		//     1.,  0.,  0.,  0.,
		//     0., -1.,  0.,  0.,
		//     0.,  0., -1.,  0.,
		//     0.,  0.,  0.,  1.,
		// ]);

		(FIX.matmul(&R), t)
	};

	AprilTagPose { R, t }
}

#[derive(Copy, Clone, Debug)]
pub struct OrthogonalIterationResult {
	/// Best pose solution
	pub solution1: AprilTagPoseWithError,
	/// Second-best pose solution
	pub solution2: Option<AprilTagPoseWithError>,
}

/// Estimate tag pose using orthogonal iteration.
pub fn estimate_tag_pose_orthogonal_iteration(detection: &AprilTagDetection, params: &PoseParams, n_iters: usize) -> OrthogonalIterationResult {
	let p = {
		let scale = params.tagsize/2.0;
		[
			Vec3::of(-scale,  scale, 0.),
			Vec3::of( scale,  scale, 0.),
			Vec3::of( scale, -scale, 0.),
			Vec3::of(-scale, -scale, 0.),
		]
	};
	let v = detection.corners
		.as_array()
		.map(|[x, y]| Vec3::of((x - params.cx)/params.fx, (y - params.cy)/params.fy, 1.));

	let solution1 = {
		let mut pose1 = estimate_pose_for_tag_homography(detection, params);
		let err1 = orthogonal_iteration(&v, &p, &mut pose1.t, &mut pose1.R, n_iters);
	
		AprilTagPoseWithError {
			pose: pose1,
			error: err1,
		}
	};
	let solution2 = match fix_pose_ambiguities(&v, &p, &solution1.pose.t, &solution1.pose.R) {
		Some(mut R) => {
			let mut t = Vec3::zero();
			let err2 = orthogonal_iteration(&v, &p, &mut t, &mut R, n_iters);
			Some(AprilTagPoseWithError {
				pose: AprilTagPose { R, t },
				error: err2,
			})
		},
		None => None,
	};

	OrthogonalIterationResult {
		solution1,
		solution2,
	}
}

/// Estimate tag pose.
pub fn estimate_tag_pose(detection: &AprilTagDetection, params: &PoseParams) -> AprilTagPoseWithError {
	let OrthogonalIterationResult {
		solution1,
		solution2
	} = estimate_tag_pose_orthogonal_iteration(detection, params, 50);

	if let Some(solution2) = solution2 {
		if solution2.error < solution1.error {
			return solution2;
		}
	}
	solution1
}
