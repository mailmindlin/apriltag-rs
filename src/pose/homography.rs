use crate::util::math::{mat::Mat33, Vec3};

/// assuming that the projection matrix is:
/// [ fx 0  cx 0 ]
/// [  0 fy cy 0 ]
/// [  0  0  1 0 ]
///
/// And that the homography is equal to the projection matrix times the
/// model matrix, recover the model matrix (which is returned). Note
/// that the third column of the model matrix is missing in the
/// expresison below, reflecting the fact that the homography assumes
/// all points are at z=0 (i.e., planar) and that the element of z is
/// thus omitted.  (3x1 instead of 4x1).
///
/// [ fx 0  cx 0 ] [ R00  R01  TX ]    [ H00 H01 H02 ]
/// [  0 fy cy 0 ] [ R10  R11  TY ] =  [ H10 H11 H12 ]
/// [  0  0  1 0 ] [ R20  R21  TZ ] =  [ H20 H21 H22 ]
///                [  0    0    1 ]
///
/// fx*R00 + cx*R20 = H00   (note, H only known up to scale; some additional adjustments required; see code.)
/// fx*R01 + cx*R21 = H01
/// fx*TX  + cx*TZ  = H02
/// fy*R10 + cy*R20 = H10
/// fy*R11 + cy*R21 = H11
/// fy*TY  + cy*TZ  = H12
/// R20 = H20
/// R21 = H21
/// TZ  = H22
pub fn homography_to_pose(H: &Mat33, fx: f64, fy: f64, cx: f64, cy: f64) -> (Mat33, Vec3) {
	// Note that every variable that we compute is proportional to the scale factor of H.
	let mut R20 = H[(2, 0)];
	let mut R21 = H[(2, 1)];
	let mut TZ  = H[(2, 2)];
	let mut R00 = (H[(0, 0)] - cx*R20) / fx;
	let mut R01 = (H[(0, 1)] - cx*R21) / fx;
	let mut TX  = (H[(0, 2)] - cx*TZ)  / fx;
	let mut R10 = (H[(1, 0)] - cy*R20) / fy;
	let mut R11 = (H[(1, 1)] - cy*R21) / fy;
	let mut TY  = (H[(1, 2)] - cy*TZ)  / fy;

	let s = {
		// compute the scale by requiring that the rotation columns are unit length
		// (Use geometric average of the two length vectors we have)
		let length1 = f64::sqrt(R00*R00 + R10*R10 + R20*R20);
		let length2 = f64::sqrt(R01*R01 + R11*R11 + R21*R21);
		let s = f64::sqrt((length1 * length2) as f64).recip();
	
		// get sign of S by requiring the tag to be in front the camera;
		// we assume camera looks in the -Z direction.
		if TZ > 0. { -s } else { s }
	};

	R20 *= s;
	R21 *= s;
	TZ  *= s;
	R00 *= s;
	R01 *= s;
	TX  *= s;
	R10 *= s;
	R11 *= s;
	TY  *= s;

	// now recover [R02 R12 R22] by noting that it is the cross product of the other two columns.
	let R02 = R10*R21 - R20*R11;
	let R12 = R20*R01 - R00*R21;
	let R22 = R00*R11 - R10*R01;

	let R = Mat33::of([
		R00, R01, R02,
		R10, R11, R12,
		R20, R21, R22
	]);

	let t = Vec3::of(TX, TY, TZ);

	// Improve rotation matrix by applying polar decomposition.
	let R = if true {
		// do polar decomposition. This makes the rotation matrix
		// "proper", but probably increases the reprojection error. An
		// iterative alignment step would be superior.

		let svd = R.svd();
		svd.U.matmul_transpose(&svd.V)
	} else {
		R
	};

	(R, t)
}