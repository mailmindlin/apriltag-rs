use std::{alloc::AllocError, fmt::Debug};

use crate::pose::{self, AprilTagDetectionInfo, AprilTagPose, PoseParams};

use super::{matd_ptr, apriltag_detection_t, FFIConvertError};

unsafe fn try_write<T, V: TryInto<T>>(ptr: *mut T, v: V) where <V as TryInto<T>>::Error: Debug {
    if let Some(dst) = ptr.as_mut() {
        match v.try_into() {
            Ok(v) => {
                *dst = v;
            },
            Err(e) => {
                eprintln!("{e:?}");
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn estimate_pose_for_tag_homography(info: *const apriltag_detection_info_t, pose: *mut apriltag_pose_t) {
    let info = AprilTagDetectionInfo::try_from(info).unwrap();

    let res = pose::estimate_pose_for_tag_homography(&info);

    try_write(pose, res);
}

#[no_mangle]
pub unsafe extern "C" fn estimate_tag_pose_orthogonal_iteration(
    info: *const apriltag_detection_info_t,
    err1: *mut libc::c_double,
    pose1: *mut apriltag_pose_t,
    err2: *mut libc::c_double,
    pose2: *mut apriltag_pose_t,
    nIters: libc::c_int,
) {
    let result = pose::estimate_tag_pose_orthogonal_iteration(&info, nIters as usize);
    let info = AprilTagDetectionInfo::try_from(info).unwrap();

    // Write back to out-params
    if let Some(err1) = err1.as_mut() {
        *err1 = result.solution1.1;
    }
    try_write(pose1, result.solution1.0);
    if let Some((pose2_src, err2_src)) = result.solution2 {
        if let Some(err2_dst) = err2.as_mut() {
            *err2_dst = err2_src;
        }
        try_write(pose2, pose2_src);
    } else {
        if let Some(err2_dst) = err2.as_mut() {
            *err2_dst = f64::INFINITY;
        }
    }
}

#[repr(C)]
pub struct apriltag_pose_t {
    R: matd_ptr,
    t: matd_ptr,
}

impl TryFrom<AprilTagPose> for apriltag_pose_t {
    type Error = AllocError;
    fn try_from(value: AprilTagPose) -> Result<Self, Self::Error> {
        let R = matd_ptr::new(3, 3, value.R.data())?;
        let t = matd_ptr::new(1, 3, &[value.t.0, value.t.1, value.t.2])?;
        Ok(Self { R, t })
    }
}

#[repr(C)]
pub struct apriltag_detection_info_t {
    det: *const apriltag_detection_t,
    tagsize: libc::c_double, // In meters.
    fx: libc::c_double, // In pixels.
    fy: libc::c_double, // In pixels.
    cx: libc::c_double, // In pixels.
    cy: libc::c_double, // In pixels.
}

impl TryFrom<*const apriltag_detection_info_t> for AprilTagDetectionInfo {
    type Error = FFIConvertError;
    fn try_from(value: *const apriltag_detection_info_t) -> Result<Self, Self::Error> {
        let value = unsafe { value.as_ref() }.ok_or(FFIConvertError::NullPointer)?;
        let detection = unsafe { value.det.as_ref() }.ok_or(FFIConvertError::NullPointer)?;
        Ok(Self {
            detection: detection.try_into()?,
            tagsize: value.tagsize,
            fx: value.fx,
            fy: value.fy,
            cx: value.cx,
            cy: value.cy,
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn estimate_tag_pose(info: *const apriltag_detection_info_t, pose: *mut apriltag_pose_t) -> f64 {
    let info = match AprilTagDetectionInfo::try_from(info) {
        Ok(info) => info,
        Err(e) => {
            #[cfg(debug_assertions)]
            eprintln!("Error: called estimate_tag_pose with null info: {e:?}");
            return f64::NAN;
        }
    };
    let solution = pose::estimate_tag_pose(&info.detection, &info.extrinsics);
    try_write(pose, solution.pose);
    solution.error
}