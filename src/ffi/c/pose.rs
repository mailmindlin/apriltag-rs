use crate::pose::{self, ApriltagDetectionInfo, ApriltagPose};

use super::{matd_t, apriltag_detection_t, FFIConvertError};

#[no_mangle]
pub unsafe extern "C" fn estimate_pose_for_tag_homography(info: *const apriltag_detection_info_t, pose: *mut apriltag_pose_t) {
    let info = ApriltagDetectionInfo::try_from(info).unwrap();

    let res = pose::estimate_pose_for_tag_homography(&info);

    if let Some(pose) = pose.as_mut() {
        *pose = res.into();
    }
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
    let info = ApriltagDetectionInfo::try_from(info).unwrap();
    let result = pose::estimate_tag_pose_orthogonal_iteration(&info, nIters as usize);

    // Write back to out-params
    if let Some(err1) = err1.as_mut() {
        *err1 = result.solution1.1;
    }
    if let Some(pose1) = pose1.as_mut() {
        *pose1 = result.solution1.0.into();
    }
    if let Some((pose2_src, err2_src)) = result.solution2 {
        if let Some(err2_dst) = err2.as_mut() {
            *err2_dst = err2_src;
        }
        if let Some(pose2_dst) = pose2.as_mut() {
            *pose2_dst = pose2_src.into();
        }
    } else {
        if let Some(err2_dst) = err2.as_mut() {
            *err2_dst = f64::INFINITY;
        }
    }
}

#[repr(C)]
pub struct apriltag_pose_t {
    R: *mut matd_t,
    t: *mut matd_t,
}

impl From<ApriltagPose> for apriltag_pose_t {
    fn from(value: ApriltagPose) -> Self {
        Self {
            R: matd_t::convert(&value.R),
            t: matd_t::convert(&value.t),
        }
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

impl TryFrom<*const apriltag_detection_info_t> for ApriltagDetectionInfo {
    type Error = FFIConvertError;
    fn try_from(value: *const apriltag_detection_info_t) -> Result<Self, Self::Error> {
        let value = unsafe { value.as_ref() }.ok_or(FFIConvertError::NullPointer)?;
        Ok(Self {
            detection: value.det.try_into()?,
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
    let info = ApriltagDetectionInfo::try_from(info).unwrap();
    let (sol_pose, err) = pose::estimate_tag_pose(&info);
    if let Some(pose_out) = pose.as_mut() {
        *pose_out = sol_pose.into();
    }
    err
}