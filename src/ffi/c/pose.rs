use crate::pose::{self, ApriltagDetectionInfo, ApriltagPose};

use super::{matd_t, apriltag_detection_t};

#[no_mangle]
pub unsafe extern "C" fn estimate_pose_for_tag_homography(info: *const apriltag_detection_info_t, pose: *mut apriltag_pose_t) {
    let info_raw = info.as_ref().unwrap();
    todo!()
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
    todo!()
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

impl From<&apriltag_detection_info_t> for ApriltagDetectionInfo {
    fn from(value: &apriltag_detection_info_t) -> Self {
        Self {
            detection: todo!(),
            tagsize: value.tagsize,
            fx: value.fx,
            fy: value.fy,
            cx: value.cx,
            cy: value.cy,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn estimate_tag_pose(info: *const apriltag_detection_info_t, pose: *mut apriltag_pose_t) -> f64 {
    let info_raw = info.as_ref().unwrap();
    let info = ApriltagDetectionInfo::from(info_raw);
    let (sol_pose, err) = pose::estimate_tag_pose(&info);
    if let Some(pose_out) = pose.as_mut() {
        *pose_out = sol_pose.into();
    }
    err
}