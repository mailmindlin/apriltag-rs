use std::alloc::AllocError;

use crate::pose::{self, ApriltagDetectionInfo, ApriltagPose};

use super::{matd_ptr, apriltag_detection_t, FFIConvertError, shim::{InPtr, OutPtr, cffi_wrapper, ReadPtr}};

#[no_mangle]
pub unsafe extern "C" fn estimate_pose_for_tag_homography<'a>(info: InPtr<'a, apriltag_detection_info_t>, pose: OutPtr<'a, apriltag_pose_t>) {
    cffi_wrapper(|| {
        let info = info.try_read("info")?;

        let res = pose::estimate_pose_for_tag_homography(&info);
        drop(info);

        pose.maybe_try_write(res)?;

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn estimate_tag_pose_orthogonal_iteration<'a>(
    info: InPtr<'a, apriltag_detection_info_t>,
    err1: OutPtr<'a, libc::c_double>,
    pose1: OutPtr<'a, apriltag_pose_t>,
    err2: OutPtr<'a, libc::c_double>,
    pose2: OutPtr<'a, apriltag_pose_t>,
    nIters: libc::c_int,
) {
    cffi_wrapper(|| {
        let info = info.try_read("info")?;
        let result = pose::estimate_tag_pose_orthogonal_iteration(&info, nIters as usize);

        // Write back to out-params
        err1.maybe_write(result.solution1.1);
        pose1.maybe_try_write(result.solution1.0)?;
        if let Some((pose2_src, err2_src)) = result.solution2 {
            err2.maybe_write(err2_src);
            pose2.maybe_try_write(pose2_src)?;
        } else {
            err2.maybe_write(f64::INFINITY);
        }
        Ok(())
    })
}

#[repr(C)]
pub struct apriltag_pose_t {
    R: matd_ptr,
    t: matd_ptr,
}

impl TryFrom<ApriltagPose> for apriltag_pose_t {
    type Error = AllocError;
    fn try_from(value: ApriltagPose) -> Result<Self, Self::Error> {
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

impl TryFrom<*const apriltag_detection_info_t> for ApriltagDetectionInfo {
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
pub unsafe extern "C" fn estimate_tag_pose<'a>(info: InPtr<'a, apriltag_detection_info_t>, pose: OutPtr<'a, apriltag_pose_t>) -> f64 {
    cffi_wrapper(|| {
        let info = info.try_read("info")?;
        let (sol_pose, err) = pose::estimate_tag_pose(&info);
        pose.maybe_try_write(sol_pose)?;
        Ok(err)
    })
}