use std::alloc::AllocError;

use crate::pose::{self, AprilTagDetectionInfo, AprilTagPose, PoseParams};

use super::{matd_ptr, apriltag_detection_t, FFIConvertError, shim::{InPtr, OutPtr, cffi_wrapper, ReadPtr}};


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
            extrinsics: PoseParams {
                tagsize: value.tagsize,
                fx: value.fx,
                fy: value.fy,
                cx: value.cx,
                cy: value.cy,
            }
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn estimate_pose_for_tag_homography<'a>(info: InPtr<'a, apriltag_detection_info_t>, pose: OutPtr<'a, apriltag_pose_t>) {
    cffi_wrapper(|| {
        let value = info.try_read::<AprilTagDetectionInfo>("info")?;

        let res = pose::estimate_pose_for_tag_homography(&value.detection, &value.extrinsics);
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
        let info: AprilTagDetectionInfo = info.try_read("info")?;
        let result = pose::estimate_tag_pose_orthogonal_iteration(&info.detection, &info.extrinsics, nIters as usize);

        // Write back to out-params
        err1.maybe_write(result.solution1.error);
        pose1.maybe_try_write(result.solution1.pose)?;
        if let Some(solution2) = result.solution2 {
            err2.maybe_write(solution2.error);
            pose2.maybe_try_write(solution2.pose)?;
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

impl TryFrom<AprilTagPose> for apriltag_pose_t {
    type Error = AllocError;
    fn try_from(value: AprilTagPose) -> Result<Self, Self::Error> {
        let R = matd_ptr::new(3, 3, value.R.data())?;
        let t = matd_ptr::new(1, 3, &[value.t.0, value.t.1, value.t.2])?;
        Ok(Self { R, t })
    }
}

#[no_mangle]
pub unsafe extern "C" fn estimate_tag_pose<'a>(info: InPtr<'a, apriltag_detection_info_t>, pose: OutPtr<'a, apriltag_pose_t>) -> f64 {
    cffi_wrapper(|| {
        let info: AprilTagDetectionInfo = info.try_read("info")?;
        let solution = pose::estimate_tag_pose(&info.detection, &info.extrinsics);
        pose.maybe_try_write(solution.pose)?;
        Ok(solution.error)
    })
}