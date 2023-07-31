use jni::{sys::{jdouble, jvalue}, JNIEnv, objects::{JClass, JValue, JObject, JIntArray, JMethodID, JObjectArray}, descriptors::Desc};

use crate::{AprilTagDetectionInfo, ffi::jni::util::jni_wrap, estimate_tag_pose, util::{math::mat::Mat33, mem::calloc}, Detections, pose::PoseWithError};

use super::{util::{JavaError, JPtr}, CLASS_POSE};

pub(super) fn new_mat33<'a>(env: &mut JNIEnv<'a>, native: &Mat33) -> Result<JObject<'a>, JavaError> {
    let res = env.new_object("com/mindlin/apriltagrs/util/Mat33", "(DDDDDDDDD)V", &[
        JValue::Double(native.0[0]),
        JValue::Double(native.0[1]),
        JValue::Double(native.0[2]),
        JValue::Double(native.0[3]),
        JValue::Double(native.0[4]),
        JValue::Double(native.0[5]),
        JValue::Double(native.0[6]),
        JValue::Double(native.0[7]),
        JValue::Double(native.0[8]),
    ])?;
    Ok(res)
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagPoseEstimator_nativeEstimatePoses<'local>(env: JNIEnv<'local>, _class: JClass<'local>, tagsize: jdouble, fx: jdouble, fy: jdouble, cx: jdouble, cy: jdouble, detections: JPtr<'local, Box<Detections>>, idxs: JIntArray<'local>) -> JObjectArray<'local> {
    jni_wrap(env, detections, |env, dets| {
        let pose_class = Desc::<JClass>::lookup(CLASS_POSE, env)?;
        let pose_ctor = Desc::<JMethodID>::lookup((&pose_class, "(DDDDDDDDDDDDD)V"), env)?;

        let (len, idxs) = if idxs.is_null() {
            // We want the poses of all of them
            (dets.detections.len(), None)
        } else {
            let idxs_len = env.get_array_length(&idxs)?;
            let mut idxs_buf = calloc(idxs_len as _);
            env.get_int_array_region(&idxs, 0, &mut idxs_buf)?;
            env.delete_local_ref(idxs)?;
            (idxs_buf.len(), Some(idxs_buf))
        };

        let result_arr = env.new_object_array(len as _, format!("[L{CLASS_POSE};"), JObject::null())?;

        let mut estimate_pose = |index, dst_idx| -> Result<_, JavaError> {
            let detection: &crate::AprilTagDetection = &dets.detections[index];
            let info = AprilTagDetectionInfo {
                //TODO: avoid copying?
                detection: detection.clone(),
                tagsize,
                fx,
                fy,
                cx,
                cy,
            };
            let PoseWithError {
                pose,
                error,
            } = estimate_tag_pose(&info);
            let element = unsafe { env.new_object_unchecked(&pose_class, pose_ctor, &[
                jvalue { d: pose.R.0[0] },
                jvalue { d: pose.R.0[1] },
                jvalue { d: pose.R.0[2] },
                jvalue { d: pose.R.0[3] },
                jvalue { d: pose.R.0[4] },
                jvalue { d: pose.R.0[5] },
                jvalue { d: pose.R.0[6] },
                jvalue { d: pose.R.0[7] },
                jvalue { d: pose.t.0 },
                jvalue { d: pose.t.1 },
                jvalue { d: pose.t.2 },
                jvalue { d: error },
            ]) }?;
            // Ok(env.auto_local(element))
            env.set_object_array_element(&result_arr, dst_idx as _, element)?;
            Ok(())
        };

        match idxs {
            Some(idxs) => {
                for (i, idx) in idxs.into_iter().enumerate() {
                    let idx = *idx;
                    if idx < 0 || dets.detections.len() <= idx as usize {
                        return Err(JavaError::IllegalArgumentException(format!("Invalid index (actual: {idx}; expected: 0..{len}")));
                    }
                    estimate_pose(idx as _, i)?;
                }
            },
            None => {
                for i in 0..len {
                    estimate_pose(i, i)?;
                }
            }
        }
        Ok(result_arr)
    })
}