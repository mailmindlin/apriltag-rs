use jni::sys::jobject;

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagPoseEstimator_nativeEstimatePose<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ) -> jobject {
    todo!()
}