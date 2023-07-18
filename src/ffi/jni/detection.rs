//! Methods for AprilTagDetection.java

use std::sync::Arc;

use jni::{JNIEnv, objects::{JClass, JObject}, sys::{jint, jfloat, jlong}};

use crate::AprilTagDetection;

use super::{util::{JPtr, jni_wrap, jni_wrap_simple}, pose::new_mat33};

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_destroy<'local>(mut env: JNIEnv<'local>, this: JObject<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) {

}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetFamilyPointer<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> jlong {
    jni_wrap_simple(env, |env| {
        // We just need the pointer (to do a lookup), no need to get a reference
        let ptr = Arc::into_raw(ptr.as_ref()?.family);
        assert!(std::mem::size_of_val(&ptr) <= std::mem::size_of::<jlong>());
        Ok(unsafe { std::mem::transmute_copy::<_, usize>(&ptr) } as _)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetTagId<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> jint {
    jni_wrap_simple(env, |env| {
        let data = ptr.as_ref()?;
        Ok(data.id as _)
    })
}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetHammingDistance<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> jint {
    jni_wrap_simple(env, |env| {
        let data = ptr.as_ref()?;
        Ok(data.hamming as _)
    })
}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetDecisionMargin<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> jfloat {
    jni_wrap_simple(env, |env| {
        let data = ptr.as_ref()?;
        Ok(data.decision_margin)
    })
}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetHomogrophy<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> JObject<'local> {
    jni_wrap::<'local, Arc<AprilTagDetection>, _>(env, ptr, |env, data| {
        new_mat33(env, &data.H)
    })
}