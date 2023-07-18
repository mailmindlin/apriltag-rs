//! Methods for AprilTagDetection.java

use std::sync::Arc;

use jni::{JNIEnv, objects::JClass, sys::{jint, jfloat}};

use crate::{AprilTagFamily, AprilTagDetection};

use super::util::{JPtr, jni_wrap};

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetectio_destroy<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, nthreads: jint, quadDecimate: jfloat, quadSigma: jfloat, refineEdges: jboolean, decodeSharpening: jdouble, debugPath: JString<'local>, familyPtrs: JLongArray<'local>, bitsCorrecteds: JIntArray<'local>) -> JPtr<'static, Data> {

}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetFamilyPointer<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> JPtr<'static, Arc<AprilTagFamily>>{

}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetTagId<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> jint {
    jni_wrap(&mut env, ptr, |det| {
        det.id as _
    })
}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetHammingDistance<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> jint {
    jni_wrap(&mut env, ptr, |det| {
        det.hamming_distance
    })
}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetDecisionMargin<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> jfloat {

}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetHomogrophy<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Arc<AprilTagDetection>>) -> JPtr<'static, Arc<AprilTagFamily>> {

}