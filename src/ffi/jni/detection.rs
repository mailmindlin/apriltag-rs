//! Methods for AprilTagDetection.java

use std::sync::Arc;

use jni::{JNIEnv, objects::{JClass, JObject, JValue}, sys::{jint, jfloat}};

use crate::{AprilTagDetection, TimeProfile, Detections, AprilTagFamily};

use super::{util::{jni_wrap, jni_wrap_simple, JavaError, JPtr, JPtrManaged}, pose::new_mat33, CLASS_TIMEPROFILE};

/// Make TimeProfile Java object from rust
fn make_timeprofile<'l>(env: &mut JNIEnv<'l>, tp: &TimeProfile) -> Result<JObject<'l>, JavaError> {
    let (names_arr, timestamps_arr) = {
        let num_entries = tp.entries().len();

        // String[] names
        let mut names_arr = {
            let arr = env.new_object_array(num_entries as _, "Ljava/lang/String;", JObject::null())?;
            env.auto_local(arr)
        };
        // long[] timestamps
        let mut timestamps_arr = env.auto_local(env.new_long_array(num_entries as i32 * 2)?);
        
        let mut next_write = 0;
        const BUFFER_SIZE: usize = 64;
        let mut timestamps_buffer = Vec::new();
        if let Err(e) = timestamps_buffer.try_reserve(BUFFER_SIZE) {
            return Err(JavaError::OutOfMemoryError(format!("Unable to allocate buffer: {e}")));
        }

        for (idx, entry) in tp.entries().iter().enumerate() {
            // Put name into array
            {
                let name = env.auto_local(env.new_string(entry.name())?);
                env.set_object_array_element(&mut names_arr, idx as _, name)?;
            }

            // Put duration into array (u128 -> 2x u64)
            {
                let rel_stamp = entry.timestamp().duration_since(*tp.start());
                timestamps_buffer.push(rel_stamp.as_secs() as i64);
                let nanos = rel_stamp.subsec_nanos() as u64;
                timestamps_buffer.push(nanos as i64);
            }

            // Flush duration buffer
            if timestamps_buffer.len() >= BUFFER_SIZE {
                env.set_long_array_region(&mut timestamps_arr, next_write, &timestamps_buffer)?;
                next_write += timestamps_buffer.len() as i32;
                timestamps_buffer.clear();
            }
        }
        // Flush
        if timestamps_buffer.len() > 0 {
            env.set_long_array_region(&mut timestamps_arr, next_write, &timestamps_buffer)?;
            next_write += timestamps_buffer.len() as i32;
            timestamps_buffer.clear();
        }
        assert_eq!(next_write, tp.entries().len() as i32 * 2);

        (names_arr, timestamps_arr)
    };

    // var tp = new TimeProfile(String[] names, long[] timestamps);
    let tp = env.new_object(CLASS_TIMEPROFILE, "([Ljava/lang/String;[J)V", &[
        JValue::Object(&names_arr),
        JValue::Object(&timestamps_arr),
    ])?;
    Ok(tp)
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetections_nativeGetTimeProfile<'local>(env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Box<Detections>>) -> JObject<'local> {
    jni_wrap(env, ptr, |env, dets| {
        make_timeprofile(env, &dets.tp)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetections_destroy<'local>(env: JNIEnv<'local>, _this: JObject<'local>, ptr: JPtrManaged<'local, Box<Detections>>) {
    jni_wrap(env, ptr, |_env, data| {
        drop(data);
        Ok(())
    })
}

/// Safely get the detection at the index
fn get_detection<'a>(ptr: JPtr<'a, Box<Detections>>, index: jint) -> Result<&'a AprilTagDetection, JavaError> {
    let dets = ptr.as_ref()?;
    if index < 0 || dets.detections.len() < (index as usize) {
        Err(JavaError::IllegalArgumentException(format!("Invalid index (actual: {index}; expected: 0..{}", dets.detections.len())))
    } else {
        Ok(&dets.detections[index as usize])
    }
}

/// Get a pointer to the AprilTagFamily for the given index
/// 
/// @parameter ptr (Locked) detections
/// @parameter index Detection index
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetections_nativeGetFamilyPointer<'local>(env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Box<Detections>>, index: jint) -> JPtr<'local, *const AprilTagFamily> {
    jni_wrap_simple(env, |_env| {
        let det = get_detection(ptr, index)?;
        let ptr = Arc::as_ptr(&det.family);

        JPtr::from_ptr(ptr)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetTagId<'local>(env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Box<Detections>>, index: jint) -> jint {
    jni_wrap_simple(env, |_env| {
        let det = get_detection(ptr, index)?;
        Ok(det.id as _)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetHammingDistance<'local>(env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Box<Detections>>, index: jint) -> jint {
    jni_wrap_simple(env, |_env| {
        let det = get_detection(ptr, index)?;
        Ok(det.hamming as _)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetDecisionMargin<'local>(env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Box<Detections>>, index: jint) -> jfloat {
    jni_wrap_simple(env, |_env| {
        let det = get_detection(ptr, index)?;
        Ok(det.decision_margin as _)
    })
}
#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetection_nativeGetHomography<'local>(env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Box<Detections>>, index: jint) -> JObject<'local> {
    jni_wrap_simple(env, |env| {
        let det = get_detection(ptr, index)?;
        new_mat33(env, &det.H)
    })
}