use std::{mem, sync::RwLock};

use jni::{JNIEnv, sys::{jlong, jint, jboolean, jdouble, JNI_TRUE}, objects::{JClass, JString}};

use crate::{detector::ApriltagDetector, families::AprilTagFamily};

mod util;

use self::util::{JavaError, jni_wrap_simple, JniDetectorData, get_ptr_mut, jni_wrap};

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_create(mut env: JNIEnv, _class: JClass) -> jlong {
    jni_wrap_simple(&mut env, |env| {
        if mem::size_of::<jlong>() < mem::size_of::<*mut ApriltagDetector>() {
            //Throw exception
            return Err(JavaError::Exception("Invalid pointer size".to_owned()));
        }
    
        let ptr = {
            let detector = ApriltagDetector::default();
            
            let data = box JniDetectorData {
                detector: RwLock::new(ApriltagDetector::default()),
            };
    
            Box::into_raw(data)
        };
    
        //TODO: double check the transmute was safe
        let jniptr = unsafe { mem::transmute(ptr) };
    
        // Double check that the cast worked
        if let Ok(decoded) = get_ptr_mut(jniptr) {
            let decoded_test = decoded as *mut JniDetectorData;
            if ptr == decoded {
                return Ok(jniptr);
            }
        }
        // Release data
        std::mem::drop(unsafe { Box::from_raw(ptr) });
        Err(JavaError::Exception("Pointer decode failed".to_owned()))
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_setParams<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: jlong, nthreads: jint, quadDecimate: jdouble, quadSigma: jdouble, refineEdges: jboolean, decodeSharpening: jdouble, debug: jboolean) {
    jni_wrap(&mut env, ptr, |env, det| {
        det.params.nthreads = nthreads.try_into()
            .map_err(|e| JavaError::IllegalArgumentException(format!("Illegal argument: nthreads: {:?}", e)))?;
        
        det.params.quad_decimate = quadDecimate as f32;
        det.params.quad_sigma = quadSigma as f32;
        det.params.refine_edges = refineEdges == JNI_TRUE;
        det.params.decode_sharpening = decodeSharpening;
        det.params.debug = debug == JNI_TRUE;
        // det.refine_decode = decode

        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_destroy(env: JNIEnv, _class: JClass, ptr: jlong) {
    let ptr: *mut JniDetectorData = unsafe { std::mem::transmute(ptr) };
    if ptr.is_null() {
        env.throw_new("java/lang/NullPointerException", "Missing pointer");
        return;
    }
    let data = unsafe { Box::from_raw(ptr) };
    std::mem::drop(data);
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_addFamily<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: jlong, family: JString<'local>, bits_corrected: jint) {
    jni_wrap(&mut env, ptr, |env, detector| {
        let family = {
            let family_name: String = env.get_string(family)?.into();

            AprilTagFamily::for_name(&family_name)
                .ok_or_else(|| JavaError::IllegalArgumentException(format!("Unknown AprilTag family: {}", family_name)))?
        };

        detector.add_family_bits(family, bits_corrected as usize);

        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_clearFamilies(mut env: JNIEnv, _class: JClass, ptr: jlong) {
    jni_wrap(&mut env, ptr, |env, detector| {
        detector.clear_families();

        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_removeFamily<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: jlong, family: JString<'local>) {
    jni_wrap(&mut env, ptr, |env, detector| {
        //TODO
        
        Ok(())
    })
}
