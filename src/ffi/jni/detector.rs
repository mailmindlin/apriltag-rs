use std::{mem::{size_of, self}, sync::Arc};

use jni::{JNIEnv, objects::{JClass, JString, JByteBuffer, JIntArray, JLongArray}, sys::{jint, jdouble, jboolean, jfloat, jlong}};

use crate::{AprilTagDetector, AprilTagFamily, AddFamilyError, util::ImageY8, detector::{DetectorBuilder, DetectorBuildError}, ffi::util::ManagedPtr};

use super::util::{jni_wrap_simple, jni_wrap, JavaError, JPtr};

type Data = Box<AprilTagDetector>;

fn add_families(env: &mut JNIEnv, builder: &mut DetectorBuilder, familyPtrs: &JLongArray, bitsCorrecteds: &JIntArray) -> Result<(), JavaError> {
    let len_fp = env.get_array_length(familyPtrs)?;
    let len_bc = env.get_array_length(bitsCorrecteds)?;
    if len_fp != len_bc {
        return Err(JavaError::IllegalArgumentException("familyPtrs and bitsCorrecteds length mismatch".to_owned()));
    }
    if len_fp <= 0 {
        return Ok(());
    }

    let mut arr_fps = unsafe { Box::new_zeroed_slice(len_fp as usize).assume_init() };
    let mut arr_bcs = unsafe { Box::new_zeroed_slice(len_fp as usize).assume_init() };
    env.get_long_array_region(familyPtrs, 0, &mut arr_fps)?;
    env.get_int_array_region(bitsCorrecteds, 0, &mut arr_bcs)?;

    for (idx, (fp, bc)) in arr_fps.into_iter().zip(arr_bcs.into_iter()).enumerate() {
        let fp: ManagedPtr<Arc<AprilTagFamily>> = unsafe { ManagedPtr::wrap_ptr(*fp as *const AprilTagFamily) };
        let family = match fp.borrow() {
            Some(family) => family,
            None => return Err(JavaError::NullPointerException(format!("Null AprilTagFamily pointer (index {idx})"))),
        };
        let bits_corrected = *bc as usize;

        match builder.add_family_bits(family, bits_corrected) {
            Ok(()) => {},
            Err(AddFamilyError::QuickDecodeAllocation) => {
                //TODO: error type
                return Err(JavaError::IllegalArgumentException(format!("Error allocating QuickDecode table (maybe reduce bits_corrected; index {idx})")));
            },
            Err(AddFamilyError::BigHamming(hamming)) => {
                return Err(JavaError::IllegalArgumentException(format!("Bits corrected too big ({hamming}; index {idx})")));
            },
            Err(AddFamilyError::TooManyCodes(num_codes)) => {
                return Err(JavaError::IllegalArgumentException(format!("Too many codes for AprilTag family ({num_codes} codes; index {idx})")));
            },
        }
    }
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_nativeCreate<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, nthreads: jint, quadDecimate: jfloat, quadSigma: jfloat, refineEdges: jboolean, decodeSharpening: jdouble, debugPath: JString<'local>, familyPtrs: JLongArray<'local>, bitsCorrecteds: JIntArray<'local>) -> JPtr<'static, Data> {
    jni_wrap_simple(&mut env, |env| {
        if size_of::<jlong>() < size_of::<*mut AprilTagDetector>() {
            //Throw exception
            return Err(JavaError::GenericException("Invalid pointer size".to_owned()));
        }

        let detector = {
            let mut builder = AprilTagDetector::builder();
            builder.config.nthreads = nthreads as usize;
            builder.config.quad_decimate = quadDecimate;
            builder.config.quad_sigma = quadSigma;
            builder.config.refine_edges = refineEdges != 0;
            builder.config.decode_sharpening = decodeSharpening;
            builder.config.debug;

            add_families(env, &mut builder, &familyPtrs, &bitsCorrecteds)?;

            match builder.build() {
                Ok(det) => Box::new(det),
                Err(DetectorBuildError::Threadpool(e)) => {
                    return Err(JavaError::GenericException(format!("Unable to create threadpool: {e:?}")));
                },
                Err(e) => {
                    return Err(JavaError::GenericException(format!("{e:?}")));
                }
            }
        };
    
        let jniptr = JPtr::from(detector);

        Ok(jniptr)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_nativeDetect<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>, buf: JByteBuffer<'local>, width: jint, height: jint, stride: jint) {
    jni_wrap(&mut env, ptr, |env, det| {
        let buf = super::util::get_buffer(env, buf)?.to_vec();
        let img = ImageY8::wrap(buf.into_boxed_slice(), width as _, height as _, stride as _);
        let res = det.detect(&img);
        
        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_nativeDestroy<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>) {
    jni_wrap_simple(&mut env, |env| {
        let inner = ptr.take()?;
        mem::drop(inner);
        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_addFamily<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>, family: JPtr<'local, Arc<AprilTagFamily>>, bits_corrected: jint) {
    jni_wrap(&mut env, ptr, |env, detector| {
        let family = {
            let family: *const AprilTagFamily = unsafe { mem::transmute(family) };
            if family.is_null() {
                return Err(JavaError::NullPointerException("Null AprilTag family".into()));
            }
            unsafe { Arc::from_raw(family) }
        };

        match detector.add_family_bits(family, bits_corrected as usize) {
            Ok(()) => Ok(()),
            Err(AddFamilyError::TooManyCodes(num_codes)) =>
                Err(JavaError::IllegalArgumentException(format!("Too many codes ({}, max 2**16) in AprilTag family", num_codes))),
            Err(AddFamilyError::BigHamming(hamming)) =>
                Err(JavaError::IllegalArgumentException(format!("Hamming out of bounds (actual: {}, expected: 0..3)", hamming))),
            Err(AddFamilyError::QuickDecodeAllocation) =>
                Err(JavaError::GenericException(format!("Unable to allocate memory for AprilTag family"))),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_clearFamilies<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>) {
    jni_wrap(&mut env, ptr, |_env, detector| {
        detector.clear_families();

        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_removeFamily<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>, family: JString<'local>) {
    jni_wrap(&mut env, ptr, |_env, detector| {
        //TODO
        
        Ok(())
    })
}