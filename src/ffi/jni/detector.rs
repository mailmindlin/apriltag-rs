use std::{mem::size_of, sync::Arc, num::TryFromIntError};

use jni::{JNIEnv, objects::{JClass, JString, JByteBuffer, JIntArray, JLongArray, JObject, JValue}, sys::{jint, jdouble, jboolean, jfloat, jlong, JNI_FALSE}};

use crate::{AprilTagDetector, AprilTagFamily, AddFamilyError, util::{ImageY8, mem::calloc}, detector::{DetectorBuilder, DetectorBuildError}, ffi::util::ManagedPtr, DetectError, AprilTagQuadThreshParams};

use super::{util::{jni_wrap_simple, JavaError, JPtr, JPtrManaged}, CLASS_DETECTIONS};

type Data = Box<AprilTagDetector>;

fn add_families(env: &mut JNIEnv, builder: &mut DetectorBuilder, familyPtrs: &JLongArray, bitsCorrecteds: &JIntArray) -> Result<(), JavaError> {
    let len_fp = env.get_array_length(familyPtrs)?;
    let len_bc = env.get_array_length(bitsCorrecteds)?;
    if len_fp != len_bc {
        return Err(JavaError::IllegalArgumentException("familyPtrs and bitsCorrecteds length mismatch".to_owned()));
    }
    if len_fp == 0 {
        return Ok(());
    }

    let mut arr_fps = calloc(len_fp.try_into().unwrap());
    let mut arr_bcs = calloc(len_fp.try_into().unwrap());
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

fn convert_arg<R, T>(src: T, name: &str) -> Result<R, JavaError> where R: TryFrom<T, Error=TryFromIntError> {
    match src.try_into() {
        Ok(v) => Ok(v),
        Err(e) => Err(JavaError::IllegalArgumentException(format!("{name} out of range ({e})"))),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_nativeCreate<'local>(env: JNIEnv<'local>, _class: JClass<'local>,
        nthreads: jint, quadDecimate: jfloat, quadSigma: jfloat, refineEdges: jboolean, decodeSharpening: jdouble, debugPath: JString<'local>, // Config
        minClusterPixels: jint, maxNumMaxima: jint, cosCriticalRad: jfloat, maxLineFitMSE: jfloat, minWhiteBlackDiff: jint, deglitch: jboolean, // QTP
        familyPtrs: JLongArray<'local>, bitsCorrecteds: JIntArray<'local>,
    ) -> JPtrManaged<'static, Data> {
    jni_wrap_simple(env, |env| {
        if size_of::<jlong>() < size_of::<*mut AprilTagDetector>() {
            //Throw exception
            return Err(JavaError::AssertionError("Invalid pointer size".to_owned()));
        }

        let detector = {
            let mut builder = AprilTagDetector::builder();

            // Set config
            builder.config.nthreads = convert_arg(nthreads, "nthreads")?;
            builder.config.quad_decimate = quadDecimate;
            builder.config.quad_sigma = quadSigma;
            builder.config.refine_edges = refineEdges != JNI_FALSE;
            builder.config.decode_sharpening = decodeSharpening;
            if debugPath.is_null() {
                builder.config.debug = false;
                builder.config.debug_path = None;
            } else {
                builder.config.debug = true;
                let debugPath = env.get_string(&debugPath)?;
                builder.config.debug_path = Some(String::from(debugPath));
            }

            builder.config.qtp = AprilTagQuadThreshParams {
                min_cluster_pixels: convert_arg(minClusterPixels, "minClusterPixels")?,
                max_nmaxima: convert_arg(maxNumMaxima, "maxNumMaxima")?,
                cos_critical_rad: cosCriticalRad,
                max_line_fit_mse: maxLineFitMSE,
                min_white_black_diff: convert_arg(minWhiteBlackDiff, "minWhiteBlackDiff")?,
                deglitch: deglitch != JNI_FALSE,
            };

            add_families(env, &mut builder, &familyPtrs, &bitsCorrecteds)?;

            match builder.build() {
                Ok(det) => Box::new(det),
                Err(DetectorBuildError::Threadpool(e)) => {
                    return Err(JavaError::Custom {
                        class: "ThreadPoolBuildException",
                        msg: format!("Unable to create threadpool: {e:?}"),
                    });
                },
                Err(DetectorBuildError::BufferAllocationFailure) => {
                    return Err(JavaError::Custom {
                        class: "AprilTagDetectorBuildException",
                        msg: format!("Unable to allocate buffer"),
                    });
                },
                #[cfg(feature="opencl")]
                Err(DetectorBuildError::OpenCLError(e)) => {
                    return Err(JavaError::Custom {
                        class: "OpenCLException",
                        msg: format!("Error initializing OpenCL: {e:?}",
                    )});
                },
                Err(DetectorBuildError::OpenCLNotAvailable) => {
                    return Err(JavaError::Custom {
                        class: "OpenCLUnavailableException",
                        msg: "OpenCL not available".to_string()
                    });
                }
                Err(e) => {
                    return Err(JavaError::Custom {
                        class: "AprilTagDetectorBuildException",
                        msg: format!("{e:?}"),
                    });
                }
            }
        };
    
        let jniptr = JPtrManaged::from(detector);

        Ok(jniptr)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_nativeDetect<'local>(env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>, buf: JByteBuffer<'local>, width: jint, height: jint, stride: jint, family_lookup: JObject<'local>) -> JObject<'local> {
    jni_wrap_simple(env, |env| {
        let det = ptr.as_ref()?;
        let res = {
            let buf = super::util::get_buffer(env, &buf)?.to_vec();
            let img = ImageY8::wrap(buf.into_boxed_slice(), width as _, height as _, stride as _);
            det.detect(&img)
        };
        match res {
            Ok(dets) => {
                let dets = Box::new(dets);
                let detections_count: i32 = dets.detections.len()
                    .try_into()
                    .map_err(|e| JavaError::OutOfMemoryError(format!("detections_count overflow: {e}")))?;
                let nquads: i32 = dets.nquads as i32;

                let dets_ptr = JPtrManaged::from(dets);
                let res = {
                    let res = env.new_object(CLASS_DETECTIONS, "(JIILjava/lang/Map;)V", &[
                        JValue::Long(dets_ptr.as_jni()),
                        JValue::Int(nquads),
                        JValue::Int(detections_count),
                        JValue::Object(&family_lookup),
                    ])?;
                    // We constructed the object, and the value is owned by Java now
                    std::mem::forget(dets_ptr);
                    res
                };
                Ok(res)
            },
            Err(DetectError::AllocError)
                => Err(JavaError::OutOfMemoryError("Unable to allocate buffer(s)".into())),
            Err(DetectError::ImageTooSmall)
                => Err(JavaError::IllegalArgumentException("Image was too small".into())),
            Err(DetectError::ImageTooBig)
                => Err(JavaError::IllegalArgumentException("Image was too big".into())),
            Err(DetectError::OpenCLError)
                => Err(JavaError::IllegalArgumentException("OpenCL error".into())),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_destroy<'local>(env: JNIEnv<'local>, _this: JObject<'local>, ptr: JPtrManaged<'local, Data>) {
    jni_wrap_simple(env, |_env| {
        drop(ptr);
        Ok(())
    })
}

// #[no_mangle]
// pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_addFamily<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>, family: JPtr<'local, Arc<AprilTagFamily>>, bits_corrected: jint) {
//     jni_wrap(env, ptr, |env, detector| {
//         let family = {
//             let family: *const AprilTagFamily = unsafe { mem::transmute(family) };
//             if family.is_null() {
//                 return Err(JavaError::NullPointerException("Null AprilTag family".into()));
//             }
//             unsafe { Arc::from_raw(family) }
//         };

//         match detector.add_family_bits(family, bits_corrected as usize) {
//             Ok(()) => Ok(()),
//             Err(AddFamilyError::TooManyCodes(num_codes)) =>
//                 Err(JavaError::IllegalArgumentException(format!("Too many codes ({}, max 2**16) in AprilTag family", num_codes))),
//             Err(AddFamilyError::BigHamming(hamming)) =>
//                 Err(JavaError::IllegalArgumentException(format!("Hamming out of bounds (actual: {}, expected: 0..3)", hamming))),
//             Err(AddFamilyError::QuickDecodeAllocation) =>
//                 Err(JavaError::GenericException(format!("Unable to allocate memory for AprilTag family"))),
//         }
//     })
// }

// #[no_mangle]
// pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_clearFamilies<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>) {
//     jni_wrap(&mut env, ptr, |_env, detector| {
//         detector.clear_families();

//         Ok(())
//     })
// }

// #[no_mangle]
// pub extern "system" fn Java_com_mindlin_apriltags_AprilTagDetector_removeFamily<'local>(mut env: JNIEnv<'local>, _class: JClass<'local>, ptr: JPtr<'local, Data>, family: JString<'local>) {
//     jni_wrap(&mut env, ptr, |_env, detector| {
//         //TODO
        
//         Ok(())
//     })
// }