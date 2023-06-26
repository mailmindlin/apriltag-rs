use std::sync::Arc;

use jni::{objects::{JString, JClass, JObject}, JNIEnv};

use crate::AprilTagFamily;

use super::util::{jni_wrap_simple, JavaError, JPtr, jni_wrap};

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeForName<'a>(mut env: JNIEnv<'a>, _class: JClass<'a>, name: JString<'a>) -> JPtr<'static, Arc<AprilTagFamily>> {
    jni_wrap_simple(&mut env, |env| {
        let name: String = env.get_string(name)?.into();
        match AprilTagFamily::for_name(&name) {
            Some(family) => Ok(JPtr::from(family)),
            None => Err(JavaError::IllegalArgumentException(format!("Unknown AprilTag family: {}", name))),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeDestroy<'a>(mut env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtr<'a, Arc<AprilTagFamily>>) {
    jni_wrap_simple(&mut env, |_env| {
        let family = ptr.take()?;
        drop(family);
        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeGetName<'a>(mut env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtr<'a, Arc<AprilTagFamily>>) -> JString<'a> {
    jni_wrap(&mut env, ptr, |env, family| {
        let str = env.new_string(family.name.clone())?;
        Ok(str)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeGetCodes<'a>(mut env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtr<'a, Arc<AprilTagFamily>>) -> JObject<'a> {
    jni_wrap(&mut env, ptr, |env, family| {
        if family.codes.len() > i32::MAX as usize {
            return Err(JavaError::IllegalStateException("Too many codes".into()));
        }

        let res = {
            let raw = env.new_long_array(family.codes.len() as _)?;
            env.auto_local(unsafe { JObject::from_raw(raw) })
        };

        {
            let arr = env.get_long_array_elements(res.as_obj().into_raw(), jni::objects::ReleaseMode::CopyBack)?;
            let dst = unsafe { std::slice::from_raw_parts_mut(arr.as_ptr() as *mut u64, family.codes.len()) };
            dst.copy_from_slice(&family.codes);
        }
        Ok(res.as_obj())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeGetBits<'a>(mut env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtr<'a, Arc<AprilTagFamily>>) -> JObject<'a> {
    jni_wrap(&mut env, ptr, |env, family| {
        let len = family.bits.len() * 2;
        if len > i32::MAX as usize {
            return Err(JavaError::IllegalStateException("Too many codes".into()));
        }

        let res = {
            let raw = env.new_int_array(len as _)?;
            env.auto_local(unsafe { JObject::from_raw(raw) })
        };

        {
            let arr = env.get_int_array_elements(res.as_obj().into_raw(), jni::objects::ReleaseMode::CopyBack)?;
            let dst = unsafe { std::slice::from_raw_parts_mut(arr.as_ptr() as *mut u32, len) };
            for (dst, (x, y)) in dst.array_chunks_mut::<2>().zip(family.bits.iter()) {
                *dst = [*x, *y];
            }
        }
        Ok(res.as_obj())
    })
}