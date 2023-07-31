use std::{sync::Arc, ops::DerefMut};

use jni::{objects::{JString, JClass, JPrimitiveArray}, JNIEnv};

use crate::AprilTagFamily;

use super::util::{jni_wrap_simple, JavaError, jni_wrap, JPtrManaged, JPtr};

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeForName<'a>(env: JNIEnv<'a>, _class: JClass<'a>, name: JString<'a>) -> JPtrManaged<'static, Arc<AprilTagFamily>> {
    jni_wrap_simple(env, |env| {
        let name: String = env.get_string(&name)?.into();
        match AprilTagFamily::for_name(&name) {
            Some(family) => Ok(JPtrManaged::from(family)),
            None => Err(JavaError::IllegalArgumentException(format!("Unknown AprilTag family: {}", name))),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeDestroy<'a>(env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtrManaged<'a, Arc<AprilTagFamily>>) {
    jni_wrap_simple(env, |_env| {
        let family = ptr.take()?;
        drop(family);
        Ok(())
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeGetName<'a>(env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtr<'a, Arc<AprilTagFamily>>) -> JString<'a> {
    jni_wrap_simple(env, |env| {
        let family = ptr.as_ref()?;
        let str = env.new_string(family.name.clone())?;
        Ok(str)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeGetCodes<'a>(env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtr<'a, Arc<AprilTagFamily>>) -> JPrimitiveArray<'a, i64> {
    jni_wrap(env, ptr, |env, family| {
        if family.codes.len() > i32::MAX as usize {
            return Err(JavaError::IllegalStateException("Too many codes".into()));
        }

        let res = env.new_long_array(family.codes.len() as _)?;

        {
            let mut arr = unsafe { env.get_array_elements(&res, jni::objects::ReleaseMode::CopyBack) }?;
            let dst_signed: &mut [i64] = arr.deref_mut();
            let dst: &mut [u64] = unsafe { std::mem::transmute(dst_signed)};
            dst.copy_from_slice(&family.codes);
        }
        Ok(res)
    })
}

#[no_mangle]
pub extern "system" fn Java_com_mindlin_apriltags_AprilTagFamily_nativeGetBits<'a>(env: JNIEnv<'a>, _class: JClass<'a>, ptr: JPtr<'a, Arc<AprilTagFamily>>) -> JPrimitiveArray<'a, i32> {
    jni_wrap(env, ptr, |env, family| {
        let len = family.bits.len() * 2;
        if len > i32::MAX as usize {
            return Err(JavaError::IllegalStateException("Too many codes".into()));
        }

        let res = env.new_int_array(len as _)?;

        {
            let mut arr = unsafe { env.get_array_elements(&res, jni::objects::ReleaseMode::CopyBack) }?;
            for (dst, (x, y)) in arr.array_chunks_mut::<2>().zip(family.bits.iter()) {
                *dst = [*x as i32, *y as i32];
            }
        }
        Ok(res)
    })
}