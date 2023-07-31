use std::{ops::Deref, slice, mem::transmute, ptr::NonNull};

use jni::{JNIEnv, objects::{JByteBuffer, JClass, JByteArray, JObject}, sys::{jint, jvalue, JNI_FALSE, JNI_ABORT}, signature::{ReturnType, Primitive}, errors::Error as JNIError};

use super::JavaError;

pub(in super::super) struct ArrayElements<'a> {
	env: JNIEnv<'a>,
	ptr: NonNull<u8>,
	off: usize,
	len: usize,
	arr: JObject<'a>,
	is_copy: bool,
}

impl<'a> ArrayElements<'a> {
	fn new<'loc>(env: &mut JNIEnv<'a>, arr: impl AsRef<JObject<'loc>>, off: usize, len: usize) -> Result<Self, JavaError> {
		let arr = env.auto_local(env.new_local_ref(arr)?);
		let env = unsafe { env.unsafe_clone() };
		let (ptr, is_copy) = unsafe {
			let raw_env = env.get_native_interface();
			let raw_ref = raw_env
				.as_ref().ok_or_else(|| JNIError::NullPtr("Null env"))?
				.as_ref().ok_or_else(|| JNIError::NullPtr("Null env"))?;
			let GetByteArrayElements = raw_ref.GetByteArrayElements
				.ok_or_else(|| JNIError::JNIEnvMethodNotFound("Missing GetByteArrayElements"))?;
			let mut is_copy = JNI_FALSE;
			let ptr = GetByteArrayElements(raw_env, arr.as_raw(), &mut is_copy);
			let is_copy = is_copy != JNI_FALSE;
			(ptr as *mut u8, is_copy)
		};
		let ptr = NonNull::new(ptr)
			.ok_or_else(|| JNIError::NullPtr("Null array"))?;
		Ok(Self {
			env,
			ptr,
			off,
			len,
			arr: arr.forget(),
			is_copy,
		})
	}
}

impl<'a> Deref for ArrayElements<'a> {
    type Target = [u8];

    fn deref(&self) -> &'a Self::Target {
        unsafe {
			let base = self.ptr.as_ptr().add(self.off);
			std::slice::from_raw_parts(base, self.len)
		}
    }
}

impl<'a> Drop for ArrayElements<'a> {
    fn drop(&mut self) {
        unsafe {
			let raw_env = self.env.get_native_interface();
			let raw_ref = raw_env
				.as_ref().unwrap()
				.as_ref().unwrap();
			let ReleaseByteArrayElements = raw_ref.ReleaseByteArrayElements
				.expect("Missing ReleaseByteArrayElements");
			ReleaseByteArrayElements(raw_env, self.arr.as_raw(), self.ptr.as_ptr() as _, JNI_ABORT);
		};
		let arr = std::mem::take(&mut self.arr);
		self.env.delete_local_ref(arr)
			.unwrap();
    }
}
pub(in super::super) enum BufferView<'a> {
	Owned(Vec<u8>),
	Array(ArrayElements<'a>),
	Direct(&'a [u8]),
}

impl<'a> BufferView<'a> {
    pub(in super::super) fn to_owned(self) -> BufferView<'static> {
		BufferView::Owned(self.to_vec())
    }

	pub(in super::super) fn to_vec(self) -> Vec<u8> {
		match self {
			BufferView::Owned(vec) => vec,
			_ => self.deref().to_vec(),
		}
	}
	const fn is_copy(&self) -> bool {
		match self {
			Self::Owned(..) => false,
			Self::Array(arr) => arr.is_copy,
			Self::Direct(..) => false,
		}
	}
}

impl<'a> Deref for BufferView<'a> {
	type Target = [u8];

	fn deref(&self) -> &Self::Target {
		match self {
			Self::Owned(vec) => &vec,
			Self::Array(arr) => &arr,
			Self::Direct(s) => s,
		}
	}
}

fn check_buffer_range(position: jint, limit: jint) -> Result<(usize, usize), JavaError> {
	if position < 0 {
		return Err(JavaError::IllegalStateException("Negative buffer position".into()));
	}
	if limit < 0 {
		return Err(JavaError::IllegalStateException("Negative buffer limit".into()));
	}

	if limit < position {
		return Err(JavaError::IllegalArgumentException("buffer limit < position".into()));
	}

	Ok((position as _, limit as _))
}

fn get_buffer_range<'a>(env: &mut JNIEnv<'a>, class: &JClass<'a>, buf: &JByteBuffer<'a>) -> Result<(usize, usize), JavaError> {
	let position = unsafe { env.call_method_unchecked(buf, (class, "position", "()I"), ReturnType::Primitive(Primitive::Int), &[]) }?.i().unwrap();
	let limit    = unsafe { env.call_method_unchecked(buf, (class, "limit", "()I"), ReturnType::Primitive(Primitive::Int), &[]) }?.i().unwrap();

	check_buffer_range(position, limit)
}

fn get_buffer_direct<'a>(env: &JNIEnv<'a>, buf: &JByteBuffer<'a>) -> Result<Option<&'a [u8]>, JavaError> {
	match env.get_direct_buffer_address(&buf) {
		Ok(ptr) => {
			// Is direct, so we can just get it
			let cap = env.get_direct_buffer_capacity(buf)?;
			let slice = unsafe { slice::from_raw_parts(ptr, cap) };
			Ok(Some(slice))
		},
		Err(JNIError::NullPtr(msg)) if msg == "get_direct_buffer_address return value" => {
			Ok(None)
		},
		Err(e) => Err(e.into())
	}
}

pub(in super::super) fn get_array_elements<'a, 'b>(env: &mut JNIEnv<'a>, array: impl AsRef<JByteArray<'a>>, offset: usize, length: usize, force_copy: bool) -> Result<BufferView<'a>, JavaError> {
	if length == 0 {
		return Ok(BufferView::Owned(vec![]));
	}

	let arr_cap = {
		let arr_cap = env.get_array_length(array.as_ref())?;
		assert!(arr_cap >= 0);
		arr_cap as usize
	};
	if arr_cap < offset + length {
		return Err(JavaError::GenericException("Java cap too small".into()));
	}

	if force_copy || 3 * length / 2 > arr_cap || length < 64 {
		// Always get region if length << cap
		let mut buf = vec![0u8; length];
		env.get_byte_array_region(array, offset.try_into().unwrap(), unsafe { transmute(buf.as_mut_slice()) })?;
		return Ok(BufferView::Owned(buf));
	}

	let res = BufferView::Array(ArrayElements::new(env, array.as_ref(), offset, length)?);

	if res.is_copy() && length / 2 > arr_cap {
		// Copy to smaller buffer
		Ok(res.to_owned())
	} else {
		Ok(res)
	}
}

pub(in super::super) fn get_buffer<'a, 'b>(env: &mut JNIEnv<'a>, buf: &'b JByteBuffer<'a>) -> Result<BufferView<'b>, JavaError> {
	// Get class once
	let bb_class = env.auto_local(env.get_object_class(buf)?);
	if let Some(direct) = get_buffer_direct(env, buf)? {
		let (position, limit) = get_buffer_range(env, &bb_class, buf)?;
		let partial = &direct[position..limit];
		return Ok(BufferView::Direct(partial));
	}

	// Try array()
	if unsafe { env.call_method_unchecked(buf, (&bb_class, "hasArray", "()Z"), ReturnType::Primitive(Primitive::Boolean), &[]) }?.z().unwrap() {
		let array_offset = unsafe { env.call_method_unchecked(buf, (&bb_class, "arrayOffset", "()I"), ReturnType::Primitive(Primitive::Int), &[]) }?.i().unwrap();
		if array_offset < 0 {
			return Err(JavaError::IllegalStateException("Negative array offset".into()));
		}
		let array: JByteArray = unsafe { env.call_method_unchecked(buf, (&bb_class, "array", "()[B"), ReturnType::Array, &[]) }?
			.l()
			.unwrap()
			.into();
		let array = env.auto_local(array);
		
		let (position, limit) = get_buffer_range(env, &bb_class, buf)?;

		return get_array_elements(env, array, position + array_offset as usize, limit - position, false);
	}

	// Use bulk get()
	let remaining = unsafe { env.call_method_unchecked(buf, (&bb_class, "remaining", "()I"), ReturnType::Primitive(Primitive::Int), &[]) }?.i().unwrap();
	if remaining < 0 {
		return Err(JavaError::IllegalArgumentException("Buffer negative remaining".into()));
	}

	let arr = env.auto_local(env.new_byte_array(remaining)?);
	{
		let x = jvalue { l: arr.as_raw() };
		let res = unsafe { env.call_method_unchecked(buf, (&bb_class, "get", "([B)Ljava/nio/ByteBuffer;"), ReturnType::Object, &[x]) }?.l()?;
		env.delete_local_ref(res)?;
	}
	get_array_elements(env, arr, 0, remaining as _, true)
}