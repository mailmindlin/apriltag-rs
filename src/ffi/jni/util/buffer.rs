use std::{ops::Deref, slice, mem::transmute};

use jni::{JNIEnv, objects::{JByteBuffer, JClass, JObject, JByteArray, AutoElements}, sys::{jbyte, jint, jvalue}, signature::{ReturnType, Primitive}, descriptors::Desc, errors::Error as JNIError};

use super::JavaError;

pub(in super::super) enum BufferView<'a> {
	Owned(Vec<u8>),
	Array(AutoElements<'a, 'a, 'a, jbyte>, usize, usize),
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
	pub(in super::super) fn into_boxed_slice(self) -> Box<[u8]> {
		self.to_vec().into_boxed_slice()
	}
}

impl<'a> Deref for BufferView<'a> {
	type Target = [u8];

	fn deref(&self) -> &Self::Target {
		match self {
			Self::Owned(vec) => &vec,
			Self::Array(arr, off, len) => unsafe {
				let ptr: *const u8 = transmute(arr.as_ptr());
				let base = ptr.add(*off);
				slice::from_raw_parts(base, *len)
			},
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

fn get_buffer_range<'a>(env: &mut JNIEnv<'a>, class: impl Desc<'a, JClass>, buf: JByteBuffer<'a>) -> Result<(usize, usize), JavaError> {
	let class = Desc::<JClass>::lookup(class, env)?;
	let position = unsafe { env.call_method_unchecked(buf, (class, "position", "()I"), ReturnType::Primitive(Primitive::Int), &[]) }?.i().unwrap();
	let limit    = unsafe { env.call_method_unchecked(buf, (class, "limit", "()I"), ReturnType::Primitive(Primitive::Int), &[]) }?.i().unwrap();

	check_buffer_range(position, limit)
}

fn get_buffer_direct<'a>(env: &JNIEnv<'a>, buf: JByteBuffer<'a>) -> Result<Option<&'a [u8]>, JavaError> {
	match env.get_direct_buffer_address(&buf) {
		Ok(ptr) => {
			// Is direct, so we can just get it
			let cap = env.get_direct_buffer_capacity(&buf)?;
			let slice = unsafe { slice::from_raw_parts(ptr, cap) };
			Ok(Some(slice))
		},
		Err(JNIError::NullPtr(msg)) if msg == "get_direct_buffer_address return value" => {
			Ok(None)
		},
		Err(e) => Err(e.into())
	}
}

pub(in super::super) fn get_array_elements<'a>(env: &JNIEnv<'a>, array: &'a JByteArray<'a>, offset: usize, length: usize, force_copy: bool) -> Result<BufferView<'a>, JavaError> {
	if length == 0 {
		return Ok(BufferView::Owned(vec![]));
	}

	let arr_cap = {
		let arr_cap = env.get_array_length(array)?;
		assert!(arr_cap >= 0);
		arr_cap as usize
	};
	if arr_cap < offset + length {
		return Err(JavaError::GenericException("Java cap too small".into()));
	}

	if force_copy || 3 * length / 2 > arr_cap || length < 64 {
		// Always get region if length << cap
		let mut buf = vec![0u8; length];
		env.get_byte_array_region(*array, offset.try_into().unwrap(), unsafe { transmute(buf.as_mut_slice()) })?;
		return Ok(BufferView::Owned(buf));
	}

	let elements = unsafe { env.get_array_elements(array, jni::objects::ReleaseMode::NoCopyBack)? };
	let copied = elements.is_copy();
	let res = BufferView::Array(elements, offset, length);

	if copied && length / 2 > arr_cap {
		// Copy to smaller buffer
		Ok(res.to_owned())
	} else {
		Ok(res)
	}
}

pub(in super::super) fn get_buffer<'a>(env: &mut JNIEnv<'a>, buf: JByteBuffer<'a>) -> Result<BufferView<'a>, JavaError> {
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
		let array = env.auto_local(unsafe {env.call_method_unchecked(buf, (&bb_class, "array", "()[B"), ReturnType::Array, &[]) }?.l().unwrap());
		
		let (position, limit) = get_buffer_range(env, &bb_class, buf)?;

		return get_array_elements(env, (&array).into(), position + array_offset as usize, limit - position, false);
	}

	// Use bulk get()
	let remaining = unsafe {env.call_method_unchecked(buf, (&bb_class, "remaining", "()I"), ReturnType::Primitive(Primitive::Int), &[]) }?.i().unwrap();
	if remaining < 0 {
		return Err(JavaError::IllegalArgumentException("Buffer negative remaining".into()));
	}

	let arr = {
		let inner = env.new_byte_array(remaining)?;
		env.auto_local(inner)
	};
	{
		let x = jvalue { l: arr.as_obj().into_raw() };
		let res = unsafe {  env.call_method_unchecked(buf, (&bb_class, "get", "([B)Ljava/nio/ByteBuffer;"), ReturnType::Object, &[x]) }?.l()?;
		env.delete_local_ref(res)?;
	}
	// env.get_byte_array_elements(arr, jni::objects::ReleaseMode::NoCopyBack)?
	get_array_elements(env, arr.as_obj(), 0, remaining as _, true)
}