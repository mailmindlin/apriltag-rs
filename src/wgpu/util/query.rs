use std::{borrow::Cow, time::{Duration, Instant}};

use crate::{wgpu::{error::GpuBufferFetchError, WgpuDetectError}, TimeProfile};

use super::{GpuBuffer1, GpuBufferFetch, GpuContext};

struct GpuTimestampInner {
	set: wgpu::QuerySet,
	buffer: GpuBuffer1<u64>,
	names: Vec<String>,
	num_queries: usize,
}

impl GpuTimestampInner {
	fn new(ctx: &GpuContext, num_queries: usize) -> Result<Self, WgpuDetectError> {
		Ok(Self {
			set: ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
				label: Some("Timestamp query set"),
				count: num_queries.try_into().unwrap(),
				ty: wgpu::QueryType::Timestamp,
			}),
			// COPY_SRC only — STORAGE is irrelevant here and can conflict with
			// Metal's counter-resolve buffer allocation.
			buffer: ctx.temp_buffer1_usage(
				num_queries,
				wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
				"timestamp query"
			)?,
			num_queries,
			names: vec![],
		})
	}

	fn next_query_id(&mut self, name: String) -> Option<u32> {
		let idx = self.names.len();
		if idx >= self.num_queries {
			#[cfg(feature="debug")]
			println!("WARNING: out of timestamp query ids for {name}");
			None
		} else {
			self.names.push(name);
			Some(idx.try_into().unwrap())
		}
	}

	/// Record a pass-level timestamp pair via `ComputePassTimestampWrites`.
	/// Requires `TIMESTAMP_QUERY_INSIDE_PASSES`.
	fn next_pass<'a>(&'a mut self, name: &str) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
		let beginning_of_pass_write_index = self.next_query_id(format!("{name}->start"));
		let end_of_pass_write_index = self.next_query_id(format!("{name}->end"));
		if beginning_of_pass_write_index.is_none() && end_of_pass_write_index.is_none() {
			None
		} else {
			Some(wgpu::ComputePassTimestampWrites {
				query_set: &self.set,
				beginning_of_pass_write_index,
				end_of_pass_write_index,
			})
		}
	}

	/// Record a single encoder-level timestamp.
	/// Requires `TIMESTAMP_QUERY_INSIDE_ENCODERS`.
	fn write_timestamp(&mut self, encoder: &mut wgpu::CommandEncoder, name: String) {
		if let Some(id) = self.next_query_id(name) {
			encoder.write_timestamp(&self.set, id);
		}
	}

	fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
		if self.names.is_empty() {
			return;
		}
		encoder.resolve_query_set(
			&self.set,
			// Only resolve the slots we actually wrote.
			0..(self.names.len() as _),
			&self.buffer.buffer,
			0,
		);
	}

	async fn wait_for_results(&self, context: &GpuContext) -> Result<TimeProfile, GpuBufferFetchError> {
		let res = self.buffer.fetch_buffer(context).await?;
		let period = context.queue.get_timestamp_period();
		let mut tp = TimeProfile::default();
		let t0 = Instant::now();
		tp.set_start(t0);

		let gpu_t0 = match res.first() {
			None => return Ok(tp),
			Some(first) => *first,
		};

		// Discard results if all timestamps are zero — the resolve didn't fire.
		if res.iter().all(|&v| v == 0) {
			return Ok(tp);
		}

		for (name, item) in self.names.iter().zip(res.iter()) {
			let offset_ticks = item.saturating_sub(gpu_t0);
			let offset_nanos = (offset_ticks as f64) * (period as f64);
			let tN = t0 + Duration::from_nanos(offset_nanos as u64);
			tp.stamp_at(Cow::Owned(name.clone()), tN);
		}
		Ok(tp)
	}
}

/// Which GPU timestamp mechanism is in use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimestampMode {
	/// No timestamp support available.
	None,
	/// `ComputePassTimestampWrites` — requires `TIMESTAMP_QUERY_INSIDE_PASSES`.
	InsidePasses,
	/// `CommandEncoder::write_timestamp` — requires `TIMESTAMP_QUERY_INSIDE_ENCODERS`.
	/// Timestamps are written before/after each pass by the caller.
	InsideEncoders,
}

pub(in super::super) struct GpuTimestampQueries {
	inner: Option<GpuTimestampInner>,
	mode: TimestampMode,
}

impl GpuTimestampQueries {
	pub(crate) fn empty() -> Self {
		Self { inner: None, mode: TimestampMode::None }
	}

	pub(crate) fn new_inside_passes(context: &GpuContext, num_queries: usize) -> Result<Self, WgpuDetectError> {
		Ok(Self {
			inner: Some(GpuTimestampInner::new(context, num_queries)?),
			mode: TimestampMode::InsidePasses,
		})
	}

	pub(crate) fn new_inside_encoders(context: &GpuContext, num_queries: usize) -> Result<Self, WgpuDetectError> {
		Ok(Self {
			inner: Some(GpuTimestampInner::new(context, num_queries)?),
			mode: TimestampMode::InsideEncoders,
		})
	}

	/// Return a `ComputePassDescriptor` with timestamp writes embedded when in
	/// `InsidePasses` mode; plain descriptor otherwise.
	pub(crate) fn make_cpd<'a>(&'a mut self, name: &'a str) -> wgpu::ComputePassDescriptor<'a> {
		let timestamp_writes = if self.mode == TimestampMode::InsidePasses {
			self.inner.as_mut().and_then(|inner| inner.next_pass(name))
		} else {
			None
		};
		wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes }
	}

	/// Write a single encoder-level timestamp. No-op unless in `InsideEncoders` mode.
	/// Call this immediately before `begin_compute_pass` (for "start") and immediately
	/// after the compute pass ends (for "end").
	pub(crate) fn write_timestamp(&mut self, encoder: &mut wgpu::CommandEncoder, name: String) {
		if self.mode == TimestampMode::InsideEncoders {
			if let Some(inner) = &mut self.inner {
				inner.write_timestamp(encoder, name);
			}
		}
	}

	pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
		if let Some(inner) = &self.inner {
			inner.resolve(encoder)
		}
	}

	pub(in super::super) async fn wait_for_results(&self, context: &GpuContext) -> Result<Option<TimeProfile>, GpuBufferFetchError> {
		if let Some(inner) = &self.inner {
			inner.wait_for_results(context).await.map(Some)
		} else {
			Ok(None)
		}
	}
}
