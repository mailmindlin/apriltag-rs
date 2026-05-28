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
			buffer: ctx.temp_buffer1_usage(
				num_queries,
				ctx.buffer_usage(true) | wgpu::BufferUsages::QUERY_RESOLVE,
				Some("timestamp query")
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

	fn next<'a>(&'a mut self, name: &str) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
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

	fn write_timestamp(&mut self, encoder: &mut wgpu::CommandEncoder, name: String) {
		if let Some(id) = self.next_query_id(name) {
			encoder.write_timestamp(&self.set, id);
		}
	}

	fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
		#[cfg(debug_assertions)]
		if self.names.len() != self.num_queries {
			println!("Queries: requested {} but used {}", self.num_queries, self.names.len());
		}
		encoder.resolve_query_set(
			&self.set,
			// TODO(https://github.com/gfx-rs/wgpu/issues/3993): Musn't be larger than the number valid queries in the set.
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

		for (name, item) in self.names.iter().zip(res.into_iter()) {
			let offset_ticks = item - gpu_t0;
			let offset_nanos = (offset_ticks as f32) * period;
			let tN = t0 + Duration::from_nanos(offset_nanos as u64);
			tp.stamp_at(Cow::Owned(name.clone()), tN);
		}
		Ok(tp)
	}
}
pub(in super::super) struct GpuTimestampQueries(Option<GpuTimestampInner>);

impl GpuTimestampQueries {
	pub(crate) fn empty() -> Self {
		Self(None)
	}
	pub(crate) fn new(context: &GpuContext, num_queries: usize) -> Result<Self, WgpuDetectError> {
		let inner = GpuTimestampInner::new(context, num_queries)?;
		Ok(Self(Some(inner)))
	}

	pub(crate) fn next<'a>(&'a mut self, name: &str) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
		self.0.as_mut()
			.and_then(|inner| inner.next(name))
	}

	pub(crate) fn make_cpd<'a>(&'a mut self, name: &'a str) -> wgpu::ComputePassDescriptor<'a> {
		wgpu::ComputePassDescriptor {
			label: Some(name),
			timestamp_writes: self.next(name),
		}
	}

	pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
		if let Some(inner) = &self.0 {
			inner.resolve(encoder)
		}
	}
	pub(crate) fn write_timestamp(&mut self, encoder: &mut wgpu::CommandEncoder, name: String) {
		if let Some(inner) = &mut self.0 {
			inner.write_timestamp(encoder, name);
		}
	}

	pub(in super::super) async fn wait_for_results(&self, context: &GpuContext) -> Result<Option<TimeProfile>, GpuBufferFetchError> {
		if let Some(inner) = &self.0 {
			inner.wait_for_results(context).await.map(Some)
		} else {
			Ok(None)
		}
	}
}