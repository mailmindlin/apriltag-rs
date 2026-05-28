use std::borrow::Cow;

use wgpu::{RequestAdapterOptions, PowerPreference};

async fn run() {
	let inst = wgpu::Instance::default();
	let mut opts = RequestAdapterOptions::default();
	opts.power_preference = PowerPreference::HighPerformance;
	let adapter = inst.request_adapter(&opts).await
		.unwrap();
	println!("{adapter:?} {:?} {:?}", adapter.features(), adapter.get_info());
	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				label: None,
				required_features: wgpu::Features::empty(),
				required_limits: wgpu::Limits::downlevel_defaults(),
			},
			None,
		)
		.await
		.unwrap();
	println!("{device:?}");

	// let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
	// 	label: Some("01"),
	// 	source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
	// });
}

pub fn main() {
	use futures::executor::block_on;

	let a = run();
	block_on(a);
}