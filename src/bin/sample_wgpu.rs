use std::borrow::Cow;

#[cfg(feature="wgpu")]
use wgpu::{ExperimentalFeatures, PowerPreference, RequestAdapterOptions};

#[cfg(feature="wgpu")]
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
				experimental_features: wgpu::ExperimentalFeatures::disabled(),
				memory_hints: wgpu::MemoryHints::Performance,
				trace: wgpu::Trace::Off,
			}
		)
		.await
		.unwrap();
	println!("{device:?}");

	// let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
	// 	label: Some("01"),
	// 	source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
	// });
}

#[cfg(feature="wgpu")]
pub fn main() {
	use futures::executor::block_on;

	let a = run();
	block_on(a);
}

#[cfg(not(feature="wgpu"))]
pub fn main() {
	panic!("requires feature `wgpu`")
}