use wgpu::{Adapter, InstanceDescriptor, PowerPreference, RequestAdapterOptions, TextureFormat, TextureUsages};

use crate::{detector::ImageDimensionError, wgpu::WgpuBuildError, DetectorConfig};

#[derive(Clone, Debug)]
pub enum InvalidAdapterReason {
	/// Compute shaders aren't supported
	NoComputeShaders,
	MissingTextureFeature(TextureFormat, TextureUsages),
	TextureSize(ImageDimensionError),
	RequestMismatch,
}

#[derive(Clone, Debug)]
pub struct InvalidAdapterError {
	pub adapter_backend: &'static str,
	pub adapter_name: String,
	pub reason: InvalidAdapterReason,
}

/// Evaluate each adapter, eather returning a weight (higher weight wins) or a reason why it can't be selected
fn evaluate_adapter(adapter: &wgpu::Adapter, config: &DetectorConfig) -> Result<u32, InvalidAdapterReason> {
    let caps = adapter.get_downlevel_capabilities();
    if !caps.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
        return Err(InvalidAdapterReason::NoComputeShaders);
    }
    drop(caps);

    let limits = adapter.limits();
    let max_dim = limits.max_texture_dimension_2d as usize;
    if let Err(e) = config.source_dimensions.check(..max_dim, ..max_dim) {
        return Err(InvalidAdapterReason::TextureSize(e));
    }
    drop(limits);

    {
        // We need read/write/storage/texture for R8Uint
        let r8u = adapter.get_texture_format_features(TextureFormat::R8Uint);
        let requested_usages = TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        if !r8u.allowed_usages.contains(requested_usages) {
            return Err(InvalidAdapterReason::MissingTextureFeature(TextureFormat::R8Uint, requested_usages.difference(r8u.allowed_usages)));
        }
    }

    {
        // We need read/write/storage/texture for Rg8Uint
        let rg8u = adapter.get_texture_format_features(TextureFormat::R16Uint);
        let requested_usages = if config.debug() {
            TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING
        } else {
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING
        };
        if !rg8u.allowed_usages.contains(requested_usages) {
            return Err(InvalidAdapterReason::MissingTextureFeature(TextureFormat::Rg8Uint, requested_usages.difference(rg8u.allowed_usages)));
        }
    }

    let mut points = 0;
    let info = adapter.get_info();
    match info.device_type {
        wgpu::DeviceType::Other => {},
        wgpu::DeviceType::IntegratedGpu => {
            if config.acceleration.prefer_high_power() {
                points += 10;
            } else {
                points += 50;
            }
        },
        wgpu::DeviceType::DiscreteGpu => {
            if config.acceleration.prefer_high_power() {
                points += 50;
            } else {
                points += 10;
            }
        },
        wgpu::DeviceType::VirtualGpu => {
            //TODO: what is a virtual GPU?
            points += 5;
        },
        wgpu::DeviceType::Cpu => {
            if !config.acceleration.allow_cpu() {
                return Err(InvalidAdapterReason::RequestMismatch);
            }
        }
    }

    let features = adapter.features();

    if !config.allow_concurrency {
        //TODO: points if we can cache textures?
    }

    if config.debug() {
        // Extra points for debug features
        if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            points += 2;
        }
        if features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
            points += 1;
        }
    }

    Ok(points)
}

enum AdapterData {
    /// All adapters are bad (keep reasons for debugging)
    AllBad(Vec<InvalidAdapterError>),
    Best {
        value: Adapter,
        weight: u32,
    },
}

impl Default for AdapterData {
    fn default() -> Self {
        Self::AllBad(vec![])
    }
}

impl AdapterData {
    /// Update with new adapter/weight pair
    fn update_adapter(&mut self, adapter: wgpu::Adapter, weight: u32) {
        let should_replace = match &self {
            Self::AllBad(_) => true,
            // Argmax
            Self::Best { weight: prev_weight, .. } => weight > *prev_weight
        };
        if should_replace {
            *self = AdapterData::Best { value: adapter, weight }
        }
    }
    /// Update with new failure
    fn add_error(&mut self, adapter: wgpu::Adapter, reason: InvalidAdapterReason) {
        if let Self::AllBad(reasons) = self {
            let info = adapter.get_info();
            reasons.push(InvalidAdapterError {
                adapter_backend: info.backend.to_str(),
                adapter_name: info.name,
                reason,
            });
        }
    }

    fn update(&mut self, result: Result<u32, InvalidAdapterReason>, adapter: wgpu::Adapter) {
        match result {
            Ok(weight) => self.update_adapter(adapter, weight),
            Err(reason) => self.add_error(adapter, reason),
        }
    }
}

impl From<AdapterData> for Result<Adapter, WgpuBuildError> {
    fn from(value: AdapterData) -> Self {
        match value {
            AdapterData::AllBad(reasons) if reasons.is_empty() => Err(WgpuBuildError::NoAdapters),
            AdapterData::AllBad(mut reasons) => {
                reasons.shrink_to_fit();
                Err(WgpuBuildError::BadAdapters(reasons))
            },
            AdapterData::Best { value, .. } => Ok(value),
        }
    }
}

fn make_request_options(config: &DetectorConfig) -> RequestAdapterOptions {
    let mut opts = RequestAdapterOptions::default();
    if config.acceleration.prefer_high_power() {
        opts.power_preference = PowerPreference::HighPerformance;
    }
    if config.acceleration.allow_cpu() && config.acceleration.is_required() {
        opts.force_fallback_adapter = true;
    }
    opts
}

async fn request_adapter_async(instance: &wgpu::Instance, config: &DetectorConfig) -> Result<Adapter, WgpuBuildError> {
    let options = make_request_options(config);
    match instance.request_adapter(&options).await {
        Some(adapter) => Ok(adapter),
        None => Err(WgpuBuildError::NoAdapters)
    }
}

/// Get adapter compatible with config
pub(super) async fn select_adapter(config: &DetectorConfig) -> Result<Adapter, WgpuBuildError> {
    let mut desc = InstanceDescriptor::default();
    desc.backends = wgpu::Instance::enabled_backend_features();
    desc.backends -= wgpu::Backends::VULKAN;


    #[cfg(feature="debug")]
    println!("WGPU enabled backends: {:?}", desc.backends);

    let inst = wgpu::Instance::new(desc);

    let result = {
        let adapters = inst.enumerate_adapters(wgpu::Backends::all());
        let mut accumulator = AdapterData::default();
        for adapter in adapters.into_iter() {
            accumulator.update(evaluate_adapter(&adapter, config), adapter);
        }
        accumulator.into()
    };
    
    if let Err(WgpuBuildError::NoAdapters) = result {
        // On wasm, we won't get any values from enumerate_adapters
        // Try falling back on request_adapter
        request_adapter_async(&inst, config).await
    } else {
        result
    }
}

pub(super) async fn request_device(adapter: wgpu::Adapter, config: &DetectorConfig) -> Result<(wgpu::Device, wgpu::Queue), WgpuBuildError> {
    // println!("Adapter limits: {:?}", adapter.limits());
	
    let features = {
        use wgpu::Features;
        let mut request_features = Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        
        // Debug features
        if config.debug() || true {
            request_features |= Features::TIMESTAMP_QUERY | Features::PIPELINE_STATISTICS_QUERY;
        }

        // Enable MAPPABLE_PRIMARY_BUFFERS for iGPUs
        //TODO: check the performance implications of this
        if adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu {
            request_features |= Features::MAPPABLE_PRIMARY_BUFFERS;
        }

        // Filter only available features
        adapter.features().intersection(request_features)
    };

    let limits = {
        let adapter_limits = adapter.limits();
        wgpu::Limits {
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d,
            max_bind_groups: 2,
            max_bindings_per_bind_group: 4,
            max_compute_workgroup_storage_size: adapter_limits.max_compute_invocations_per_workgroup,
            max_compute_invocations_per_workgroup: adapter_limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: 1,
            max_compute_workgroups_per_dimension: adapter_limits.max_compute_workgroups_per_dimension,
            ..wgpu::Limits::downlevel_defaults()
        }
    };

    let res = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("AprilTag WebGPU"),
                required_features: features,
                required_limits: limits,
            },
            None,
        )
        .await?;
    Ok(res)
}