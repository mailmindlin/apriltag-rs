use std::{any::Any, collections::HashMap};

use crate::wgpu::WgpuBuildError;

pub(in super::super) struct ProgramBuilder {
    name: &'static str,
    replacements: HashMap<String, String>,
    text: String,
}

impl ProgramBuilder {
    pub(in super::super) fn new(name: &'static str) -> Self {
        Self {
            name,
            replacements: HashMap::new(),
            text: String::new(),
        }
    }

    /// Set a string replacement u32
    pub(in super::super) fn set_u32(&mut self, name: &str, value: u32) {
        self.replacements.insert(name.into(), format!("{value}u"));
    }

    pub(in super::super) fn append(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        // Append newline when necessary
        if !(self.text.is_empty() || self.text.ends_with("\n")) {
            self.text += "\n";
        }

        if self.replacements.is_empty() {
            self.text += text;
        } else {
            // Apply substitutions
            let mut remaining = &text[..];
            while let Some(idx) = remaining.find("$") {
                let pfx = &remaining[..idx];
                self.text += pfx;
                remaining = &remaining[idx+1..];

                let end_idx = remaining.find(|c: char| !c.is_alphanumeric() && c != '_')
                    .expect("String subst variable ends");
                let key = &remaining[..end_idx];
                match self.replacements.get(key) {
                    Some(value) => {
                        self.text += value;
                    },
                    None => panic!("Missing subsitution for variable {key} in program {}", self.name),
                }
                remaining = &remaining[end_idx..];
            }
            self.text += remaining;
        }
    }

    pub(in super::super) fn finish(self) -> String {
        // println!("Program {}: {}", self.name, self.text);
        self.text
    }

    pub(in super::super) async fn build<'a>(self, device: &'a wgpu::Device) -> Result<ShaderModule<'a>, WgpuBuildError> {
        let name = self.name;
        let text = self.finish();
        
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(text)),
        });
        if let Some(err) = device.pop_error_scope().await {
            if let wgpu::Error::Validation { description, .. } = &err {
                println!("{description}");
            }
            return Err(WgpuBuildError::RuntimeError(err));
        }
        
        Ok(ShaderModule {
            device,
            module,
            constants: HashMap::new(),
        })
    }
}

pub(in super::super) struct ComputePipelineDescriptor<'a> {
    pub(crate) label: Option<&'a str>,
    pub(crate) layout: Option<&'a wgpu::PipelineLayout>,
    pub(crate) entry_point: &'a str,
}

pub(in super::super) struct ShaderModule<'a> {
    device: &'a wgpu::Device,
    module: wgpu::ShaderModule,
    constants: HashMap<String, f64>,
}

impl<'a> ShaderModule<'a> {
    #[deprecated]
    pub(crate) fn create_compute_pipeline_<'b>(&self, descriptor: &ComputePipelineDescriptor<'b>) -> wgpu::ComputePipeline {
        let descriptor_wgpu = wgpu::ComputePipelineDescriptor {
            label: descriptor.label,
            layout: descriptor.layout,
            module: &self.module,
            entry_point: &descriptor.entry_point,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &self.constants,
                zero_initialize_workgroup_memory: true,
            },
        };

        let res = self.device.create_compute_pipeline(&descriptor_wgpu);
        res
    }

    pub(crate) async fn create_compute_pipeline<'b>(&self, descriptor: ComputePipelineDescriptor<'b>) -> Result<wgpu::ComputePipeline, WgpuBuildError> {
        let descriptor_wgpu = wgpu::ComputePipelineDescriptor {
            label: descriptor.label,
            layout: descriptor.layout,
            module: &self.module,
            entry_point: &descriptor.entry_point,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &self.constants,
                zero_initialize_workgroup_memory: true,
            },
        };

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let res = self.device.create_compute_pipeline(&descriptor_wgpu);
        if let Some(err) = self.device.pop_error_scope().await {
            println!("wgpu error: {err}");
            println!("Type: {:?}", err.type_id());
            Err(WgpuBuildError::RuntimeError(err))
        } else {
            Ok(res)
        }
    }

    pub(crate) async fn create_shader_pipeline<'b>(&self, descriptor: &ComputePipelineDescriptor<'b>) -> wgpu::ComputePipeline {
        let descriptor_wgpu = wgpu::ComputePipelineDescriptor {
            label: descriptor.label,
            layout: descriptor.layout,
            module: &self.module,
            entry_point: &descriptor.entry_point,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &self.constants,
                zero_initialize_workgroup_memory: true,
            },
        };

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let res = self.device.create_compute_pipeline(&descriptor_wgpu);
        if let Some(err) = self.device.pop_error_scope().await {
            println!("wgpu error: {err}");
            println!("Type: {:?}", err.type_id());
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::ProgramBuilder;

    #[test]
    fn test_str_replace() {
        let mut builder = ProgramBuilder::new("sample");
        builder.set_u32("foo", 0);
        builder.append("a $foo b");
        assert_eq!(builder.finish(), "a 0u b");
    }
}