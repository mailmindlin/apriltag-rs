use hashbrown::HashMap;


#[derive(Default)]
pub(super) struct ConstBuilder(String);

impl ConstBuilder {
    pub(super) fn set_u32(&mut self, name: &str, value: u32) {
        let line = format!("const {name}: u32 = {value}u;\n");
        self.0.push_str(&line);
    }

    pub(super) fn set_i32(&mut self, name: &str, value: i32) {
        let line = format!("const {name}: i32 = {value};\n");
        self.0.push_str(&line);
    }

    pub(super) fn set_f32(&mut self, name: &str, value: f32) {
        let line = format!("const {name}: f32 = {value};\n");
        self.0.push_str(&line);
    }

    pub(super) fn finish(self, program: &str) -> String {
        self.0 + program
    }
}

pub(super) struct ProgramBuilder {
    name: &'static str,
    replacements: HashMap<String, String>,
    text: String,
}

impl ProgramBuilder {
    pub(super) fn new(name: &'static str) -> Self {
        Self {
            name,
            replacements: HashMap::new(),
            text: String::new(),
        }
    }

    pub(super) fn set_u32(&mut self, name: &str, value: u32) {
        self.replacements.insert(name.into(), format!("{value}u"));
    }

    pub(super) fn append(&mut self, text: &str) {
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

    pub(super) fn finish(self) -> String {
        // println!("Program {}: {}", self.name, self.text);
        self.text
    }

    pub(super) fn build(self, device: &wgpu::Device) -> wgpu::ShaderModule {
        let name = self.name;
        let text = self.finish();
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(text)),
        })
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

pub(super) struct DataStore<T>(Option<T>);

impl<T> DataStore<T> {
    pub(super) fn store<'a>(&'a mut self, value: T) -> &'a T {
        self.0 = Some(value);
        self.0.as_ref().unwrap()
    }
}

impl<T> AsRef<Option<T>> for DataStore<T> {
    fn as_ref(&self) -> &Option<T> {
        &self.0
    }
}

impl<T> Default for DataStore<T> {
    fn default() -> Self {
        Self(None)
    }
}