use std::{sync::Arc, ops::Deref};

use rayon::ThreadPoolBuildError;

use crate::{AprilTagFamily, quickdecode::QuickDecode, AddFamilyError};
use super::{AprilTagDetector, DetectorConfig};

#[derive(Default, Clone)]
pub struct DetectorBuilder {
	pub config: DetectorConfig,
	tag_families: Vec<Arc<QuickDecode>>,
}

impl From<AprilTagDetector> for DetectorBuilder {
    fn from(value: AprilTagDetector) -> Self {
        Self {
            config: value.params,
            tag_families: value.tag_families
        }
    }
}

impl DetectorBuilder {
	pub fn new(config: DetectorConfig) -> Self {
		Self {
			config,
			tag_families: Vec::new(),
		}
	}

	/// Add a family to the apriltag detector.
	/// 
	/// A single instance should only be provided to one apriltag detector instance.
	pub fn add_family_bits(&mut self, family: Arc<AprilTagFamily>, bits_corrected: usize) -> Result<(), AddFamilyError> {
        let qd = QuickDecode::new(family, bits_corrected)?;
		self.tag_families.push(Arc::new(qd));
		Ok(())
	}

	pub fn with_family(mut self, fam: Arc<AprilTagFamily>, bits_corrected: usize) -> Result<Self, AddFamilyError> {
		self.add_family_bits(fam, bits_corrected)?;
		Ok(self)
	}

	pub fn clear_families(&mut self) {
		self.tag_families.clear();
	}

	pub fn remove_family(&mut self, fam: &AprilTagFamily) {
		if let Some(idx) = self.tag_families.iter().position(|qd| qd.family.deref().eq(fam)) {
			self.tag_families.remove(idx);
		}
	}

	pub fn build(self) -> Result<AprilTagDetector, DetectorBuildError> {
        AprilTagDetector::new(self.config, self.tag_families)
	}
}

#[derive(Debug)]
#[non_exhaustive]
pub enum DetectorBuildError {
	Threadpool(ThreadPoolBuildError),
}

impl From<ThreadPoolBuildError> for DetectorBuildError {
    fn from(value: ThreadPoolBuildError) -> Self {
        Self::Threadpool(value)
    }
}