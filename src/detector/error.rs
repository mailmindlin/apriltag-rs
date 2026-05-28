use std::alloc::AllocError;

use rayon::ThreadPoolBuildError;
use thiserror::Error;

use crate::util::image::ImageAllocError;

#[derive(Copy, Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum ImageDimensionError {
    #[error("Width too small (actual: {actual}, minimum: {minimum})")]
    WidthTooSmall {
        actual: usize,
        minimum: usize,
    },
    #[error("Width too small (actual: {actual}, minimum: {minimum})")]
    HeightTooSmall {
        actual: usize,
        minimum: usize,
    },
    #[error("Width was too large (actual: {actual}, maximum: {maximum})")]
    WidthTooBig {
        actual: usize,
        maximum: usize,
    },
    #[error("Height was too large (actual: {actual}, maximum: {maximum})")]
    HeightTooBig {
        actual: usize,
        maximum: usize,
    },
    #[error("Too many pixels in image (actual: {actual}, maximum: {maximum})")]
    TooManyPixels {
        actual: usize,
        maximum: usize,
    }
}

/// Error generated when [detecting AprilTags](crate::AprilTagDetector::detect)
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum DetectError {
    #[error("Input image was the wrong size")]
    BadSourceImageDimensions(#[source] ImageDimensionError),
	#[error("Buffer allocation error")]
	OutOfMemory(#[from] AllocError),
    #[error("Image allocation error")]
	ImageAlloc(#[from] #[source] ImageAllocError),
    #[error("Hardware acceleration error")]
	Acceleration,
}


/// Error generated when attempting to [build](crate::DetectorBuilder::build) an [AprilTagDetector]
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DetectorBuildError {
	#[error("No AprilTag families were provided (minimum: 1)")]
	NoTagFamilies,
    #[error("There was an error when creating the thread pool")]
	Threadpool(#[from] ThreadPoolBuildError),
	#[error("There was an error allocating static buffers")]
	BufferAllocationFailure,
    #[error("The requested source dimensions were bad")]
	InvalidSourceDimensions(#[source] ImageDimensionError),
	#[error("Hardware acceleration was required, but not available")]
    AccelerationNotAvailable,
    #[cfg(feature="wgpu")]
    #[error(transparent)]
	WGPU(#[from] crate::wgpu::WgpuBuildError),
	#[cfg(feature="opencl")]
    #[error(transparent)]
	OpenCLError(crate::ocl::Error),
}

impl PartialEq for DetectorBuildError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Threadpool(_), Self::Threadpool(_)) => false, // ThreadPoolBuildError is not comparable
            (Self::InvalidSourceDimensions(l0), Self::InvalidSourceDimensions(r0)) => l0 == r0,
            #[cfg(feature="opencl")]
            (Self::OpenCLError(l0), Self::OpenCLError(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

#[cfg(feature="opencl")]
impl From<ocl::Error> for DetectorBuildError {
    fn from(value: ocl::Error) -> Self {
        Self::OpenCLError(value)
    }
}