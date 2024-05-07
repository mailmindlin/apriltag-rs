from . import apriltag_rs_native as raw
from typing import Optional, Sequence, Union, Literal
import numpy as np

from .apriltag_rs_native import (
	AprilTagFamily,
	DetectorConfig,
	Detection,
	TimeProfile,
	Detections,
	AprilTagPose,
	AprilTagPoseWithError,
	PoseEstimator,
)

class DetectorBuilder:
	def __init__(self, *,
		families: Optional[Sequence[Union[str, AprilTagFamily, tuple[str, int], tuple[AprilTagFamily, int]]]] = None,
		nthreads: Optional[int] = None,
		quad_decimate: Optional[float] = None,
		quad_sigma: Optional[float] = None,
		refine_edges: Optional[bool] = None,
		decode_sharpening: Optional[float] = None,
		debug: Optional[bool] = None,
		camera_params: Optional[Sequence[float]] = None,
	):
		self._inner = raw.DetectorBuilder()
		if families is not None:
			for family in families:
				self.add_family()
		
		if nthreads is not None:
			pass
	
	@property
	def config(self) -> DetectorConfig:
		return self._inner.config

	@config.setter
	def config(self, value: DetectorConfig):
		self._inner.config = value
	
	@property
	def acceleration(self) -> Literal["prefer", "prefer_gpu", "required", "required_gpu"] | None:
		"What kind of hardware acceleration should we use?"
		return self._inner.acceleration
	
	@acceleration.setter
	def acceleration(self, value: Literal["prefer", "prefer_gpu", "required", "required_gpu"] | str | None):
		self._inner.acceleration = value

	def add_family(self, family: Union[AprilTagFamily, str], hamming_bits: Optional[int] = 2):
		if isinstance(family, str):
			family = AprilTagFamily.for_name(family)
		
		if (not isinstance(hamming_bits, int)) or not (0 <= hamming_bits <= 3):
			raise ValueError(f'Invalid hamming distance: {hamming_bits}')
		
		self._inner.add_family(
			family,
			hamming_bits,
		)
	
	def build(self) -> 'Detector':
		return self._inner.build()

# Wrap around raw.Detector()
class Detector:
	@staticmethod
	def builder() -> DetectorBuilder:
		return DetectorBuilder()
	
	def __init__(self, *,
			families: Optional[Sequence[Union[str, AprilTagFamily, tuple[str, int], tuple[AprilTagFamily, int]]]] = None,
			nthreads: Optional[int] = None,
			quad_decimate: Optional[float] = None,
			quad_sigma: Optional[float] = None,
			refine_edges: Optional[bool] = None,
			decode_sharpening: Optional[float] = None,
			debug: Optional[bool] = None,
			camera_params: Optional[Sequence[float]] = None,
	):
		builder = self.builder()
		if nthreads is not None:
			builder.config.nthreads = nthreads
		
		self._inner = raw.Detector(
			nthreads=nthreads,
			quad_decimate=quad_decimate,
			quad_sigma=quad_sigma,
			refine_edges=refine_edges,
			decode_sharpening=decode_sharpening,
			debug=debug,
			camera_params=camera_params,
		)

		if families is not None:
			for family in families:
				if isinstance(family, tuple):
					family, hamming = family
				else:
					hamming = 2
				self.add_family(family, hamming)
	
	@property
	def config(self) -> DetectorConfig:
		"Get configuration for this detector"
		return self._inner.config
	
	def detect(self, image: np.ndarray[np.uint8]) -> Detections:
		assert len(image.shape) == 2
		assert image.dtype == np.uint8
		return self._inner.detect(image)