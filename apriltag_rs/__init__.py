from . import apriltag_rs as raw
from typing import Optional, Sequence, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
	class AprilTagFamily:
		@staticmethod
		def for_name(name: str) -> Optional['AprilTagFamily']:
			return raw.AprilTagFamily.for_name(name)

		codes: list[int]
		bits: list[(int, int)]
		width_at_border: int
		total_width: int
		reversed_border: bool
		min_hamming: int
		name: str

		def __init__(self, raw) -> None:
			self._inner = raw
		
		def to_image(self, idx: int) -> np.ndarray[np.uint8]:
			pass
	

	class Detection:
		tag_id: int
		tag_family: str
		hamming: int
		decision_margin: float
		center: tuple[float, float]
		corners: list[tuple[float, float]]

	class Detections:
		nquads: int
		detections: list[Detection]
else:
	AprilTagFamily = raw.AprilTagFamily
	Detections = raw.Detections
	Detection = raw.Detection

class Detector:
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
	def nthreads(self) -> int:
		return self._inner.nthreads
	
	@nthreads.setter
	def nthreads(self, value: int):
		self._inner.nthreads = value
	
	@property
	def quad_decimate(self) -> float:
		return self._inner.quad_decimate
	
	@quad_decimate.setter
	def quad_decimate(self, value: float):
		self._inner.quad_decimate = value
	
	@property
	def quad_sigma(self) -> float:
		return self._inner.quad_sigma
	
	@quad_sigma.setter
	def quad_sigma(self, value: float):
		self._inner.quad_sigma = value
	
	@property
	def refine_edges(self) -> bool:
		"""
		When true, the edges of the each quad are adjusted to "snap to" strong
		gradients nearby. This is useful when decimation is employed, as it can
		increase the quality of the initial quad estimate substantially.
		Generally recommended to be on (1).
		
		Very computationally inexpensive. Option is ignored if 
		`quad_decimate = 1`.
		"""
		return self._inner.refine_edges
	
	@refine_edges.setter
	def refine_edges(self, value: bool):
		self._inner.refine_edges = value
	
	@property
	def decode_sharpening(self) -> float:
		return self._inner.decode_sharpening
	
	@decode_sharpening.setter
	def decode_sharpening(self, value: float):
		self._inner.decode_sharpening = value
	
	@property
	def debug(self) -> bool:
		"Whether or not debug mode is enabled"
		return self._inner.debug
	
	@debug.setter
	def debug(self, value: bool):
		self._inner.debug = value
	
	def add_family(self, family: Union[AprilTagFamily, str], hamming_bits: Optional[int] = 2):
		if isinstance(family, str):
			family = AprilTagFamily.for_name(family)
		
		if (not isinstance(hamming_bits, int)) or not (0 <= hamming_bits <= 3):
			raise ValueError(f'Invalid hamming distance: {hamming_bits}')
		
		self._inner.add_family(
			family,
			hamming_bits,
		)
	
	def detect(self, image: np.ndarray[np.uint8]) -> Detections:
		assert len(image.shape) == 2
		assert image.dtype == np.uint8
		return self._inner.detect(image)