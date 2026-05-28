from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Iterator, Union, Optional, Any, Iterable
import numpy as np

AprilTagFamilyName = Literal['tag16h5', 'tag25h9', 'tag36h10', 'tag36h11', 'tagCircle21h7']

class AprilTagFamily:
    @staticmethod
    def names() -> list[AprilTagFamilyName]:
        "Get known names of AprilTag families"
        ...
    @staticmethod
    def for_name(name: AprilTagFamilyName) -> 'AprilTagFamily':
        """
        Get AprilTagFamily by name.
        Raises
        ------
        ValueError
            If `name` is not a valid/known AprilTag family
        """

    codes: list[int]
    "List of codes (bitfield integers)"
    bits: list[tuple[int, int]]
    "List of bit positions"
    width_at_border: int
    total_width: int
    reversed_border: bool
    min_hamming: int
    "Minimum hamming distance between any two codes in this family"
    name: str
    "Family name"
    
    def to_image(self, id: int, /) -> np.ndarray[tuple[int, int], np.uint8]:
        """
        Generate greyscale image of AprilTag at index
        Parameters
        ----------
        id: int
            AprilTag ID

        Returns
        -------
        np.ndarray[np.uint8]

        Raises
        ------
        ValueError
            If `id` is not a valid index
        """

    def __repr__(self) -> str: ...

class DetectorConfig:
    "Configuration for the AprilTag detector"
    @property
    def nthreads(self) -> int:
        "Number of theads to use (zero = automatic)"
    
    @nthreads.setter
    def nthreads(self, value: int):
        ...
    
    @property
    def quad_decimate(self) -> float:
        """
        Detection of quads can be done on a lower-resolution image,
        improving speed at a cost of pose accuracy and a slight
        decrease in detection rate. Decoding the binary payload is
        still done at full resolution.
        """
    
    @quad_decimate.setter
    def quad_decimate(self, value: float):
        ...
    
    @property
    def quad_sigma(self) -> float:
        """
        What Gaussian blur should be applied to the segmented image
        (used for quad detection?)  Parameter is the standard deviation
        in pixels.  Very noisy images benefit from non-zero values
        (e.g. 0.8).
        """
    
    @quad_sigma.setter
    def quad_sigma(self, value: float):
        ...
    
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
    
    @refine_edges.setter
    def refine_edges(self, value: bool):
        ...
    
    @property
    def decode_sharpening(self) -> float:
        """
        How much sharpening should be done to decoded images? This
        can help decode small tags but may or may not help in odd
        lighting conditions or low light conditions.

        The default value is 0.25.
        """
    
    @decode_sharpening.setter
    def decode_sharpening(self, value: float):
        ...
    
    @property
    def debug(self) -> bool:
        "Whether or not debug mode is enabled"
    
    @debug.setter
    def debug(self, value: bool):
        ...

    @property
    def debug_path(self) -> Optional[str]:
        "Path to write debug images to"
    
    @debug_path.setter
    def debug_path(self, value: Optional[str]):
        ...
    
    def __repr__(self) -> str: ...

    def as_dict(self) -> dict[str, Any]: ...

class Detection:
    tag_id: int
    tag_family: str
    family: AprilTagFamily
    hamming: int
    decision_margin: float
    center: tuple[float, float]
    H: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]
    corners: list[tuple[float, float]]
    def __repr__(self) -> str: ...

class TimeProfile:
    @property
    def total_duration(self) -> float: ...
    def as_list(self) -> list[tuple[float, str]]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, item: int) -> tuple[float, str]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Detections:
    @property
    def nquads(self) -> int:
        "Number of quads found (idk why this is here)"
    @property
    def time_profile(self) -> TimeProfile:
        "Timing information from detection"
    @property
    def detections(self) -> list[Detection]:
        "List of detections"
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Detection]: ...
    def __getitem__(self, idx: int) -> Detection: ...
    def __repr__(self) -> str: ...

class AprilTagPose:
    def R(self) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]: ...
    def t(self) -> np.ndarray[Literal[3], np.dtype[np.float64]]: ...

class AprilTagPoseWithError:
    def pose(self) -> AprilTagPose: ...
    def error(self) -> float: ...

class PoseEstimator:
    def __init__(self, cx: float, cy: float, fx: float, fy: float, tagsize: float): ...
    def estimate_pose_for_tag_homography(detection: Detection) -> AprilTagPose: ...
    def estimate_tag_pose_orthogonal_iteration(detection: Detection, n_iters: int = ...) -> Iterable[AprilTagPoseWithError]: ...
    def estimate_tag_pose(detection: Detection) -> AprilTagPoseWithError: ...

class Detector:
	def __init__(self): ...
	@property
	def config(self) -> DetectorConfig:
		"Get configuration for this detector"
	
	def detect(self, image: np.ndarray[tuple[int, int], np.uint8]) -> Detections:
		"Detect AprilTags in image"
		...

class DetectorBuilder:
	def __init__(self): ...
	
	@property
	def config(self) -> DetectorConfig:
		...

	@config.setter
	def config(self, value: DetectorConfig):
		...
	
	@property
	def acceleration(self) -> Literal["prefer", "prefer_gpu", "required", "required_gpu"] | None:
		"What kind of hardware acceleration should we use?"
	
	@acceleration.setter
	def acceleration(self, value: Literal["prefer", "prefer_gpu", "required", "required_gpu"] | None):
		...

	def add_family(self, family: Union[AprilTagFamily, AprilTagFamilyName], hamming_bits: Optional[int] = 2):
		...
	
	def build(self) -> 'Detector':
		...