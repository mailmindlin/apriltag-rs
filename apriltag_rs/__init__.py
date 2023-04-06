from . import apriltag_rs as raw
from typing import Optional, Sequence
import numpy as np

class Detection:
    pass

class Detector:
    def __init__(self, *, families: Optional[Sequence[str]] = None, nthreads: Optional[int] = None, quad_decimate: Optional[float] = None, quad_sigma: Optional[float] = None, refine_edges: Optional[bool] = None, decode_sharpening: Optional[float] = None, debug: Optional[bool] = None, camera_params: Optional[Sequence] = None):
        self._inner = raw.Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=debug,
            camera_params=camera_params,
        )
    
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
    
    def detect(self, image) -> Sequence[Detection]:
        assert len(image.shape) == 2
        assert image.dtype == np.uint8
        