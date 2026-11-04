"""
Simply Blur Detector: a small CNN that tells sharp images from blurry ones.
"""
from .utils import *  # noqa: F401,F403  (check_normal, equalize_hist, preprocessing, get_data, ...)
from .utils import __all__ as _utils_all

from .detection_model import DetectionModel, build_model

__version__ = '1.0.0'

__all__ = ['DetectionModel', 'build_model', *_utils_all]
