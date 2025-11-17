"""
Computer Vision Module
Optimized for RTX 5090
"""

from .classifier import ImageClassifier
from .detector import ObjectDetector
from .segmentation import ImageSegmentation
from .trainer import VisionTrainer

__all__ = ["ImageClassifier", "ObjectDetector", "ImageSegmentation", "VisionTrainer"]
