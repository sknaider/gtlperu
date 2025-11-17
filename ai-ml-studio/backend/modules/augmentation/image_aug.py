"""
Image Augmentation with Albumentations
Optimized for training on RTX 5090
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image


class ImageAugmentor:
    """Advanced image augmentation"""

    def __init__(self, mode: str = "medium", image_size: int = 224):
        """
        Initialize augmentor

        Args:
            mode: Augmentation strength (light, medium, heavy, auto)
            image_size: Target image size
        """
        self.mode = mode
        self.image_size = image_size
        self.transform = self._get_transform()

    def _get_transform(self):
        """Get augmentation pipeline"""
        if self.mode == "light":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])

        elif self.mode == "medium":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])

        elif self.mode == "heavy":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.OneOf([
                    A.GaussNoise(),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ], p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ])

        else:  # auto
            return self._auto_augment()

    def _auto_augment(self):
        """AutoAugment policy"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
                A.RGBShift(p=1),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
            ], p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    def augment(self, image: np.ndarray) -> Dict[str, Any]:
        """Apply augmentation"""
        return self.transform(image=image)

    def augment_pil(self, image: Image.Image) -> Dict[str, Any]:
        """Augment PIL image"""
        img_array = np.array(image)
        return self.augment(img_array)
