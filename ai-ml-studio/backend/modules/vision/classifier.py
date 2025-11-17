"""
Image Classification with SOTA Models
Optimized for RTX 5090
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import timm  # PyTorch Image Models
from loguru import logger


class ImageClassifier:
    """Image classifier with modern architectures"""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 1000,
        pretrained: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize image classifier

        Args:
            model_name: Model architecture name
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            device: Device to use
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device(device)

        logger.info(f"Loading model: {model_name}")

        # Load model using timm (supports 1000+ models)
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Get model-specific transforms
        self.transform = self._get_transforms()

        # Enable optimizations for RTX 5090
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            # Compile model for faster inference (PyTorch 2.0+)
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except:
                pass

        logger.info(f"Model loaded on {self.device}")

    def _get_transforms(self):
        """Get preprocessing transforms"""
        # Get model-specific config
        data_config = timm.data.resolve_model_data_config(self.model)
        return timm.data.create_transform(**data_config, is_training=False)

    @torch.no_grad()
    def predict(
        self, image: Image.Image, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Predict class probabilities

        Args:
            image: PIL Image
            top_k: Number of top predictions to return

        Returns:
            List of (class_id, probability) tuples
        """
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.cuda.amp.autocast():  # Automatic mixed precision
            output = self.model(input_tensor)

        # Get probabilities
        probs = torch.softmax(output, dim=1)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

        results = [
            (int(idx), float(prob))
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]

        return results

    @torch.no_grad()
    def predict_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> List[List[Tuple[int, float]]]:
        """
        Batch prediction for multiple images

        Args:
            images: List of PIL Images
            batch_size: Batch size for inference

        Returns:
            List of predictions for each image
        """
        all_results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Preprocess batch
            inputs = torch.stack([self.transform(img) for img in batch]).to(
                self.device
            )

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Get top predictions for each image
            for prob in probs:
                top_probs, top_indices = torch.topk(prob, k=5)
                results = [
                    (int(idx), float(p)) for idx, p in zip(top_indices, top_probs)
                ]
                all_results.append(results)

        return all_results

    def predict_from_path(self, image_path: Path, top_k: int = 5):
        """Predict from image file path"""
        image = Image.open(image_path).convert("RGB")
        return self.predict(image, top_k=top_k)

    def get_features(self, image: Image.Image) -> torch.Tensor:
        """Extract features from image (before classification head)"""
        # Remove classification head temporarily
        if hasattr(self.model, "classifier"):
            original_classifier = self.model.classifier
            self.model.classifier = nn.Identity()
        elif hasattr(self.model, "fc"):
            original_fc = self.model.fc
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "head"):
            original_head = self.model.head
            self.model.head = nn.Identity()

        # Extract features
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(input_tensor)

        # Restore classifier
        if hasattr(self.model, "classifier"):
            self.model.classifier = original_classifier
        elif hasattr(self.model, "fc"):
            self.model.fc = original_fc
        elif hasattr(self.model, "head"):
            self.model.head = original_head

        return features


# Factory function for different model architectures
def create_classifier(
    architecture: str = "efficientnet_b0", num_classes: int = 1000, **kwargs
) -> ImageClassifier:
    """
    Create image classifier with specified architecture

    Popular architectures:
    - efficientnet_b0 to efficientnet_b7
    - resnet18, resnet50, resnet101
    - vit_base_patch16_224 (Vision Transformer)
    - convnext_tiny, convnext_base
    - swin_base_patch4_window7_224

    Args:
        architecture: Model architecture name
        num_classes: Number of classes
        **kwargs: Additional arguments

    Returns:
        ImageClassifier instance
    """
    return ImageClassifier(model_name=architecture, num_classes=num_classes, **kwargs)
