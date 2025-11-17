"""
Test Module Imports
Verify all modules can be imported without errors
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_core_imports():
    """Test core module imports"""
    try:
        from backend.core import config
        from backend.core import cache
        from backend.core import database
        from backend.core import claude_client
        from backend.core import hardware_optimizer
        assert True
    except ImportError as e:
        pytest.fail(f"Core import failed: {e}")


def test_training_module():
    """Test training module imports"""
    try:
        from backend.modules import training
        from backend.modules.training import Trainer, TrainingConfig
        assert True
    except ImportError as e:
        pytest.fail(f"Training module import failed: {e}")


def test_finetuning_module():
    """Test finetuning module imports"""
    try:
        from backend.modules import finetuning
        from backend.modules.finetuning import LoRAFineTuner, QLoRAFineTuner
        assert True
    except ImportError as e:
        pytest.fail(f"Finetuning module import failed: {e}")


def test_datasets_module():
    """Test datasets module imports"""
    try:
        from backend.modules import datasets
        from backend.modules.datasets import DatasetGenerator
        assert True
    except ImportError as e:
        pytest.fail(f"Datasets module import failed: {e}")


def test_rag_module():
    """Test RAG module imports"""
    try:
        from backend.modules import rag
        from backend.modules.rag import RAGSystem, EmbeddingModel, VectorDatabase
        assert True
    except ImportError as e:
        pytest.fail(f"RAG module import failed: {e}")


def test_vision_module():
    """Test vision module imports"""
    try:
        from backend.modules import vision
        from backend.modules.vision import ImageClassifier
        assert True
    except ImportError as e:
        pytest.fail(f"Vision module import failed: {e}")


def test_augmentation_module():
    """Test augmentation module imports"""
    try:
        from backend.modules import augmentation
        from backend.modules.augmentation import ImageAugmentor
        assert True
    except ImportError as e:
        pytest.fail(f"Augmentation module import failed: {e}")


def test_tuning_module():
    """Test tuning module imports"""
    try:
        from backend.modules import tuning
        from backend.modules.tuning import OptunaTuner
        assert True
    except ImportError as e:
        pytest.fail(f"Tuning module import failed: {e}")


def test_api_module():
    """Test API module imports"""
    try:
        from backend.api import app
        assert app is not None
    except ImportError as e:
        pytest.fail(f"API module import failed: {e}")


def test_gradio_interfaces():
    """Test Gradio interfaces import"""
    try:
        import apps.gradio_interfaces
        assert True
    except ImportError as e:
        pytest.fail(f"Gradio interfaces import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
