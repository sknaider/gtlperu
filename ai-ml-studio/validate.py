#!/usr/bin/env python3
"""
AI ML Studio - Validation Script
Checks that all modules can be imported and basic functionality works
"""

import sys
from pathlib import Path
from colorama import init, Fore, Style

init(autoreset=True)

# Track results
passed = 0
failed = 0
warnings = 0


def test(name, func):
    """Run a test and track results"""
    global passed, failed
    try:
        func()
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} {name}")
        passed += 1
        return True
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} {name}: {e}")
        failed += 1
        return False


def warn(message):
    """Print warning"""
    global warnings
    print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} {message}")
    warnings += 1


print("=" * 70)
print("AI ML STUDIO - VALIDATION")
print("=" * 70)
print()

# Test 1: Core modules
print(f"{Fore.CYAN}Testing Core Modules...{Style.RESET_ALL}")


def test_core():
    from backend.core import config, cache, database, claude_client, hardware_optimizer


test("Core modules", test_core)

# Test 2: Training module
print(f"\n{Fore.CYAN}Testing Training Module...{Style.RESET_ALL}")


def test_training():
    from backend.modules.training import Trainer, TrainingConfig


test("Training module", test_training)

# Test 3: Fine-tuning module
print(f"\n{Fore.CYAN}Testing Fine-tuning Module...{Style.RESET_ALL}")


def test_finetuning():
    from backend.modules.finetuning import LoRAFineTuner, QLoRAFineTuner


test("Fine-tuning module", test_finetuning)

# Test 4: Datasets module
print(f"\n{Fore.CYAN}Testing Datasets Module...{Style.RESET_ALL}")


def test_datasets():
    from backend.modules.datasets import DatasetGenerator


test("Datasets module", test_datasets)

# Test 5: RAG module
print(f"\n{Fore.CYAN}Testing RAG Module...{Style.RESET_ALL}")


def test_rag():
    from backend.modules.rag import RAGSystem, EmbeddingModel, VectorDatabase


test("RAG module", test_rag)

# Test 6: Vision module
print(f"\n{Fore.CYAN}Testing Vision Module...{Style.RESET_ALL}")


def test_vision():
    from backend.modules.vision import ImageClassifier


test("Vision module", test_vision)

# Test 7: Augmentation module
print(f"\n{Fore.CYAN}Testing Augmentation Module...{Style.RESET_ALL}")


def test_augmentation():
    from backend.modules.augmentation import ImageAugmentor


test("Augmentation module", test_augmentation)

# Test 8: Tuning module
print(f"\n{Fore.CYAN}Testing Tuning Module...{Style.RESET_ALL}")


def test_tuning():
    from backend.modules.tuning import OptunaTuner


test("Tuning module", test_tuning)

# Test 9: API module
print(f"\n{Fore.CYAN}Testing API Module...{Style.RESET_ALL}")


def test_api():
    from backend.api import app
    assert app is not None


test("API module", test_api)

# Test 10: Environment file
print(f"\n{Fore.CYAN}Checking Configuration...{Style.RESET_ALL}")

if not Path(".env").exists():
    warn("No .env file found. Copy .env.example to .env and configure it.")
else:
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} .env file exists")
    passed += 1

# Test 11: Check PyTorch
print(f"\n{Fore.CYAN}Checking PyTorch...{Style.RESET_ALL}")


def test_pytorch():
    import torch

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
    else:
        warn("CUDA not available. GPU acceleration disabled.")


test("PyTorch import", test_pytorch)

# Summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"{Fore.GREEN}Passed:{Style.RESET_ALL} {passed}")
print(f"{Fore.RED}Failed:{Style.RESET_ALL} {failed}")
print(f"{Fore.YELLOW}Warnings:{Style.RESET_ALL} {warnings}")

if failed > 0:
    print(f"\n{Fore.RED}❌ VALIDATION FAILED{Style.RESET_ALL}")
    print("Please fix the errors above before proceeding.")
    sys.exit(1)
else:
    print(f"\n{Fore.GREEN}✅ VALIDATION PASSED{Style.RESET_ALL}")
    print("All core modules are working correctly!")

    if warnings > 0:
        print(f"\n{Fore.YELLOW}Note:{Style.RESET_ALL} There are {warnings} warnings. Review them above.")

    print("\nNext steps:")
    print("1. Configure .env with your API keys")
    print("2. Run: python start.py")
    print("3. Choose an option from the menu")
