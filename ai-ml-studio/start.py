#!/usr/bin/env python3
"""
AI ML Studio - Optimized Startup Script
Hardware: RTX 5090 + Ryzen 9 9950X + 128GB RAM
"""

import sys
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)


def main():
    """Main startup sequence"""

    logger.info("=" * 70)
    logger.info("🚀 AI ML STUDIO - STARTING UP")
    logger.info("=" * 70)

    # Enable hardware optimizations
    logger.info("\n📊 Enabling hardware optimizations...")
    try:
        from backend.core.hardware_optimizer import enable_optimizations

        enable_optimizations()
    except Exception as e:
        logger.warning(f"Could not enable optimizations: {e}")

    # Check system status
    logger.info("\n🔍 Checking system components...")

    # Check GPU
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("✗ No CUDA GPU detected")
    except:
        logger.warning("✗ Could not check GPU")

    # Check database
    try:
        from backend.core.database import check_db_connection

        if check_db_connection():
            logger.info("✓ Database: Connected")
        else:
            logger.warning("✗ Database: Not connected")
    except:
        logger.warning("✗ Could not check database")

    # Check cache
    try:
        from backend.core.cache import cache

        stats = cache.get_stats()
        if stats:
            logger.info(f"✓ Redis: Connected ({stats.get('total_keys', 0)} keys)")
        else:
            logger.warning("✗ Redis: Not connected")
    except:
        logger.warning("✗ Could not check cache")

    logger.info("\n" + "=" * 70)
    logger.info("✅ STARTUP COMPLETE!")
    logger.info("=" * 70)

    # Show menu
    show_menu()


def show_menu():
    """Show startup menu"""
    print("\n🎯 What would you like to do?")
    print("\n1. Start Web UI (Gradio)")
    print("2. Start API Server (FastAPI)")
    print("3. Start Jupyter Lab")
    print("4. Run Benchmark")
    print("5. Interactive Python Shell")
    print("6. Exit")

    choice = input("\nChoice [1-6]: ").strip()

    if choice == "1":
        start_web_ui()
    elif choice == "2":
        start_api_server()
    elif choice == "3":
        start_jupyter()
    elif choice == "4":
        run_benchmark()
    elif choice == "5":
        start_shell()
    else:
        logger.info("Goodbye!")


def start_web_ui():
    """Start Gradio web interface"""
    logger.info("🌐 Starting Gradio Web UI...")
    logger.info("Open http://localhost:7860 in your browser")

    try:
        from apps.gradio_interfaces import launch_all_interfaces

        launch_all_interfaces(share=False)
    except Exception as e:
        logger.error(f"Failed to start Web UI: {e}")


def start_api_server():
    """Start FastAPI server"""
    logger.info("🚀 Starting FastAPI Server...")
    logger.info("API docs: http://localhost:8000/docs")

    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def start_jupyter():
    """Start Jupyter Lab"""
    logger.info("📓 Starting Jupyter Lab...")

    import subprocess

    subprocess.run([
        "jupyter",
        "lab",
        "--ip=0.0.0.0",
        "--port=8888",
        "--no-browser",
        "--NotebookApp.token=''",
    ])


def run_benchmark():
    """Run hardware benchmark"""
    logger.info("⚡ Running hardware benchmark...")

    from backend.core.hardware_optimizer import hardware_optimizer

    hardware_optimizer.benchmark()

    input("\nPress Enter to continue...")
    main()


def start_shell():
    """Start interactive Python shell"""
    logger.info("🐍 Starting interactive shell...")
    logger.info("Importing common modules...")

    import code

    # Import useful modules
    import torch
    import numpy as np
    import pandas as pd
    from backend.modules.rag import RAGSystem
    from backend.modules.datasets.generator import DatasetGenerator
    from backend.modules.vision.classifier import ImageClassifier

    local_vars = {
        "torch": torch,
        "np": np,
        "pd": pd,
        "RAGSystem": RAGSystem,
        "DatasetGenerator": DatasetGenerator,
        "ImageClassifier": ImageClassifier,
    }

    banner = """
AI ML Studio Interactive Shell
-------------------------------
Available modules:
  - torch, np, pd
  - RAGSystem, DatasetGenerator, ImageClassifier

Example:
  >>> rag = RAGSystem("my_collection")
  >>> gen = DatasetGenerator()
    """

    code.interact(banner=banner, local=local_vars)


if __name__ == "__main__":
    main()
