"""
Hardware-Specific Optimizations
Optimized for:
- CPU: AMD Ryzen 9 9950X (16 cores, 32 threads)
- GPU: NVIDIA RTX 5090 (24GB+ VRAM)
- RAM: 128GB DDR5
"""

import torch
import os
from loguru import logger
from typing import Optional


class HardwareOptimizer:
    """Optimize settings for high-end hardware"""

    def __init__(self):
        """Initialize hardware optimizer"""
        self.cpu_cores = 16
        self.cpu_threads = 32
        self.gpu_vram_gb = 24
        self.system_ram_gb = 128

        logger.info("Initializing hardware optimizations...")
        self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware"""
        # CPU info
        import multiprocessing

        available_cores = multiprocessing.cpu_count()
        logger.info(f"Detected {available_cores} CPU threads")

        # GPU info
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Detected {gpu_count} CUDA GPUs")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

    def optimize_pytorch(self):
        """Apply PyTorch optimizations for RTX 5090"""
        logger.info("Applying PyTorch optimizations...")

        # Enable TF32 for Ampere/Ada/Hopper GPUs (RTX 5090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ Enabled TF32 for faster computation")

        # Enable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = True
        logger.info("✓ Enabled cuDNN benchmark mode")

        # Enable cuDNN deterministic mode if needed
        # torch.backends.cudnn.deterministic = True

        # Set optimal number of threads for Ryzen 9 9950X
        torch.set_num_threads(self.cpu_threads)
        logger.info(f"✓ Set PyTorch threads to {self.cpu_threads}")

        # Enable JIT fusion
        torch.jit.enable_onednn_fusion(True)
        logger.info("✓ Enabled JIT fusion")

    def optimize_dataloader(
        self, dataset_size: Optional[int] = None
    ) -> dict:
        """
        Get optimal DataLoader settings

        Args:
            dataset_size: Size of dataset

        Returns:
            Dictionary with optimal settings
        """
        # With 128GB RAM and RTX 5090, we can use large batches
        settings = {
            "batch_size": 128,  # Large batch size for RTX 5090
            "num_workers": 16,  # Use all physical cores
            "pin_memory": True,  # Faster data transfer to GPU
            "persistent_workers": True,  # Keep workers alive
            "prefetch_factor": 4,  # Prefetch 4 batches per worker
        }

        logger.info("Optimal DataLoader settings:")
        for key, value in settings.items():
            logger.info(f"  {key}: {value}")

        return settings

    def optimize_training(self) -> dict:
        """
        Get optimal training settings

        Returns:
            Dictionary with training configurations
        """
        settings = {
            # Large batch sizes for RTX 5090
            "batch_size": 128,
            "gradient_accumulation_steps": 1,  # No need with large VRAM

            # Mixed precision for faster training
            "mixed_precision": "bf16",  # BFloat16 for RTX 5090

            # Memory optimizations
            "gradient_checkpointing": False,  # Disable with 24GB VRAM
            "use_8bit_optimizer": False,  # No need with 128GB RAM

            # Compilation
            "torch_compile": True,  # Use torch.compile for 2x speedup
            "compile_mode": "max-autotune",  # Maximum optimization

            # Multi-GPU (if available)
            "distributed": False,  # Single GPU is enough
        }

        logger.info("Optimal training settings:")
        for key, value in settings.items():
            logger.info(f"  {key}: {value}")

        return settings

    def optimize_inference(self) -> dict:
        """
        Get optimal inference settings

        Returns:
            Dictionary with inference configurations
        """
        settings = {
            "batch_size": 256,  # Even larger for inference
            "use_amp": True,  # Automatic mixed precision
            "torch_compile": True,  # Compile for faster inference
            "use_channels_last": True,  # Better memory layout
        }

        logger.info("Optimal inference settings:")
        for key, value in settings.items():
            logger.info(f"  {key}: {value}")

        return settings

    def get_optimal_embedding_batch_size(self) -> int:
        """Get optimal batch size for embedding models"""
        # With RTX 5090, we can process large batches
        return 512

    def get_optimal_lora_config(self) -> dict:
        """
        Get optimal LoRA configuration for fine-tuning

        Returns:
            LoRA configuration optimized for RTX 5090
        """
        # With 24GB VRAM, we can use higher ranks and full precision
        config = {
            "r": 64,  # Higher rank for better quality
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "batch_size": 8,  # Good balance
            "gradient_accumulation_steps": 1,
            "fp16": False,  # Use bf16 instead
            "bf16": True,  # Better for RTX 5090
            "max_grad_norm": 1.0,
        }

        return config

    def enable_all_optimizations(self):
        """Enable all available optimizations"""
        logger.info("="*60)
        logger.info("ENABLING ALL HARDWARE OPTIMIZATIONS")
        logger.info("="*60)

        self.optimize_pytorch()

        # Set environment variables
        os.environ["OMP_NUM_THREADS"] = str(self.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.cpu_threads)

        logger.info("="*60)
        logger.info("ALL OPTIMIZATIONS ENABLED!")
        logger.info("="*60)

    def benchmark(self):
        """Run quick benchmark"""
        logger.info("Running hardware benchmark...")

        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return

        # Matrix multiplication benchmark
        size = 8192
        device = torch.device("cuda")

        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warmup
        for _ in range(5):
            c = torch.matmul(a, b)

        # Benchmark
        import time

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):
            c = torch.matmul(a, b)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        tflops = (2 * size**3 * 100) / elapsed / 1e12

        logger.info(f"Matrix multiplication: {tflops:.2f} TFLOPS")
        logger.info(f"Time: {elapsed:.3f}s for 100 iterations")


# Global optimizer instance
hardware_optimizer = HardwareOptimizer()


def enable_optimizations():
    """Enable all optimizations (call at startup)"""
    hardware_optimizer.enable_all_optimizations()
