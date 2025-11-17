"""
LoRA (Low-Rank Adaptation) Fine-tuning
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from pathlib import Path


@dataclass
class LoRAConfig:
    """LoRA configuration"""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM

    # LoRA Parameters
    r: int = 8  # Rank
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[list] = None  # None = auto-detect
    bias: str = "none"  # none, all, lora_only

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512

    # Output
    output_dir: str = "./models/lora"
    save_steps: int = 100

    # Advanced
    fp16: bool = True
    gradient_checkpointing: bool = True


class LoRAFineTuner:
    """LoRA Fine-tuner for Large Language Models"""

    def __init__(self, config: LoRAConfig):
        """
        Initialize LoRA fine-tuner

        Args:
            config: LoRA configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load base model and apply LoRA"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for training
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }

        peft_config = LoraConfig(
            task_type=task_type_map[self.config.task_type],
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        logger.info("Model loaded with LoRA")

    def prepare_dataset(self, dataset):
        """
        Prepare dataset for training

        Args:
            dataset: HuggingFace dataset or list of examples

        Returns:
            Tokenized dataset
        """

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )

        if hasattr(dataset, "map"):
            # HuggingFace dataset
            return dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
        else:
            # Custom dataset
            from datasets import Dataset

            dataset = Dataset.from_dict({"text": dataset})
            return dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )

    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model with LoRA

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        if self.model is None:
            self.load_model()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset)

        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            save_steps=self.config.save_steps,
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["tensorboard"],
            gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train
        logger.info("Starting LoRA training...")
        self.trainer.train()

        # Save final model
        self.save_model(Path(self.config.output_dir) / "final")

    def save_model(self, output_dir: Path):
        """
        Save fine-tuned model

        Args:
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

    def load_finetuned_model(self, adapter_path: str):
        """
        Load fine-tuned model

        Args:
            adapter_path: Path to LoRA adapters
        """
        from peft import PeftModel

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto",
        )

        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        logger.info(f"Fine-tuned model loaded from {adapter_path}")

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text using fine-tuned model

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
