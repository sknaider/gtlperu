"""
QLoRA (Quantized LoRA) Fine-tuning
4-bit quantization for memory-efficient fine-tuning
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from typing import Optional
from dataclasses import dataclass
from loguru import logger
from pathlib import Path


@dataclass
class QLoRAConfig:
    """QLoRA configuration"""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    task_type: str = "CAUSAL_LM"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"  # float16, bfloat16
    bnb_4bit_quant_type: str = "nf4"  # fp4, nf4
    bnb_4bit_use_double_quant: bool = True

    # LoRA Parameters
    r: int = 64  # Higher rank for QLoRA
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[list] = None
    bias: str = "none"

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1  # Smaller batch for memory
    gradient_accumulation_steps: int = 16
    max_length: int = 512

    # Output
    output_dir: str = "./models/qlora"
    save_steps: int = 100

    # Advanced
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03


class QLoRAFineTuner:
    """QLoRA Fine-tuner for Large Language Models"""

    def __init__(self, config: QLoRAConfig):
        """
        Initialize QLoRA fine-tuner

        Args:
            config: QLoRA configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load base model with 4-bit quantization and apply LoRA"""
        logger.info(f"Loading model with QLoRA: {self.config.model_name}")

        # Quantization config
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

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
            inference_mode=False,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        logger.info("Model loaded with QLoRA (4-bit)")

    def prepare_dataset(self, dataset):
        """
        Prepare dataset for training

        Args:
            dataset: Training data

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
            return dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
        else:
            from datasets import Dataset

            dataset = Dataset.from_dict({"text": dataset})
            return dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )

    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model with QLoRA

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        if self.model is None:
            self.load_model()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset)

        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)

        # Training arguments optimized for QLoRA
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            optim="paged_adamw_32bit",  # Optimized for QLoRA
            save_steps=self.config.save_steps,
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["tensorboard"],
            fp16=True,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train
        logger.info("Starting QLoRA training...")
        self.trainer.train()

        # Save final model
        self.save_model(Path(self.config.output_dir) / "final")

    def save_model(self, output_dir: Path):
        """Save fine-tuned model"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters only (base model is quantized)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"QLoRA model saved to {output_dir}")

    def load_finetuned_model(self, adapter_path: str):
        """Load fine-tuned QLoRA model"""
        from peft import PeftModel

        # Recreate quantization config
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        # Load quantized base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        logger.info(f"QLoRA model loaded from {adapter_path}")

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using fine-tuned model"""
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

    def merge_and_save(self, output_dir: Path):
        """
        Merge LoRA weights with base model and save

        Warning: This requires loading full model in 16-bit
        """
        logger.info("Merging LoRA weights with base model...")

        # This will load full model (requires more memory)
        merged_model = self.model.merge_and_unload()

        output_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Merged model saved to {output_dir}")
