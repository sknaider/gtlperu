"""
Synthetic Dataset Generator using Claude API
"""

from backend.core.claude_client import claude_client
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
import json
from pathlib import Path
import asyncio


class DatasetGenerator:
    """Generate synthetic datasets using Claude API"""

    def __init__(self):
        """Initialize dataset generator"""
        self.client = claude_client

    def generate_text_classification_dataset(
        self,
        task_description: str,
        categories: List[str],
        num_samples: int = 100,
        examples: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate text classification dataset

        Args:
            task_description: Description of classification task
            categories: List of categories/labels
            num_samples: Number of samples to generate
            examples: Optional few-shot examples

        Returns:
            List of generated samples with text and labels
        """
        logger.info(f"Generating {num_samples} text classification samples...")

        samples_per_batch = 10
        num_batches = (num_samples + samples_per_batch - 1) // samples_per_batch

        all_samples = []

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            batch_size = min(samples_per_batch, num_samples - len(all_samples))

            prompt = self._create_classification_prompt(
                task_description,
                categories,
                batch_size,
                examples,
            )

            try:
                response = self.client.generate_json(prompt)

                if "samples" in response:
                    all_samples.extend(response["samples"])
                else:
                    logger.warning(f"Batch {batch_idx} returned unexpected format")

            except Exception as e:
                logger.error(f"Error generating batch {batch_idx}: {e}")

        return all_samples[:num_samples]

    def _create_classification_prompt(
        self,
        task_description: str,
        categories: List[str],
        num_samples: int,
        examples: Optional[List[Dict]] = None,
    ) -> str:
        """Create prompt for classification dataset generation"""

        prompt = f"""Generate {num_samples} diverse and realistic samples for the following text classification task:

Task: {task_description}

Categories: {', '.join(categories)}

"""

        if examples:
            prompt += "Examples of the desired format:\n"
            for ex in examples[:3]:
                prompt += f"- Text: \"{ex['text']}\"\n  Label: {ex['label']}\n"
            prompt += "\n"

        prompt += f"""Requirements:
1. Generate exactly {num_samples} samples
2. Each sample should have:
   - "text": The input text (varied and realistic)
   - "label": One of the categories
3. Ensure diversity in:
   - Text length and complexity
   - Vocabulary and style
   - Clear representation of each category
4. Make samples realistic and useful for training

Return as JSON with this structure:
{{
    "samples": [
        {{"text": "...", "label": "..."}},
        ...
    ]
}}
"""

        return prompt

    def generate_conversation_dataset(
        self,
        domain: str,
        num_conversations: int = 50,
        turns_per_conversation: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Generate conversation dataset

        Args:
            domain: Domain/topic of conversations (e.g., "customer support")
            num_conversations: Number of conversations to generate
            turns_per_conversation: Number of turns per conversation

        Returns:
            List of conversations
        """
        logger.info(f"Generating {num_conversations} conversations...")

        conversations = []

        for i in tqdm(range(num_conversations), desc="Generating conversations"):
            prompt = f"""Generate a realistic {turns_per_conversation}-turn conversation in the {domain} domain.

Requirements:
1. Make it natural and realistic
2. Include {turns_per_conversation} turns (alternating user and assistant)
3. Include variety in topics and scenarios within {domain}
4. Make responses helpful and contextual

Return as JSON:
{{
    "conversation": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
    ],
    "topic": "brief description of conversation topic"
}}
"""

            try:
                response = self.client.generate_json(prompt)

                if "conversation" in response:
                    conversations.append(response)

            except Exception as e:
                logger.error(f"Error generating conversation {i}: {e}")

        return conversations

    def generate_qa_dataset(
        self,
        context_source: str,
        num_pairs: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate question-answer pairs

        Args:
            context_source: Source/domain for generating QA pairs
            num_pairs: Number of QA pairs to generate

        Returns:
            List of QA pairs
        """
        logger.info(f"Generating {num_pairs} QA pairs...")

        pairs_per_batch = 5
        num_batches = (num_pairs + pairs_per_batch - 1) // pairs_per_batch

        all_pairs = []

        for batch_idx in tqdm(range(num_batches), desc="Generating QA pairs"):
            batch_size = min(pairs_per_batch, num_pairs - len(all_pairs))

            prompt = f"""Generate {batch_size} diverse question-answer pairs about {context_source}.

Requirements:
1. Questions should be:
   - Clear and specific
   - Varied in complexity
   - Realistic
2. Answers should be:
   - Accurate and informative
   - Appropriately detailed
   - Natural sounding

Return as JSON:
{{
    "pairs": [
        {{"question": "...", "answer": "..."}},
        ...
    ]
}}
"""

            try:
                response = self.client.generate_json(prompt)

                if "pairs" in response:
                    all_pairs.extend(response["pairs"])

            except Exception as e:
                logger.error(f"Error generating QA batch {batch_idx}: {e}")

        return all_pairs[:num_pairs]

    def generate_instruction_dataset(
        self,
        task_type: str,
        num_examples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate instruction-following dataset

        Args:
            task_type: Type of tasks (e.g., "coding", "writing", "reasoning")
            num_examples: Number of examples to generate

        Returns:
            List of instruction-response pairs
        """
        logger.info(f"Generating {num_examples} instruction examples...")

        examples_per_batch = 5
        num_batches = (num_examples + examples_per_batch - 1) // examples_per_batch

        all_examples = []

        for batch_idx in tqdm(range(num_batches), desc="Generating instructions"):
            batch_size = min(examples_per_batch, num_examples - len(all_examples))

            prompt = f"""Generate {batch_size} diverse instruction-response pairs for {task_type} tasks.

Requirements:
1. Instructions should be:
   - Clear and specific
   - Varied in complexity
   - Realistic user requests
2. Responses should be:
   - Complete and helpful
   - Follow instructions accurately
   - High quality

Return as JSON:
{{
    "examples": [
        {{"instruction": "...", "response": "..."}},
        ...
    ]
}}
"""

            try:
                response = self.client.generate_json(prompt)

                if "examples" in response:
                    all_examples.extend(response["examples"])

            except Exception as e:
                logger.error(f"Error generating instruction batch {batch_idx}: {e}")

        return all_examples[:num_examples]

    def save_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: Path,
        format: str = "jsonl",
    ):
        """
        Save generated dataset to file

        Args:
            dataset: Generated dataset
            output_path: Output file path
            format: Output format (jsonl, json, csv)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")

        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)

        elif format == "csv":
            import pandas as pd

            df = pd.DataFrame(dataset)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Dataset saved to {output_path}")

    async def generate_async(
        self,
        task_description: str,
        num_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate dataset asynchronously (faster)

        Args:
            task_description: Description of task
            num_samples: Number of samples

        Returns:
            Generated dataset
        """
        # Implement async generation for better performance
        # This would use async Claude API calls
        pass
