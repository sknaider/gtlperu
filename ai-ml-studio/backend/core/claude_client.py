"""
Claude API Client for AI ML Studio
"""

from anthropic import Anthropic, AsyncAnthropic
from typing import List, Dict, Optional, AsyncIterator
from .config import settings
from loguru import logger
import json


class ClaudeClient:
    """Client for interacting with Claude API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key (uses settings if not provided)
        """
        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            logger.warning("No Anthropic API key provided")

        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        self.model = settings.claude_model
        self.max_tokens = settings.claude_max_tokens
        self.temperature = settings.claude_temperature

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text using Claude

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop sequences

        Returns:
            Generated text
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }

            if system:
                kwargs["system"] = system

            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            response = self.client.messages.create(**kwargs)

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error generating with Claude: {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Async generate text using Claude

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop sequences

        Returns:
            Generated text
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }

            if system:
                kwargs["system"] = system

            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            response = await self.async_client.messages.create(**kwargs)

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error generating with Claude: {e}")
            raise

    async def stream_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream text generation using Claude

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Text chunks
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }

            if system:
                kwargs["system"] = system

            async with self.async_client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Error streaming with Claude: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Generate JSON response using Claude

        Args:
            prompt: User prompt (should request JSON output)
            system: System prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON response
        """
        try:
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."

            response = self.generate(
                prompt=json_prompt,
                system=system,
                max_tokens=max_tokens,
            )

            # Extract JSON from response
            # Handle cases where Claude might add explanation before/after JSON
            response = response.strip()

            # Try to find JSON in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse entire response
                return json.loads(response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise
        except Exception as e:
            logger.error(f"Error generating JSON with Claude: {e}")
            raise

    async def generate_json_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Async generate JSON response using Claude

        Args:
            prompt: User prompt (should request JSON output)
            system: System prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON response
        """
        try:
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."

            response = await self.generate_async(
                prompt=json_prompt,
                system=system,
                max_tokens=max_tokens,
            )

            # Extract JSON from response
            response = response.strip()

            # Try to find JSON in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return json.loads(response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise
        except Exception as e:
            logger.error(f"Error generating JSON with Claude: {e}")
            raise

    def generate_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Generate multiple responses (sequential for now)

        Args:
            prompts: List of prompts
            system: System prompt
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature

        Returns:
            List of generated responses
        """
        responses = []

        for prompt in prompts:
            try:
                response = self.generate(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Error in batch generation for prompt: {e}")
                responses.append(f"ERROR: {str(e)}")

        return responses


# Global Claude client instance
claude_client = ClaudeClient()
