"""Unified LLM clients for PACT."""

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import anthropic
except ImportError:  # Optional unless an Anthropic-backed client is selected.
    anthropic = None
import huggingface_hub
import openai
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Environment
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Usage tracking
USAGE_KEYS = (
    "api_calls",
    "successful_calls",
    "failed_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


class LLMClient:
    """Base client with unified usage tracking."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._usage_lock = threading.Lock()
        self._usage = {key: 0 for key in USAGE_KEYS}

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        raise NotImplementedError

    # Usage API
    def get_usage(self) -> Dict[str, int]:
        """Return cumulative usage for this client."""
        with self._usage_lock:
            return dict(self._usage)

    def get_usage_since(self, snapshot: Dict[str, int]) -> Dict[str, int]:
        """Return usage accumulated after a previous snapshot."""
        current = self.get_usage()
        return {key: current[key] - snapshot.get(key, 0) for key in USAGE_KEYS}

    def reset_usage(self) -> None:
        """Reset cumulative usage."""
        with self._usage_lock:
            for key in USAGE_KEYS:
                self._usage[key] = 0

    # Internal usage helpers
    def _record_api_call(self) -> None:
        with self._usage_lock:
            self._usage["api_calls"] += 1

    def _record_failure(self) -> None:
        with self._usage_lock:
            self._usage["failed_calls"] += 1

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        """Normalize token usage from common provider response formats."""
        usage = getattr(response, "usage", None)

        if usage is None:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        input_tokens = (
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", None)

        if total_tokens is None:
            total_tokens = int(input_tokens) + int(output_tokens)

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }

    def _record_response(self, response: Any, success: bool) -> None:
        usage = self._extract_usage(response)

        with self._usage_lock:
            if success:
                self._usage["successful_calls"] += 1
            else:
                self._usage["failed_calls"] += 1

            self._usage["input_tokens"] += usage["input_tokens"]
            self._usage["output_tokens"] += usage["output_tokens"]
            self._usage["total_tokens"] += usage["total_tokens"]

    def _record_manual_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        success: bool,
    ) -> None:
        """Record usage for models without provider-side usage metadata."""
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)

        with self._usage_lock:
            if success:
                self._usage["successful_calls"] += 1
            else:
                self._usage["failed_calls"] += 1

            self._usage["input_tokens"] += input_tokens
            self._usage["output_tokens"] += output_tokens
            self._usage["total_tokens"] += input_tokens + output_tokens

    # Shared OpenAI-compatible helpers
    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str],
    ):
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    def _invoke_openai_compatible(
        self,
        client,
        provider_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Execute exactly one provider request.

        Retry belongs to the caller/stage layer so transport retry and
        semantic/JSON retry do not multiply each other.
        """
        messages = self._build_messages(prompt, system_prompt)

        self._record_api_call()

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
        except Exception:
            self._record_failure()
            logger.exception(
                "%s | request failed | model=%s",
                provider_name,
                self.model_name,
            )
            return None

        content = None

        if (
            response
            and getattr(response, "choices", None)
            and getattr(response.choices[0], "message", None)
        ):
            content = response.choices[0].message.content

        success = bool(content)
        self._record_response(response, success=success)

        if not success:
            logger.warning(
                "%s | empty response | model=%s",
                provider_name,
                self.model_name,
            )
            return None

        return content

    async def _ainvoke_openai_compatible(
        self,
        client,
        provider_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Async version of one OpenAI-compatible provider request."""
        messages = self._build_messages(prompt, system_prompt)

        self._record_api_call()

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
        except Exception:
            self._record_failure()
            logger.exception(
                "%s async | request failed | model=%s",
                provider_name,
                self.model_name,
            )
            return None

        content = None

        if (
            response
            and getattr(response, "choices", None)
            and getattr(response.choices[0], "message", None)
        ):
            content = response.choices[0].message.content

        success = bool(content)
        self._record_response(response, success=success)

        if not success:
            logger.warning(
                "%s async | empty response | model=%s",
                provider_name,
                self.model_name,
            )
            return None

        return content


# OpenAI-compatible clients
class OpenAILLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        base_url = os.getenv("OPENAI_BASE_URL")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        return self._invoke_openai_compatible(
            client=self.client,
            provider_name="OpenAI",
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        return await self._ainvoke_openai_compatible(
            client=self.async_client,
            provider_name="OpenAI",
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


class GeminiLLMClient(LLMClient):
    """
    Gemini client through an OpenAI-compatible endpoint.

    Required:
        GEMINI_API_KEY
        GEMINI_BASE_URL
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

        api_key = os.getenv("GEMINI_API_KEY")
        base_url = os.getenv("GEMINI_BASE_URL")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        if not base_url:
            raise ValueError("GEMINI_BASE_URL is not configured.")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        return self._invoke_openai_compatible(
            client=self.client,
            provider_name="Gemini",
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


class DeepseekLLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured.")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "DEEPSEEK_BASE_URL",
                "https://api.deepseek.com",
            ),
        )

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:

        return self._invoke_openai_compatible(
            client=self.client,
            provider_name="DeepSeek",
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


class HuggingFaceLLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        token = os.getenv("HUGGINGFACE_TOKEN")
        self.client = huggingface_hub.InferenceClient(token=token)

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        return self._invoke_openai_compatible(
            client=self.client,
            provider_name="HuggingFace",
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


# Anthropic
class AnthropicLLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        if anthropic is None:
            raise ImportError(
                "Install the optional 'anthropic' package to use AnthropicLLMClient."
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")

        self.client = anthropic.Anthropic(api_key=api_key)

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("max_tokens", 4096)

        if system_prompt:
            request_kwargs["system"] = system_prompt

        self._record_api_call()

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                **request_kwargs,
            )
        except Exception:
            self._record_failure()
            logger.exception(
                "Anthropic | request failed | model=%s",
                self.model_name,
            )
            return None

        text_parts = []

        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)

        content = "\n".join(text_parts).strip()
        success = bool(content)

        self._record_response(response, success=success)

        if not success:
            logger.warning(
                "Anthropic | empty response | model=%s",
                self.model_name,
            )
            return None

        return content


# Local Transformers
class LocalLLMClient(LLMClient):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name)

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs,
    ) -> Optional[str]:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        self._record_api_call()

        try:
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
            ).to(self.device)

            input_ids = inputs["input_ids"]
            input_tokens = int(input_ids.shape[-1])

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        except Exception:
            self._record_failure()
            logger.exception(
                "LocalLLM | generation failed | model=%s",
                self.model_name,
            )
            return None

        generated_ids = outputs[0]
        new_tokens = generated_ids[input_tokens:]
        output_tokens = int(new_tokens.shape[-1])

        content = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        success = bool(content)

        self._record_manual_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
        )

        if not success:
            logger.warning(
                "LocalLLM | empty response | model=%s",
                self.model_name,
            )
            return None

        return content
