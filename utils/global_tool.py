import logging
import time
from typing import Callable
import json
import os
import sys
from llms import LLMClient, OpenAILLMClient, DeepseekLLMClient

SUBJECT_EXAMPLE_NUM = 3
EXAMPLE_NUM = 10
title = "harmful_scenarios"
policy_title = "harmful_scenarios"


def get_llm_client(model_name: str) -> LLMClient:
    """Create the API client matching the supplied model name."""
    # Map model families to their API clients.
    model_map = {
        "gpt": OpenAILLMClient,
        "deepseek": DeepseekLLMClient,
    }

    # Select the client by model-name prefix.
    for prefix, client_class in model_map.items():
        if model_name.lower().startswith(prefix.lower()):
            return client_class(model_name=model_name)

    # Reject unsupported model families.
    raise ValueError(f"Unsupported model type: {model_name}")


def retry_operation_with_length_check(
    operation: Callable, expected_length: int, max_retries: int, operation_name: str
):
    """Retry an operation until its result has the expected length."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            result = operation()
            if len(result) != expected_length:
                raise ValueError(
                    f"Result length mismatch: expected {expected_length}, got {len(result)}"
                )

            logging.info(f"{operation_name} succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            last_exception = e
            logging.warning(
                f"Attempt {attempt + 1} failed for {operation_name}: {str(e)}"
            )
            time.sleep(1 + attempt)  # Increase the delay after each failed attempt.

    logging.error(f"Failed to complete {operation_name} after {max_retries} attempts")
    raise last_exception


# Persist intermediate results.
def load_save(response, file_path):
    # Resolve the output directory.
    directory = os.path.dirname(file_path)

    # Create the output directory if needed.
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Preserve readable indentation inside the JSON block.
    formatted_json = (
        "```json\n" + json.dumps(response, indent=4, ensure_ascii=False) + "\n```"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{formatted_json}")
    logging.info(f"Results saved to {file_path}")


# Configure console logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
