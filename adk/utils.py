# File: hsn_agent_adk/adk/utils.py

import logging
import openai
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type
)
from .exceptions import OpenAIAPIError

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def validate_api_key():
    """
    Ensures that OPENAI_API_KEY is set in the environment.
    """
    if not openai.api_key:
        logger.error("OPENAI_API_KEY not found in environment.")
        raise OpenAIAPIError("Please set the OPENAI_API_KEY environment variable.")


@retry(
    reraise=True,
    wait=wait_exponential(min=1, max=20),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError))
)
def safe_chat_completion(**kwargs) -> dict:
    """
    A thin wrapper around openai.ChatCompletion.create with exponential backoff on rate limits
    or transient server errors. Returns the raw response dict.
    """
    try:
        logger.debug("Calling OpenAI ChatCompletion with args: %s", {k: v for k, v in kwargs.items() if k != "functions"})
        response = openai.ChatCompletion.create(**kwargs)
        return response
    except (openai.error.RateLimitError, openai.error.APIError) as e:
        logger.warning(f"OpenAI API error (will retry): {e}")
        raise
    except openai.error.InvalidRequestError as e:
        logger.error(f"Invalid request to OpenAI: {e}")
        raise OpenAIAPIError(f"Invalid request: {e}")
    except Exception as e:
        logger.exception("Unexpected error calling OpenAI")
        raise OpenAIAPIError(f"Unexpected error: {e}")
