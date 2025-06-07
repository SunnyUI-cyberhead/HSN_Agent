# File: hsn_agent_adk/adk/exceptions.py

class ModelLoadError(Exception):
    """Raised when the HSN classifier model or tokenizers fail to load."""
    pass


class OpenAIAPIError(Exception):
    """Raised when an OpenAI API call fails after retries."""
    pass


class HSNPredictorError(Exception):
    """General exception for prediction/inference issues."""
    pass
